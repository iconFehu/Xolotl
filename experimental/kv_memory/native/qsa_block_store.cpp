#include "qsa_block_store.hpp"
#include "qsa_diagnostic_policy.hpp"
#include "qsa_compact_layout.hpp"
#include "qsa_hot_reset_contract.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/kernels/qsa.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <limits>
#include <map>
#include "qsa_operation_profile.hpp"
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <fstream>
#include <iomanip>

namespace strata::qsa_proto {
namespace {
constexpr int kSites = 3, kCount = 7;
constexpr int64_t kMaxDeletedCells = 65536;
constexpr uint64_t kBankLimit = 512ull * 1024 * 1024;
constexpr int64_t kDraftGuard = 4096;
const char* const kSiteNames[kSites] = {"decode", "verify", "prefill"};

struct Layer {
    const core::QsaState* state = nullptr;
    int64_t ordinal = 0;
    uint8_t* mask = nullptr;
    uint8_t* watch = nullptr;
    int32_t* steps = nullptr;
    uint64_t* counts[kSites] = {};
    std::vector<uint8_t> deleted;
    std::vector<uint8_t> watched;
};
struct LayerImage {
    int64_t ordinal = 0;
    bool rotated = false;
    int mode = 0;
    std::array<std::vector<uint8_t>, 5> data; // K,V,Kscale,Vscale,pooled
};
struct Group {
    std::string id, lineage;
    int64_t p0 = 0, p1 = 0, saved_tokens = 0;
    uint64_t bytes = 0, hash = 0;
    bool removed = false;
    std::vector<LayerImage> layers;
};
struct Registry {
    core::SessionState* session = nullptr;
    core::ModelGeometry geometry;
    const core::QsaState* draft = nullptr;
    int64_t max_queries = 0, mtp_window = -1, deleted_cells = 0;
    uint64_t bank_bytes = 0;
    bool active = false, bad = false;
    int64_t external_p0=0,external_p1=0,external_tokens=0;
    std::string external_lineage;
    std::vector<Layer> layers;
    std::unordered_map<const int32_t*, size_t> by_table;
    std::map<std::string, Group> groups;
};
Registry registry;

void cuda_check(cudaError_t e, const char* operation) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("qsa prototype ") + operation + ": " + cudaGetErrorString(e));
}
void sync() { cuda_check(cudaDeviceSynchronize(), "synchronize"); }
void copy(void* dst, const void* src, size_t n) {
    if (n) cuda_check(cudaMemcpy(dst, src, n, cudaMemcpyDefault), "copy");
}
template<class T> void allocate(T*& p, size_t count) {
    if (count > SIZE_MAX / sizeof(T)) throw std::runtime_error("qsa prototype allocation overflow");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)), "allocate");
    cuda_check(cudaMemset(p, 0, count * sizeof(T)), "initialize allocation");
}
OpResult failure(const std::string& error, bool bad = false) {
    if (bad) registry.bad = true;
    OpResult r; r.error = error; r.poisoned = registry.bad; return r;
}
OpResult success(const Group* group = nullptr) {
    OpResult r; r.ok = true; r.poisoned = registry.bad;
    if (group) { r.id = group->id; r.p0 = group->p0; r.p1 = group->p1;
        r.token_count = group->saved_tokens; r.bytes = group->bytes; r.payload_hash = group->hash; }
    return r;
}
void require_ready() {
    if (!registry.active) throw std::runtime_error("qsa prototype is not initialized");
    if (registry.bad) throw std::runtime_error("qsa prototype session is poisoned; inference must not continue");
}
void valid_identity(const std::string& id, const std::string& lineage) {
    if (id.empty() || id.size() > 64 || !std::all_of(id.begin(), id.end(), [](unsigned char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
               (c >= '0' && c <= '9') || c == '_' || c == '-'; })) throw std::runtime_error("invalid block group ID");
    if (lineage.empty() || lineage.size() > 4096) throw std::runtime_error("missing or oversized caller-verified lineage");
}
void valid_range(int64_t p0, int64_t p1, int64_t tokens) {
    if (tokens < 4096 || tokens > registry.session->max_cells || p0 < 4 || p1 <= p0 || p0 % 4 || p1 % 4 ||
        p1 > (tokens / 4 - 1) * 4) throw std::runtime_error("range must be complete old 4-cell blocks excluding block0/current tail");
    if (registry.draft) {
        if (registry.mtp_window <= 0) throw std::runtime_error("explicit finite MTP window must be configured before block operations");
        if (tokens < registry.mtp_window + kDraftGuard || p1 > tokens - registry.mtp_window - kDraftGuard)
            throw std::runtime_error("range overlaps protected MTP window/guard");
    }
}
std::array<size_t, 5> page_sizes() {
    const auto& g = registry.geometry;
    return {size_t(4 * g.n_head_kv * g.head_dim), size_t(4 * g.n_head_kv * g.head_dim),
            size_t(4 * g.n_head_kv * (g.head_dim / 64) * 2), size_t(4 * g.n_head_kv * (g.head_dim / 64) * 2),
            size_t(g.idx_key_dim * 4)};
}
std::array<void*, 5> pools(const core::QsaState& st) {
    if (st.kv_mode == 0) return {st.k_q, st.v_q, st.k_scale, st.v_scale, st.idx_pooled};
    return {st.host.k_q, st.host.v_q, st.host.k_scale, st.host.v_scale, st.idx_pooled};
}
void validate_state(const core::QsaState& st) {
    if (!st.kv_int8 || st.kv_q4 || st.kv_hybrid || (st.kv_mode != 0 && st.kv_mode != 1) ||
        !st.k_q || !st.v_q || !st.k_scale || !st.v_scale || !st.idx_pooled || !st.page_table ||
        !st.idx_tail || !st.idx_dead || !st.idx_block_pos || st.kv_elastic >= 0 ||
        st.max_cells != registry.session->max_cells || st.n_pages != (st.max_cells + 3) / 4 ||
        st.idx_pooled_rows < st.max_cells / 4 + 2 || st.n_slots <= 0)
        throw std::runtime_error("only complete INT8 main-QSA mode0/1 states are supported");
    if (st.kv_mode == 0 && st.n_slots != st.n_pages) throw std::runtime_error("mode0 is not identity-sized");
    if (st.kv_mode == 1 && (!st.host.k_q || !st.host.v_q || !st.host.k_scale || !st.host.v_scale ||
        st.n_slots < 5120 || st.n_slots >= st.n_pages ||
        st.map.page_table != st.page_table || st.map.n_blocks != st.n_pages || st.map.n_slots != st.n_slots ||
        !st.map.slot_block || !st.map.slot_stamp || !st.map.slot_ref || !st.map.ctl ||
        !st.map.miss_block || !st.map.miss_slot))
        throw std::runtime_error("invalid authoritative host pool or streaming map");
}
void validate_maps(int64_t b0, int64_t b1) {
    profiling::Phase phase("target_map_validation");
    for (const auto& layer : registry.layers) {
        const auto& st = *layer.state;
        std::vector<int32_t> pages(size_t(b1 - b0));
        copy(pages.data(), st.page_table + b0, pages.size() * sizeof(int32_t));
        std::vector<int32_t> owners;
        if(qsa_pool_v2::compact_enabled() && st.kv_mode==1){owners.resize(size_t(st.n_slots));copy(owners.data(),st.map.slot_block,owners.size()*sizeof(int32_t));}
        for (int64_t b = b0; b < b1; ++b) {
            const int32_t slot = pages[size_t(b - b0)];
            if (st.kv_mode == 0) { if (slot != b) throw std::runtime_error("mode0 page table is not identity"); }
            else if (slot < -1 || slot >= st.n_slots) throw std::runtime_error("streaming page is unresolved/busy or out of range");
            else if (slot >= 0) {
                int32_t reverse = -1; if(!owners.empty())reverse=owners[size_t(slot)];else copy(&reverse, st.map.slot_block + slot, sizeof(reverse));
                if (reverse != b) throw std::runtime_error("streaming residency map is inconsistent");
            }
        }
    }
}
uint64_t image_hash(const Group& group) {
    profiling::Phase phase("archive_checksum");
    core::SessionIdentityBuilder h(0x515341424c4f434bull);
    h.str("id", group.id); h.str("lineage", group.lineage);
    h.i64("p0", group.p0); h.i64("p1", group.p1); h.i64("saved_tokens", group.saved_tokens);
    for (const auto& layer : group.layers) {
        h.i64("layer", layer.ordinal); h.i64("mode", layer.mode); h.i64("rotation", layer.rotated);
        for (size_t i = 0; i < layer.data.size(); ++i) h.bytes(std::string("part") + std::to_string(i), layer.data[i].data(), layer.data[i].size());
    }
    return h.digest();
}
void validate_group(const Group& group, int64_t tokens, const std::string& lineage) {
    profiling::Phase phase("archive_shape_lineage_checks");
    valid_identity(group.id, lineage); valid_range(group.p0, group.p1, tokens);
    if (group.lineage != lineage || tokens < group.saved_tokens || group.layers.size() != registry.layers.size() || image_hash(group) != group.hash)
        throw std::runtime_error("archive lineage, extent, layer count or payload checksum mismatch");
    const auto sizes = page_sizes(); const size_t pages = size_t((group.p1 - group.p0) / 4);
    for (size_t i = 0; i < group.layers.size(); ++i) {
        const auto& saved = group.layers[i]; const auto& current = registry.layers[i];
        validate_state(*current.state);
        if (saved.ordinal != current.ordinal || saved.mode != current.state->kv_mode || saved.rotated != current.state->kv_rot)
            throw std::runtime_error("archive layer storage format mismatch");
        for (size_t p = 0; p < sizes.size(); ++p) if (saved.data[p].size() != pages * sizes[p])
            throw std::runtime_error("archive part size mismatch");
    }
}
bool same_active_payload(const Group& group, bool expect_zero, uint64_t* digest = nullptr) {
    profiling::Phase phase("target_payload_validation");
    core::SessionHasher hash(0x515341415554484full);
    const auto per = page_sizes(); const int64_t b0 = group.p0 / 4;
    for (size_t j = 0; j < registry.layers.size(); ++j) {
        const auto active = pools(*registry.layers[j].state);
        for (size_t p = 0; p < per.size(); ++p) {
            const auto& expected = group.layers[j].data[p];
            std::vector<uint8_t> current(expected.size());
            if((profiling::fast() || qsa_pool_v2::compact_enabled()) && registry.layers[j].state->kv_mode==1 && p<4) std::memcpy(current.data(),static_cast<uint8_t*>(active[p])+size_t(b0)*per[p],current.size());
            else copy(current.data(), static_cast<uint8_t*>(active[p]) + size_t(b0) * per[p], current.size());
            if (digest) { const uint64_t n=current.size(); hash.update(&n,sizeof n); hash.update(current.data(),current.size()); }
            if (expect_zero) { if (std::any_of(current.begin(), current.end(), [](uint8_t x) { return x != 0; })) return false; }
            else if (current != expected) return false;
        }
    }
    if (digest) *digest=hash.digest();
    return true;
}
void update_masks(int64_t b0, int64_t b1, uint8_t value) {
    profiling::Phase phase("availability_mask");
    for (auto& layer : registry.layers) {
        std::fill(layer.deleted.begin() + b0, layer.deleted.begin() + b1, value);
        copy(layer.mask + b0, layer.deleted.data() + b0, size_t(b1 - b0));
    }
}
void invalidate_target(int64_t b0, int64_t b1) {
    profiling::Phase phase("mapping_eviction");
    if(profiling::fast()) { (void)qsa_pool_v2::evict_range_fast(b0,b1); return; }
    if (qsa_pool_v2::initialized()) {
        (void)qsa_pool_v2::evict_range(b0, b1);
        return;
    }
    for (const auto& layer : registry.layers) {
        const auto& st = *layer.state;
        if (st.kv_mode == 1) detail::launch_invalidate(st.page_table, st.map.slot_block, st.map.slot_stamp,
            st.map.slot_ref, st.n_pages, st.n_slots, b0, b1, nullptr);
    }
}
uint64_t running_guard(int64_t tokens) {
    if(profiling::fast()) return 0;
    profiling::Phase phase("full_running_state_validation");
    const auto g = guard_fingerprint(*registry.session, registry.geometry, registry.draft, tokens);
    if (!g.ok) throw std::runtime_error(g.error);
    return g.combined;
}
}

bool initialized() noexcept { return registry.active; }
bool poisoned() noexcept { return registry.bad; }
void shutdown() noexcept {
    // Destruction cannot report an error, but it must not free captured inputs
    // while previously submitted work is still using them. No replay is legal
    // after this point, even if the owning graph object has not been destroyed.
    if (!registry.layers.empty()) cudaDeviceSynchronize();
    for (auto& layer : registry.layers) {
        if (layer.mask) cudaFree(layer.mask);
        if (layer.watch) cudaFree(layer.watch);
        if (layer.steps) cudaFree(layer.steps);
        for (auto* p : layer.counts) if (p) cudaFree(p);
    }
    registry = Registry{};
}

OpResult initialize(core::SessionState& ss, const core::ModelGeometry& g,
                    const core::QsaState* draft, int64_t max_queries) {
    if (registry.active || !registry.layers.empty()) return failure("already initialized; shutdown first");
    try {
        if (g.family != ModelFamily::Qwen4Exp || g.n_qsa_layers() != 12 || g.n_head_kv != 2 || g.head_dim != 256 ||
            g.idx_block != 4 || g.idx_key_dim != 128 || ss.layer_lo != 0 || ss.layer_hi != g.n_layers ||
            ss.qsa_ord0 != 0 || ss.qsa_alloc != 12 || !ss.qsa_states || !ss.gdn_state ||
            ss.gdn_ord0 != 0 || ss.gdn_alloc != g.n_gdn_layers() || ss.max_cells <= 0 || ss.max_cells > 524288 ||
            max_queries <= 0 || max_queries > 16384)
            throw std::runtime_error("prototype requires one complete Flash-Next session and max_queries in 1..16384");
        registry.session = &ss; registry.geometry = g; registry.draft = draft; registry.max_queries = max_queries;
        registry.layers.resize(size_t(ss.qsa_alloc));
        for (size_t j = 0; j < registry.layers.size(); ++j) {
            auto& layer = registry.layers[j]; layer.state = &ss.qsa_states[j]; layer.ordinal = int64_t(j);
            validate_state(*layer.state);
            layer.deleted.resize(size_t(layer.state->n_pages), 0);
            layer.watched.resize(size_t(layer.state->n_pages), 0);
            allocate(layer.mask, layer.deleted.size());
            allocate(layer.watch, layer.watched.size());
            allocate(layer.steps, size_t(max_queries) * strata::kernels::kStepCount);
            for (int site = 0; site < kSites; ++site) allocate(layer.counts[site], kCount);
            if (!registry.by_table.emplace(layer.state->page_table, j).second) throw std::runtime_error("duplicate main-QSA table identity");
        }
        registry.active = true;
        return success();
    } catch (const std::exception& e) { const std::string error = e.what(); shutdown(); return failure(error); }
}

OpResult configure_mtp_window(int64_t window) {
    try { require_ready(); if (!registry.groups.empty()) throw std::runtime_error("MTP window cannot change after a save");
        if (window <= 0 || window > registry.session->max_cells) throw std::runtime_error("finite positive MTP window required");
        registry.mtp_window = window; return success();
    } catch (const std::exception& e) { return failure(e.what()); }
}

const int32_t* filter_selection(const core::QsaState& state, int32_t* ids, const int32_t* steps,
                               int64_t nq, int64_t cap, void* stream, const char* site) {
    if (!registry.active) return steps;
    require_ready();
    const auto found = registry.by_table.find(state.page_table);
    if (found == registry.by_table.end()) throw std::runtime_error("unregistered QSA state in active prototype");
    if (!ids || !steps || nq <= 0 || nq > registry.max_queries || cap <= 0 || cap > 2051) throw std::runtime_error("invalid prototype filter extent");
    int which = -1;
    for (int i = 0; i < kSites; ++i) if (site && std::strcmp(site, kSiteNames[i]) == 0) which = i;
    if (which < 0) throw std::runtime_error("unknown prototype filter site");
    auto& layer = registry.layers[found->second];
    detail::launch_filter(layer.mask, layer.watch, state.n_pages, ids, steps, layer.steps, nq, cap, layer.counts[which], stream);
    return layer.steps;
}

std::vector<SiteCounters> counters() {
    require_ready(); sync(); std::vector<SiteCounters> out;
    for (const auto& layer : registry.layers) for (int i = 0; i < kSites; ++i) {
        uint64_t v[kCount] = {}; copy(v, layer.counts[i], sizeof(v));
        out.push_back({kSiteNames[i], layer.ordinal, v[0], v[1], v[2], v[3], v[4], v[5], v[6]});
    }
    return out;
}
void reset_counters() {
    require_ready(); sync();
    for (const auto& layer : registry.layers) for (auto* p : layer.counts) cuda_check(cudaMemset(p, 0, kCount * sizeof(uint64_t)), "reset counters");
}

OpResult discard_session_archives(){
 bool mutated=false;try{
  require_ready();sync();const auto per=page_sizes();uint64_t bytes=0;
  auto identity=[&]{core::SessionHasher h(0x484f544144445232ull);auto add=[&](const void* p){auto x=reinterpret_cast<uintptr_t>(p);h.update(&x,sizeof(x));};
   add(registry.session->gdn_state);add(registry.session->ple_hist);
   for(const auto& l:registry.layers){add(l.mask);add(l.watch);add(l.steps);for(auto* p:l.counts)add(p);add(l.state->page_table);add(l.state->idx_pooled);add(l.state->idx_tail);add(l.state->idx_block_pos);const auto p=pools(*l.state);for(auto* x:p)add(x);}
   if(registry.draft){const auto p=pools(*registry.draft);for(auto* x:p)add(x);add(registry.draft->page_table);}return h.digest();};
  const auto before=identity();
  auto zero_host=[&](const core::QsaState& st){
   const std::array<void*,4> ptr={st.host.k_q,st.host.v_q,st.host.k_scale,st.host.v_scale};
   for(size_t p=0;p<4;++p){const size_t n=size_t(st.n_pages)*per[p];std::memset(ptr[p],0,n);bytes+=n;}
  };
  for(const auto& l:registry.layers)qsa_hot_reset::validate_host(*l.state,registry.session->max_cells,false);
  if(!registry.draft)throw std::runtime_error("hot discard requires draft state");
  qsa_hot_reset::validate_host(*registry.draft,registry.session->max_cells,true);
  mutated=true;
  for(auto& l:registry.layers){
   zero_host(*l.state);std::fill(l.deleted.begin(),l.deleted.end(),0);std::fill(l.watched.begin(),l.watched.end(),0);
   cuda_check(cudaMemset(l.mask,0,l.deleted.size()),"discard mask");cuda_check(cudaMemset(l.watch,0,l.watched.size()),"discard watch");
   cuda_check(cudaMemset(l.steps,0,size_t(registry.max_queries)*strata::kernels::kStepCount*sizeof(int32_t)),"discard copied steps");
   for(auto* p:l.counts)cuda_check(cudaMemset(p,0,kCount*sizeof(uint64_t)),"discard selection counters");
  }
  if(!registry.draft)throw std::runtime_error("hot discard requires draft state");zero_host(*registry.draft);
  sync();registry.groups.clear();registry.bank_bytes=0;registry.deleted_cells=0;
  registry.external_p0=registry.external_p1=registry.external_tokens=0;registry.external_lineage.clear();
  auto r=success();r.bytes=bytes;r.running_before=before;r.running_after=identity();if(r.running_before!=r.running_after)throw std::runtime_error("registered graph address identity changed");return r;
 }catch(const std::exception& e){return failure(e.what(),mutated);}
}

OpResult save_by_id_impl(const std::string& id, int64_t p0, int64_t p1, int64_t tokens, const std::string& lineage) {
    bool instrumentation_mutated = false;
    try {
        require_ready(); valid_identity(id, lineage); valid_range(p0, p1, tokens);
        if (registry.groups.count(id)) throw std::runtime_error("block ID already exists");
        if (registry.groups.size() >= 64) throw std::runtime_error("RAM bank group limit reached");
        const int64_t b0 = p0 / 4, b1 = p1 / 4;
        for (const auto& layer : registry.layers) for (int64_t b = b0; b < b1; ++b)
            if (layer.deleted[size_t(b)]) throw std::runtime_error("cannot save an absent block");
        const auto per = page_sizes(); uint64_t bytes = 0;
        for (const auto n : per) bytes += uint64_t(b1 - b0) * n * registry.layers.size();
        if (bytes > kBankLimit || registry.bank_bytes > kBankLimit - bytes) throw std::runtime_error("RAM bank byte budget exceeded");
        sync(); validate_maps(b0, b1);
        const uint64_t guard = running_guard(tokens);
        Group group; group.id = id; group.lineage = lineage; group.p0 = p0; group.p1 = p1; group.saved_tokens = tokens; group.bytes = bytes;
        { profiling::Phase phase("archive_transfer");
        for (const auto& layer : registry.layers) {
            LayerImage image; image.ordinal = layer.ordinal; image.mode = layer.state->kv_mode; image.rotated = layer.state->kv_rot;
            const auto src = pools(*layer.state);
            for (size_t p = 0; p < per.size(); ++p) {
                image.data[p].resize(size_t(b1 - b0) * per[p]);
                if(profiling::fast() && layer.state->kv_mode==1 && p<4) std::memcpy(image.data[p].data(),static_cast<uint8_t*>(src[p])+size_t(b0)*per[p],image.data[p].size());
                else copy(image.data[p].data(), static_cast<uint8_t*>(src[p]) + size_t(b0) * per[p], image.data[p].size());
            }
            group.layers.push_back(std::move(image));
        }
        }
        group.hash = image_hash(group);
        const uint64_t after = running_guard(tokens);
        if (guard != after) return failure("save changed protected running state", true);
        instrumentation_mutated = true;
        for (auto& layer : registry.layers) {
            std::fill(layer.watched.begin() + b0, layer.watched.begin() + b1, 1);
            copy(layer.watch + b0, layer.watched.data() + b0, size_t(b1 - b0));
        }
        auto inserted = registry.groups.emplace(id, std::move(group)); registry.bank_bytes += bytes;
        if (qsa_pool_v2::initialized()) qsa_pool_v2::watch_range(b0, b1);
        OpResult r = success(&inserted.first->second); r.running_before = guard; r.running_after = after; return r;
    } catch (const std::exception& e) { return failure(e.what(), instrumentation_mutated); }
}

OpResult remove_by_id_impl(const std::string& id, int64_t tokens, const std::string& lineage) {
    bool mutated = false;
    try {
        require_ready(); const auto it = registry.groups.find(id);
        if (it == registry.groups.end()) throw std::runtime_error("unknown block ID");
        auto& group = it->second; validate_group(group, tokens, lineage);
        if (group.removed) throw std::runtime_error("block group already absent");
        const int64_t cells = group.p1 - group.p0, b0 = group.p0 / 4, b1 = group.p1 / 4;
        if (registry.deleted_cells + cells > kMaxDeletedCells) throw std::runtime_error("removed-cell cap exceeds 65536");
        for (const auto& layer : registry.layers) for (int64_t b = b0; b < b1; ++b)
            if (layer.deleted[size_t(b)]) throw std::runtime_error("overlapping absent group");
        sync(); validate_maps(b0, b1);
        if (!same_active_payload(group, false)) throw std::runtime_error("active block bytes changed since save");
        const uint64_t before = running_guard(tokens);
        const auto per = page_sizes(); const size_t largest = size_t(b1 - b0) * *std::max_element(per.begin(), per.end());
        std::vector<uint8_t> zeros(largest, 0);
        mutated = true; update_masks(b0, b1, 1); invalidate_target(b0, b1);
        { profiling::Phase phase("zero_target_transfer");
        for (const auto& layer : registry.layers) {
            const auto dst = pools(*layer.state);
            for (size_t p = 0; p < per.size(); ++p) {
                if(profiling::fast() && layer.state->kv_mode==1 && p<4) std::memset(static_cast<uint8_t*>(dst[p])+size_t(b0)*per[p],0,size_t(b1-b0)*per[p]);
                else copy(static_cast<uint8_t*>(dst[p]) + size_t(b0) * per[p], zeros.data(), size_t(b1 - b0) * per[p]);
            }
        }
        sync();
        }
        if (!same_active_payload(group, true)) throw std::runtime_error("remove did not erase all active authoritative bytes");
        const uint64_t after = running_guard(tokens);
        if (before != after) throw std::runtime_error("remove changed protected running state");
        group.removed = true; registry.deleted_cells += cells;
        auto r = success(&group); r.token_count = tokens; r.running_before = before; r.running_after = after; return r;
    } catch (const std::exception& e) { return failure(e.what(), mutated); }
}

OpResult load_by_id_impl(const std::string& id, int64_t tokens, const std::string& lineage) {
    bool mutated = false;
    try {
        require_ready(); const auto it = registry.groups.find(id);
        if (it == registry.groups.end()) throw std::runtime_error("unknown block ID");
        auto& group = it->second; validate_group(group, tokens, lineage);
        if (!group.removed) throw std::runtime_error("block group is already active");
        const int64_t b0 = group.p0 / 4, b1 = group.p1 / 4;
        for (const auto& layer : registry.layers) for (int64_t b = b0; b < b1; ++b)
            if (!layer.deleted[size_t(b)]) throw std::runtime_error("removed mask ownership mismatch");
        sync(); validate_maps(b0, b1);
        if (!same_active_payload(group, true)) throw std::runtime_error("absent block was silently repopulated");
        const uint64_t before = running_guard(tokens); const auto per = page_sizes();
        mutated = true; invalidate_target(b0, b1);
        { profiling::Phase phase("restore_target_transfer");
        for (size_t j = 0; j < registry.layers.size(); ++j) {
            const auto dst = pools(*registry.layers[j].state);
            for (size_t p = 0; p < per.size(); ++p) {
                const auto& data = group.layers[j].data[p];
                if((profiling::fast() || qsa_pool_v2::compact_enabled()) && registry.layers[j].state->kv_mode==1 && p<4) std::memcpy(static_cast<uint8_t*>(dst[p])+size_t(b0)*per[p],data.data(),data.size());
                else copy(static_cast<uint8_t*>(dst[p]) + size_t(b0) * per[p], data.data(), data.size());
            }
        }
        sync();
        }
        if (!same_active_payload(group, false)) throw std::runtime_error("loaded bytes do not match RAM archive");
        const uint64_t after = running_guard(tokens);
        if (before != after) throw std::runtime_error("load changed protected running state");
        update_masks(b0, b1, 0); sync(); // availability published only after all data checks pass
        group.removed = false; registry.deleted_cells -= group.p1 - group.p0;
        auto r = success(&group); r.token_count = tokens; r.running_before = before; r.running_after = after; return r;
    } catch (const std::exception& e) { return failure(e.what(), mutated); }
}

BankStatus bank_status(){require_ready();return {registry.bank_bytes,uint64_t(registry.groups.size()),uint64_t(registry.deleted_cells)};}
OpResult unsave_by_id_impl(const std::string& id,int64_t tokens,const std::string& lineage){
    try{require_ready();auto it=registry.groups.find(id);if(it==registry.groups.end())throw std::runtime_error("unknown block ID");
        const auto& group=it->second;validate_group(group,tokens,lineage);
        if(group.removed)throw std::runtime_error("cannot free an absent archive");
        sync();if(!same_active_payload(group,false))throw std::runtime_error("cannot free changed archive");
        const auto before=running_guard(tokens);auto r=success(&group);r.token_count=tokens;
        const auto after=running_guard(tokens);if(before!=after)throw std::runtime_error("free guard differs");
        r.running_before=before;r.running_after=after;
        registry.bank_bytes-=group.bytes;registry.groups.erase(it);return r;
    }catch(const std::exception& e){return failure(e.what(),false);}
}
std::vector<OpResult> inspect() {
    require_ready(); std::vector<OpResult> out;
    for (const auto& item : registry.groups) {
        auto r = success(&item.second); r.error = item.second.removed ? "absent" : "active"; out.push_back(std::move(r));
    }
    return out;
}

PoolByIdResult evict_by_id_impl(const std::string& id, int64_t tokens, const std::string& lineage) {
    PoolByIdResult out; bool mutated = false;
    try {
        require_ready();
        if (!qsa_pool_v2::initialized()) throw std::runtime_error("GPU-only by-ID eviction requires the independent mode1 pool");
        const auto it = registry.groups.find(id);
        if (it == registry.groups.end()) throw std::runtime_error("unknown block ID");
        const auto& group = it->second; validate_group(group, tokens, lineage);
        if (group.removed) throw std::runtime_error("cannot evict a logically absent group");
        sync();
        if (!same_active_payload(group, false)) throw std::runtime_error("authoritative bytes differ from archive before eviction");
        const auto before = running_guard(tokens);
        mutated = true;
        const auto e = qsa_pool_v2::evict_range(group.p0 / 4, group.p1 / 4);
        if (!same_active_payload(group, false)) throw std::runtime_error("GPU eviction changed authoritative RAM/index payload");
        const auto after = running_guard(tokens);
        if (before != after) throw std::runtime_error("GPU eviction changed GDN/PLE/index-running/draft guard");
        out.operation = success(&group); out.operation.token_count = tokens;
        out.operation.running_before = before; out.operation.running_after = after;
        out.before = e.before; out.after = e.after; out.released = e.released;
        out.authoritative_before = e.authoritative_before; out.authoritative_after = e.authoritative_after;
        out.exact_map_transition = e.exact_map_transition;
    } catch (const std::exception& e) { out.operation = failure(e.what(), mutated); }
    return out;
}

PoolByIdResult resident_by_id_impl(const std::string& id, int64_t tokens, const std::string& lineage) {
    PoolByIdResult out; bool mutated = false;
    int32_t *ids = nullptr, *steps = nullptr;
    try {
        require_ready();
        if (!qsa_pool_v2::initialized()) throw std::runtime_error("GPU prefetch requires the independent mode1 pool");
        const auto it = registry.groups.find(id);
        if (it == registry.groups.end()) throw std::runtime_error("unknown block ID");
        const auto& group = it->second; validate_group(group, tokens, lineage);
        if (group.removed) throw std::runtime_error("cannot prefetch a logically absent group");
        if (group.p1 - group.p0 > 2048) throw std::runtime_error("prefetch group exceeds the bounded selected-ID capacity");
        sync();
        if (!same_active_payload(group, false, &out.authoritative_before)) throw std::runtime_error("authoritative bytes differ from archive before prefetch");
        const auto before = running_guard(tokens);
        { profiling::Phase timer("full_pool_snapshot_validation"); out.before = qsa_pool_v2::snapshot(false,!qsa_pool_v2::compact_enabled()); }
        const int32_t cap = static_cast<int32_t>(group.p1 - group.p0);
        std::vector<int32_t> selection(size_t(cap), 0), descriptor(strata::kernels::kStepCount, 0);
        for (int32_t i = 0; i < cap; ++i) selection[size_t(i)] = int32_t(group.p0 + i);
        descriptor[0] = int32_t(tokens - 1);
        descriptor[strata::kernels::kStepNKv] = int32_t(tokens);
        descriptor[strata::kernels::kStepNBid] = int32_t(tokens / 4);
        descriptor[strata::kernels::kStepWidth] = cap;
        allocate(ids, selection.size()); allocate(steps, descriptor.size());
        copy(ids, selection.data(), selection.size()*sizeof(int32_t));
        copy(steps, descriptor.data(), descriptor.size()*sizeof(int32_t));
        mutated = true;
        const auto per = page_sizes();
        for (size_t j = 0; j < registry.layers.size(); ++j) {
            const auto& st = *registry.layers[j].state;
            { profiling::Phase phase("GPU_prefetch");
            core::qsa_kv_resolve(st, registry.geometry, ids, steps, 1, cap, nullptr);
            sync(); }
            profiling::Phase readback_phase("GPU_resident_byte_validation");
            sync();
            const std::array<const void*,4> resident = {st.k_q, st.v_q, st.k_scale, st.v_scale};
            if(qsa_pool_v2::compact_enabled()) {
                const int64_t first=group.p0/4,n=(group.p1-group.p0)/4;
                std::vector<int32_t> pages(size_t(n),-1),owners(size_t(st.n_slots),-1);
                copy(pages.data(),st.page_table+first,pages.size()*sizeof(int32_t));
                copy(owners.data(),st.map.slot_block,owners.size()*sizeof(int32_t));
                for(int64_t i=0;i<n;++i){auto slot=pages[size_t(i)];if(slot<0 || slot>=st.n_slots || owners[size_t(slot)]!=first+i)throw std::runtime_error("batched resident map mismatch");}
                for(size_t part=0;part<4;++part){
                    std::vector<uint8_t> actual(size_t(st.n_slots)*per[part]);
                    copy(actual.data(),resident[part],actual.size());
                    for(int64_t i=0;i<n;++i)if(std::memcmp(actual.data()+size_t(pages[size_t(i)])*per[part],group.layers[j].data[part].data()+size_t(i)*per[part],per[part]))throw std::runtime_error("batched resident bytes differ");
                }
            } else for (int64_t block = group.p0/4; block < group.p1/4; ++block) {
                int32_t slot = -1; copy(&slot, st.page_table+block, sizeof(slot));
                if (slot < 0 || slot >= st.n_slots) throw std::runtime_error("selected-ID prefetch did not create a resident mapping");
                int32_t reverse = -1; copy(&reverse, st.map.slot_block+slot, sizeof(reverse));
                if (reverse != block) throw std::runtime_error("prefetch mapping is not bidirectional");
                for (size_t part=0; part<4; ++part) {
                    std::vector<uint8_t> actual(per[part]);
                    copy(actual.data(), static_cast<const uint8_t*>(resident[part])+size_t(slot)*per[part], actual.size());
                    const auto& expected = group.layers[j].data[part];
                    const auto at = size_t(block-group.p0/4)*per[part];
                    if (!std::equal(actual.begin(), actual.end(), expected.begin()+at)) throw std::runtime_error("resident K/V code or scale differs from RAM archive");
                }
            }
        }
        { profiling::Phase timer("full_pool_snapshot_validation"); out.after = qsa_pool_v2::snapshot(false,!qsa_pool_v2::compact_enabled()); }
        if (!same_active_payload(group, false, &out.authoritative_after)) throw std::runtime_error("prefetch changed authoritative RAM/index payload");
        const auto after = running_guard(tokens);
        if (before != after) throw std::runtime_error("prefetch changed protected running state");
        out.operation = success(&group); out.operation.token_count = tokens;
        out.operation.running_before = before; out.operation.running_after = after;
        out.resident_verified = true;
    } catch (const std::exception& e) { out.operation = failure(e.what(), mutated); }
    if (ids) cudaFree(ids); if (steps) cudaFree(steps);
    return out;
}


template<class F> OpResult profiled_op(F&& f) {
    profiling::Session session;
    if(session.value.fast && !qsa_pool_v2::initialized()) throw std::runtime_error("fast control requires isolated mode1 pool");
    auto r=f();r.profile=session.finish();r.guard_checked=r.ok && !r.profile.fast;return r;
}
OpResult save_by_id(const std::string& id,int64_t p0,int64_t p1,int64_t tokens,const std::string& lineage) {
    return profiled_op([&]{return save_by_id_impl(id,p0,p1,tokens,lineage);});
}
OpResult unsave_by_id(const std::string& id,int64_t tokens,const std::string& lineage){return profiled_op([&]{return unsave_by_id_impl(id,tokens,lineage);});}
OpResult remove_by_id(const std::string& id,int64_t tokens,const std::string& lineage) {
    return profiled_op([&]{return remove_by_id_impl(id,tokens,lineage);});
}
OpResult load_by_id(const std::string& id,int64_t tokens,const std::string& lineage) {
    return profiled_op([&]{return load_by_id_impl(id,tokens,lineage);});
}
PoolByIdResult evict_by_id(const std::string& id,int64_t tokens,const std::string& lineage) {
    profiling::Session session(true);auto r=evict_by_id_impl(id,tokens,lineage);
    r.operation.profile=session.finish();r.operation.guard_checked=r.operation.ok;return r;
}
PoolByIdResult resident_by_id(const std::string& id,int64_t tokens,const std::string& lineage) {
    profiling::Session session(true);auto r=resident_by_id_impl(id,tokens,lineage);
    r.operation.profile=session.finish();r.operation.guard_checked=r.operation.ok;return r;
}

namespace {
void exact_read(std::istream& f,void* data,size_t n){if(n)f.read(static_cast<char*>(data),n);if(!f)throw std::runtime_error("persistent group truncated");}
void verify_zero_kv(int64_t p0,int64_t p1){
 const auto per=page_sizes();std::vector<uint8_t> buffer;
 for(const auto& l:registry.layers){const auto src=pools(*l.state);for(size_t p=0;p<4;++p){buffer.resize(size_t((p1-p0)/4)*per[p]);copy(buffer.data(),static_cast<const uint8_t*>(src[p])+size_t(p0/4)*per[p],buffer.size());if(std::any_of(buffer.begin(),buffer.end(),[](uint8_t x){return x!=0;}))throw std::runtime_error("external missing KV must be zero");}}
}
}
ExternalStatus external_status(){return {registry.external_tokens>0,registry.external_p0,registry.external_p1,registry.external_tokens,registry.deleted_cells};}
OpResult external_begin(int64_t p0,int64_t p1,int64_t tokens,const std::string& lineage){
 return profiled_op([&]{bool mutated=false;try{
  require_ready();valid_identity("external",lineage);valid_range(p0,p1,tokens);
  if(registry.external_tokens||!registry.groups.empty()||registry.deleted_cells||p1-p0>kMaxDeletedCells)throw std::runtime_error("external mask requires a fresh empty bank");
  sync();verify_zero_kv(p0,p1);validate_maps(p0/4,p1/4);
  mutated=true;invalidate_target(p0/4,p1/4);update_masks(p0/4,p1/4,1);sync();
  registry.external_p0=p0;registry.external_p1=p1;registry.external_tokens=tokens;registry.external_lineage=lineage;registry.deleted_cells=p1-p0;
  auto r=success();r.p0=p0;r.p1=p1;r.token_count=tokens;return r;
 }catch(const std::exception& e){return failure(e.what(),mutated);}});
}
OpResult export_group(const std::string& id,const std::string& file,int64_t tokens,const std::string& lineage){
 return profiled_op([&]{try{
  require_ready();auto it=registry.groups.find(id);if(it==registry.groups.end()||it->second.removed)throw std::runtime_error("export requires a present saved group");
  const auto& x=it->second;validate_group(x,tokens,lineage);if(x.p1-x.p0>2048)throw std::runtime_error("persistent group cap 2048");
  std::ofstream f(file,std::ios::binary|std::ios::trunc);if(!f)throw std::runtime_error("group output open");
  f<<"QSA_GROUP_V1 "<<std::quoted(x.id)<<' '<<std::quoted(x.lineage)<<' '<<x.p0<<' '<<x.p1<<' '<<x.saved_tokens<<' '<<x.bytes<<' '<<x.hash<<' '<<x.layers.size()<<'\n';
  for(const auto& l:x.layers){f<<l.ordinal<<' '<<int(l.rotated)<<' '<<l.mode;for(const auto& d:l.data)f<<' '<<d.size();f<<'\n';for(const auto& d:l.data)f.write(reinterpret_cast<const char*>(d.data()),d.size());}
  f.flush();if(!f)throw std::runtime_error("group output write");return success(&x);
 }catch(const std::exception& e){return failure(e.what());}});
}
OpResult import_group(const std::string& file,int64_t tokens,const std::string& lineage){
 return profiled_op([&]{bool mutated=false;try{
  require_ready();if(!registry.external_tokens||tokens!=registry.external_tokens||lineage!=registry.external_lineage||!registry.groups.empty())throw std::runtime_error("persistent import needs the matching external base");
  std::ifstream f(file,std::ios::binary);std::string magic;Group x;size_t layers=0;
  f>>magic>>std::quoted(x.id)>>std::quoted(x.lineage)>>x.p0>>x.p1>>x.saved_tokens>>x.bytes>>x.hash>>layers;
  if(!f||magic!="QSA_GROUP_V1"||layers!=registry.layers.size()||x.p0<registry.external_p0||x.p1>registry.external_p1||x.p1-x.p0>2048||x.saved_tokens!=tokens||f.get()!='\n')throw std::runtime_error("persistent group header mismatch");
  valid_identity(x.id,x.lineage);valid_range(x.p0,x.p1,tokens);const auto per=page_sizes();uint64_t bytes=0;
  for(size_t j=0;j<layers;++j){LayerImage l;int rot=-1;std::array<uint64_t,5> sizes{};f>>l.ordinal>>rot>>l.mode;for(auto& n:sizes)f>>n;
   if(!f||l.ordinal!=int64_t(j)||(rot!=0&&rot!=1)||f.get()!='\n')throw std::runtime_error("persistent layer metadata");l.rotated=rot==1;
   for(size_t p=0;p<5;++p){const uint64_t want=uint64_t((x.p1-x.p0)/4)*per[p];if(sizes[p]!=want)throw std::runtime_error("persistent group part bound");l.data[p].resize(size_t(want));exact_read(f,l.data[p].data(),size_t(want));bytes+=want;}x.layers.push_back(std::move(l));}
  if(f.peek()!=EOF||bytes!=x.bytes)throw std::runtime_error("persistent group trailing/size mismatch");validate_group(x,tokens,lineage);
  for(const auto& l:registry.layers)for(int64_t b=x.p0/4;b<x.p1/4;++b)if(!l.deleted[size_t(b)])throw std::runtime_error("persistent group already loaded");
  sync();validate_maps(x.p0/4,x.p1/4);verify_zero_kv(x.p0,x.p1);
  for(size_t j=0;j<layers;++j){auto src=pools(*registry.layers[j].state);const auto& expected=x.layers[j].data[4];std::vector<uint8_t> actual(expected.size());copy(actual.data(),static_cast<uint8_t*>(src[4])+size_t(x.p0/4)*per[4],actual.size());if(actual!=expected)throw std::runtime_error("persistent pooled index dependency mismatch");}
  mutated=true;invalidate_target(x.p0/4,x.p1/4);
  for(size_t j=0;j<layers;++j){auto dst=pools(*registry.layers[j].state);for(size_t p=0;p<4;++p)copy(static_cast<uint8_t*>(dst[p])+size_t(x.p0/4)*per[p],x.layers[j].data[p].data(),x.layers[j].data[p].size());}
  sync();if(!same_active_payload(x,false))throw std::runtime_error("persistent import payload differs");update_masks(x.p0/4,x.p1/4,0);sync();registry.deleted_cells-=x.p1-x.p0;return success(&x);
 }catch(const std::exception& e){return failure(e.what(),mutated);}});
}

bool compact_validate(const core::SavedConversation& image,std::string& error){
 try{
  require_ready();const auto& g=registry.geometry;auto& ss=*registry.session;
  const std::array<int64_t,18> expected={g.n_embd,g.n_layers,g.qsa_interval,g.ssm_state_size,g.ssm_k_heads,g.ssm_v_heads,g.ssm_d_conv,g.ssm_conv_channels,g.ssm_value_dim,g.n_head,g.n_head_kv,g.head_dim,g.idx_q_heads,g.idx_key_dim,g.hc,g.hc_lr,g.n_expert,g.n_ff};
  const auto n=image.live.ids.size();
  if(image.geometry!=expected||image.layer_lo!=ss.layer_lo||image.layer_hi!=ss.layer_hi||!image.stage_images.empty()||!image.live.stage_parts.empty()||!image.live.imgs.empty()||n<106496||n>size_t(ss.max_cells)||image.kv.size()!=registry.layers.size()+1||image.checkpoints.empty()||registry.external_tokens||!registry.groups.empty()||registry.deleted_cells)throw std::runtime_error("compact base metadata or target admission mismatch");
  if(std::any_of(image.live.ids.begin(),image.live.ids.end(),[](int32_t x){return x<0;}))throw std::runtime_error("compact negative token ID");
  if(!core::conversation_checkpoint_validate(image.live,ss,g,error))throw std::runtime_error(error);
  for(const auto& c:image.checkpoints){if(!c.stage_parts.empty()||!c.imgs.empty()||c.ids.size()<66048||c.ids.size()>n||!std::equal(c.ids.begin(),c.ids.end(),image.live.ids.begin())||!core::conversation_checkpoint_validate(c,ss,g,error))throw std::runtime_error("compact checkpoint metadata/prefix mismatch: "+error);}
  const uint64_t pages=(n+3)/4;
  for(size_t j=0;j<registry.layers.size();++j){const auto& st=*registry.layers[j].state;validate_state(st);if(st.kv_mode!=1)throw std::runtime_error("compact base requires authoritative host pools");
   core::SessionKvSource source;if(!core::conversation_kv_source(source,st,g,int64_t(n),true,error))throw std::runtime_error(error);
   const auto& k=image.kv[j];if(k.format!=source.format||k.cells!=source.cells||k.heads!=source.heads||k.head_dim!=source.head_dim||k.page_size!=source.page_size||k.pooled_rows!=source.pooled_rows||k.idx_dim!=source.idx_dim)throw std::runtime_error("compact layer format mismatch");
   const std::array<const core::ConversationBuffer*,5> parts={&k.k,&k.v,&k.k_scale,&k.v_scale,&k.pooled};
   for(size_t p=0;p<5;++p){const uint64_t want=p<4?qsa_compact::main_part(source.sizes[p],pages).size():source.sizes[p];if(parts[p]->size()!=want)throw std::runtime_error("compact layer part size mismatch");}
  }
  if(!registry.draft||!core::conversation_kv_validate(image.kv.back(),*registry.draft,g,int64_t(n),false,error))throw std::runtime_error("compact draft dependency mismatch: "+error);
  return true;
 }catch(const std::exception& e){error=e.what();return false;}
}
OpResult compact_restore(const core::SavedConversation& image){
 return profiled_op([&]{bool mutated=false;try{
  std::string error;if(!compact_validate(image,error))throw std::runtime_error(error);sync();
  const auto& g=registry.geometry;auto& ss=*registry.session;const int64_t n=int64_t(image.live.ids.size());const uint64_t pages=(n+3)/4;
  mutated=true;for(const auto& layer:registry.layers)strata::kernels::kv_stream_reset(layer.state->map,nullptr);sync();
  for(size_t j=0;j<registry.layers.size();++j){const auto& st=*registry.layers[j].state;auto dst=pools(st);core::SessionKvSource shape;if(!core::conversation_kv_source(shape,st,g,n,true,error))throw std::runtime_error(error);
   const auto& k=image.kv[j];const std::array<const core::ConversationBuffer*,5> parts={&k.k,&k.v,&k.k_scale,&k.v_scale,&k.pooled};
   for(size_t p=0;p<5;++p){
    if(p<4){const auto layout=qsa_compact::main_part(shape.sizes[p],pages);std::memset(static_cast<uint8_t*>(dst[p])+layout.lo,0,size_t(layout.gap()));
     parts[p]->visit(0,parts[p]->size(),[&](const uint8_t* data,size_t bytes,size_t at){for(auto x:layout.slices(at,bytes))copy(static_cast<uint8_t*>(dst[p])+x.original,data+(x.packed-at),size_t(x.bytes));return true;});
    }else parts[p]->visit(0,parts[p]->size(),[&](const uint8_t* data,size_t bytes,size_t at){copy(static_cast<uint8_t*>(dst[p])+at,data,bytes);return true;});
   }
  }
  if(!core::conversation_kv_restore(image.kv.back(),*registry.draft,g,n,false,error)||!core::conversation_checkpoint_restore(image.live,ss,g,error))throw std::runtime_error(error);
  sync();return success();
 }catch(const std::exception& e){return failure(e.what(),mutated);}});
}

uint64_t main_payload_hash(int64_t tokens) {
    require_ready();
    if (tokens<0 || tokens>registry.session->max_cells) throw std::runtime_error("invalid full-main fingerprint extent");
    sync(); core::SessionHasher hash(0x5153414d41494e32ull);
    const auto per=page_sizes(); const uint64_t pages=uint64_t(tokens/4), tail=uint64_t(tokens%4);
    std::vector<uint8_t> buffer(1024*1024);
    auto add=[&](const uint8_t* ptr,uint64_t bytes){
        hash.update(&bytes,sizeof bytes);
        for(uint64_t offset=0;offset<bytes;){const size_t n=size_t(std::min<uint64_t>(buffer.size(),bytes-offset));copy(buffer.data(),ptr+offset,n);hash.update(buffer.data(),n);offset+=n;}
    };
    hash.update(&tokens,sizeof tokens);
    for(const auto& layer:registry.layers){
        hash.update(&layer.ordinal,sizeof layer.ordinal);const auto src=pools(*layer.state);
        for(size_t part=0;part<4;++part){
            const auto* ptr=static_cast<const uint8_t*>(src[part]);add(ptr,pages*per[part]);
            for(int64_t head=0;head<registry.geometry.n_head_kv;++head){
                const uint64_t head_page=per[part]/uint64_t(registry.geometry.n_head_kv);
                add(ptr+pages*per[part]+uint64_t(head)*head_page,tail*(head_page/4));
            }
        }
        add(static_cast<const uint8_t*>(src[4]),pages*per[4]);
    }
    return hash.digest();
}

GuardFingerprint guard_fingerprint(const core::SessionState& ss, const core::ModelGeometry& g,
                                   const core::QsaState* draft, int64_t upto) {
    GuardFingerprint out;++strata::qsa_diagnostic_policy::guards;
    try {
        if (upto < 0 || upto > ss.max_cells) throw std::runtime_error("invalid guard extent");
        sync(); core::ConversationCheckpoint checkpoint; std::string error;
        if (!core::conversation_checkpoint_save(checkpoint, ss, g, error)) throw std::runtime_error(error);
        out.gdn = core::session_hash64(checkpoint.gdn.data(), checkpoint.gdn.size(), 1);
        core::SessionHasher ple(2); ple.update(checkpoint.ple.data(), checkpoint.ple.size());
        ple.update(ss.ple_prev, sizeof(ss.ple_prev)); ple.update(&ss.ple_token, sizeof(ss.ple_token)); out.ple = ple.digest();
        core::SessionHasher index(3); index.update(checkpoint.tails.data(), checkpoint.tails.size());
        index.update(checkpoint.dead.data(), checkpoint.dead.size()); index.update(checkpoint.block_pos.data(), checkpoint.block_pos.size());
        out.index_running = index.digest();
        if (draft && upto > 0) {
            core::SessionKvSource source;
            if (!core::conversation_kv_source(source, *draft, g, upto, false, error)) throw std::runtime_error(error);
            std::array<uint8_t, 65536> buffer; core::SessionHasher hash(4);
            for (size_t part = 0; part < source.sizes.size(); ++part) {
                const uint64_t size = source.sizes[part]; hash.update(&size, sizeof(size));
                for (uint64_t at = 0; at < size;) {
                    const size_t n = size_t(std::min<uint64_t>(buffer.size(), size - at));
                    if (!source.read(part, size_t(at), buffer.data(), n)) throw std::runtime_error(source.error());
                    hash.update(buffer.data(), n); at += n;
                }
            }
            out.draft = hash.digest();
        }
        core::SessionHasher all(5); all.update(&out.gdn, sizeof(out.gdn)); all.update(&out.ple, sizeof(out.ple));
        all.update(&out.index_running, sizeof(out.index_running)); all.update(&out.draft, sizeof(out.draft));
        out.combined = all.digest(); out.ok = true;
    } catch (const std::exception& e) { out.error = e.what(); }
    return out;
}
} // namespace strata::qsa_proto
