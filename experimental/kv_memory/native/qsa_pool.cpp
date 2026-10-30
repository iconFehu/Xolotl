#include "qsa_pool.hpp"
#include "strata/kernels/kv_q8.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

namespace strata::qsa_pool_v2 {
namespace {
struct RegisteredLayer { const core::QsaState* state = nullptr; int64_t ordinal = 0; };
std::vector<RegisteredLayer> layers;
std::set<int32_t> watched_blocks;
core::ModelGeometry geometry;
bool active = false;
void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string("QSA pool v2 ") + what + ": " + cudaGetErrorString(status));
}
void fence() { check(cudaDeviceSynchronize(), "idle fence"); }
void require() { if (!active) throw std::runtime_error("QSA pool v2 is not initialized"); }
template<class T> std::vector<T> read(const T* p, int64_t count) {
    if (!p || count < 0 || uint64_t(count) > SIZE_MAX / sizeof(T)) throw std::runtime_error("invalid pool diagnostic read");
    std::vector<T> out(size_t(count), T{});
    if (count) check(cudaMemcpy(out.data(), p, size_t(count) * sizeof(T), cudaMemcpyDefault), "diagnostic read");
    return out;
}
uint64_t hash_bytes(uint64_t hash, const void* bytes, size_t size) {
    const auto* p = static_cast<const uint8_t*>(bytes);
    for (size_t i = 0; i < size; ++i) { hash ^= p[i]; hash *= 1099511628211ull; }
    return hash;
}
uint64_t per_block() { return strata::kernels::kv_block_bytes(core::qsa_shapes(geometry), strata::kernels::kKvInt8); }
void validate(const core::QsaState& st) {
    if (st.kv_mode != 1 || !st.kv_int8 || st.kv_q4 || st.kv_hybrid || st.kv_elastic >= 0 ||
        st.max_cells != 524288 || st.n_pages != 131072 || st.n_slots < 5120 || st.n_slots >= st.n_pages ||
        !st.k_q || !st.v_q || !st.k_scale || !st.v_scale || !st.idx_pooled || st.idx_pooled_rows < st.n_pages + 2 ||
        !st.host.k_q || !st.host.v_q || !st.host.k_scale || !st.host.v_scale ||
        st.map.page_table != st.page_table || st.map.n_blocks != st.n_pages || st.map.n_slots != st.n_slots ||
        !st.map.slot_block || !st.map.slot_stamp || !st.map.slot_ref || !st.map.miss_block || !st.map.miss_slot || !st.map.ctl)
        throw std::runtime_error("QSA pool v2 requires complete fixed-capacity INT8 mode1 SC117 state");
}
LayerSnapshot layer_snapshot(const RegisteredLayer& entry, bool events, bool details) {
    const auto& st = *entry.state; validate(st);
    const auto pages = read(st.page_table, st.n_pages);
    const auto owners = read(st.map.slot_block, st.n_slots);
    const auto refs = read(st.map.slot_ref, st.n_slots);
    const auto control = read(st.map.ctl, strata::kernels::kKvCtlInts);
    std::vector<int32_t> seen(size_t(st.n_slots), 0);
    LayerSnapshot out; out.ordinal = entry.ordinal; out.blocks = st.n_pages; out.slots = st.n_slots;
    out.slot_pool_bytes = uint64_t(st.n_slots) * per_block();
    out.authoritative_host_bytes = uint64_t(st.n_pages) * per_block();
    out.pooled_index_bytes = uint64_t(st.idx_pooled_rows) * geometry.idx_key_dim * sizeof(float);
    out.map_hash = hash_bytes(14695981039346656037ull, pages.data(), pages.size() * sizeof(int32_t));
    out.map_hash = hash_bytes(out.map_hash, owners.data(), owners.size() * sizeof(int32_t));
    for (int64_t b = 0; b < st.n_pages; ++b) {
        const int32_t sl = pages[size_t(b)];
        if (sl == -1) continue;
        if (sl == -2) { ++out.transient_pages; continue; }
        if (sl < 0 || sl >= st.n_slots) { ++out.invalid_forward; continue; }
        ++out.mapped_blocks;
        if (++seen[size_t(sl)] > 1) ++out.duplicate_owners;
        if (owners[size_t(sl)] != b) ++out.invalid_forward;
    }
    for (int64_t sl = 0; sl < st.n_slots; ++sl) {
        const int32_t b = owners[size_t(sl)];
        if (b == -1) {
            ++out.free_slots;
            if (seen[size_t(sl)] || refs[size_t(sl)] != 0) ++out.invalid_reverse;
        } else {
            ++out.occupied_slots;
            if (b < 0 || b >= st.n_pages || pages[size_t(b)] != sl || seen[size_t(sl)] != 1)
                ++out.invalid_reverse;
            if (refs[size_t(sl)] < 0 || refs[size_t(sl)] > 1) ++out.invalid_reverse;
        }
    }
    out.overflow = control[3] != 0;
    out.map_ok = !out.invalid_forward && !out.invalid_reverse && !out.duplicate_owners && !out.transient_pages &&
        !out.overflow && out.mapped_blocks == out.occupied_slots && out.free_slots + out.occupied_slots == st.n_slots;
    out.trace = trace_read(st.page_table, events);
    out.details_included=details;
    if (!events) out.trace.events.clear();
    if (!watched_blocks.empty()) {
        const auto view = trace_for(st.page_table);
        if (!view.block_completed_copies) throw std::runtime_error("pool observer lacks complete block copy counters");
        const auto completed = read(view.block_completed_copies, st.n_pages);
        if(details) out.watched_blocks.reserve(watched_blocks.size());
        for(const int32_t block:watched_blocks) {
            const auto count=completed[size_t(block)]; ++out.watched_count;out.watched_nonzero+=(count>0);out.watched_copy_sum+=count;
            out.watched_id_hash=hash_bytes(out.watched_id_hash,&block,sizeof(block));
            out.watched_copy_hash=hash_bytes(out.watched_copy_hash,&block,sizeof(block));
            out.watched_copy_hash=hash_bytes(out.watched_copy_hash,&count,sizeof(count));
            if(details)out.watched_blocks.push_back({block,count});
        }
    }
    return out;
}
uint64_t authoritative_hash(int64_t b0, int64_t b1) {
    const uint64_t codes = 4ull * geometry.n_head_kv * geometry.head_dim;
    const uint64_t scales = 4ull * geometry.n_head_kv * (geometry.head_dim / 64) * sizeof(uint16_t);
    const uint64_t index = uint64_t(geometry.idx_key_dim) * sizeof(float);
    const std::array<uint64_t, 5> per = {codes, codes, scales, scales, index};
    std::array<uint8_t, 65536> buffer{};
    uint64_t hash = 14695981039346656037ull;
    hash = hash_bytes(hash, &b0, sizeof(b0)); hash = hash_bytes(hash, &b1, sizeof(b1));
    for (const auto& entry : layers) {
        const auto& st = *entry.state;
        const std::array<const void*, 5> parts = {st.host.k_q, st.host.v_q, st.host.k_scale, st.host.v_scale, st.idx_pooled};
        hash = hash_bytes(hash, &entry.ordinal, sizeof(entry.ordinal));
        for (size_t p = 0; p < per.size(); ++p) {
            const uint64_t bytes = uint64_t(b1 - b0) * per[p];
            hash = hash_bytes(hash, &p, sizeof(p)); hash = hash_bytes(hash, &bytes, sizeof(bytes));
            const auto* src = static_cast<const uint8_t*>(parts[p]) + uint64_t(b0) * per[p];
            for (uint64_t at = 0; at < bytes;) {
                const size_t n = size_t(std::min<uint64_t>(buffer.size(), bytes - at));
                check(cudaMemcpy(buffer.data(), src + at, n, cudaMemcpyDefault), "authoritative byte read");
                hash = hash_bytes(hash, buffer.data(), n); at += n;
            }
        }
    }
    return hash;
}
void require_maps(const std::vector<LayerSnapshot>& all) {
    if (std::any_of(all.begin(), all.end(), [](const LayerSnapshot& s) { return !s.map_ok; }))
        throw std::runtime_error("QSA pool v2 observed invalid/busy/overflow bidirectional residency mapping");
}
}

bool initialized() noexcept { return active; }
void initialize(core::SessionState& session, const core::ModelGeometry& g, uint64_t capacity) {
    if (active || !layers.empty()) throw std::runtime_error("pool v2 already initialized");
    if (g.family != ModelFamily::Qwen4Exp || g.n_qsa_layers() != 12 || g.n_head_kv != 2 || g.head_dim != 256 ||
        g.idx_block != 4 || g.idx_key_dim != 128 || session.max_cells != 524288 || session.qsa_alloc != 12 ||
        session.qsa_ord0 != 0 || session.layer_lo != 0 || session.layer_hi != g.n_layers || !session.qsa_states)
        throw std::runtime_error("pool v2 requires the complete one-GPU 512K Flash-Next session");
    geometry = g;
    try {
        fence();
        for (int64_t j = 0; j < session.qsa_alloc; ++j) {
            const auto& st = session.qsa_states[j]; validate(st);
            layers.push_back({&st, j});
            trace_register(st.map, per_block(), capacity);
        }
        active = true;
        auto initial = snapshot(false); require_maps(initial);
        if (std::any_of(initial.begin(), initial.end(), [](const LayerSnapshot& s) { return s.occupied_slots != 0; }))
            throw std::runtime_error("register pool observers before first forward/capture, while every map is empty");
    } catch (...) { shutdown(); throw; }
}
void shutdown() noexcept {
    if (!layers.empty()) cudaDeviceSynchronize();
    for (const auto& entry : layers) if (entry.state) trace_unregister(entry.state->page_table);
    layers.clear(); watched_blocks.clear(); active = false;
}
std::vector<LayerSnapshot> snapshot(bool events, bool details) {
    require(); fence();
    std::vector<LayerSnapshot> out; out.reserve(layers.size());
    for (const auto& entry : layers) out.push_back(layer_snapshot(entry, events, details));
    return out;
}
void reset_trace() {
    require(); fence();
    for (const auto& entry : layers) trace_reset(entry.state->page_table);
}
void discard_watched(){require();fence();watched_blocks.clear();reset_trace();}
void watch_range(int64_t b0, int64_t b1) {
    require();
    if (b0 < 0 || b1 <= b0 || b1 > layers.front().state->n_pages)
        throw std::runtime_error("invalid original-position watched block range");
    // Stage the complete union first: an allocation failure cannot leave a
    // partial watched set published. No device allocation or upload is needed.
    auto next = watched_blocks;
    for (int64_t block = b0; block < b1; ++block) next.insert(int32_t(block));
    watched_blocks.swap(next);
}

// Opt-in control fast path. Targeted bidirectional maps and all reverse owners
// are verified. Full CLOCK/table/stamp fingerprints remain in evict_range().
int64_t evict_map_fast(const strata::kernels::KvStreamMap& map,int64_t b0,int64_t b1) {
    if(b0<1 || b1<=b0 || b1>map.n_blocks) throw std::runtime_error("invalid fast map extent");
    fence();int64_t released=0;
        const auto pages=read(map.page_table+b0,b1-b0);
        const auto owners=read(map.slot_block,map.n_slots);
        auto expected=owners;
        for(int64_t b=b0;b<b1;++b) {
            const auto slot=pages[size_t(b-b0)];
            if(slot < -1 || slot>=map.n_slots) throw std::runtime_error("busy/invalid fast eviction map");
            if(slot>=0) {
                if(owners[size_t(slot)]!=b) throw std::runtime_error("fast eviction reverse map mismatch");
                expected[size_t(slot)]=-1; ++released;
            }
        }
        for(int64_t slot=0;slot<map.n_slots;++slot) {
            const auto b=owners[size_t(slot)];
            if(b>=b0 && b<b1 && pages[size_t(b-b0)]!=slot) throw std::runtime_error("orphan fast eviction owner");
        }
        if(released==0) return 0; // Target and reverse owners already verified.
        trace_evict_range(map,b0,b1,nullptr); fence();
        const auto after=read(map.page_table+b0,b1-b0);
        if(std::any_of(after.begin(),after.end(),[](int32_t v){return v!=-1;}) ||
           read(map.slot_block,map.n_slots)!=expected) throw std::runtime_error("fast eviction changed wrong owner or retained target");
    return released;
}
int64_t evict_range_fast(int64_t b0,int64_t b1) {
    require();int64_t released=0;
    for(const auto& entry:layers) released+=evict_map_fast(entry.state->map,b0,b1);
    return released;
}
EvictionResult evict_range(int64_t b0, int64_t b1) {
    require();
    if (b0 < 1 || b1 <= b0 || b1 > layers.front().state->n_pages) throw std::runtime_error("invalid original-position old-text pool extent");
    fence();
    EvictionResult out; out.b0 = b0; out.b1 = b1;
    out.before = snapshot(false); require_maps(out.before);
    out.authoritative_before = authoritative_hash(b0, b1);
    struct MapImage { std::vector<int32_t> table, owners, stamps, refs, control; };
    std::vector<MapImage> expected;
    expected.reserve(layers.size());
    for (const auto& entry : layers) {
        const auto& st = *entry.state;
        MapImage image{read(st.page_table, st.n_pages), read(st.map.slot_block, st.n_slots),
            read(st.map.slot_stamp, st.n_slots), read(st.map.slot_ref, st.n_slots),
            read(st.map.ctl, strata::kernels::kKvCtlInts)};
        for (int64_t b = b0; b < b1; ++b) {
            const int32_t slot = image.table[size_t(b)];
            if (slot >= 0) {
                out.released.push_back({entry.ordinal, int32_t(b), slot});
                image.owners[size_t(slot)] = -1; image.stamps[size_t(slot)] = -1; image.refs[size_t(slot)] = 0;
            }
            image.table[size_t(b)] = -1;
        }
        expected.push_back(std::move(image));
        trace_evict_range(st.map, b0, b1, nullptr);
    }
    fence();
    out.after = snapshot(false); require_maps(out.after);
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& st = *layers[i].state;
        const auto& image = expected[i];
        if (read(st.page_table, st.n_pages) != image.table || read(st.map.slot_block, st.n_slots) != image.owners ||
            read(st.map.slot_stamp, st.n_slots) != image.stamps || read(st.map.slot_ref, st.n_slots) != image.refs ||
            read(st.map.ctl, strata::kernels::kKvCtlInts) != image.control)
            throw std::runtime_error("GPU eviction changed untargeted mapping/clock fields or left a stale targeted map");
        const auto table = read(st.page_table + b0, b1 - b0);
        if (std::any_of(table.begin(), table.end(), [](int32_t slot) { return slot != -1; }))
            throw std::runtime_error("GPU eviction left an original-position target mapped");
        const int64_t released = int64_t(std::count_if(out.released.begin(), out.released.end(), [&](const SlotRelease& r) { return r.ordinal == layers[i].ordinal; }));
        if (out.after[i].slots != out.before[i].slots || out.after[i].slot_pool_bytes != out.before[i].slot_pool_bytes ||
            out.after[i].occupied_slots != out.before[i].occupied_slots - released ||
            out.after[i].free_slots != out.before[i].free_slots + released)
            throw std::runtime_error("GPU eviction did not release precisely the original resident slots");
    }
    out.authoritative_after = authoritative_hash(b0, b1);
    if (out.authoritative_after != out.authoritative_before)
        throw std::runtime_error("GPU-only eviction changed authoritative host/index bytes");
    out.exact_map_transition = true;
    return out;
}
} // namespace strata::qsa_pool_v2
