// Synthetic CUDA test: no model, tokenizer, HTTP, keys or deployment lifecycle.
// Building is permitted independently; RUN ONLY in the authorized GPU window.
#include "qsa_pool_trace.hpp"
#include "qsa_pool.hpp"
#include "strata/kernels/kv_q8.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace k = strata::kernels;
namespace p = strata::qsa_pool_v2;
namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}
void need(bool condition, const char* why) { if (!condition) throw std::runtime_error(why); }
template<class T> std::vector<T> read(const T* src, size_t n) {
    std::vector<T> out(n);
    ck(cudaMemcpy(out.data(), src, n * sizeof(T), cudaMemcpyDefault), "read synthetic state"); return out;
}
struct Fixture {
    k::QsaShapes shape = k::qsa_real_shapes();
    k::KvStreamMap map{};
    k::KvHostPools host{};
    k::QsaAttnPools slots{};
    std::array<size_t, 4> per{};
    std::array<uint8_t*, 4> cpu{}, devhost{}, gpu{};
    std::vector<void*> allocations;
    int32_t *ids = nullptr, *steps = nullptr;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    bool tracing = false;
    static constexpr int cap = 2048;
    template<class T> T* alloc(size_t count) {
        T* value = nullptr; ck(cudaMalloc(reinterpret_cast<void**>(&value), count * sizeof(T)), "allocate synthetic GPU buffer");
        allocations.push_back(value); ck(cudaMemset(value, 0, count * sizeof(T)), "zero synthetic GPU buffer"); return value;
    }
    explicit Fixture(int64_t count, bool observe) {
        shape.n_head_kv = 2; shape.head_dim = 256; shape.page_size = 4;
        map.n_slots = count; map.n_blocks = count + 4096;
        per = {2048, 2048, 64, 64};
        for (size_t a = 0; a < per.size(); ++a) {
            void* h = nullptr; void* d = nullptr;
            ck(cudaHostAlloc(&h, size_t(map.n_blocks) * per[a], cudaHostAllocMapped), "allocate mapped authoritative RAM");
            cpu[a] = static_cast<uint8_t*>(h);
            for (int64_t b = 0; b < map.n_blocks; ++b) for (size_t i = 0; i < per[a]; ++i)
                cpu[a][size_t(b) * per[a] + i] = uint8_t((uint64_t(b) * 37 + i * 17 + a * 61) % 251);
            ck(cudaHostGetDevicePointer(&d, h, 0), "map authoritative RAM"); devhost[a] = static_cast<uint8_t*>(d);
            gpu[a] = alloc<uint8_t>(size_t(map.n_slots) * per[a]);
        }
        host.k_q = reinterpret_cast<int8_t*>(devhost[0]); host.v_q = reinterpret_cast<int8_t*>(devhost[1]);
        host.k_scale = reinterpret_cast<uint16_t*>(devhost[2]); host.v_scale = reinterpret_cast<uint16_t*>(devhost[3]);
        slots.k_q = reinterpret_cast<int8_t*>(gpu[0]); slots.v_q = reinterpret_cast<int8_t*>(gpu[1]);
        slots.k_scale = reinterpret_cast<uint16_t*>(gpu[2]); slots.v_scale = reinterpret_cast<uint16_t*>(gpu[3]);
        map.page_table = alloc<int32_t>(size_t(map.n_blocks)); slots.page_table = map.page_table;
        map.slot_block = alloc<int32_t>(size_t(map.n_slots)); map.slot_stamp = alloc<int32_t>(size_t(map.n_slots));
        map.slot_ref = alloc<int32_t>(size_t(map.n_slots)); map.miss_block = alloc<int32_t>(size_t(map.n_slots));
        map.miss_slot = alloc<int32_t>(size_t(map.n_slots)); map.ctl = alloc<int32_t>(k::kKvCtlInts);
        ids = alloc<int32_t>(cap); steps = alloc<int32_t>(k::kStepCount);
        k::kv_stream_reset(map, nullptr); ck(cudaDeviceSynchronize(), "reset synthetic map");
        need(!p::trace_for(map.page_table).counters, "pool observer is unexpectedly enabled by default");
        if (observe) { p::trace_register(map, k::kv_block_bytes(shape, k::kKvInt8), 4096); tracing = true; }
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create synthetic capture stream");
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin synthetic graph capture");
        k::kv_stream_resolve(map, slots, host, k::kKvInt8, ids, steps, 1, cap, shape, stream);
        ck(cudaStreamEndCapture(stream, &graph), "end synthetic graph capture");
        ck(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate synthetic graph");
    }
    ~Fixture() {
        cudaDeviceSynchronize();
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        if (stream) cudaStreamDestroy(stream);
        if (tracing) p::trace_unregister(map.page_table);
        for (void* value : allocations) cudaFree(value);
        for (auto* value : cpu) if (value) cudaFreeHost(value);
    }
    void select(int64_t first, int n) {
        need(first >= 0 && n > 0 && n <= cap && first + n <= map.n_blocks, "bad synthetic selection");
        std::vector<int32_t> selected(size_t(n), 0);
        for (int i = 0; i < n; ++i) selected[size_t(i)] = int32_t((first + i) * 4);
        int32_t step[k::kStepCount] = {};
        step[k::kStepPos] = int32_t(map.n_blocks * 4 - 1);
        step[k::kStepNKv] = int32_t(map.n_blocks * 4);
        step[k::kStepWidth] = n;
        ck(cudaMemcpyAsync(ids, selected.data(), size_t(n) * sizeof(int32_t), cudaMemcpyHostToDevice, stream), "write selection IDs");
        ck(cudaMemcpyAsync(steps, step, sizeof(step), cudaMemcpyHostToDevice, stream), "write selection steps");
        ck(cudaGraphLaunch(executable, stream), "replay original CLOCK resolve/copy graph");
        ck(cudaStreamSynchronize(stream), "finish graph replay");
        audit();
        const auto table = read(map.page_table, size_t(map.n_blocks));
        for (int i = 0; i < n; ++i) {
            const int64_t b = first + i; const int32_t sl = table[size_t(b)];
            need(sl >= 0, "selected original-position block was not faulted resident");
            for (size_t a = 0; a < per.size(); ++a) {
                const auto bytes = read(gpu[a] + size_t(sl) * per[a], per[a]);
                need(std::memcmp(bytes.data(), cpu[a] + size_t(b) * per[a], per[a]) == 0,
                     "resident codes/scales differ from original-position authoritative RAM bytes");
            }
        }
    }
    int64_t audit() {
        const auto table = read(map.page_table, size_t(map.n_blocks)), owner = read(map.slot_block, size_t(map.n_slots));
        std::vector<int> seen(size_t(map.n_slots), 0);
        int64_t occupied = 0;
        for (int64_t b = 0; b < map.n_blocks; ++b) {
            const int32_t sl = table[size_t(b)];
            need(sl >= -1 && sl < map.n_slots, "invalid/transient forward page table");
            if (sl >= 0) { need(owner[size_t(sl)] == b && ++seen[size_t(sl)] == 1, "forward/reverse duplicate mismatch"); ++occupied; }
        }
        for (int64_t sl = 0; sl < map.n_slots; ++sl) {
            const int32_t b = owner[size_t(sl)];
            need(b == -1 || (b >= 0 && b < map.n_blocks && table[size_t(b)] == sl && seen[size_t(sl)] == 1), "reverse/forward mismatch");
        }
        need(!k::kv_stream_counters(map).overflow, "original CLOCK resolver overflowed");
        return occupied;
    }
    uint64_t host_hash() const {
        uint64_t h = 14695981039346656037ull;
        for (size_t a = 0; a < per.size(); ++a) for (size_t i = 0; i < size_t(map.n_blocks) * per[a]; ++i) {
            h ^= cpu[a][i]; h *= 1099511628211ull;
        }
        return h;
    }
};

void run(int64_t slots, bool observed) {
    Fixture f(slots, observed);
    const uint64_t host_before = f.host_hash();
    auto reset_probe=[&]{
        f.select(0,4);k::kv_stream_reset(f.map,nullptr);ck(cudaDeviceSynchronize(),"whole-map reset regression fence");
        need(f.audit()==0&&read(f.map.page_table,size_t(f.map.n_blocks))[0]==-1,"whole-map reset retained block0 or a reverse owner");
        need(f.host_hash()==host_before,"whole-map reset changed authoritative bytes");f.select(0,4);need(f.audit()==4,"block0 did not reload after full reset");
        std::puts("  COMPACT_WHOLE_MAP_RESET_PASS block0,forward/reverse maps,authoritative bytes,captured reload");
    };
    for (int64_t b = 0; b < slots; b += 1024) f.select(b, 1024);
    need(f.audit() == slots, "warmup did not occupy the configured fixed slot pool");
    if (!observed) {
        f.select(slots, 1024); f.select(64, 4);
        need(!p::trace_for(f.map.page_table).counters, "off path allocated an observer");
        need(f.host_hash() == host_before, "off path changed authoritative RAM");
        std::printf("  slots=%lld observer=off: captured resolve, payload bytes, CLOCK replacement PASS\n", (long long)slots);
        reset_probe();return;
    }
    auto warm = p::trace_read(f.map.page_table);
    need(warm.counters[p::kEventDropped] > 0, "bounded event ring never exercised its drop counter");
    const auto completed_view = p::trace_for(f.map.page_table);
    need(completed_view.block_completed_copies != nullptr, "complete lifetime block-copy counters are missing");
    const auto warm_completed = read(completed_view.block_completed_copies, size_t(f.map.n_blocks));
    uint64_t warm_sum = 0; for (const auto n : warm_completed) warm_sum += n;
    need(warm_sum == uint64_t(slots), "complete lifetime counts lost warmup copies after event overflow");
    const int32_t late_target = int32_t(slots - 1);
    need(warm_completed[size_t(late_target)] == 1 && std::none_of(warm.events.begin(), warm.events.end(),
        [late_target](const p::Event& e) { return e.kind == 3 && e.block == late_target; }),
        "late target copy was not proven independently of the truncated event ring");
    p::trace_reset(f.map.page_table);
    need(read(completed_view.block_completed_copies, size_t(f.map.n_blocks)) == warm_completed,
         "trace reset changed completed host-copy lifetime counters");
    const auto old_table = read(f.map.page_table, size_t(f.map.n_blocks));
    std::array<int32_t, 4> released{};
    for (int i = 0; i < 4; ++i) released[size_t(i)] = old_table[size_t(64 + i)];
    const auto old_stamp=read(f.map.slot_stamp,size_t(slots)),old_ref=read(f.map.slot_ref,size_t(slots)),old_ctl=read(f.map.ctl,size_t(k::kKvCtlInts));
    need(p::evict_map_fast(f.map,64,68)==4,"fast map eviction did not release exactly four slots");
    auto expected_stamp=old_stamp,expected_ref=old_ref;
    for(auto slot:released){expected_stamp[size_t(slot)]=-1;expected_ref[size_t(slot)]=0;}
    need(read(f.map.slot_stamp,size_t(slots))==expected_stamp && read(f.map.slot_ref,size_t(slots))==expected_ref && read(f.map.ctl,size_t(k::kKvCtlInts))==old_ctl,"fast eviction changed non-target CLOCK metadata");
    bool refused=false;try{p::evict_map_fast(f.map,0,4);}catch(...){refused=true;}need(refused,"fast map accepted block0");
    need(f.audit() == slots - 4, "GPU-only eviction did not recycle exactly four occupied slots");
    const auto empty_table = read(f.map.page_table, size_t(f.map.n_blocks)), empty_owner = read(f.map.slot_block, size_t(slots));
    for (int i = 0; i < 4; ++i) need(empty_table[size_t(64 + i)] == -1 && empty_owner[size_t(released[size_t(i)])] == -1,
                                    "GPU eviction left a forward or reverse mapping live");
    need(f.host_hash() == host_before, "GPU-only eviction changed authoritative RAM");
    need(read(completed_view.block_completed_copies, size_t(f.map.n_blocks)) == warm_completed,
         "GPU-only eviction falsely counted a completed host copy");
    f.select(slots, 1024);
    const auto reused_owner = read(f.map.slot_block, size_t(slots));
    for (int i = 0; i < 4; ++i) need(reused_owner[size_t(released[size_t(i)])] >= slots,
                                    "a released physical slot was not actually reused for a different logical block");
    f.select(64, 4); // exact original-position old blocks, loaded from host without prefill/model forward
    const auto final_completed = read(completed_view.block_completed_copies, size_t(f.map.n_blocks));
    uint64_t final_sum = 0; for (const auto n : final_completed) final_sum += n;
    need(final_sum == uint64_t(slots + 1024 + 4), "complete lifetime total differs from actual host-copy work");
    for (int i = 0; i < 4; ++i) need(final_completed[size_t(64 + i)] == warm_completed[size_t(64 + i)] + 1,
        "exact old-target completed-copy delta failed after an event-ring reset");
    const auto trace = p::trace_read(f.map.page_table);
    need(trace.counters[p::kManualEvictions] == 4 && trace.counters[p::kManualSlotReuses] == 4,
         "manual eviction/reuse counters do not match the proven physical slots");
    need(trace.counters[p::kReloads] >= 4, "trace reset destroyed old-block lifetime load counts");
    need(trace.counters[p::kCopiedBytes] == trace.counters[p::kPlaced] * 4224,
         "copy-completion byte counters do not match the real completed INT8 block copies");
    need(trace.counters[p::kEventDropped] == 0, "post-reset bounded event log overflowed");
    int original_reloads = 0, completed = 0, recycled = 0;
    for (const auto& e : trace.events) {
        if (e.kind == 1 && e.block >= 64 && e.block < 68 && e.prior_loads > 0) ++original_reloads;
        if (e.kind == 1 && e.origin == 2 && e.previous_owner >= 64 && e.previous_owner < 68 && e.block >= slots) ++recycled;
        if (e.kind == 3) ++completed;
    }
    need(original_reloads == 4 && recycled == 4 && uint64_t(completed) == trace.counters[p::kPlaced],
         "bounded events do not prove by-ID old reload, freed-slot reuse and completed host copies");
    need(f.host_hash() == host_before, "restore/reuse queries modified authoritative RAM");
    std::printf("  slots=%lld observer=on: captured replay, 4 freed slots reused, 4 exact old-block reloads, complete target-copy counters survive event overflow/reset, %llu bytes copied PASS\n",
        (long long)slots, (unsigned long long)trace.counters[p::kCopiedBytes]);
    reset_probe();
}
}
int main() {
    try {
        for (const int64_t slots : {8192ll, 16384ll}) { run(slots, false); run(slots, true); }
        std::puts("QSA_POOL_V2_SYNTHETIC_PASS"); return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "QSA_POOL_V2_SYNTHETIC_FAIL: %s\n", e.what()); return 1; }
}
