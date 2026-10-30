#include "qsa_pool_device.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>

namespace strata::qsa_pool_v2 {
namespace {
std::map<const int32_t*, TraceView> views;
void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string("QSA pool v2 ") + what + ": " + cudaGetErrorString(status));
}
template<class T> void alloc(T*& value, uint64_t count, int initial = 0) {
    if (!count || count > SIZE_MAX / sizeof(T)) throw std::runtime_error("invalid pool trace allocation extent");
    check(cudaMalloc(reinterpret_cast<void**>(&value), size_t(count) * sizeof(T)), "allocate trace");
    check(cudaMemset(value, initial, size_t(count) * sizeof(T)), "initialize trace");
}
void release(TraceView& v) noexcept {
    for (void* p : {static_cast<void*>(v.counters), static_cast<void*>(v.block_load_count),
                   static_cast<void*>(v.block_completed_copies),
                   static_cast<void*>(v.slot_history), static_cast<void*>(v.slot_generation),
                   static_cast<void*>(v.slot_manually_vacated), static_cast<void*>(v.events)})
        if (p) cudaFree(p);
    v = TraceView{};
}
__global__ void evict_kernel(strata::kernels::KvStreamMap m, TraceView v, int64_t b0, int64_t b1) {
    const int64_t b = b0 + int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= b1) return;
    const int32_t sl = m.page_table[b];
    if (sl < 0) return; // idle CPU validated -1 and the complete inverse maps
    if (sl >= m.n_slots || m.slot_block[sl] != b) return; // CPU post-check fails closed
    if (v.counters) {
        v.slot_history[sl] = int32_t(b);
        v.slot_manually_vacated[sl] = 1;
        Event e{}; e.epoch = m.ctl[0]; e.block = int32_t(b); e.slot = sl;
        e.previous_owner = int32_t(b); e.slot_generation = v.slot_generation[sl]; e.kind = 2;
        device::event(v, e); device::add(v, kManualEvictions);
    }
    m.slot_block[sl] = -1;
    m.slot_stamp[sl] = -1;
    m.slot_ref[sl] = 0;
    m.page_table[b] = -1;
}
}

TraceView trace_for(const int32_t* page_table) noexcept {
    const auto found = views.find(page_table);
    return found == views.end() ? TraceView{} : found->second;
}
void trace_register(const strata::kernels::KvStreamMap& m, uint64_t bytes, uint64_t capacity) {
    if (!m.page_table || !m.slot_block || !m.slot_stamp || !m.slot_ref || !m.ctl ||
        m.n_blocks <= 0 || m.n_slots < 5120 || m.n_slots >= m.n_blocks || !bytes || !capacity || capacity > 65536)
        throw std::runtime_error("invalid complete mode1 map for pool trace registration");
    if (views.count(m.page_table)) throw std::runtime_error("pool trace map already registered");
    TraceView v{}; v.capacity = capacity; v.block_bytes = bytes;
    try {
        alloc(v.counters, kCounterCount);
        alloc(v.block_load_count, uint64_t(m.n_blocks));
        alloc(v.block_completed_copies, uint64_t(m.n_blocks));
        alloc(v.slot_history, uint64_t(m.n_slots), -1);
        alloc(v.slot_generation, uint64_t(m.n_slots));
        alloc(v.slot_manually_vacated, uint64_t(m.n_slots));
        alloc(v.events, capacity);
        views.emplace(m.page_table, v);
    } catch (...) { release(v); throw; }
}
void trace_unregister(const int32_t* page_table) noexcept {
    const auto found = views.find(page_table);
    if (found == views.end()) return;
    cudaDeviceSynchronize();
    release(found->second); views.erase(found);
}
void trace_reset(const int32_t* page_table) {
    const auto v = trace_for(page_table);
    if (!v.counters) throw std::runtime_error("pool trace reset requested for an unregistered map");
    check(cudaDeviceSynchronize(), "fence trace reset");
    check(cudaMemset(v.counters, 0, kCounterCount * sizeof(unsigned long long)), "reset trace counters");
    // Lifetime assigned/completed-copy counts, slot history/generations and manually-vacated bits
    // deliberately survive. No CUDA graph pointer or replacement bit is changed.
}
TraceRead trace_read(const int32_t* page_table, bool include_events) {
    const auto v = trace_for(page_table);
    if (!v.counters) throw std::runtime_error("pool trace read requested for an unregistered map");
    check(cudaDeviceSynchronize(), "fence trace read");
    TraceRead out;
    check(cudaMemcpy(out.counters, v.counters, sizeof(out.counters), cudaMemcpyDeviceToHost), "read trace counters");
    const size_t n = include_events ? size_t(std::min<uint64_t>(out.counters[kEventCursor], v.capacity)) : 0;
    out.events.resize(n);
    if (n) check(cudaMemcpy(out.events.data(), v.events, n * sizeof(Event), cudaMemcpyDeviceToHost), "read trace events");
    return out;
}
void trace_evict_range(const strata::kernels::KvStreamMap& m, int64_t b0, int64_t b1, void* stream) {
    if (b0 < 0 || b1 <= b0 || b1 > m.n_blocks) throw std::runtime_error("invalid idle pool eviction extent");
    const auto v = trace_for(m.page_table);
    if (!v.counters) throw std::runtime_error("pool eviction requires its registered fixed-address observer");
    evict_kernel<<<unsigned((b1 - b0 + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(m, v, b0, b1);
    check(cudaGetLastError(), "evict range launch");
}
} // namespace strata::qsa_pool_v2
