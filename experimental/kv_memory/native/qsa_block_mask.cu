#include "qsa_block_store.hpp"
#include "strata/kernels/qsa.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace strata::qsa_proto::detail {
namespace {
void check(const char* what) {
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(std::string("qsa prototype ") + what + ": " + cudaGetErrorString(status));
}

// One thread owns a row: in-place stable compaction cannot overwrite a future
// unread ID, and no parallel read/write race is possible. This is a correctness
// prototype, not a throughput-optimized compactor.
__global__ void filter_kernel(const uint8_t* deleted, const uint8_t* watched, int64_t blocks, int32_t* ids,
                              const int32_t* original, int32_t* filtered,
                              int64_t nq, int64_t cap, unsigned long long* counts) {
    const int64_t q = blockIdx.x;
    if (threadIdx.x != 0 || q >= nq) return;
    using namespace strata::kernels;
    const int32_t* src = original + q * kStepCount;
    int32_t* dst = filtered + q * kStepCount;
    for (int j = 0; j < kStepCount; ++j) dst[j] = src[j];
    atomicAdd(counts, 1ull);
    const int64_t width = src[kStepWidth], n_kv = src[kStepNKv];
    if (width < 0 || width > cap || n_kv < 0 || n_kv > blocks * 4) {
        dst[kStepWidth] = 0;
        atomicAdd(counts + 3, 1ull);
        atomicAdd(counts + 4, 1ull);
        return;
    }
    int32_t* row = ids + q * cap;
    int64_t written = 0;
    unsigned long long removed = 0, invalid = 0, violations = 0, before = 0, after = 0;
    for (int64_t i = 0; i < width; ++i) {
        const int32_t cell = row[i];
        if (cell < 0 || cell >= n_kv || cell / 4 >= blocks) { ++invalid; continue; }
        if (watched[cell / 4]) ++before;
        if (deleted[cell / 4]) { ++removed; continue; }
        row[written++] = cell;
    }
    for (int64_t i = 0; i < written; ++i) {
        if (deleted[row[i] / 4]) ++violations;
        if (watched[row[i] / 4]) ++after;
    }
    for (int64_t i = written; i < width; ++i) row[i] = -1;
    // Invalid upstream IDs are never allowed to reach a reader. Control-plane
    // audit must reject invalid/empty counters; no model result can pass that gate.
    dst[kStepWidth] = invalid ? 0 : static_cast<int32_t>(written);
    if (removed) atomicAdd(counts + 1, removed);
    if (violations) atomicAdd(counts + 2, violations);
    if (invalid) atomicAdd(counts + 3, invalid);
    if (written == 0 || invalid) atomicAdd(counts + 4, 1ull);
    if (before) atomicAdd(counts + 5, before);
    if (after) atomicAdd(counts + 6, after);
}

__global__ void invalidate_kernel(int32_t* table, int32_t* slot_block,
                                  int32_t* stamp, int32_t* ref,
                                  int64_t blocks, int64_t slots, int64_t b0, int64_t b1) {
    const int64_t b = b0 + (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= b1 || b >= blocks) return;
    const int32_t slot = table[b];
    if (slot >= 0 && slot < slots && slot_block[slot] == b) {
        slot_block[slot] = -1;
        stamp[slot] = 0;
        ref[slot] = 0;
    }
    table[b] = -1;
}
}

void launch_filter(const uint8_t* deleted, const uint8_t* watched, int64_t blocks, int32_t* ids,
                   const int32_t* original_steps, int32_t* filtered_steps,
                   int64_t n_queries, int64_t cap, uint64_t* counters, void* stream) {
    filter_kernel<<<static_cast<unsigned>(n_queries), 1, 0, static_cast<cudaStream_t>(stream)>>>(
        deleted, watched, blocks, ids, original_steps, filtered_steps, n_queries, cap,
        reinterpret_cast<unsigned long long*>(counters));
    check("filter launch");
}

void launch_invalidate(int32_t* page_table, int32_t* slot_block,
                       int32_t* slot_stamp, int32_t* slot_ref,
                       int64_t n_blocks, int64_t n_slots, int64_t b0, int64_t b1,
                       void* stream) {
    if (b1 <= b0) return;
    invalidate_kernel<<<static_cast<unsigned>((b1 - b0 + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        page_table, slot_block, slot_stamp, slot_ref, n_blocks, n_slots, b0, b1);
    check("invalidate launch");
}
} // namespace strata::qsa_proto::detail
