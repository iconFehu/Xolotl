#pragma once
#include "qsa_pool_trace.hpp"
#include <cuda_runtime.h>

namespace strata::qsa_pool_v2::device {
__device__ inline void add(TraceView v, Counter c, unsigned long long n = 1) {
    if (v.counters && n) atomicAdd(v.counters + c, n);
}
__device__ inline void event(TraceView v, Event e) {
    if (!v.counters) return;
    const unsigned long long sequence = atomicAdd(v.counters + kEventCursor, 1ull);
    if (sequence >= v.capacity) { atomicAdd(v.counters + kEventDropped, 1ull); return; }
    e.sequence = sequence;
    v.events[sequence] = e;
}
__device__ inline void assigned(TraceView v, int epoch, int b, int sl, int old) {
    if (!v.counters) return;
    const int previous = old >= 0 ? old : v.slot_history[sl];
    const bool manual = old < 0 && v.slot_manually_vacated[sl] != 0;
    const uint32_t prior = atomicAdd(v.block_load_count + b, 1u);
    const uint32_t generation = atomicAdd(v.slot_generation + sl, 1u) + 1u;
    if (prior) add(v, kReloads);
    if (old >= 0) add(v, kClockEvictions);
    if (previous >= 0 && previous != b) add(v, kSlotRecycles);
    if (manual && previous != b) add(v, kManualSlotReuses);
    Event e{}; e.epoch = epoch; e.block = b; e.slot = sl;
    e.previous_owner = previous; e.prior_loads = prior; e.slot_generation = generation;
    e.kind = 1; e.origin = old >= 0 ? 1u : manual ? 2u : previous >= 0 ? 3u : 0u;
    event(v, e);
    v.slot_history[sl] = b;
    v.slot_manually_vacated[sl] = 0;
}
__device__ inline void copied(TraceView v, int epoch, int b, int sl) {
    if (!v.counters) return;
    // The caller's copy-block barrier precedes this hook. Keep a complete
    // lifetime record even when event() below drops its bounded sample.
    atomicAdd(v.block_completed_copies + b, 1u);
    add(v, kCopiedBytes, v.block_bytes);
    Event e{}; e.epoch = epoch; e.block = b; e.slot = sl;
    e.slot_generation = v.slot_generation[sl]; e.bytes = v.block_bytes; e.kind = 3;
    event(v, e);
}
} // namespace strata::qsa_pool_v2::device
