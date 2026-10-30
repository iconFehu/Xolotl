#pragma once

#include "strata/kernels/kv_stream.hpp"
#include <cstdint>
#include <vector>

// An opt-in, fixed-address observer of the EXISTING mode1 CLOCK allocator.
// No hook changes selection, replacement choices, copy order or pool capacity.
namespace strata::qsa_pool_v2 {
enum Counter : int {
    kResolves, kLookups, kRequestedMisses, kPlaced, kHits, kReloads,
    kClockEvictions, kSlotRecycles, kManualSlotReuses, kCopiedBytes,
    kManualEvictions, kEventCursor, kEventDropped, kCounterCount
};
struct Event {
    uint64_t sequence = 0;
    uint64_t bytes = 0;
    int32_t epoch = 0, block = -1, slot = -1, previous_owner = -1;
    uint32_t prior_loads = 0, slot_generation = 0;
    // 1: selected-ID fault assigned a slot (bytes=0); 2: explicit idle GPU-only
    // eviction; 3: host copy kernel completed that (epoch,block,slot,generation).
    uint32_t kind = 0;
    // Fault's prior slot state: 0 never occupied, 1 live CLOCK victim,
    // 2 explicitly vacated slot, 3 empty slot with older lifetime ownership.
    uint32_t origin = 0;
};
struct TraceView {
    unsigned long long* counters = nullptr;
    uint32_t* block_load_count = nullptr;
    // Complete host->slot copies, independent of bounded event sampling. Never
    // reset by trace_reset; graph copies update this only after their stores.
    uint32_t* block_completed_copies = nullptr;
    int32_t* slot_history = nullptr;
    uint32_t* slot_generation = nullptr;
    uint32_t* slot_manually_vacated = nullptr;
    Event* events = nullptr;
    uint64_t capacity = 0, block_bytes = 0;
};
struct TraceRead {
    uint64_t counters[kCounterCount] = {};
    std::vector<Event> events;
};

// Host lookup is performed only when building/launching an original graph call;
// captured nodes hold the immutable TraceView's fixed device addresses. Unknown
// maps return a completely empty view, making instrumentation default-off.
TraceView trace_for(const int32_t* page_table) noexcept;
void trace_register(const strata::kernels::KvStreamMap& map, uint64_t block_bytes,
                    uint64_t event_capacity = 4096);
void trace_unregister(const int32_t* page_table) noexcept;
void trace_reset(const int32_t* page_table); // no lifetime/history reset
TraceRead trace_read(const int32_t* page_table, bool include_events = true);
void trace_evict_range(const strata::kernels::KvStreamMap& map,
                       int64_t b0, int64_t b1, void* stream);
} // namespace strata::qsa_pool_v2
