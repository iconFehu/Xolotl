#pragma once

#include "qsa_pool_trace.hpp"
#include "strata/core/session.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::qsa_pool_v2 {
inline bool compact_enabled() { const char* e=std::getenv("STRATA_QSA_V5_COMPACT"); if(e && std::strcmp(e,"0") && std::strcmp(e,"1")) throw std::runtime_error("invalid compact flag"); return e && !std::strcmp(e,"1"); }
struct WatchedBlock {
    int32_t block = -1;
    uint32_t completed_copies = 0;
};
struct LayerSnapshot {
    int64_t ordinal = 0, blocks = 0, slots = 0;
    uint64_t slot_pool_bytes = 0, authoritative_host_bytes = 0, pooled_index_bytes = 0;
    int64_t occupied_slots = 0, free_slots = 0, mapped_blocks = 0;
    int64_t invalid_forward = 0, invalid_reverse = 0, duplicate_owners = 0, transient_pages = 0;
    bool map_ok = false, overflow = false;
    uint64_t map_hash = 0;
    bool details_included = true;
    uint64_t watched_count=0, watched_nonzero=0, watched_copy_sum=0, watched_id_hash=14695981039346656037ull, watched_copy_hash=14695981039346656037ull;
    TraceRead trace;
    std::vector<WatchedBlock> watched_blocks;
};
struct SlotRelease {
    int64_t ordinal = 0;
    int32_t block = -1, slot = -1;
};
struct EvictionResult {
    int64_t b0 = 0, b1 = 0;
    std::vector<SlotRelease> released;
    std::vector<LayerSnapshot> before, after;
    // K/V host + pooled-index bytes over [b0,b1), before and after. This is an
    // idle diagnostic, NOT included in model TTFT or per-forward counters.
    uint64_t authoritative_before = 0, authoritative_after = 0;
    bool exact_map_transition = false;
};

// Caller must opt in and register before ANY graph capture. Complete one-GPU
// SC117 INT8 mode1 main-QSA states only; draft ring untouched. No v1/production
// ownership, no CUDA allocation replacement, and no model forward is performed.
void initialize(core::SessionState& session, const core::ModelGeometry& geometry,
                uint64_t event_capacity = 4096);
bool initialized() noexcept;
// Drain and destroy all captured graphs before shutdown. Never recapture with
// old views after freeing their fixed-address allocations.
void shutdown() noexcept;
std::vector<LayerSnapshot> snapshot(bool include_events = true, bool include_watched = true);
void reset_trace();
void discard_watched(); // idle-only, no device address changes
// CPU-only registration, original-position blocks, deduplicated for all twelve
// layers. It does not zero counters or mutate model/selection/residency state.
// Call only after a successful QBSAVE in an initialized mode1 prototype.
void watch_range(int64_t b0, int64_t b1);

// Strictly idle, full-device fence; validates all bidirectional mappings and
// authoritative bytes. Logically active data remains readable and may fault
// back through original selected-ID resolve. Does not touch v1 deleted mask,
// K/V host, pooled index, GDN, PLE, draft, slot pool capacity or allocator.
// Caller validates by-ID lineage/archive checksum/range and full running guards.
int64_t evict_map_fast(const strata::kernels::KvStreamMap& map,int64_t b0,int64_t b1);
int64_t evict_range_fast(int64_t b0, int64_t b1);
EvictionResult evict_range(int64_t b0, int64_t b1);
} // namespace strata::qsa_pool_v2
