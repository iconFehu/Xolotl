#pragma once

#include "strata/core/session.hpp"
#include "strata/core/conversation_cache.hpp"
#include "qsa_pool.hpp"
#include "qsa_operation_profile.hpp"
#include <cstdint>
#include <string>
#include <vector>

// CUDA-only, opt-in, process-local prototype. Caller owns admission/idle fencing
// and an append-only lineage (no hidden checkpoint restore while blocks are absent).
namespace strata::qsa_proto {

struct OpResult {
    OperationProfile profile;
    bool guard_checked = false;
    bool ok = false;
    bool poisoned = false;
    std::string error;
    std::string id;
    int64_t p0 = 0, p1 = 0, token_count = 0;
    uint64_t bytes = 0, payload_hash = 0;
    uint64_t running_before = 0, running_after = 0;
};

struct SiteCounters {
    std::string site;
    int64_t qsa_ordinal = 0;
    uint64_t queries = 0, seen_deleted = 0, remaining_violations = 0;
    uint64_t invalid_ids = 0, empty_queries = 0;
    uint64_t watched_before = 0, watched_after = 0; // union of all saved ranges, not per-ID attribution
};

struct GuardFingerprint {
    bool ok = false;
    std::string error;
    uint64_t gdn = 0, ple = 0, index_running = 0, draft = 0, combined = 0;
};

// Call before capture: masks and per-layer filtered step arrays have fixed device
// addresses. Registration uses each main QSA's page_table identity. INT8 mode0/1,
// full single-GPU session only. A second initialize is refused until shutdown.
// Destroy/drain every graph using these allocations before shutdown; graphs
// captured under one registration must never be reused after reinitialization.
OpResult initialize(core::SessionState& session, const core::ModelGeometry& geometry,
                    const core::QsaState* draft = nullptr, int64_t max_queries = 16384);
void shutdown() noexcept;
bool initialized() noexcept;
bool poisoned() noexcept;

// Mandatory before block operations if initialize received a draft state. Zero
// means all-history draft attention and is deliberately unsupported in v1.
OpResult configure_mtp_window(int64_t window);

// Stable compaction of IDs. ORIGINAL steps are never changed (verify shares them
// across layers). Returns this layer's persistent copied/filtered steps. No active
// registration returns original steps. Unknown registered state/site fails closed.
// Sites are exactly "decode", "verify", "prefill". No host synchronization here.
// counters() must be checked before accepting a probe result: invalid_ids,
// remaining_violations and empty_queries must all be zero.
const int32_t* filter_selection(const core::QsaState& state, int32_t* ids,
                               const int32_t* steps, int64_t n_queries, int64_t cap,
                               void* stream, const char* site);
std::vector<SiteCounters> counters(); // synchronized diagnostic read; throws on CUDA errors
void reset_counters();

// Groups cover the same [p0,p1) across all 12 main QSA layers. Positions stay
// unchanged; ranges must be complete 4-cell blocks, not block0/current tail or the
// MTP read window. v1 caps the union of removed cells at 1024 and requires >=4096
// committed tokens, so compaction cannot silently empty a normal selection.
// lineage must be a nonempty model/config/prefix identity verified by the caller.
// Archive is external to active authoritative pools, in process-owned CPU vectors.
// Save additionally enables an instrumentation-only watched union for its range.
OpResult save_by_id(const std::string& id, int64_t p0, int64_t p1,
                    int64_t token_count, const std::string& lineage);
OpResult remove_by_id(const std::string& id, int64_t token_count,
                      const std::string& lineage);
OpResult load_by_id(const std::string& id, int64_t token_count,
                    const std::string& lineage);
std::vector<OpResult> inspect();
OpResult unsave_by_id(const std::string& id,int64_t tokens,const std::string& lineage);
struct BankStatus { uint64_t bytes=0,groups=0,deleted_cells=0; };
BankStatus bank_status();
// Explicit destructive session discard after the caller's verifier/GPU idle fence.
// Keeps all registered device addresses (and therefore graph views) unchanged.
OpResult discard_session_archives();

struct PoolByIdResult {
    OpResult operation;
    bool resident_verified = false;
    bool exact_map_transition = false;
    uint64_t authoritative_before = 0, authoritative_after = 0;
    std::vector<qsa_pool_v2::LayerSnapshot> before, after;
    std::vector<qsa_pool_v2::SlotRelease> released;
};
PoolByIdResult evict_by_id(const std::string& id, int64_t token_count, const std::string& lineage);
PoolByIdResult resident_by_id(const std::string& id, int64_t token_count, const std::string& lineage);
// Diagnostic only: written main KV/scales plus completed pooled-index rows.
// Excludes unwritten tail cells; running index tail is covered by guard_fingerprint.
uint64_t main_payload_hash(int64_t token_count);

// Read-only existing snapshot helpers: GDN/PLE/index running state and the draft's
// authoritative payload (not the MTP class's internal counters/ring duplication).
// No checkpoint or whole-session restore is performed. Diagnostic CPU copies are
// deliberately expensive and are not a performance-path measurement.
GuardFingerprint guard_fingerprint(const core::SessionState& session,
                                   const core::ModelGeometry& geometry,
                                   const core::QsaState* draft, int64_t upto);

// Compact file has original metadata but physically omits main K/V interval.
bool compact_validate(const core::SavedConversation& image,std::string& error);
OpResult compact_restore(const core::SavedConversation& image);
// Opt-in persistent logical groups; caller serializes and fences all operations.
OpResult export_group(const std::string& id,const std::string& file,int64_t tokens,const std::string& lineage);
OpResult import_group(const std::string& file,int64_t tokens,const std::string& lineage);
OpResult external_begin(int64_t p0,int64_t p1,int64_t tokens,const std::string& lineage);
struct ExternalStatus { bool active=false; int64_t p0=0,p1=0,tokens=0,missing=0; };
ExternalStatus external_status();

namespace detail {
// CUDA wrappers implemented in qsa_block_mask.cu; seven u64 counters per site:
// queries, seen_deleted, remaining_violations, invalid_ids, empty_queries,
// watched_before, watched_after.
void launch_filter(const uint8_t* deleted, const uint8_t* watched, int64_t blocks, int32_t* ids,
                   const int32_t* original_steps, int32_t* filtered_steps,
                   int64_t n_queries, int64_t cap, uint64_t* counters, void* stream);
void launch_invalidate(int32_t* page_table, int32_t* slot_block,
                       int32_t* slot_stamp, int32_t* slot_ref,
                       int64_t n_blocks, int64_t n_slots, int64_t b0, int64_t b1,
                       void* stream);
}
} // namespace strata::qsa_proto
