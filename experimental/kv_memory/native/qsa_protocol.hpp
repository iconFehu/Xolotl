#pragma once
#include "qsa_block_store.hpp"
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace strata::qsa_proto::wire {
inline std::string quote(const std::string& text) {
    std::string out = "\"";
    for (unsigned char c : text) {
        if (c == '\\' || c == '"') { out += '\\'; out += char(c); }
        else if (c < 32) { char b[7]; std::snprintf(b, sizeof b, "\\u%04x", unsigned(c)); out += b; }
        else out += char(c);
    }
    return out + '"';
}
inline std::string hex(uint64_t value) {
    char out[17]; std::snprintf(out, sizeof out, "%016llx", (unsigned long long)value); return out;
}
inline uint64_t live_hash(const std::vector<int32_t>& ids) {
    uint64_t h = 14695981039346656037ull;
    for (const int32_t id : ids) for (int shift = 0; shift < 32; shift += 8) {
        h ^= uint8_t(uint32_t(id) >> shift); h *= 1099511628211ull;
    }
    return h;
}
inline std::string result(const OpResult& r) {
    std::ostringstream out;
    out << "\"ok\":" << (r.ok ? "true" : "false") << ",\"poisoned\":" << (r.poisoned ? "true" : "false")
        << ",\"error\":" << quote(r.error) << ",\"id\":" << quote(r.id)
        << ",\"p0\":" << r.p0 << ",\"p1\":" << r.p1 << ",\"token_count\":" << r.token_count
        << ",\"bytes\":" << r.bytes << ",\"payload_hash\":" << quote(hex(r.payload_hash))
        << ",\"running_before\":" << (r.guard_checked?quote(hex(r.running_before)):"null")
        << ",\"running_after\":" << (r.guard_checked?quote(hex(r.running_after)):"null");
    out << ",\"diagnostic_mode\":" << quote(r.profile.fast ? "fast" : "full")
        << ",\"running_guard_checked\":" << (r.guard_checked?"true":"false")
        << ",\"server_total_ms\":" << r.profile.total_ms << ",\"server_exclusive_ms\":{";
    bool first=true;for(const auto& item:r.profile.exclusive_ms){if(!first)out<<',';first=false;out<<quote(item.first)<<':'<<item.second;}
    out << '}';
    return out.str();
}
inline void emit(const std::string& op, const std::string& fields) {
    std::printf("QBRESULT {\"op\":%s,%s}\n", quote(op).c_str(), fields.c_str()); std::fflush(stdout);
}
inline void reject(const std::string& op, const std::string& error) {
    OpResult r; r.error = error; r.poisoned = poisoned(); emit(op, result(r));
}
inline std::string guard(const GuardFingerprint& g) {
    return std::string("{\"ok\":") + (g.ok ? "true" : "false") + ",\"error\":" + quote(g.error) +
        ",\"gdn\":" + quote(hex(g.gdn)) + ",\"ple\":" + quote(hex(g.ple)) +
        ",\"index_running\":" + quote(hex(g.index_running)) + ",\"draft\":" + quote(hex(g.draft)) +
        ",\"combined\":" + quote(hex(g.combined)) + "}";
}
inline std::string trace(const std::vector<SiteCounters>& all) {
    std::ostringstream out; out << "["; bool first = true;
    for (const auto& c : all) {
        if (!first) out << ','; first = false;
        out << "{\"site\":" << quote(c.site) << ",\"qsa_ordinal\":" << c.qsa_ordinal
            << ",\"queries\":" << c.queries << ",\"seen_deleted\":" << c.seen_deleted
            << ",\"remaining_violations\":" << c.remaining_violations << ",\"invalid_ids\":" << c.invalid_ids
            << ",\"empty_queries\":" << c.empty_queries << ",\"watched_before\":" << c.watched_before
            << ",\"watched_after\":" << c.watched_after << "}";
    }
    return out.str() + "]";
}
inline const char* const* pool_counter_names() {
    static const char* names[]={"resolves","lookups","requested_misses","placed","hits","reloads",
        "clock_evictions","slot_recycles","manual_slot_reuses","copied_bytes","manual_evictions","event_cursor","event_dropped"};
    return names;
}
inline std::string pool_layers(const std::vector<strata::qsa_pool_v2::LayerSnapshot>& layers) {
    std::ostringstream out;out<<'[';bool first=true;
    for(const auto& s:layers){
        if(!first)out<<',';first=false;
        out<<"{\"ordinal\":"<<s.ordinal<<",\"blocks\":"<<s.blocks<<",\"slots\":"<<s.slots
           <<",\"slot_pool_bytes\":"<<s.slot_pool_bytes<<",\"authoritative_host_bytes\":"<<s.authoritative_host_bytes
           <<",\"pooled_index_bytes\":"<<s.pooled_index_bytes<<",\"occupied_slots\":"<<s.occupied_slots
           <<",\"free_slots\":"<<s.free_slots<<",\"mapped_blocks\":"<<s.mapped_blocks
           <<",\"invalid_forward\":"<<s.invalid_forward<<",\"invalid_reverse\":"<<s.invalid_reverse
           <<",\"duplicate_owners\":"<<s.duplicate_owners<<",\"transient_pages\":"<<s.transient_pages
           <<",\"map_ok\":"<<(s.map_ok?"true":"false")<<",\"overflow\":"<<(s.overflow?"true":"false")
           <<",\"map_hash\":"<<quote(hex(s.map_hash))<<",\"details_included\":"<<(s.details_included?"true":"false")<<",\"watched_count\":"<<s.watched_count<<",\"watched_nonzero\":"<<s.watched_nonzero<<",\"watched_copy_sum\":"<<s.watched_copy_sum<<",\"watched_id_hash\":"<<quote(hex(s.watched_id_hash))<<",\"watched_copy_hash\":"<<quote(hex(s.watched_copy_hash))<<",\"counters\":[";
        for(int i=0;i<strata::qsa_pool_v2::kCounterCount;++i){if(i)out<<',';out<<s.trace.counters[i];}
        out<<"],\"events\":[";bool fe=true;
        for(const auto& e:s.trace.events){if(!fe)out<<',';fe=false;
            out<<"{\"sequence\":"<<e.sequence<<",\"bytes\":"<<e.bytes<<",\"epoch\":"<<e.epoch
               <<",\"block\":"<<e.block<<",\"slot\":"<<e.slot<<",\"previous_owner\":"<<e.previous_owner
               <<",\"prior_loads\":"<<e.prior_loads<<",\"slot_generation\":"<<e.slot_generation
               <<",\"kind\":"<<e.kind<<",\"origin\":"<<e.origin<<'}';}
        out<<"],\"watched_blocks\":[";bool fw=true;
        for(const auto& w:s.watched_blocks){if(!fw)out<<',';fw=false;
            out<<"{\"block\":"<<w.block<<",\"completed_copies\":"<<w.completed_copies<<'}';}
        out<<"]}";
    }return out.str()+']';
}
inline std::string pool_result(const PoolByIdResult& p){
    std::ostringstream out;out<<result(p.operation)<<",\"resident_verified\":"<<(p.resident_verified?"true":"false")
        <<",\"exact_map_transition\":"<<(p.exact_map_transition?"true":"false")
        <<",\"authoritative_before\":"<<quote(hex(p.authoritative_before))<<",\"authoritative_after\":"<<quote(hex(p.authoritative_after))
        <<",\"pool_before\":"<<pool_layers(p.before)<<",\"pool_after\":"<<pool_layers(p.after)<<",\"released_slots\":[";
    bool first=true;for(const auto& r:p.released){if(!first)out<<',';first=false;out<<"{\"ordinal\":"<<r.ordinal<<",\"block\":"<<r.block<<",\"slot\":"<<r.slot<<'}';}
    return out.str()+']';
}
}
