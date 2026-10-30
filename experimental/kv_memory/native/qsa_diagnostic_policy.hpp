#pragma once
#include <cstdint>
#include <sstream>
namespace strata::qsa_diagnostic_policy {
// Only readbacks/export timing is switched. Numeric, MTP validity and transfer
// admission paths are unchanged. Default remains full diagnostics.
inline bool full=true;
inline uint64_t guards=0,drafts=0,logit_rows=0;
inline void reset(){guards=drafts=logit_rows=0;}
inline std::string counters(){std::ostringstream s;s<<"{\"guards\":"<<guards<<",\"drafts\":"<<drafts<<",\"logit_rows\":"<<logit_rows<<'}';return s.str();}
struct Scope {bool previous=full;Scope(){full=true;}~Scope(){full=previous;}};
}
