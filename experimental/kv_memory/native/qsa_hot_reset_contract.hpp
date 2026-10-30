#pragma once
#include "strata/core/session.hpp"
#include <stdexcept>
namespace strata::qsa_hot_reset {
inline void validate_host(const core::QsaState& st,int64_t cells,bool draft){
 if(cells<=0||cells>524288||st.kv_mode!=(draft?2:1)||!st.kv_int8||st.kv_q4||st.kv_hybrid||st.max_cells!=cells||st.n_pages!=(cells+3)/4||st.n_slots<=0||st.n_slots>=st.n_pages||!st.host.k_q||!st.host.v_q||!st.host.k_scale||!st.host.v_scale)
  throw std::runtime_error(draft?"hot discard requires authoritative finite-window INT8 draft ring mode2":"hot discard requires authoritative INT8 main stream mode1");
}
}
