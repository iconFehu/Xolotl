#ifdef NDEBUG
#undef NDEBUG
#endif
#include "qsa_hot_reset_contract.hpp"
#include <cassert>
#include <cstdio>
int main(){
 strata::core::QsaState st{};st.kv_mode=1;st.kv_int8=true;st.max_cells=524288;st.n_pages=131072;st.n_slots=8192;
 st.host.k_q=reinterpret_cast<decltype(st.host.k_q)>(uintptr_t(1));st.host.v_q=reinterpret_cast<decltype(st.host.v_q)>(uintptr_t(1));st.host.k_scale=reinterpret_cast<decltype(st.host.k_scale)>(uintptr_t(1));st.host.v_scale=reinterpret_cast<decltype(st.host.v_scale)>(uintptr_t(1));
 auto check=[&](bool draft,bool allowed){bool good=true;try{strata::qsa_hot_reset::validate_host(st,524288,draft);}catch(...){good=false;}assert(good==allowed);};
 check(false,true);check(true,false);st.kv_mode=2;check(true,true);check(false,false);
 st.n_pages--;check(true,false);st.n_pages++;st.host.k_q=nullptr;check(true,false);
 std::puts("HOT_RESET_CONTRACT_CPU_PASS main-mode1,draft-ring-mode2,cross-mode-refused,bounds,missing-pointer");
}
