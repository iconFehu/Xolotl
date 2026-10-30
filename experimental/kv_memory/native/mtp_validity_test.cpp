#ifdef NDEBUG
#undef NDEBUG
#endif
#include "qsa_mtp_validity.hpp"
#include <cassert>
#include <iostream>
#include <numeric>
#include <cstdlib>
#ifndef _WIN32
static void _putenv_s(const char* k,const char* v){setenv(k,v,1);}
#endif
int main(){
 _putenv_s("STRATA_QSA_V2_VALIDITY","1");_putenv_s("STRATA_QSA_V2_WINDOW_TRACE","");
 namespace policy=strata::qsa_diagnostic_policy;
 policy::full=false;try{policy::Scope scope;assert(policy::full);throw std::runtime_error("scope restoration");}catch(const std::runtime_error&){}assert(!policy::full);
 using strata::qsa_window_diag::MtpValidity;MtpValidity safety;std::vector<int64_t> no_ids,small={1,2,3,4};safety.rebase(no_ids,small,128);assert(safety.check(1,0,3,2)>=0);safety.prefill(0,3,6,-1);assert(safety.check(1,0,3,2)==-1);
 policy::guards=3;policy::reset();assert(policy::guards==0);policy::full=true;
 std::cout<<"DIAGNOSTIC_POLICY_CPU_PASS scope exception restoration, fast mode keeps MTP validity checks\n";
 MtpValidity l;
 std::vector<int64_t> empty,a(80),b(120);std::iota(a.begin(),a.end(),0);std::iota(b.begin(),b.end(),0);
 l.rebase(empty,a,128);l.prefill(0,79,6,16);assert(l.check(1,0,79,32)==-1);assert(!l.valid[0]&&l.valid[48]);
 l.rebase(a,b,128);l.prefill(79,40,6,24);assert(l.check(2,79,119,32)==-1);assert(l.fresh[90]);
 l.rebase(b,a,128);l.prefill(64,15,6,-16);assert(l.check(3,64,79,32)==-1);assert(l.valid[48]&&!l.fresh[48]&&l.fresh[64]);
 auto shortp=std::vector<int64_t>(a.begin(),a.begin()+20);l.rebase(a,shortp,128);l.prefill(16,3,6,-76);assert(l.check(4,16,19,32)==0);
 l.clear();l.rebase(empty,shortp,128);l.prefill(0,19,6,-76);assert(l.check(1,0,19,32)==-1);
 auto changed=shortp;changed[10]=999;l.rebase(shortp,changed,128);assert(!l.valid[9]&&l.valid[8]);assert(l.check(2,16,19,32)==9);
 l.clear();assert(l.check(1,0,19,32)==0);
 MtpValidity bulk;bulk.rebase(empty,a,128);bulk.prefill(0,20,6,50,true);
 for(int i=0;i<20;++i)assert(!bulk.valid[i]);
 bulk.prefill(20,59,6,50,true);for(int i=0;i<50;++i)assert(!bulk.valid[i]);for(int i=50;i<79;++i)assert(bulk.valid[i]);
 assert(bulk.check(1,0,79,32)==48); // active rows48/49 were never written
 bulk.clear();bulk.rebase(empty,a,128);bulk.prefill(0,79,6,-1,true);assert(bulk.check(1,0,79,32)==-1);
 auto deep=std::vector<int64_t>(a.begin(),a.begin()+20);bulk.rebase(a,deep,128);bulk.clear();bulk.rebase(empty,a,128);bulk.prefill(0,79,6,50,true);bulk.rebase(a,deep,128);bulk.prefill(16,3,6,-1,true);assert(bulk.check(2,16,19,32)==0);
 std::cout<<"MTP_VALIDITY_CPU_PASS cold, advance, valid rewind, missing rewind, cold fallback, shifted-token lineage, cancel invalidation\n";
}
