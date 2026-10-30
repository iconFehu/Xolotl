#pragma once
// Experimental sequence-local validity, never infer validity from nonzero bytes.
#include "qsa_window_diag.hpp"
#include <vector>
namespace strata::qsa_window_diag {
struct MtpValidity {
 std::vector<uint8_t> valid,fresh;int64_t common=0;
 bool on()const{const char* e=std::getenv("STRATA_QSA_V2_VALIDITY");return e&&std::string(e)=="1";}
 void clear(){valid.clear();fresh.clear();common=0;}
 template<class A,class B>void rebase(const A& old,const B& next,int64_t capacity){
  if(!on())return;if(valid.size()!=size_t(capacity))valid.assign(size_t(capacity),0);fresh.assign(size_t(capacity),0);
  common=0;while(common<int64_t(old.size())&&common<int64_t(next.size())&&old[size_t(common)]==next[size_t(common)])++common;
  // Draft cell i depends on main residual i AND token i+1.
  const int64_t cut=std::max<int64_t>(0,common-1);std::fill(valid.begin()+cut,valid.end(),0);
 }
 void mark(int64_t p,int64_t n){if(!on())return;if(p<0||n<0||p+n>int64_t(valid.size()))throw std::runtime_error("MTP validity write range invalid");std::fill(valid.begin()+p,valid.begin()+p+n,1);std::fill(fresh.begin()+p,fresh.begin()+p+n,1);}
 void prefill(int64_t p,int64_t n,int64_t max_t,int64_t first_needed,bool batched=false){
  if(!on())return;if(max_t<1)throw std::runtime_error("MTP validity group invalid");
  // E-9's true means handled, including a no-op when r0>=n. It writes
  // EXACTLY [max(cell0,first_needed),cell0+n), not the whole chunk.
  if(batched){const auto start=std::max(p,first_needed);if(start<p+n)mark(start,p+n-start);return;}
  for(int64_t c=0;c<n;c+=max_t){const auto T=std::min(max_t,n-c);if(p+c+T>first_needed)mark(p+c,T);}
 }
 int64_t check(int request,int64_t resume,int64_t q,int64_t window){
  if(!on())return -1;const int64_t lo=std::max<int64_t>(0,q+1-window);int64_t missing=-1,kept=0,written=0;
  for(int64_t i=lo;i<q;++i){if(i>=int64_t(valid.size())||!valid[size_t(i)]){if(missing<0)missing=i;}else if(fresh[size_t(i)])++written;else ++kept;}
  const auto path=trace_path();std::ostringstream o;o<<"{\"schema\":\"mtp-validity/v1\",\"request\":"<<request<<",\"resume\":"<<resume<<",\"q\":"<<q<<",\"active_from\":"<<lo<<",\"common_prefix\":"<<common<<",\"missing\":"<<missing<<",\"reused_valid_cells\":"<<kept<<",\"fresh_valid_cells\":"<<written<<"}\n";
  if(!path.empty()){std::FILE* f=std::fopen((path+".validity.jsonl").c_str(),"ab");if(!f)throw std::runtime_error("MTP validity log open failed");const auto s=o.str();const bool ok=std::fwrite(s.data(),1,s.size(),f)==s.size();std::fclose(f);if(!ok)throw std::runtime_error("MTP validity log write failed");}
  return missing;
 }
};
inline MtpValidity mtp_validity;
}
