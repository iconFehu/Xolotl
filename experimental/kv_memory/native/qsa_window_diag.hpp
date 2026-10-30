#pragma once
#include "qsa_protocol.hpp"
#include "qsa_diagnostic_policy.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <cuda_runtime.h>
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace strata::qsa_window_diag {
inline int process_id(){
#if defined(_WIN32)
 return _getpid();
#else
 return getpid();
#endif
}
inline std::string trace_path(){const char* p=std::getenv("STRATA_QSA_V2_WINDOW_TRACE");return p&&*p?std::string(p)+".pid"+std::to_string(process_id()):std::string();}
inline uint64_t bytes_hash(const void* data, size_t n, uint64_t h=14695981039346656037ull) {
    const auto* p=static_cast<const uint8_t*>(data);
    for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ull;}return h;
}
inline uint32_t bits(float f){uint32_t out;std::memcpy(&out,&f,4);return out;}
inline double state_point(const char* phase,int request,int64_t position,int64_t produced,
 const strata::core::SessionState& ss,const strata::core::ModelGeometry& g,
 const strata::core::QsaState* draft,const std::vector<int32_t>& residency,const std::vector<int32_t>* context=nullptr){
 if(!strata::qsa_diagnostic_policy::full)return 0;const std::string path=trace_path();if(path.empty())return 0;
 int limit=32;if(const char* v=std::getenv("STRATA_QSA_V2_TRACE_MAXTOKENS"))limit=std::clamp(std::atoi(v),1,128);
 if(produced>=limit)return 0;
 const auto t0=std::chrono::steady_clock::now();
 const auto fp=strata::qsa_proto::guard_fingerprint(ss,g,draft,draft?position:0);
 if(!fp.ok)throw std::runtime_error("diagnostic state guard failed: "+fp.error);
 strata::core::ConversationCheckpoint c;std::string err;
 if(!strata::core::conversation_checkpoint_save(c,ss,g,err))throw std::runtime_error(err);
 std::ostringstream out;out<<"{\"schema\":\"qsa-state-diag/v1\",\"pid\":"<<process_id()<<",\"request\":"<<request<<",\"phase\":"<<strata::qsa_proto::wire::quote(phase)
 <<",\"position\":"<<position<<",\"produced\":"<<produced<<",\"draft_checked\":"<<(draft?"true":"false")<<",\"context_hash\":"<<strata::qsa_proto::wire::quote(context?strata::qsa_proto::wire::hex(bytes_hash(context->data(),context->size()*sizeof(int32_t))):"")<<",\"guard\":"<<strata::qsa_proto::wire::guard(fp)<<",\"gdn_layers\":[";
 const size_t bytes=ss.gdn_alloc?c.gdn.size()/size_t(ss.gdn_alloc):0;
 for(int64_t i=0;i<ss.gdn_alloc;++i){if(i)out<<',';out<<strata::qsa_proto::wire::quote(strata::qsa_proto::wire::hex(bytes_hash(c.gdn.data()+size_t(i)*bytes,bytes)));}
 out<<"],\"ple_parts\":{\"checkpoint_bytes\":"<<c.ple.size()<<",\"checkpoint_hash\":"<<strata::qsa_proto::wire::quote(strata::qsa_proto::wire::hex(bytes_hash(c.ple.data(),c.ple.size())))
 <<",\"prev\":["<<ss.ple_prev[0]<<','<<ss.ple_prev[1]<<"],\"token\":"<<ss.ple_token<<'}';
 out<<",\"index_parts\":{\"tails\":"<<strata::qsa_proto::wire::quote(strata::qsa_proto::wire::hex(bytes_hash(c.tails.data(),c.tails.size())))
 <<",\"dead\":"<<strata::qsa_proto::wire::quote(strata::qsa_proto::wire::hex(bytes_hash(c.dead.data(),c.dead.size())))
 <<",\"block_pos\":"<<strata::qsa_proto::wire::quote(strata::qsa_proto::wire::hex(bytes_hash(c.block_pos.data(),c.block_pos.size())))<<'}';
 if(std::strcmp(phase,"prefill")==0){
  const std::string file=path+".r"+std::to_string(request)+".ple.bin";std::FILE* f=std::fopen(file.c_str(),"wb");
  if(!f)throw std::runtime_error("PLE diagnostic open failed");const auto written=std::fwrite(c.ple.data(),1,c.ple.size(),f);std::fclose(f);if(written!=c.ple.size())throw std::runtime_error("PLE diagnostic write failed");out<<",\"ple_file\":"<<strata::qsa_proto::wire::quote(file);
 }
 out<<",\"index_block_positions\":[";for(size_t i=0;i<c.block_pos.size()/sizeof(int32_t);++i){if(i)out<<',';int32_t v;std::memcpy(&v,c.block_pos.data()+i*sizeof(v),sizeof(v));out<<v;}
 out<<"],\"residency_hash\":"<<strata::qsa_proto::wire::quote(strata::qsa_proto::wire::hex(bytes_hash(residency.data(),residency.size()*sizeof(int32_t))));
 if(std::strcmp(phase,"prefill")==0){out<<",\"residency\":[";for(size_t i=0;i<residency.size();++i){if(i)out<<',';out<<residency[i];}out<<']';}
 out<<"}\n";const std::string line=out.str();std::FILE* f=std::fopen((path+".states.jsonl").c_str(),"ab");if(!f)throw std::runtime_error("state trace open failed");const auto n=std::fwrite(line.data(),1,line.size(),f);std::fclose(f);if(n!=line.size())throw std::runtime_error("state trace write failed");
 return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
}
// Diagnostic lane only. No sampling/state change; its synchronous copies and IO
// are timed separately and deducted from DraftPolicy.observe's measured cost.
inline double record(strata::core::Verifier& ver, int request, int64_t produced,
                     int64_t p,int T,int T_mtp,bool suffix,int match,int chain,int accepted,
                     const std::vector<int32_t>& consumed,const std::vector<int32_t>& window,
                     const std::vector<int32_t>& predicted,const std::vector<float>& probability,
                     const std::vector<int32_t>& residency,int64_t residual_values) {
    if(!strata::qsa_diagnostic_policy::full)return 0;const std::string private_path=trace_path();if(private_path.empty())return 0;const char* path=private_path.c_str();
    int limit=32;if(const char* v=std::getenv("STRATA_QSA_V2_TRACE_MAXTOKENS"))limit=std::clamp(std::atoi(v),1,128);
    const bool detailed=produced<limit;
    const auto begin=std::chrono::steady_clock::now();
    const auto quote=strata::qsa_proto::wire::quote;
    const auto hex=strata::qsa_proto::wire::hex;
    std::ostringstream out;
    out<<"{\"schema\":\"qsa-window-diag/v1\",\"request\":"<<request<<",\"produced_before\":"<<produced
       <<",\"position\":"<<p<<",\"T\":"<<T<<",\"T_mtp\":"<<T_mtp<<",\"from_suffix\":"<<(suffix?"true":"false")
       <<",\"suffix_match\":"<<match<<",\"chain\":"<<chain<<",\"accepted\":"<<accepted
       <<",\"residency_hash\":"<<quote(hex(bytes_hash(residency.data(),residency.size()*sizeof(int32_t))))
       <<",\"window\":[";
    for(int i=0;i<T;++i){if(i)out<<',';out<<window[size_t(i)];}out<<"],\"predicted\":[";
    for(int i=0;i<T;++i){if(i)out<<',';out<<predicted[size_t(i)];}out<<"],\"dprob_bits\":[";
    for(int i=0;i<T_mtp-1;++i){if(i)out<<',';out<<bits(probability[size_t(i)]);}out<<"],\"committable_rows\":[";
    uint64_t history=bytes_hash(consumed.data(),consumed.size()*sizeof(int32_t));
    const int rows=detailed?int(std::min<int64_t>(accepted+1,limit-produced)):0;
    for(int t=0;t<rows;++t){
        history=bytes_hash(&window[size_t(t)],sizeof(int32_t),history);
        ++strata::qsa_diagnostic_policy::logit_rows;
        std::vector<float> logits(size_t(ver.vocab()));
        if(!ver.copy_logits(t,logits.data()))throw std::runtime_error("v2 diagnostic logits copy failed");
        std::vector<size_t> order(logits.size());std::iota(order.begin(),order.end(),0);
        const size_t top=std::min<size_t>(16,order.size());
        std::partial_sort(order.begin(),order.begin()+top,order.end(),[&](size_t a,size_t b){return logits[a]>logits[b]||(logits[a]==logits[b]&&a<b);});
        std::vector<float> residual(size_t(residual_values),0);
        if(cudaMemcpy(residual.data(),ver.final_R(t),residual.size()*sizeof(float),cudaMemcpyDeviceToHost)!=cudaSuccess)
            throw std::runtime_error("v2 diagnostic residual copy failed");
        if(t)out<<',';
        out<<"{\"row\":"<<t<<",\"output_index\":"<<(produced+t)<<",\"context_hash\":"<<quote(hex(history))
           <<",\"logits_hash\":"<<quote(hex(bytes_hash(logits.data(),logits.size()*sizeof(float))))
           <<",\"residual_hash\":"<<quote(hex(bytes_hash(residual.data(),residual.size()*sizeof(float))))
           <<",\"top_margin\":"<<(double(logits[order[0]])-double(logits[order[1]]))<<",\"top\":[";
        for(size_t i=0;i<top;++i){if(i)out<<',';out<<"{\"id\":"<<order[i]<<",\"value\":"<<double(logits[order[i]])<<",\"bits\":"<<bits(logits[order[i]])<<'}';}
        out<<']';
        if(produced+t<32){
            const std::string bin=std::string(path)+".r"+std::to_string(request)+".o"+std::to_string(produced+t)+".f32";
            std::FILE* f=std::fopen(bin.c_str(),"wb");
            if(!f)throw std::runtime_error("v2 diagnostic file open failed");
            const size_t written=std::fwrite(logits.data(),sizeof(float),logits.size(),f);std::fclose(f);
            if(written!=logits.size())throw std::runtime_error("v2 diagnostic logits file write failed");
            out<<",\"full_logits_file\":"<<quote(bin)<<",\"vocab\":"<<logits.size();
        }
        out<<'}';
    }
    out<<"]}\n";
    std::FILE* f=std::fopen(path,"ab");if(!f)throw std::runtime_error("v2 window trace open failed");
    const std::string line=out.str();const size_t written=std::fwrite(line.data(),1,line.size(),f);std::fclose(f);
    if(written!=line.size())throw std::runtime_error("v2 window trace write failed");
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
}
}
