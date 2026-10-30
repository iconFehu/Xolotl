#pragma once
// Opt-in, read-only diagnostic. Interpret logical cells using [page,head,row,dim],
// exactly as kv_q8 scatter/gather; never pretend physical ring order is history.
#include "qsa_window_diag.hpp"
#include "strata/kernels/qsa.hpp"
#include <array>
#include <limits>
namespace strata::qsa_window_diag {
inline void draft_copy(void* to,const void* from,size_t n){if(n&&(!from||cudaMemcpy(to,from,n,cudaMemcpyDefault)!=cudaSuccess))throw std::runtime_error("draft diagnostic copy failed");}
inline double draft_point(const char* phase,int request,int64_t valid_upto,int64_t main_upto,
 const strata::core::QsaState& st,const strata::core::ModelGeometry& g,int64_t window,int64_t first_needed,
 const std::vector<int32_t>& context){
 if(!strata::qsa_diagnostic_policy::full)return 0;const auto path=trace_path();if(path.empty())return 0;++strata::qsa_diagnostic_policy::drafts;const auto t0=std::chrono::steady_clock::now();
 if(cudaDeviceSynchronize()!=cudaSuccess)throw std::runtime_error("draft diagnostic fence failed");
 if(!st.kv_int8||st.kv_q4||st.kv_hybrid||st.kv_mode!=2||window<=0||valid_upto<0||main_upto<valid_upto||main_upto>st.max_cells||st.n_slots<=0)
  throw std::runtime_error("draft diagnostic requires finite-window INT8 ring and explicit valid extent");
 const int64_t page=strata::kernels::qsa_real_shapes().page_size,heads=g.n_head_kv,dim=g.head_dim;
 if(page!=4||heads!=2||dim!=256)throw std::runtime_error("draft diagnostic geometry differs");
 const int64_t rounded=((main_upto+page-1)/page)*page;
 // window_ids uses n_kv=q+1 for the next write/query q. The query writes its
 // own cell first, so the pre-existing KV it may read is [q+1-window,q).
 const int64_t active0=std::max<int64_t>(0,valid_upto+1-window);
 const std::array<const void*,4> hp={st.host.k_q,st.host.v_q,st.host.k_scale,st.host.v_scale};
 const std::array<const void*,4> gp={st.k_q,st.v_q,st.k_scale,st.v_scale};
 const std::array<size_t,4> perhead={size_t(dim),size_t(dim),size_t(dim/64)*2,size_t(dim/64)*2};
 const std::array<const char*,4> names={"K_codes","V_codes","K_scales","V_scales"};
 std::array<std::vector<uint8_t>,4> host,ring;std::vector<int32_t> table(size_t(st.n_pages),0);
 draft_copy(table.data(),st.page_table,table.size()*sizeof(int32_t));
 strata::core::SessionHasher serialized(4);
 const std::string file=path+".r"+std::to_string(request)+"."+phase+".cells.u64";
 std::FILE* dump=std::fopen(file.c_str(),"wb");if(!dump)throw std::runtime_error("draft cell diagnostic open failed");
 bool write_ok=true;uint64_t mismatched_bytes=0,checked_bytes=0;int64_t first_bad_cell=-1;int first_bad_part=-1;
 std::ostringstream out;out<<"{\"schema\":\"draft-active-diag/v1\",\"request\":"<<request<<",\"pid\":"<<process_id()<<",\"phase\":"<<strata::qsa_proto::wire::quote(phase)
 <<",\"valid_upto\":"<<valid_upto<<",\"main_upto\":"<<main_upto<<",\"rounded_upto\":"<<rounded<<",\"active_from\":"<<active0<<",\"window\":"<<window<<",\"prefill_first_needed\":"<<first_needed
 <<",\"context_hash\":"<<strata::qsa_proto::wire::quote(strata::qsa_proto::wire::hex(bytes_hash(context.data(),context.size()*4)))<<",\"page_size\":4,\"heads\":"<<heads<<",\"head_dim\":"<<dim<<",\"ring_slots\":"<<st.n_slots<<",\"parts\":[";
 for(size_t part=0;part<4;++part){
  const uint64_t n=uint64_t(rounded)*uint64_t(heads)*perhead[part];host[part].resize(size_t(n));draft_copy(host[part].data(),hp[part],size_t(n));serialized.update(&n,sizeof(n));serialized.update(host[part].data(),size_t(n));
  ring[part].resize(size_t(st.n_slots*page*heads)*perhead[part]);draft_copy(ring[part].data(),gp[part],ring[part].size());
  uint64_t outside=14695981039346656037ull,active=outside,unused=outside,ringhash=outside;uint64_t nz_old=0,nz_active=0,nz_unused=0;std::vector<uint64_t> cellhash(size_t(rounded),0);
  for(int64_t cell=0;cell<rounded;++cell){
   uint64_t one=14695981039346656037ull;uint64_t* hash=cell<active0?&outside:cell<valid_upto?&active:&unused;
   uint64_t* nz=cell<active0?&nz_old:cell<valid_upto?&nz_active:&nz_unused;
   for(int64_t h=0;h<heads;++h){
    const size_t logical=size_t(((cell/page)*heads+h)*page+cell%page)*perhead[part];const uint8_t* src=host[part].data()+logical;
    one=bytes_hash(src,perhead[part],one);*hash=bytes_hash(src,perhead[part],*hash);for(size_t j=0;j<perhead[part];++j)*nz+=src[j]!=0;
    if(cell>=active0&&cell<valid_upto){
     const int32_t slot=table[size_t(cell/page)];if(slot<0||slot>=st.n_slots||slot!=(cell/page)%st.n_slots){std::fclose(dump);throw std::runtime_error("draft active logical-to-ring mapping invalid");}
     const size_t physical=size_t((int64_t(slot)*heads+h)*page+cell%page)*perhead[part];const uint8_t* r=ring[part].data()+physical;
     ringhash=bytes_hash(r,perhead[part],ringhash);checked_bytes+=perhead[part];for(size_t j=0;j<perhead[part];++j)if(src[j]!=r[j]){++mismatched_bytes;if(first_bad_cell<0){first_bad_cell=cell;first_bad_part=int(part);}}
    }
   }
   cellhash[size_t(cell)]=one;
  }
  write_ok=write_ok&&std::fwrite(cellhash.data(),sizeof(uint64_t),cellhash.size(),dump)==cellhash.size();
  if(part)out<<',';const auto q=strata::qsa_proto::wire::quote;const auto hex=strata::qsa_proto::wire::hex;
  out<<"{\"name\":"<<q(names[part])<<",\"bytes_per_head_cell\":"<<perhead[part]<<",\"serialized_bytes\":"<<n<<",\"outside_hash\":"<<q(hex(outside))<<",\"active_hash\":"<<q(hex(active))<<",\"unused_hash\":"<<q(hex(unused))<<",\"ring_active_hash\":"<<q(hex(ringhash))<<",\"nonzero_bytes\":["<<nz_old<<','<<nz_active<<','<<nz_unused<<"]}";
 }
 std::fclose(dump);if(!write_ok)throw std::runtime_error("draft cell diagnostic write failed");
 // conversation_kv_source(index=false) still includes a zero-length pooled part.
 const uint64_t zero=0;serialized.update(&zero,sizeof(zero));
 out<<"],\"serialized_hash\":"<<strata::qsa_proto::wire::quote(strata::qsa_proto::wire::hex(serialized.digest()))<<",\"cell_hash_file\":"<<strata::qsa_proto::wire::quote(file)
 <<",\"host_ring_checked_bytes\":"<<checked_bytes<<",\"host_ring_mismatched_bytes\":"<<mismatched_bytes<<",\"first_ring_bad_cell\":"<<first_bad_cell<<",\"first_ring_bad_part\":"<<first_bad_part<<"}\n";
 const auto line=out.str();std::FILE* f=std::fopen((path+".draft.jsonl").c_str(),"ab");if(!f)throw std::runtime_error("draft diagnostic log open failed");const bool ok=std::fwrite(line.data(),1,line.size(),f)==line.size();std::fclose(f);if(!ok)throw std::runtime_error("draft diagnostic log write failed");
 if(mismatched_bytes)throw std::runtime_error("draft active host/ring bytes differ");
 return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
}
}
