#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
namespace strata::qsa_compact {
struct Slice { uint64_t packed,original,bytes; };
struct Layout {
 uint64_t full,lo,hi;
 Layout(uint64_t n,uint64_t a,uint64_t b):full(n),lo(a),hi(b){
  if(a>b||b>n)throw std::runtime_error("compact omitted range outside part");
 }
 uint64_t gap()const{return hi-lo;}
 uint64_t size()const{return full-gap();}
 std::vector<Slice> slices(uint64_t at,uint64_t count)const{
  if(at>size()||count>size()-at)throw std::runtime_error("compact packed span outside part");
  std::vector<Slice> result;
  if(at<lo&&count){auto n=std::min(count,lo-at);result.push_back({at,at,n});at+=n;count-=n;}
  if(count)result.push_back({at,at+gap(),count});
  return result;
 }
};
inline Layout main_part(uint64_t full,uint64_t pages){
 if(pages<16512||!full||full%pages)throw std::runtime_error("compact page layout incompatible");
 const auto per=full/pages;return Layout(full,128*per,16512*per);
}
}
