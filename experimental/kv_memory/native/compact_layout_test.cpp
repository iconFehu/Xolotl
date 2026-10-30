#ifdef NDEBUG
#undef NDEBUG
#endif
#include "qsa_compact_layout.hpp"
#include <cassert>
#include <cstring>
#include <cstdio>
#include <limits>
using namespace strata::qsa_compact;
int main(){
 Layout l(64,16,32);assert(l.size()==48);
 auto cross=l.slices(8,24);assert(cross.size()==2&&cross[0].original==8&&cross[0].bytes==8&&cross[1].packed==16&&cross[1].original==32&&cross[1].bytes==16);
 assert(l.slices(16,4)[0].original==32);assert(l.slices(48,0).empty());
 bool rejected=false;try{l.slices(47,2);}catch(...){rejected=true;}assert(rejected);
 rejected=false;try{Layout bad(32,16,64);}catch(...){rejected=true;}assert(rejected);
 unsigned char original[64],packed[48]{},restored[64]{};for(int i=0;i<64;++i)original[i]=static_cast<unsigned char>(i+1);
 for(auto s:l.slices(0,48))std::memcpy(packed+s.packed,original+s.original,s.bytes);
 // Segmented reads cross the hole at a different boundary from the save.
 for(uint64_t at=0;at<48;at+=7)for(auto s:l.slices(at,std::min<uint64_t>(7,48-at)))std::memcpy(restored+s.original,packed+s.packed,s.bytes);
 for(int i=0;i<64;++i)assert(restored[i]==(i>=16&&i<32?0:original[i]));
 uint64_t saved=0;for(uint64_t per:{2048,2048,64,64}){auto x=main_part(22233*per,22233);saved+=12*x.gap();}assert(saved==830472192);
 rejected=false;try{main_part(99,22233);}catch(...){rejected=true;}assert(rejected);
 Layout huge(std::numeric_limits<uint64_t>::max(),4,8);rejected=false;try{huge.slices(huge.size(),1);}catch(...){rejected=true;}assert(rejected);
 std::puts("COMPACT_LAYOUT_CPU_PASS crossing,segmented-roundtrip,original-positions,bounds,overflow,830472192-byte-omission");
}
