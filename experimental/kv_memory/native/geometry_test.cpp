#ifdef NDEBUG
#undef NDEBUG
#endif
#include "qsa_geometry.hpp"
#include <cassert>
#include <cstdio>
int main(){
 using namespace strata::geometry;
 Tape a{{1,2,3,4,5},{{2,9123}},{{0,4,false},{4,5,true}},{{0,2,false},{2,4,false},{4,5,true}}};validate(a);
 auto p=prefix(a,4);assert(p.parts.size()==1&&p.units.size()==2&&p.images.size()==1);
 int rejected=0;auto reject=[&](auto f){try{f();}catch(const std::exception&){++rejected;}};
 reject([&]{prefix(a,3);});reject([&]{auto b=a;b.units[1].a=1;validate(b);});
 reject([&]{auto b=a;b.parts[0].windows=true;validate(b);});
 reject([&]{auto b=a;b.ids[0]=7;match(a,b.ids,b.images);});
 reject([&]{auto b=a;b.images[0].hash++;match(a,b.ids,b.images);});
 reject([&]{auto b=a;b.images[0].start=8;validate(b);});
 reject([&]{auto b=a;b.units[0].max_t=0;validate(b);});
 reject([&]{auto b=a;b.units[2].batched=true;validate(b);});
 reject([&]{auto b=a;b.units[0].needed=524289;validate(b);});
 const char* file="geometry-cpu-roundtrip.txt";write(file,a);auto b=read(file);assert(a==b);std::remove(file);
 assert(rejected==9);std::puts("GEOMETRY_CPU_PASS V2 roundtrip,prefix,partial-unit,gaps,path,token,image,position,MTP-metadata");
 struct FakeDraft {int64_t n=123;void set_prompt_len(int64_t v){n=v;}} fake;
 {PromptScope<FakeDraft> scope{fake,456,false};}assert(fake.n==123);
 try{PromptScope<FakeDraft> scope{fake,123,true};fake.n=999;throw 7;}catch(int){}assert(fake.n==123);
 std::puts("MTP_PROMPT_SCOPE_CPU_PASS disabled no-op and exception restoration");
}
