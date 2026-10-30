#pragma once
// Independent diagnostic only. Stores no model weights, KV payloads, or credentials.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace strata::geometry {
template<class Drafter>struct PromptScope {Drafter& draft;int64_t n;bool active;~PromptScope(){if(active)draft.set_prompt_len(n);}};
struct Span {int64_t a=0,b=0;bool windows=false;bool operator==(const Span&)const=default;};
struct Unit {int64_t a=0,b=0;bool windows=false;int64_t needed=0;bool batched=false;int max_t=4;bool operator==(const Unit&)const=default;};
struct Image {int64_t start=0;uint64_t hash=0;bool operator==(const Image&)const=default;};
struct Tape {
 std::vector<int32_t> ids;std::vector<Image> images;std::vector<Span> parts;std::vector<Unit> units;
 bool operator==(const Tape&)const=default;
};
inline void require(bool ok,const char* message){if(!ok)throw std::runtime_error(std::string("GEOMETRY_MISMATCH: ")+message);}
inline void validate(const Tape& t){
 require(!t.ids.empty()&&t.ids.size()<=524288,"token count");
 require(!t.parts.empty()&&t.parts.size()<=8192&&t.units.size()<=524288,"span count");
 auto coverage=[&](const auto& v){
  int64_t at=0;for(const auto& s:v){require(s.a==at&&s.b>s.a&&s.b<=int64_t(t.ids.size()),"noncontiguous span");at=s.b;}
  require(at==int64_t(t.ids.size()),"incomplete span coverage");
 };coverage(t.parts);coverage(t.units);
 size_t p=0;for(const auto& u:t.units){while(u.a>=t.parts[p].b)++p;require(u.a>=t.parts[p].a&&u.b<=t.parts[p].b&&u.windows==t.parts[p].windows,"unit/part disagreement");require(u.needed>=-32832&&u.needed<=524288&&u.max_t>0&&u.max_t<=64&&(!u.windows||!u.batched),"MTP write metadata");}
 int64_t previous=-1;for(const auto& i:t.images){require(i.start>=0&&i.start<int64_t(t.ids.size())&&i.start>previous,"image positions");previous=i.start;}
}
inline Tape prefix(const Tape& in,int64_t length){
 require(length>0&&length<=int64_t(in.ids.size()),"prefix length");Tape out;
 out.ids.assign(in.ids.begin(),in.ids.begin()+length);
 for(auto i:in.images)if(i.start<length)out.images.push_back(i);
 for(auto s:in.parts){if(s.a>=length)break;s.b=std::min(s.b,length);out.parts.push_back(s);}
 for(auto s:in.units){if(s.b>length)break;out.units.push_back(s);}
 validate(out);return out;
}
inline void write(const std::string& file,const Tape& t){
 validate(t);std::ofstream f(file,std::ios::trunc);require(bool(f),"output open");
 f<<"QSA_GEOMETRY_V2\n"<<t.ids.size()<<' '<<t.images.size()<<' '<<t.parts.size()<<' '<<t.units.size()<<'\n';
 for(auto id:t.ids)f<<id<<' ';f<<'\n';for(auto i:t.images)f<<i.start<<' '<<i.hash<<'\n';
 for(auto s:t.parts)f<<s.a<<' '<<s.b<<' '<<int(s.windows)<<'\n';
 for(auto u:t.units)f<<u.a<<' '<<u.b<<' '<<int(u.windows)<<' '<<u.needed<<' '<<int(u.batched)<<' '<<u.max_t<<'\n';
 f.flush();require(bool(f),"output write");
}
inline Tape read(const std::string& file){
 std::ifstream f(file);require(bool(f),"input open");std::string magic;size_t n=0,ni=0,np=0,nu=0;
 f>>magic>>n>>ni>>np>>nu;require(bool(f)&&magic=="QSA_GEOMETRY_V2"&&n>0&&n<=524288&&ni<=4096&&np>0&&np<=8192&&nu<=524288,"input header");
 Tape t;t.ids.resize(n);t.images.resize(ni);t.parts.resize(np);t.units.resize(nu);
 for(auto& id:t.ids){int64_t v;f>>v;require(bool(f)&&v>=INT32_MIN&&v<=INT32_MAX,"input token");id=int32_t(v);}
 for(auto& i:t.images)f>>i.start>>i.hash;
 for(auto& s:t.parts){int mode=-1;f>>s.a>>s.b>>mode;require(bool(f)&&(mode==0||mode==1),"input span");s.windows=mode==1;}
 for(auto& u:t.units){int mode=-1,batch=-1;f>>u.a>>u.b>>mode>>u.needed>>batch>>u.max_t;require(bool(f)&&(mode==0||mode==1)&&(batch==0||batch==1),"input MTP unit");u.windows=mode==1;u.batched=batch==1;}
 std::string extra;require(!(f>>extra),"trailing input");validate(t);return t;
}
inline void match(const Tape& saved,const std::vector<int32_t>& ids,const std::vector<Image>& images){
 require(saved.ids==ids,"token identity");require(saved.images==images,"image identity");
}
}
