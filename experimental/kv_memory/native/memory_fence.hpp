#pragma once
#include <string>
#include <sstream>
#include <set>
#include <algorithm>
#include <cstdint>
namespace strata::memory_v1 {
struct Reply { bool handled=false,ok=false,forward=false;std::string command,error;uint64_t epoch=0; };
// Single native command stream. Ownership is checked before a command touches KV.
struct Fence {
 std::string owner;uint64_t epoch=0;
 static bool id_ok(const std::string& id){return id.size()==32&&std::all_of(id.begin(),id.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});}
 Reply consume(const std::string& line){
  std::istringstream in(line);std::string op,id,extra;in>>op;Reply r;r.handled=true;r.epoch=epoch;
  if(op=="MEM_CAPS"){r.ok=!(in>>extra);if(!r.ok)r.error="EXTRA_ARGUMENT";return r;}
  if(op=="MEM_BEGIN"){
   if(!(in>>id)||!id_ok(id)||(in>>extra)){r.error="INVALID_OWNER";return r;}
   if(!owner.empty()){r.error="TRANSACTION_BUSY";return r;}
   owner=id;r.epoch=++epoch;r.ok=true;return r;
  }
  if(op=="MEM_END"){
   if(!(in>>id)||(in>>extra)||owner.empty()||id!=owner){r.error="TRANSACTION_OWNER";return r;}
   owner.clear();r.ok=true;return r;
  }
  if(op=="MEM_CALL"){
   if(!(in>>id)||owner.empty()||id!=owner){r.error="TRANSACTION_OWNER";return r;}
   std::getline(in,r.command);auto first=r.command.find_first_not_of(' ');if(first==std::string::npos){r.error="EMPTY_COMMAND";return r;}r.command.erase(0,first);
   std::istringstream cmd(r.command);std::string kind;cmd>>kind;
   const std::set<std::string> allowed={"GEN","GENI","SAVE","RESTORE","QBGEOMPREFIX","QBMODE","QBCHECK","QBSTATUS","QBDIAG","QBNEWSEQ","QBHOTRESET","QBPOOLC","QBBANK","QBEXTERNAL","QBTRACE","QBRESETTRACE"};
   if(!allowed.count(kind)){r.error="COMMAND_NOT_ALLOWED";return r;}
   r.ok=true;r.forward=true;return r;
  }
  // Shutdown is allowed without a lease. STOP is consumed by the existing
  // cancellation reader, owned by the same server; it never grants a new lease.
  if(op=="QUIT"||op=="STOP"){r.handled=false;return r;}
  r.error="TRANSACTION_REQUIRED";return r;
 }
};
}
