#ifdef NDEBUG
#undef NDEBUG
#endif
#include "memory_fence.hpp"
#include <cassert>
#include <cstdio>
int main(){using strata::memory_v1::Fence;Fence f;const std::string a(32,'a'),b(32,'b');
 assert(f.consume("MEM_CAPS").ok);assert(!f.consume("QBSTATUS").ok);assert(!f.consume("GEN 1 1").ok);assert(!f.consume("GENI 1 image 1,2").ok);
 assert(!f.consume("MEM_BEGIN invalid").ok);assert(f.consume("MEM_BEGIN "+a).epoch==1);
 assert(!f.consume("MEM_BEGIN "+b).ok);assert(!f.consume("MEM_CALL "+b+" QBSTATUS").ok);
 assert(!f.consume("MEM_END "+b).ok);assert(f.owner==a);
 assert(!f.consume("MEM_CALL "+a+" QBREMOVE block").ok);assert(f.consume("MEM_CALL "+a+" GENI 1 image 1,2").forward);
 auto r=f.consume("MEM_CALL "+a+" QBGEOMPREFIX \"a path\"");assert(r.ok&&r.forward&&r.command=="QBGEOMPREFIX \"a path\"");
 assert(f.consume("MEM_END "+a).ok);assert(!f.consume("MEM_CALL "+a+" QBSTATUS").ok);
 assert(f.consume("MEM_BEGIN "+b).epoch==2);assert(!f.consume("MEM_CALL "+a+" RESTORE old").ok);
 assert(!f.consume("QUIT").handled);assert(f.consume("MEM_END "+b).ok);
 std::puts("MEMORY_FENCE_CPU_PASS ownership,epoch,stale-owner,raw-bypass,allowlist,geometry,shutdown");}
