#pragma once
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>

namespace strata::qsa_proto {
struct OperationProfile {
    double total_ms = 0;
    std::map<std::string,double> exclusive_ms;
    bool fast = false;
};
namespace profiling {
using Clock = std::chrono::steady_clock;
struct Session;
inline thread_local Session* current = nullptr;
struct Session {
    OperationProfile value;
    Clock::time_point start=Clock::now(), last=start;
    std::string category="other";
    Session* previous=current;
    explicit Session(bool force_full=false) {
        const char* e=std::getenv("STRATA_QSA_V3_FAST");
        if(e && std::strcmp(e,"0") && std::strcmp(e,"1")) throw std::runtime_error("invalid explicit QSA fast-path flag");
        value.fast=!force_full && e && !std::strcmp(e,"1");
        current=this;
    }
    void charge() {auto n=Clock::now();value.exclusive_ms[category]+=std::chrono::duration<double,std::milli>(n-last).count();last=n;}
    OperationProfile finish() {charge();value.total_ms=std::chrono::duration<double,std::milli>(last-start).count();return value;}
    ~Session(){current=previous;}
};
struct Phase {
    Session* session=current;
    std::string previous;
    explicit Phase(const char* name) {if(session){session->charge();previous=session->category;session->category=name;}}
    ~Phase(){if(session){session->charge();session->category=previous;}}
};
inline bool fast(){return current && current->value.fast;}
}
}
