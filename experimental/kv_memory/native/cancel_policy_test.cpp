#include "qsa_cancel_policy.hpp"
#include <iostream>
#include <stdexcept>
using strata::cancel_policy::Session;
void require(bool v) { if (!v) throw std::runtime_error("cancel policy test failed"); }
int main() {
    require(strata::cancel_policy::validate_complete_prefill(false));
    require(!strata::cancel_policy::validate_complete_prefill(true));
    Session s; require(s.allows("GEN")); s.cancel();
    for (const auto* op : {"GEN", "GENI", "SAVE", "RESTORE", "QBSAVE", "QBCHECK", "QBNEWSEQ"}) require(!s.allows(op));
    for (const auto* op : {"QBHOTRESET", "QBSTATUS", "QUIT"}) require(s.allows(op));
    // A failed reset cannot reopen generation. Only verified reset success can.
    require(s.reset_required); require(!s.allows("GEN"));
    s.reset_succeeded(); require(!s.reset_required); require(s.allows("GENI"));
    std::cout << "CANCEL_POLICY_PASS normal-guard,partial-refusal,reset-failure,verified-reset\n";
}
