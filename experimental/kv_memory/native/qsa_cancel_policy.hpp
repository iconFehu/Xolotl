#pragma once
// A cancelled prompt has no decode consumer. Its unfilled nominal draft window
// must not be validated as a completed prompt or exported as a reusable session.
#include <string>
namespace strata::cancel_policy {
inline bool validate_complete_prefill(bool cancelled) { return !cancelled; }
struct Session {
    bool reset_required = false;
    void cancel() { reset_required = true; }
    void reset_succeeded() { reset_required = false; }
    bool allows(const std::string& op) const {
        return !reset_required || op == "QBHOTRESET" || op == "QBSTATUS" || op == "QUIT";
    }
};
}
