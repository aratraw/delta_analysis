// include/delta/rational/context.h
#pragma once

#include "delta/rational/rational_class.h"

namespace delta::internal {

    // Thread-local storage for global settings
    inline thread_local bool global_eager_mode = false;
    inline thread_local Rational default_eps_value = []() -> Rational {
        // Default epsilon: 1e-30 (or 1/10^30)
        return Rational(1) / Rational("1000000000000000000000000000000");
        }();

} // namespace delta::internal

namespace delta {

    // -------------------------------------------------------------------------
    // Eager mode control
    // -------------------------------------------------------------------------
    inline bool eager_mode() {
        return internal::global_eager_mode;
    }

    inline void set_eager_mode(bool flag) {
        internal::global_eager_mode = flag;
    }

    /**
     * @brief RAII guard to temporarily enable eager evaluation.
     */
    class ScopedEagerEval {
        bool old_;
    public:
        ScopedEagerEval() : old_(internal::global_eager_mode) {
            internal::global_eager_mode = true;
        }
        ~ScopedEagerEval() {
            internal::global_eager_mode = old_;
        }
    };

    // -------------------------------------------------------------------------
    // Default epsilon control
    // -------------------------------------------------------------------------
    inline const Rational& default_eps() {
        return internal::default_eps_value;
    }

    inline void set_default_eps(const Rational& eps) {
        internal::default_eps_value = eps;
    }

} // namespace delta