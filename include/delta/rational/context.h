#pragma once

#include "rational_class.h"

namespace delta::internal {
    inline thread_local bool global_eager_mode = false;
    // default_eps = 10^-30
    inline thread_local Rational default_eps_value = Rational("1/1000000000000000000000000000000");
}

namespace delta {

    inline bool eager_mode() {
        return internal::global_eager_mode;
    }

    inline void set_eager_mode(bool flag) {
        internal::global_eager_mode = flag;
    }

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

    inline const Rational& default_eps() {
        return internal::default_eps_value;
    }

    inline void set_default_eps(const Rational& eps) {
        internal::default_eps_value = eps;
    }

} // namespace delta