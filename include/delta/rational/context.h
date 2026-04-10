// context.h
#pragma once

#include "rational_fwd.h"
#include "storage.h"
#include "utils.h"   // для dumb_int

namespace delta::internal {
    inline thread_local bool global_eager_mode = false;

    // Эпсилон = 1 / 10^30
    // Инициализируем BigStorage напрямую: числитель 1, знаменатель 10^30
    inline thread_local Value default_eps_value = []() -> Value {
        // 10^30 = 1 000 000 000 000 000 000 000 000 000 000
        dumb_int denominator("1000000000000000000000000000000");
        return Value(BigStorage(dumb_int(1), denominator));
        }();
}

namespace delta {
    inline bool eager_mode() { return internal::global_eager_mode; }
    inline void set_eager_mode(bool flag) { internal::global_eager_mode = flag; }

    class ScopedEagerEval {
        bool old_;
    public:
        ScopedEagerEval() : old_(internal::global_eager_mode) { internal::global_eager_mode = true; }
        ~ScopedEagerEval() { internal::global_eager_mode = old_; }
    };

    Rational default_eps();
    void set_default_eps(const Rational& eps);
}