// context.h
#pragma once

#include "rational_fwd.h"
#include "storage.h"   // для internal::Value

namespace delta::internal {
    inline thread_local bool global_eager_mode = false;

    // Определение default_eps_value как Value, инициализация через BigStorage (10^-30)
    inline thread_local Value default_eps_value = [] {
        boost::multiprecision::cpp_int num(1);
        boost::multiprecision::cpp_int den(1);
        for (int i = 0; i < 30; ++i) den *= 10;
        return Value(BigStorage(num, den));
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