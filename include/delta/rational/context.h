// context.h
#pragma once

#include "rational_fwd.h"
#include "storage.h"
#include "utils.h"   // для dumb_int

namespace delta::internal {

    // Эпсилон по умолчанию = 1 / 10^30
    inline thread_local Value default_eps_value = []() -> Value {
        // 10^30 = 1 000 000 000 000 000 000 000 000 000 000
        dumb_int denom("1000000000000000000000000000000");
        return Value(1) / Value(denom);
        }();

} // namespace delta::internal

namespace delta {

    inline Rational default_eps() {
        return Rational(internal::default_eps_value);
    }

    inline void set_default_eps(const Rational& eps) {
        internal::default_eps_value = eps.value();
    }

} // namespace delta