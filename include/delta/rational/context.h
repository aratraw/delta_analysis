// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

#pragma once

#include "rational_fwd.h"
#include "storage.h"

namespace delta::internal {
    // Эпсилон по умолчанию = 1e-30 = 1/10^30
    inline Value default_eps_value = Value("1/1000000000000000000000000000000");

    inline void reset_default_eps() {
        default_eps_value = Value("1/1000000000000000000000000000000");
    }
} // namespace delta::internal

namespace delta {
    inline Rational default_eps() {
        assert(internal::default_eps_value > 0);
        return Rational(internal::default_eps_value);
    }

    inline void set_default_eps(const Rational& eps) {
        internal::default_eps_value = eps.value();
    }

    inline void reset_default_eps() {
        internal::reset_default_eps();
    }
} // namespace delta