// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

#pragma once

#include <Eigen/Core>
#include "rational_class.h"
#include "transcendentals.h"

namespace Eigen {

    template<>
    struct NumTraits<delta::Rational> : GenericNumTraits<delta::Rational> {
        using Real = delta::Rational;
        using NonInteger = delta::Rational;
        using Literal = delta::Rational;

        static inline Real epsilon() { return delta::default_eps(); }
        static inline Real dummy_precision() { return delta::default_eps(); }

        enum {
            IsInteger = 0,
            IsSigned = 1,
            IsComplex = 0,
            RequireInitialization = 1,
            ReadCost = 1,
            AddCost = 1,
            MulCost = 1
        };
    };

    namespace internal {
        template<>
        struct sqrt_impl<delta::Rational> {
            static inline delta::Rational run(const delta::Rational& x) {
                return delta::sqrt(x);
            }
        };
    } // namespace internal

} // namespace Eigen