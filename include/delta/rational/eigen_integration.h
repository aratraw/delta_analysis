// include/delta/rational/eigen_integration.h
#pragma once

#include <Eigen/Core>
#include "delta/rational/rational_class.h"
#include "delta/rational/transcendentals.h"   // for delta::sqrt

namespace Eigen {

    // -------------------------------------------------------------------------
    // NumTraits for delta::Rational – tells Eigen how to handle our type.
    // -------------------------------------------------------------------------
    template<>
    struct NumTraits<delta::Rational> : GenericNumTraits<delta::Rational> {
        using Real = delta::Rational;
        using NonInteger = delta::Rational;
        using Literal = delta::Rational;

        static inline Real epsilon() {
            return delta::default_eps();
        }
        static inline Real dummy_precision() {
            return delta::default_eps();
        }

        enum {
            IsInteger = 0,
            IsSigned = 1,
            IsComplex = 0,
            RequireInitialization = 1,   // our type may allocate (big numbers, lazy nodes)
            ReadCost = 1,
            AddCost = 1,
            MulCost = 1
        };
    };

    // -------------------------------------------------------------------------
    // sqrt_impl – allows Eigen to call delta::sqrt for matrix functions.
    // -------------------------------------------------------------------------
    namespace internal {
        template<>
        struct sqrt_impl<delta::Rational> {
            static inline delta::Rational run(const delta::Rational& x) {
                return delta::sqrt(x);
            }
        };
    }

} // namespace Eigen