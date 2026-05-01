// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// eigen_integration.h
// -----------------------------------------------------------------------------
// Integration of delta::Rational with Eigen3.
//
// This header provides the necessary specializations for Eigen to treat
// delta::Rational as a valid scalar type. It enables using Rational in
// Eigen::Matrix, Eigen::Array, and all standard Eigen algorithms that only
// require basic arithmetic (+, -, *, /).
//
// For transcendental functions (sqrt, exp, log, sin, cos, tan, asin, acos,
// atan, pow), Eigen uses ADL (Argument‑Dependent Lookup). Since delta::Rational
// lives in namespace delta, and we provide all these functions there,
// no explicit specialisations are needed – they are found automatically.
//
// Example:
//   Eigen::Array<delta::Rational, Dynamic, 1> A(3);
//   A << 1_r, 2_r, 3_r;
//   auto B = A.sin();    // element-wise sine using delta::sin
//
// All functions use the global default epsilon (see context.h) for precision.
// You can change it with delta::set_default_eps().
//
// NOTE: Not every Eigen algorithm is guaranteed to work with Rational.
// In particular, linear algebra operations that rely on floating‑point
// thresholds (e.g., JacobiSVD, eigenvalue solvers) may be extremely slow
// or numerically unstable. Use with caution and prefer rational‑preserving
// algorithms where possible.
// -----------------------------------------------------------------------------

// ToDo: Research, Implement, Optimize, Test deeper Eigen integration with our Rational and LazyRational if appropriate.
#pragma once

#include <Eigen/Core>
#include "rational_class.h"
#include "transcendentals.h"

namespace Eigen {

    // ----------------------------------------------------------------------------
    // NumTraits specialization for delta::Rational
    // ----------------------------------------------------------------------------
    template<>
    struct NumTraits<delta::Rational> : GenericNumTraits<delta::Rational> {
        using Real = delta::Rational;
        using NonInteger = delta::Rational;
        using Literal = delta::Rational;

        // Precision thresholds – used by Eigen's numerical algorithms.
        // We return the global default epsilon (see context.h).
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

    // ----------------------------------------------------------------------------
    // NOTE: No explicit specializations for exp_impl, etc. are needed.
    // Eigen relies on ADL (Argument‑Dependent Lookup) to find functions like
    // sin, cos, exp, log, etc. in the namespace of the scalar type.
    // Since delta::Rational lives in namespace delta and we provide all these
    // functions in transcendentals.h, they are found automatically.
    // 
    // The sqrt_impl specialization below is an exception because Eigen's
    // internal sqrt machinery sometimes needs it for complex or custom types.
    // For Rational, we provide it to be safe.
    // ----------------------------------------------------------------------------

    namespace internal {
        template<>
        struct sqrt_impl<delta::Rational> {
            static inline delta::Rational run(const delta::Rational& x) {
                return delta::sqrt(x);
            }
        };
    } // namespace internal

} // namespace Eigen