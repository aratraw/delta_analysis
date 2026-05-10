// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0
// eigen_integration.h
// -----------------------------------------------------------------------------
// Integration of delta::Rational and delta::GaussQi with Eigen3.
//
// This header provides the necessary specializations for Eigen to treat
// delta::Rational and delta::GaussQi as valid scalar types. It enables using
// these types in Eigen::Matrix, Eigen::Array, and standard Eigen algorithms.
//
// ARCHITECTURE & USAGE:
//   1. ELEMENT-WISE TRANSCENDENTALS (automatic via ADL):
//      Eigen finds delta::sin, delta::exp, delta::sqrt, etc. automatically
//      because they live in namespace delta.
//      Example:  A.array().sin();   // applies delta::sin to each element
//
//   2. MATRIX TRANSCENDENTALS (true operator functions):
//      For true matrix functions f(A) = Σ c_k A^k, use the dedicated
//      free functions in namespace delta (defined in eigen_transcendentals.h).
//      Example:  delta::exp(A);  // matrix exponential (NOT element-wise)
//
//   NOTE: Not every Eigen algorithm is guaranteed to work with these scalar types.
//   In particular, linear algebra operations that rely on floating-point
//   thresholds (e.g., JacobiSVD, eigenvalue solvers) may be extremely slow
//   or numerically unstable. Use with caution and prefer rational-preserving
//   algorithms where possible.
// -----------------------------------------------------------------------------
#pragma once
#ifndef EIGEN_DONT_VECTORIZE
#define EIGEN_DONT_VECTORIZE
#endif
#ifndef EIGEN_DONT_ALIGN_STATICALLY
#define EIGEN_DONT_ALIGN_STATICALLY
#endif
#include <Eigen/Core>
#include "rational_class.h"
#include "gauss_qi.h"
#include "transcendentals.h"
#include "gauss_qi_transcendentals.h"
#include "context.h"            // for delta::default_eps()

namespace Eigen {

    // ========================================================================
    // Part 1: Integration for delta::Rational
    // ========================================================================
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
            // 64 байта — это 8 полноценных double. 
            // Eigen должен понимать, что просто "переложить" это число 
            // из ячейки в ячейку — это уже работа.
            ReadCost = 8,

            // Сложение: Общий знаменатель + GCD + потенциальные аллокации в cpp_int.
            // Это чудовищно дорого.
            AddCost = 250,

            // Умножение: ac/bd + GCD. 
            // Чуть дешевле сложения, но всё равно за пределами добра и зла.
            MulCost = 200
        };
    };

    namespace internal {
        // sqrt_impl is explicitly provided for safety and compatibility with
        // Eigen's internal dispatch (though ADL would also find delta::sqrt).
        template<>
        struct sqrt_impl<delta::Rational> {
            static inline delta::Rational run(const delta::Rational& x) {
                return delta::sqrt(x);
            }
        };
    } // namespace internal

    // ========================================================================
    // Part 2: Integration for delta::GaussQi
    // ========================================================================
    template<>
    struct NumTraits<delta::GaussQi> : GenericNumTraits<delta::GaussQi> {
        using Real = delta::Rational;
        using NonInteger = delta::GaussQi;
        using Literal = delta::GaussQi;

        static inline Real epsilon() { return delta::default_eps(); }
        static inline Real dummy_precision() { return delta::default_eps(); }

        // Для GaussQi
        enum {
            IsInteger = 0,
            IsSigned = 1,
            IsComplex = 1,
            RequireInitialization = 1,
            // 128 байт данных! Это два кэш-лайна.
            ReadCost = 16,

            // Комплексная арифметика на базе рациональных:
            // (a+bi)+(c+di) = (a+c) + (b+d)i
            AddCost = 500,
            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i -> 4 Mul + 2 Add + 1 Sub
            MulCost = 1200
        };
    };

    namespace internal {
        // real_impl and imag_impl are REQUIRED for Eigen's .real() and .imag()
        // methods on matrices of GaussQi. They are not found via ADL.
        template<>
        struct real_impl<delta::GaussQi> {
            static inline delta::Rational run(const delta::GaussQi& x) {
                return x.real();
            }
        };

        template<>
        struct imag_impl<delta::GaussQi> {
            static inline delta::Rational run(const delta::GaussQi& x) {
                return x.imag();
            }
        };

        // pow_impl for integer exponent (cwisePow)
        template<>
        struct pow_impl<delta::GaussQi, int> {
            static inline delta::GaussQi run(const delta::GaussQi& x, int y) {
                return delta::pow(x, y);
            }
        };

        // NOTE: exp_impl and log_impl for GaussQi are intentionally OMITTED.
        // Eigen's internal complex routines that might bypass ADL are not relied upon
        // for our use case. All transcendental functions (including exp, log) are
        // found via ADL because they live in namespace delta.
        // If you encounter a specific Eigen algorithm that requires these specializations,
        // you can add them back, but they are not needed for standard matrix operations.
    } // namespace internal

} // namespace Eigen

// ========================================================================
// Part 3: Matrix transcendental functions
// ========================================================================
// True matrix functions (exp(A), log(A), sin(A), cos(A), sqrt(A)) where
// the series/algorithm is applied to the matrix as a whole operator.
// These are NOT element-wise.
//
// Usage:
//   delta::exp(A)              // matrix exponential
//   delta::log(A)              // matrix logarithm
//   delta::sin(A)              // matrix sine
//   A.array().exp()            // element-wise exponential (ADL, automatic)
//
// Accepts any Eigen expression via Eigen::MatrixBase<Derived>.
// The epsilon parameter is always Rational (default: delta::default_eps()).
// ========================================================================
#include "eigen_transcendentals.h"