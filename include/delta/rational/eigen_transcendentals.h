// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0
// ============================================================================
// MATRIX TRANSCENDENTAL FUNCTIONS FOR RATIONAL AND GAUSSQI
// ============================================================================
// This header provides *genuine* matrix transcendental functions for
// delta::Rational and delta::GaussQi as scalar types for Eigen::Matrix.
//
//   DO NOT CONFUSE WITH ELEMENT-WISE FUNCTIONS!
//   - A.array().exp()   → element-wise exponential (uses delta::exp for each element)
//   - delta::exp(A)     → matrix exponential (series: e^A = I + A + A²/2! + ...)
//
// ----------------------------------------------------------------------------
//  SUPPORTED FUNCTIONS
// ----------------------------------------------------------------------------
//   exp(A)  – matrix exponential (scaling-and-squaring + Padé approximant)
//   log(A)  – matrix logarithm (inverse scaling + Gregory series via artanh)
//   sin(A)  – matrix sine (via exp(iA) for GaussQi, Taylor series for Rational)
//   cos(A)  – matrix cosine (via exp(iA) for GaussQi, Taylor series for Rational)
//   sqrt(A) – matrix square root (Denman‑Beavers iteration)
//
// All functions take an optional epsilon parameter of type delta::Rational.
// Defaults to delta::default_eps() (global value, initially 1e‑30).
//
// ============================================================================
//  DESIGN RATIONALE – WHY THINGS ARE DONE THIS WAY (AND NO OTHER)
// ============================================================================
//
// 1. OVERLOAD DISPATCH VIA std::enable_if_t
// ----------------------------------------------------------------------------
//    Problem: A single sin template for all types causes the compiler to find
//             the scalar overload (delta::sin(const Rational&)) when called
//             with a matrix, because it's a better match.
//    Solution: Separate overloads for Rational and GaussQi using enable_if_t.
//              NOT in the return type (MSVC fails with C2995 – duplicate definition).
//              Correct approach: add a dummy default template parameter:
//                template<typename Derived, typename = std::enable_if_t<...>>
//
// 2. FORWARD DECLARATIONS – A NECESSITY, NOT A LUXURY
// ----------------------------------------------------------------------------
//    Inside sin(RationalMatrix) we call cos(M/2). If the cos overload for
//    matrices hasn't been seen yet, the compiler only finds scalar cos(Rational).
//    Result: C2664 (cannot convert matrix to Rational).
//    **Rule**: All sin/cos overloads for MatrixBase must be declared BEFORE
//    they are used. Hence forward declarations after #includes.
//
// 3. WRAPPERS FOR ANY EIGEN EXPRESSION (EigenBase)
// ----------------------------------------------------------------------------
//    Users may write delta::sin(A / 2), where A/2 is a CwiseBinaryOp, not a
//    MatrixBase. Our primary overload takes MatrixBase and cannot be selected.
//    A wrapper is needed that accepts any EigenBase, calls .eval() (evaluates
//    the expression into a concrete matrix), and forwards to the primary overload.
//    **Important**: These wrappers must be declared AFTER forward declarations
//    of the primary functions but BEFORE their implementations, so that the call
//    to delta::sin(expr) inside the wrapper resolves to the MatrixBase overload.
//
// 4. FAST PATH FOR DIAGONAL MATRICES
// ----------------------------------------------------------------------------
//    For exp and log, fast diagonal paths existed from the start (calling scalar
//    functions). For sin/cos, it had to be added after tests revealed that
//    Taylor series for large diagonal entries (e.g., 5) produces monstrous
//    fractions and fails to achieve 1e-19 precision in reasonable iterations.
//    The diagonal path calls scalar delta::sin / delta::cos (which are correct
//    for any precision) and zeroes out off-diagonal entries.
//
// 5. PADÉ APPROXIMANT FOR exp (NOT TAYLOR SERIES)
// ----------------------------------------------------------------------------
//    Taylor series for matrix exponential requires many terms for large norms.
//    Scaling‑and‑squaring reduces the norm to ≤0.5, then Padé (m,m) gives
//    much better accuracy per term. Order m is chosen adaptively based on epsilon.
//    This is the standard algorithm (Higham, 2005).
//
// 6. GREGORY SERIES FOR log (VIA artanh)
// ----------------------------------------------------------------------------
//    After reducing the matrix close to identity (X ≈ I), compute
//    Z = (X-I)*(X+I)^{-1}, then log(X) = 2 * Σ Z^{2n+1}/(2n+1).
//    This is equivalent to the series for arctanh and converges faster than
//    the direct Mercator series for log. Termination criterion: absolute AND
//    relative (prevents infinite loops when the sum is near zero).
//
// 7. DENMAN‑BEAVERS FOR sqrt
// ----------------------------------------------------------------------------
//    Iterations Y_{k+1} = (Y_k + Z_k^{-1})/2, Z_{k+1} = (Z_k + Y_k^{-1})/2
//    converge quadratically, preserving rationality. This is preferable over
//    eigenvalue decomposition (which would require algebraic extensions for
//    rational arithmetic). Diagonal path uses scalar sqrt.
//
// ----------------------------------------------------------------------------
//  CRITICAL MISTAKES WE MADE (AND HOW TO AVOID THEM)
// ----------------------------------------------------------------------------
//
// █ Error C2664 (cannot convert Eigen expression to Rational)
//   Cause: inside sin(RationalMatrix) we called cos(M/2), but the cos overload
//          for matrices wasn't visible yet.
//   Fix: Forward declarations.
//
// █ Error C2995 (template already defined)
//   Cause: Two sin overloads with identical signatures (both without enable_if,
//          or both with enable_if in the return type, which MSVC treats as
//          nondistinct).
//   Fix: Use enable_if_t in a separate default template parameter.
//
// █ Error C2678 (no operator != between GaussQi and int)
//   Cause: In is_diagonal we compared M(i,j) != 0, where 0 is int, and GaussQi's
//          constructor from int is explicit. In rational arithmetic, zero is
//          Scalar{} (default-constructed value).
//   Fix: Write M(i,j) != Scalar{}.
//
// █ Error C1128 (too many sections) in MSVC
//   Cause: Excessive number of COFF sections due to template instantiations.
//   Fix: Add /bigobj compiler flag. Additionally, use precompiled headers (PCH)
//        for Eigen and GTest.
//
// █ Taylor series convergence failure for sin/cos
//   Cause: First version accumulated the denominator in a factorial variable `fact`,
//          which grew factorially, but term was not updated recurrently.
//   Fix: term = term * A2 / ((2k+2)*(2k+3)) – each iteration divides by the next
//        two factors. This gives O(1) per iteration and exact fractions.
//
// █ Cannot pass expression like delta::sin(A / 2)
//   Cause: A/2 is a CwiseBinaryOp, not a MatrixBase.
//   Fix: Add wrappers for EigenBase that call .eval().
//
// ----------------------------------------------------------------------------
//  INCLUSION ORDER AND MACROS (SACRED)
// ----------------------------------------------------------------------------
//   1. #includes (Eigen, type_traits, our headers)
//   2. namespace delta {
//   3.   namespace detail { ... }        // helper functions
//   4.   // FORWARD DECLARATIONS         // exp, log, sin, cos, sqrt
//   5.   // EigenBase wrappers
//   6.   // IMPLEMENTATIONS              // same order as declarations
//   7.   // (EigenBase wrappers are already declared; implemented after impls)
//   8. }
//
//   Violating this order WILL cause compilation errors.
// ----------------------------------------------------------------------------
//  DO NOT CHANGE ANY OF THESE UNLESS YOU KNOW WHAT YOU'RE DOING
// ----------------------------------------------------------------------------
//   - Do NOT change the order of declarations.
//   - Do NOT remove enable_if_t – dispatch will break.
//   - Do NOT add implicit conversions between GaussQi and int.
//   - Do NOT try to "speed up" sin/cos for Rational by calling exp(iA) – that
//     would pull in complex arithmetic and be 10x slower.
//   - Do NOT remove the fast diagonal path – tests will fail.
//   - Do NOT replace Padé for exp with Taylor series – accuracy will suffer.
//   - Do NOT remove the relative convergence criterion in Gregory series –
//     it may loop forever.
#pragma once

#include <Eigen/Dense>
#include <type_traits>
#include "rational_class.h"
#include "gauss_qi.h"
#include "transcendentals.h"
#include "gauss_qi_transcendentals.h"
#include "context.h"

namespace delta {

    // ------------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------------
    namespace detail {
        template<typename Matrix>
        Rational matrix_max_norm(const Matrix& M) {
            Rational max_abs = 0;
            for (int i = 0; i < M.rows(); ++i)
                for (int j = 0; j < M.cols(); ++j)
                    max_abs = std::max(max_abs, delta::abs(M(i, j)));
            return max_abs;
        }

        template<typename Matrix>
        bool is_diagonal(const Matrix& M) {
            using Scalar = typename Matrix::Scalar;
            const int n = M.rows();
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    if (i != j && M(i, j) != Scalar{}) return false;
            return true;
        }

        inline int pade_order(const Rational& eps) {
            double d = eps.to_double();
            if (d <= 0) return 16;
            if (d >= 1e-3) return 4;
            if (d >= 1e-7) return 6;
            if (d >= 1e-12) return 8;
            if (d >= 1e-17) return 10;
            if (d >= 1e-22) return 12;
            if (d >= 1e-27) return 14;
            return 16;
        }

        template<typename Matrix>
        Matrix exp_diagonal(const Matrix& M, const Rational& eps) {
            Matrix R = Matrix::Zero(M.rows(), M.cols());
            for (int i = 0; i < M.rows(); ++i)
                R(i, i) = delta::exp(M(i, i), eps);
            return R;
        }

        template<typename Matrix>
        Matrix log_diagonal(const Matrix& M, const Rational& eps) {
            Matrix R = Matrix::Zero(M.rows(), M.cols());
            for (int i = 0; i < M.rows(); ++i)
                R(i, i) = delta::log(M(i, i), eps);
            return R;
        }
    }

    // ========================================================================
    // FORWARD DECLARATIONS
    // ========================================================================
    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
        exp(const Eigen::MatrixBase<Derived>& A, const Rational& eps = default_eps());

    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
        log(const Eigen::MatrixBase<Derived>& A, const Rational& eps = default_eps());

    template<typename Derived>
    auto sin(const Eigen::MatrixBase<Derived>& A, const Rational& eps = default_eps())
        -> std::enable_if_t<std::is_same_v<typename Derived::Scalar, GaussQi>,
        Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>>;

    template<typename Derived>
    auto sin(const Eigen::MatrixBase<Derived>& A, const Rational& eps = default_eps())
        -> std::enable_if_t<std::is_same_v<typename Derived::Scalar, Rational>,
        Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>>;

    template<typename Derived>
    auto cos(const Eigen::MatrixBase<Derived>& A, const Rational& eps = default_eps())
        -> std::enable_if_t<std::is_same_v<typename Derived::Scalar, GaussQi>,
        Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>>;

    template<typename Derived>
    auto cos(const Eigen::MatrixBase<Derived>& A, const Rational& eps = default_eps())
        -> std::enable_if_t<std::is_same_v<typename Derived::Scalar, Rational>,
        Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>>;

    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
        sqrt(const Eigen::MatrixBase<Derived>& A, const Rational& eps = default_eps());

    // Обёртки для любых Eigen выражений
    template<typename Derived>
    auto sin(const Eigen::EigenBase<Derived>& expr, const Rational& eps = default_eps())
        -> decltype(delta::sin(expr.derived().eval(), eps));

    template<typename Derived>
    auto cos(const Eigen::EigenBase<Derived>& expr, const Rational& eps = default_eps())
        -> decltype(delta::cos(expr.derived().eval(), eps));

    // ========================================================================
    // РЕАЛИЗАЦИИ
    // ========================================================================

    // ------------------------------------------------------------------------
    // Matrix exponential
    // ------------------------------------------------------------------------
    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
        exp(const Eigen::MatrixBase<Derived>& A, const Rational& eps) {
        using Matrix = Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>;
        static_assert(Derived::RowsAtCompileTime == Derived::ColsAtCompileTime, "exp: square matrix required");

        Matrix M = A.derived();
        if (detail::is_diagonal(M))
            return detail::exp_diagonal(M, eps);

        Rational normM = detail::matrix_max_norm(M);
        int k = 0;
        Rational two_pow_k = 1;
        while (normM / two_pow_k > Rational(1, 2)) {
            two_pow_k *= 2;
            ++k;
        }
        Matrix A_scaled = M / two_pow_k;

        int m = detail::pade_order(eps);
        std::vector<Rational> c(m + 1);
        c[0] = 1;
        for (int j = 1; j <= m; ++j)
            c[j] = c[j - 1] * Rational(m - j + 1) / Rational((2 * m - j + 1) * j);

        Matrix A_pow = Matrix::Identity(M.rows(), M.cols());
        Matrix P = Matrix::Zero(M.rows(), M.cols());
        Matrix Q = Matrix::Zero(M.rows(), M.cols());

        for (int j = 0; j <= m; ++j) {
            if (c[j] != 0) {
                P += c[j] * A_pow;
                Q += ((j % 2 == 0) ? c[j] : -c[j]) * A_pow;
            }
            if (j < m) A_pow = A_pow * A_scaled;
        }

        Matrix E = Q.partialPivLu().solve(P);
        for (int i = 0; i < k; ++i) E = E * E;
        return E;
    }

    // ------------------------------------------------------------------------
    // Matrix logarithm
    // ------------------------------------------------------------------------
    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
        log(const Eigen::MatrixBase<Derived>& A, const Rational& eps) {
        using Matrix = Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>;
        static_assert(Derived::RowsAtCompileTime == Derived::ColsAtCompileTime, "log: square matrix required");

        Matrix M = A.derived();
        if (M.determinant() == 0)
            throw std::domain_error("matrix::log: singular matrix");
        if (detail::is_diagonal(M))
            return detail::log_diagonal(M, eps);

        Rational log2 = delta::log(Rational(2), eps);
        Matrix X = M;
        int k = 0;
        const int max_scale = 100;
        Matrix I = Matrix::Identity(M.rows(), M.cols());

        while (detail::matrix_max_norm(X - I) > Rational(1, 2)) {
            X = X / 2;
            ++k;
            if (k > max_scale)
                throw std::runtime_error("matrix::log: scaling did not converge");
        }

        Matrix X_minus_I = X - I;
        Matrix X_plus_I = X + I;
        Matrix X_plus_I_inv = X_plus_I.partialPivLu().inverse();
        Matrix Z = X_minus_I * X_plus_I_inv;

        Matrix Z2 = Z * Z;
        Matrix Z_pow = Z;
        Matrix sum = Z_pow;
        for (int n = 1; n <= 1000000; ++n) {
            Z_pow = Z_pow * Z2;
            Matrix term = Z_pow / Rational(2 * n + 1);
            sum += term;
            if (detail::matrix_max_norm(term) <= eps &&
                detail::matrix_max_norm(term) <= eps * (detail::matrix_max_norm(sum) + 1))
                break;
        }
        sum = sum * 2;
        Matrix result = I * (Rational(k) * log2) + sum;
        return result;
    }

    // ------------------------------------------------------------------------
    // Matrix sine for GaussQi
    // ------------------------------------------------------------------------
    template<typename Derived>
    auto sin(const Eigen::MatrixBase<Derived>& A, const Rational& eps)
        -> std::enable_if_t<std::is_same_v<typename Derived::Scalar, GaussQi>,
        Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>>
    {
        using Scalar = GaussQi;
        using Matrix = Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>;
        static_assert(Derived::RowsAtCompileTime == Derived::ColsAtCompileTime, "sin: square matrix required");

        Scalar i(0, 1);
        Matrix iA = A.derived() * i;
        Matrix exp_iA = delta::exp(iA, eps);
        Matrix exp_neg_iA = delta::exp(-iA, eps);
        return (exp_iA - exp_neg_iA) / (Scalar(2) * i);
    }

    // ------------------------------------------------------------------------
    // Matrix sine for Rational (Fixed: increased max_iter, improved convergence)
    // ------------------------------------------------------------------------
    // Matrix sine for Rational (с быстрым диагональным путём)
    template<typename Derived>
    auto sin(const Eigen::MatrixBase<Derived>& A, const Rational& eps)
        -> std::enable_if_t<std::is_same_v<typename Derived::Scalar, Rational>,
        Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>>
    {
        using Scalar = Rational;
        using Matrix = Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>;
        static_assert(Derived::RowsAtCompileTime == Derived::ColsAtCompileTime, "sin: square matrix required");

        Matrix M = A.derived();

        // Fast path for diagonal matrices
        if (detail::is_diagonal(M)) {
            Matrix result = Matrix::Zero(M.rows(), M.cols());
            for (int i = 0; i < M.rows(); ++i) {
                result(i, i) = delta::sin(M(i, i), eps);
            }
            return result;
        }

        Rational pi = delta::pi(eps);
        if (detail::matrix_max_norm(M) > pi) {
            Matrix half_sin = delta::sin(M / Scalar(2), eps);
            Matrix half_cos = delta::cos(M / Scalar(2), eps);
            return Scalar(2) * half_sin * half_cos;
        }

        Matrix result = Matrix::Zero(M.rows(), M.cols());
        Matrix term = M;
        Matrix A2 = M * M;
        Scalar sign = 1;
        const int max_iter = 10000;
        for (int k = 0; k < max_iter; ++k) {
            result += sign * term;
            Rational norm_term = detail::matrix_max_norm(term);
            if (norm_term <= eps && norm_term <= eps * (detail::matrix_max_norm(result) + 1))
                break;
            sign = -sign;
            Rational denom = Rational(2 * k + 2) * Rational(2 * k + 3);
            term = term * A2 / denom;
        }
        return result;
    }
    // ------------------------------------------------------------------------
    // Matrix cosine for GaussQi
    // ------------------------------------------------------------------------
    template<typename Derived>
    auto cos(const Eigen::MatrixBase<Derived>& A, const Rational& eps)
        -> std::enable_if_t<std::is_same_v<typename Derived::Scalar, GaussQi>,
        Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>>
    {
        using Scalar = GaussQi;
        using Matrix = Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>;
        static_assert(Derived::RowsAtCompileTime == Derived::ColsAtCompileTime, "cos: square matrix required");

        Scalar i(0, 1);
        Matrix iA = A.derived() * i;
        Matrix exp_iA = delta::exp(iA, eps);
        Matrix exp_neg_iA = delta::exp(-iA, eps);
        return (exp_iA + exp_neg_iA) / Scalar(2);
    }

    // ------------------------------------------------------------------------
    // Matrix cosine for Rational (Fixed: increased max_iter, improved convergence)
    // ------------------------------------------------------------------------
// Matrix cosine for Rational (с быстрым диагональным путём)
    template<typename Derived>
    auto cos(const Eigen::MatrixBase<Derived>& A, const Rational& eps)
        -> std::enable_if_t<std::is_same_v<typename Derived::Scalar, Rational>,
        Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>>
    {
        using Scalar = Rational;
        using Matrix = Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>;
        static_assert(Derived::RowsAtCompileTime == Derived::ColsAtCompileTime, "cos: square matrix required");

        Matrix M = A.derived();

        // Fast path for diagonal matrices
        if (detail::is_diagonal(M)) {
            Matrix result = Matrix::Zero(M.rows(), M.cols());
            for (int i = 0; i < M.rows(); ++i) {
                result(i, i) = delta::cos(M(i, i), eps);
            }
            return result;
        }

        Rational pi = delta::pi(eps);
        if (detail::matrix_max_norm(M) > pi) {
            Matrix half_sin = delta::sin(M / Scalar(2), eps);
            Matrix half_cos = delta::cos(M / Scalar(2), eps);
            return half_cos * half_cos - half_sin * half_sin;
        }

        Matrix result = Matrix::Identity(M.rows(), M.cols());
        Matrix term = Matrix::Identity(M.rows(), M.cols());
        Matrix A2 = M * M;
        Scalar sign = -1;
        const int max_iter = 10000;
        for (int k = 1; k <= max_iter; ++k) {
            Rational denom = Rational(2 * k - 1) * Rational(2 * k);
            term = term * A2 / denom;
            result += sign * term;
            Rational norm_term = detail::matrix_max_norm(term);
            Rational norm_res = detail::matrix_max_norm(result);
            if (norm_term <= eps && norm_term <= eps * (norm_res + 1))
                break;
            sign = -sign;
        }
        return result;
    }

    // ------------------------------------------------------------------------
    // Matrix square root
    // ------------------------------------------------------------------------
    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
        sqrt(const Eigen::MatrixBase<Derived>& A, const Rational& eps) {
        using Scalar = typename Derived::Scalar;
        using Matrix = Eigen::Matrix<Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>;
        static_assert(Derived::RowsAtCompileTime == Derived::ColsAtCompileTime, "sqrt: square matrix required");

        Matrix M = A.derived();
        if (detail::is_diagonal(M)) {
            Matrix R = Matrix::Zero(M.rows(), M.cols());
            for (int i = 0; i < M.rows(); ++i)
                R(i, i) = delta::sqrt(M(i, i), eps);
            return R;
        }

        Matrix Y = M;
        Matrix Z = Matrix::Identity(M.rows(), M.cols());
        const int max_iter = 1000;
        for (int iter = 0; iter < max_iter; ++iter) {
            Matrix Y_inv = Y.partialPivLu().inverse();
            Matrix Z_inv = Z.partialPivLu().inverse();
            Matrix Y_new = (Y + Z_inv) / Scalar(2);
            Matrix Z_new = (Z + Y_inv) / Scalar(2);
            if (detail::matrix_max_norm(Y_new - Y) < eps &&
                detail::matrix_max_norm(Z_new - Z) < eps) {
                return Y_new;
            }
            Y = std::move(Y_new);
            Z = std::move(Z_new);
        }
        throw std::runtime_error("matrix::sqrt: iteration did not converge");
    }

    // ------------------------------------------------------------------------
    // Универсальные обёртки для любых выражений Eigen
    // ------------------------------------------------------------------------
    template<typename Derived>
    auto sin(const Eigen::EigenBase<Derived>& expr, const Rational& eps)
        -> decltype(delta::sin(expr.derived().eval(), eps))
    {
        return delta::sin(expr.derived().eval(), eps);
    }

    template<typename Derived>
    auto cos(const Eigen::EigenBase<Derived>& expr, const Rational& eps)
        -> decltype(delta::cos(expr.derived().eval(), eps))
    {
        return delta::cos(expr.derived().eval(), eps);
    }

} // namespace delta