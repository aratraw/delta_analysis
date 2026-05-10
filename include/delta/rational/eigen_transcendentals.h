// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0
// ============================================================================
// EXPERT COMMENTARY: DESIGN TRADE-OFFS IN EIGEN INTEGRATION
// ============================================================================
//
// The code below uses a deliberate type‑erasure technique: all Eigen expressions
// are immediately converted to dynamic matrices (DynMatRational / DynMatGaussQi)
// before any non‑trivial computation.  This is a conscious departure from the
// typical Eigen style of writing heavily templated functions that accept any
// `MatrixBase<Derived>`.
//
// ----------------------------- REASONING ---------------------------------
// 1. Compilation time explosion
//    Eigen's expression templates generate a distinct type for almost every
//    arithmetic operation (sum, product, unary op, block, etc.).  When a
//    transcendental function (exp, sin, cos) is implemented as a template
//    that operates directly on `MatrixBase<Derived>`, the compiler must
//    instantiate the entire algorithm for each unique expression type.
//    For a test suite that exercises matrices of different sizes (2x2, 3x3,
//    5x5, 10x10) and different expression forms (A, A*B, A+2*B, etc.), the
//    number of instantiations grows combinatorially.  In practice, this
//    leads to compilation times of several minutes for even a few source
//    files – an unacceptable cost for iterative development.
//
// 2. Symbolic / rational arithmetic is not a typical Eigen use case
//    Eigen is optimised for high‑performance floating‑point linear algebra,
//    where small fixed‑size matrices are inlined and vectorised.  Rational
//    numbers (represented via Boost.Multiprecision) do not benefit from
//    these optimisations; any performance gain from fixed‑size templates is
//    dwarfed by the cost of arbitrary‑precision integer arithmetic.
//    Moreover, rational computations often occur on matrices that are
//    moderately large (≥ 5×5) or even dynamically sized (e.g., from physical
//    problems, finite‑difference schemes).  The runtime overhead of dynamic
//    allocation is negligible compared to the cost of exact arithmetic.
//
// 3. Code maintainability
//    The template‑heavy approach forces the entire algorithm to be placed
//    in headers, exposing every implementation detail and severely slowing
//    downstream compilation.  Type‑erased implementations can be moved to
//    regular functions, which are faster to compile and easier to debug.
//    The only remaining template layer is a thin wrapper that converts the
//    incoming expression to a dynamic matrix and dispatches to the concrete
//    implementation.  This wrapper is compiled once per scalar type, not per
//    expression.
//
// ----------------------------- TRADE-OFFS ---------------------------------
// + Significantly faster compilation (but still sort of slow)
// + Easier maintenance (algorithms written once for `DynMatRational`)
// + Still fully generic – works with any Eigen expression because `.eval()`
//   converts it to a dynamic matrix.
//
// – Loss of stack allocation for small fixed‑size matrices (2x2, 3x3).
//   All matrices become heap‑allocated after conversion.
// – Extra copy of data (the input expression must be evaluated into the
//   dynamic matrix).  For large matrices this cost is negligible; for tiny
//   matrices it may be measurable but still far lower than the cost of
//   exact rational arithmetic.
// – Cannot leverage Eigen's vectorisation for small fixed sizes (irrelevant
//   for rational numbers).
//
// ----------------------------- APPLICABILITY -------------------------------
// This design is highly recommended when:
//   • The scalar type is arbitrary‑precision (rational, big integers, etc.)
//   • Matrix sizes are not known at compile time or are moderately large
//   • Compilation time is a critical resource (typical in CI, template‑heavy
//     libraries, or shared codebases)
//
// The traditional Eigen style (heavy templating) remains the best choice for:
//   • Built‑in floating‑point types (float, double)
//   • Very small, fixed‑size matrices (2x2, 3x3) where stack allocation and
//     compiler optimisation matter
//   • Performance‑critical code that must avoid any dynamic allocation
//     overhead.
//
// ----------------------------- CONCLUSION ---------------------------------
// For the `delta` library – which provides exact rational arithmetic and
// matrix transcendental functions – the type‑erased approach is the correct
// engineering trade‑off.  It prioritises developer productivity and
// compilation speed over marginal runtime gains that are irrelevant for the
// intended use cases.  The code below follows this philosophy consistently.
//
// ============================================================================
#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <type_traits>
#include "rational_class.h"
#include "gauss_qi.h"
#include "transcendentals.h"
#include "gauss_qi_transcendentals.h"
#include "context.h"

namespace delta {

    // ========================================================================
    // TYPE-ERASED IMPLEMENTATIONS
    // ========================================================================
    namespace detail {

        using DynMatRational = Eigen::Matrix<Rational, Eigen::Dynamic, Eigen::Dynamic>;
        using DynMatGaussQi = Eigen::Matrix<GaussQi, Eigen::Dynamic, Eigen::Dynamic>;

        // --------------------------------------------------------------------
        // Helpers for Rational
        // --------------------------------------------------------------------
        inline Rational matrix_max_norm(const DynMatRational& M) {
            Rational max_abs = 0;
            for (int i = 0; i < M.rows(); ++i)
                for (int j = 0; j < M.cols(); ++j) {
                    Rational val = delta::abs(M(i, j));
                    if (max_abs < val) max_abs = val;
                }
            return max_abs;
        }

        inline bool is_diagonal(const DynMatRational& M) {
            const int n = M.rows();
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    if (i != j && M(i, j) != Rational{}) return false;
            return true;
        }

        // --------------------------------------------------------------------
        // Helpers for GaussQi
        // --------------------------------------------------------------------
        inline Rational matrix_max_norm(const DynMatGaussQi& M) {
            Rational max_abs = 0;
            for (int i = 0; i < M.rows(); ++i)
                for (int j = 0; j < M.cols(); ++j) {
                    Rational val = delta::abs(M(i, j));
                    if (max_abs < val) max_abs = val;
                }
            return max_abs;
        }

        inline bool is_diagonal(const DynMatGaussQi& M) {
            const int n = M.rows();
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    if (i != j && M(i, j) != GaussQi{}) return false;
            return true;
        }

        // --------------------------------------------------------------------
        // Common
        // --------------------------------------------------------------------
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

        // ================================================================
        // RATIONAL — forward declarations for mutual recursion
        // ================================================================
        DynMatRational sin_rational(const DynMatRational& A, const Rational& eps);
        DynMatRational cos_rational(const DynMatRational& A, const Rational& eps);

        // ================================================================
        // RATIONAL
        // ================================================================

        inline DynMatRational exp_rational(const DynMatRational& A, const Rational& eps) {
            DynMatRational M = A;
            if (is_diagonal(M)) {
                DynMatRational R = DynMatRational::Zero(M.rows(), M.cols());
                for (int i = 0; i < M.rows(); ++i)
                    R(i, i) = delta::exp(M(i, i), eps);
                return R;
            }

            Rational normM = matrix_max_norm(M);
            int k = 0;
            Rational two_pow_k = 1;
            const Rational half = Rational(1, 2);
            while (normM / two_pow_k > half) {
                two_pow_k *= 2;
                ++k;
            }
            DynMatRational A_scaled = (M / two_pow_k).eval();

            int m = pade_order(eps);
            std::vector<Rational> c(m + 1);
            c[0] = 1;
            for (int j = 1; j <= m; ++j)
                c[j] = c[j - 1] * Rational(m - j + 1) / Rational((2 * m - j + 1) * j);

            DynMatRational A_pow = DynMatRational::Identity(M.rows(), M.cols());
            DynMatRational P = DynMatRational::Zero(M.rows(), M.cols());
            DynMatRational Q = DynMatRational::Zero(M.rows(), M.cols());

            for (int j = 0; j <= m; ++j) {
                if (c[j] != 0) {
                    P += c[j] * A_pow;
                    Q += ((j % 2 == 0) ? c[j] : -c[j]) * A_pow;
                }
                if (j < m) A_pow = (A_pow * A_scaled).eval();
            }

            DynMatRational E = Q.partialPivLu().solve(P);
            for (int i = 0; i < k; ++i) E = (E * E).eval();
            return E;
        }

        inline DynMatRational log_rational(const DynMatRational& A, const Rational& eps) {
            DynMatRational M = A;
            if (M.determinant() == Rational{})
                throw std::domain_error("matrix::log: singular matrix");
            if (is_diagonal(M)) {
                DynMatRational R = DynMatRational::Zero(M.rows(), M.cols());
                for (int i = 0; i < M.rows(); ++i)
                    R(i, i) = delta::log(M(i, i), eps);
                return R;
            }

            Rational log2 = delta::log(Rational(2), eps);
            DynMatRational X = M;
            int k = 0;
            const int max_scale = 100;
            DynMatRational I = DynMatRational::Identity(M.rows(), M.cols());
            const Rational half = Rational(1, 2);

            while (true) {
                DynMatRational diff = (X - I).eval();
                Rational norm_diff = matrix_max_norm(diff);
                if (!(norm_diff > half)) break;
                X = (X / 2).eval();
                ++k;
                if (k > max_scale)
                    throw std::runtime_error("matrix::log: scaling did not converge");
            }

            DynMatRational X_minus_I = (X - I).eval();
            DynMatRational X_plus_I = (X + I).eval();
            DynMatRational Z = (X_minus_I * X_plus_I.partialPivLu().inverse()).eval();

            DynMatRational Z2 = (Z * Z).eval();
            DynMatRational Z_pow = Z;
            DynMatRational sum = Z_pow;
            for (int n = 1; n <= 1000000; ++n) {
                Z_pow = (Z_pow * Z2).eval();
                DynMatRational term = (Z_pow / Rational(2 * n + 1)).eval();
                sum += term;
                Rational norm_term = matrix_max_norm(term);
                Rational norm_sum = matrix_max_norm(sum);
                if (!(norm_term > eps) && !(norm_term > eps * (norm_sum + 1)))
                    break;
            }
            sum = (sum * 2).eval();
            return (I * Rational(k) * log2 + sum).eval();
        }

        inline DynMatRational sin_rational(const DynMatRational& A, const Rational& eps) {
            DynMatRational M = A;
            if (is_diagonal(M)) {
                DynMatRational R = DynMatRational::Zero(M.rows(), M.cols());
                for (int i = 0; i < M.rows(); ++i)
                    R(i, i) = delta::sin(M(i, i), eps);
                return R;
            }
            Rational pi = delta::pi(eps);
            Rational normM = matrix_max_norm(M);

            if (normM > pi) {
                DynMatRational half_M = (M / 2).eval();
                DynMatRational half_sin = sin_rational(half_M, eps);
                DynMatRational half_cos = cos_rational(half_M, eps);
                return (2 * half_sin * half_cos).eval();
            }

            DynMatRational result = DynMatRational::Zero(M.rows(), M.cols());
            DynMatRational term = M;
            DynMatRational A2 = (M * M).eval();
            Rational sign = 1;
            const int max_iter = 10000;

            for (int k = 0; k < max_iter; ++k) {
                result += sign * term;
                Rational norm_term = matrix_max_norm(term);
                Rational norm_result = matrix_max_norm(result);
                if (norm_term <= eps && norm_term <= eps * (norm_result + 1))
                    break;

                sign = -sign;
                // Рекуррентный знаменатель для следующего члена: (2k+2)*(2k+3)
                Rational denom = Rational(2 * k + 2) * Rational(2 * k + 3);
                term = (term * A2 / denom).eval();
            }
            return result;
        }

        inline DynMatRational cos_rational(const DynMatRational& A, const Rational& eps) {
            DynMatRational M = A;
            if (is_diagonal(M)) {
                DynMatRational R = DynMatRational::Zero(M.rows(), M.cols());
                for (int i = 0; i < M.rows(); ++i)
                    R(i, i) = delta::cos(M(i, i), eps);
                return R;
            }
            Rational pi = delta::pi(eps);
            Rational normM = matrix_max_norm(M);

            if (normM > pi) {
                DynMatRational half_M = (M / 2).eval();
                DynMatRational half_sin = sin_rational(half_M, eps);
                DynMatRational half_cos = cos_rational(half_M, eps);
                return (half_cos * half_cos - half_sin * half_sin).eval();
            }

            DynMatRational result = DynMatRational::Identity(M.rows(), M.cols());
            DynMatRational term = DynMatRational::Identity(M.rows(), M.cols());
            DynMatRational A2 = (M * M).eval();
            Rational sign = -1;
            const int max_iter = 10000;

            for (int k = 1; k <= max_iter; ++k) {
                // Рекуррентный знаменатель для члена с k: (2k-1)*(2k)
                Rational denom = Rational(2 * k - 1) * Rational(2 * k);
                term = (term * A2 / denom).eval();
                result += sign * term;

                Rational norm_term = matrix_max_norm(term);
                Rational norm_result = matrix_max_norm(result);
                if (norm_term <= eps && norm_term <= eps * (norm_result + 1))
                    break;

                sign = -sign;
            }
            return result;
        }
        inline DynMatRational sqrt_rational(const DynMatRational& A, const Rational& eps) {
            DynMatRational M = A;
            if (is_diagonal(M)) {
                DynMatRational R = DynMatRational::Zero(M.rows(), M.cols());
                for (int i = 0; i < M.rows(); ++i)
                    R(i, i) = delta::sqrt(M(i, i), eps);
                return R;
            }

            DynMatRational Y = M;
            DynMatRational Z = DynMatRational::Identity(M.rows(), M.cols());
            const int max_iter = 1000;
            for (int iter = 0; iter < max_iter; ++iter) {
                DynMatRational Y_inv = Y.partialPivLu().inverse();
                DynMatRational Z_inv = Z.partialPivLu().inverse();
                DynMatRational Y_new = ((Y + Z_inv) / 2).eval();
                DynMatRational Z_new = ((Z + Y_inv) / 2).eval();
                Rational diffY = matrix_max_norm((Y_new - Y).eval());
                Rational diffZ = matrix_max_norm((Z_new - Z).eval());
                if (!(diffY > eps) && !(diffZ > eps))
                    return Y_new;
                Y = std::move(Y_new);
                Z = std::move(Z_new);
            }
            throw std::runtime_error("matrix::sqrt: iteration did not converge");
        }

        // ================================================================
        // GAUSSQI
        // ================================================================

        inline DynMatGaussQi exp_gaussqi(const DynMatGaussQi& A, const Rational& eps) {
            DynMatGaussQi M = A;
            if (is_diagonal(M)) {
                DynMatGaussQi R = DynMatGaussQi::Zero(M.rows(), M.cols());
                for (int i = 0; i < M.rows(); ++i)
                    R(i, i) = delta::exp(M(i, i), eps);
                return R;
            }

            Rational normM = matrix_max_norm(M);
            int k = 0;
            Rational two_pow_k = 1;
            const Rational half = Rational(1, 2);
            while (normM / two_pow_k > half) {
                two_pow_k *= 2;
                ++k;
            }
            DynMatGaussQi A_scaled = (M / GaussQi(two_pow_k)).eval();

            int m = pade_order(eps);
            std::vector<Rational> c(m + 1);
            c[0] = 1;
            for (int j = 1; j <= m; ++j)
                c[j] = c[j - 1] * Rational(m - j + 1) / Rational((2 * m - j + 1) * j);

            DynMatGaussQi A_pow = DynMatGaussQi::Identity(M.rows(), M.cols());
            DynMatGaussQi P = DynMatGaussQi::Zero(M.rows(), M.cols());
            DynMatGaussQi Q = DynMatGaussQi::Zero(M.rows(), M.cols());

            for (int j = 0; j <= m; ++j) {
                if (c[j] != 0) {
                    P += c[j] * A_pow;
                    Q += ((j % 2 == 0) ? c[j] : -c[j]) * A_pow;
                }
                if (j < m) A_pow = (A_pow * A_scaled).eval();
            }

            DynMatGaussQi E = Q.partialPivLu().solve(P);
            for (int i = 0; i < k; ++i) E = (E * E).eval();
            return E;
        }

        inline DynMatGaussQi log_gaussqi(const DynMatGaussQi& A, const Rational& eps) {
            DynMatGaussQi M = A;
            if (M.determinant() == GaussQi{})
                throw std::domain_error("matrix::log: singular matrix");
            if (is_diagonal(M)) {
                DynMatGaussQi R = DynMatGaussQi::Zero(M.rows(), M.cols());
                for (int i = 0; i < M.rows(); ++i)
                    R(i, i) = delta::log(M(i, i), eps);
                return R;
            }

            Rational log2 = delta::log(Rational(2), eps);
            DynMatGaussQi X = M;
            int k = 0;
            const int max_scale = 500;
            DynMatGaussQi I = DynMatGaussQi::Identity(M.rows(), M.cols());
            const Rational half = Rational(1, 2);

            while (true) {
                DynMatGaussQi diff = (X - I).eval();
                Rational norm_diff = matrix_max_norm(diff);
                if (!(norm_diff > half)) break;
                X = (X / GaussQi(2)).eval();
                ++k;
                if (k > max_scale)
                    throw std::runtime_error("matrix::log: scaling did not converge");
            }

            DynMatGaussQi X_minus_I = (X - I).eval();
            DynMatGaussQi X_plus_I = (X + I).eval();
            DynMatGaussQi Z = (X_minus_I * X_plus_I.partialPivLu().inverse()).eval();

            DynMatGaussQi Z2 = (Z * Z).eval();
            DynMatGaussQi Z_pow = Z;
            DynMatGaussQi sum = Z_pow;
            for (int n = 1; n <= 1000000; ++n) {
                Z_pow = (Z_pow * Z2).eval();
                DynMatGaussQi term = (Z_pow / GaussQi(Rational(2 * n + 1))).eval();
                sum += term;
                Rational norm_term = matrix_max_norm(term);
                Rational norm_sum = matrix_max_norm(sum);
                if (!(norm_term > eps) && !(norm_term > eps * (norm_sum + 1)))
                    break;
            }
            sum = (sum * GaussQi(2)).eval();
            return (I * GaussQi(Rational(k) * log2) + sum).eval();
        }

        inline DynMatGaussQi sin_gaussqi(const DynMatGaussQi& A, const Rational& eps) {
            GaussQi i(0, 1);
            DynMatGaussQi iA = (A * i).eval();
            DynMatGaussQi neg_iA = (-iA).eval();
            DynMatGaussQi exp_iA = exp_gaussqi(iA, eps);
            DynMatGaussQi exp_neg_iA = exp_gaussqi(neg_iA, eps);
            return ((exp_iA - exp_neg_iA) / (GaussQi(2) * i)).eval();
        }

        inline DynMatGaussQi cos_gaussqi(const DynMatGaussQi& A, const Rational& eps) {
            GaussQi i(0, 1);
            DynMatGaussQi iA = (A * i).eval();
            DynMatGaussQi neg_iA = (-iA).eval();
            DynMatGaussQi exp_iA = exp_gaussqi(iA, eps);
            DynMatGaussQi exp_neg_iA = exp_gaussqi(neg_iA, eps);
            return ((exp_iA + exp_neg_iA) / GaussQi(2)).eval();
        }

        inline DynMatGaussQi sqrt_gaussqi(const DynMatGaussQi& A, const Rational& eps) {
            DynMatGaussQi M = A;
            if (is_diagonal(M)) {
                DynMatGaussQi R = DynMatGaussQi::Zero(M.rows(), M.cols());
                for (int i = 0; i < M.rows(); ++i)
                    R(i, i) = delta::sqrt(M(i, i), eps);
                return R;
            }

            DynMatGaussQi Y = M;
            DynMatGaussQi Z = DynMatGaussQi::Identity(M.rows(), M.cols());
            const int max_iter = 1000;
            for (int iter = 0; iter < max_iter; ++iter) {
                DynMatGaussQi Y_inv = Y.partialPivLu().inverse();
                DynMatGaussQi Z_inv = Z.partialPivLu().inverse();
                DynMatGaussQi Y_new = ((Y + Z_inv) / GaussQi(2)).eval();
                DynMatGaussQi Z_new = ((Z + Y_inv) / GaussQi(2)).eval();
                Rational diffY = matrix_max_norm((Y_new - Y).eval());
                Rational diffZ = matrix_max_norm((Z_new - Z).eval());
                if (!(diffY > eps) && !(diffZ > eps))
                    return Y_new;
                Y = std::move(Y_new);
                Z = std::move(Z_new);
            }
            throw std::runtime_error("matrix::sqrt: iteration did not converge");
        }

    } // namespace detail

    // ========================================================================
    // THIN WRAPPERS
    // ========================================================================

    template<typename Derived>
    auto exp(const Eigen::MatrixBase<Derived>& A, const Rational& eps = default_eps())
        -> Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    {
        using Scalar = typename Derived::Scalar;
        static_assert(Derived::RowsAtCompileTime == Derived::ColsAtCompileTime, "exp: square matrix required");

        if constexpr (std::is_same_v<Scalar, Rational>) {
            detail::DynMatRational dyn = A.derived();
            return detail::exp_rational(dyn, eps);
        }
        else if constexpr (std::is_same_v<Scalar, GaussQi>) {
            detail::DynMatGaussQi dyn = A.derived();
            return detail::exp_gaussqi(dyn, eps);
        }
    }

    template<typename Derived>
    auto log(const Eigen::MatrixBase<Derived>& A, const Rational& eps = default_eps())
        -> Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    {
        using Scalar = typename Derived::Scalar;
        static_assert(Derived::RowsAtCompileTime == Derived::ColsAtCompileTime, "log: square matrix required");

        if constexpr (std::is_same_v<Scalar, Rational>) {
            detail::DynMatRational dyn = A.derived();
            return detail::log_rational(dyn, eps);
        }
        else if constexpr (std::is_same_v<Scalar, GaussQi>) {
            detail::DynMatGaussQi dyn = A.derived();
            return detail::log_gaussqi(dyn, eps);
        }
    }

    template<typename Derived>
    auto sin(const Eigen::MatrixBase<Derived>& A, const Rational& eps = default_eps())
        -> Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    {
        using Scalar = typename Derived::Scalar;
        static_assert(Derived::RowsAtCompileTime == Derived::ColsAtCompileTime, "sin: square matrix required");

        if constexpr (std::is_same_v<Scalar, Rational>) {
            detail::DynMatRational dyn = A.derived();
            return detail::sin_rational(dyn, eps);
        }
        else if constexpr (std::is_same_v<Scalar, GaussQi>) {
            detail::DynMatGaussQi dyn = A.derived();
            return detail::sin_gaussqi(dyn, eps);
        }
    }

    template<typename Derived>
    auto cos(const Eigen::MatrixBase<Derived>& A, const Rational& eps = default_eps())
        -> Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    {
        using Scalar = typename Derived::Scalar;
        static_assert(Derived::RowsAtCompileTime == Derived::ColsAtCompileTime, "cos: square matrix required");

        if constexpr (std::is_same_v<Scalar, Rational>) {
            detail::DynMatRational dyn = A.derived();
            return detail::cos_rational(dyn, eps);
        }
        else if constexpr (std::is_same_v<Scalar, GaussQi>) {
            detail::DynMatGaussQi dyn = A.derived();
            return detail::cos_gaussqi(dyn, eps);
        }
    }

    template<typename Derived>
    auto sqrt(const Eigen::MatrixBase<Derived>& A, const Rational& eps = default_eps())
        -> Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
    {
        using Scalar = typename Derived::Scalar;
        static_assert(Derived::RowsAtCompileTime == Derived::ColsAtCompileTime, "sqrt: square matrix required");

        if constexpr (std::is_same_v<Scalar, Rational>) {
            detail::DynMatRational dyn = A.derived();
            return detail::sqrt_rational(dyn, eps);
        }
        else if constexpr (std::is_same_v<Scalar, GaussQi>) {
            detail::DynMatGaussQi dyn = A.derived();
            return detail::sqrt_gaussqi(dyn, eps);
        }
    }

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