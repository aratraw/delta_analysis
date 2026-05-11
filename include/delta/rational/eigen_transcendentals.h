// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0
// 
// THIS RIGHT HERE IS NOT A CODE FILE: IT IS A BATTLEFIELD FOR THE MIND.
// ============================================================================
// EXPERT COMMENTARY: MATRIX TRANSCENDENTALS IN EXACT RATIONAL ARITHMETIC
// ============================================================================
//
// 1.  ARCHITECTURAL PHILOSOPHY – WHY TYPE‑ERASURE?
// ---------------------------------------------------------------------------
// The functions below accept any Eigen dense matrix expression of square
// shape and immediately convert the input to a dynamically‑sized matrix
// (DynMatRational = Matrix<Rational,Dynamic,Dynamic> or the GaussQi
// counterpart).  This is a deliberate break from the traditional Eigen style
// of writing heavily templated functions on `MatrixBase<Derived>`.
//
// The decision is driven by three observations:
//
// (a) Compilation‑time explosion.  Eigen’s expression templates create a
//     distinct C++ type for every arithmetic sub‑expression.  A test suite
//     that exercises 2×2, 3×3, 5×5 and 10×10 matrices combined with
//     expressions like A*B, A+2*B, … would force the compiler to
//     instantiate every transcendental algorithm for dozens of expression
//     types.  Measured compilation times exceeded several minutes for a
//     handful of source files, killing developer productivity.
//
// (b) Exact rational arithmetic dominates the cost.  Rational numbers
//     (backed by Boost.Multiprecision’s arbitrary‑precision integers) do
//     not benefit from fixed‑size stack allocation or SIMD vectorisation.
//     The overhead of one dynamic allocation per function call is
//     negligible compared to the cost of GCD reductions and large‑integer
//     multiplications.
//
// (c) Maintainability.  Type‑erased implementations live in ordinary
//     functions that can be moved to .cpp files if desired.  Only a thin
//     wrapper template remains in the header, instantiated once per scalar
//     type (Rational, GaussQi), not once per expression.
//
// The trade‑off is a deliberate surrender of micro‑optimisations irrelevant
// for this scalar domain, in exchange for compilation speed and clarity.
//
//
// 2.  ALGORITHMIC LANDSCAPE
// ---------------------------------------------------------------------------
// The file provides five matrix functions for real (Rational) and complex
// (GaussQi) square matrices: exp, log, sin, cos, sqrt.
//
// • Diagonal fast‑path – element‑wise application, exact and cheap.
//
// • Exponential – scaling‑and‑squaring with Padé approximants.
//   Scale A by 2^k until ‖A/2^k‖ ≤ 0.5.  Form diagonal Padé [m/m] approximant,
//   solve Q·E = P via LU, then square k times.
//
// • Logarithm – for complex matrices, classical scaling by division by 2 fails
//   (it does not reduce phase; repeated halving drives the matrix toward -I).
//   Repeated square roots (Denman–Beavers) converge but cause catastrophic
//   growth of rational bit length, making computation effectively infinite.
//   Our solution: trace normalisation.  Replace M by B = M / (trace(M)/n),
//   centering the spectrum (average eigenvalue becomes 1).  Then apply a
//   direct diagonal Padé approximant for log(I+X) with order m computed by
//   the correct formula for the logarithm (see §5.3).  No iterative scaling.
//   For ill‑conditioned matrices, a single Cayley transform can be added
//   as a fallback (see §8).
//
// • Trigonometric functions – for Rational, Taylor series with half‑argument
//   reduction when ‖A‖ > π.  For GaussQi, via Euler identities using exp.
//
// • Square root – Newton iteration (one inversion per step, quadratic convergence).
//
//
// 3.  THE DANGER OF FLOATING‑POINT HEURISTICS
// ---------------------------------------------------------------------------
// Optimisations valid for IEEE‑754 double are often disastrous in exact
// rational arithmetic.  Every rational addition/multiplication increases
// integer bit length; the cost depends on *growth*, not operation count.
// Techniques that assume constant cost per operation (e.g., L∞ norm,
// Paterson–Stockmeyer, Bareiss) backfire spectacularly.
//
//
// 4.  ATTEMPTED OPTIMISATIONS THAT REGRESSED PERFORMANCE
// ---------------------------------------------------------------------------
//
// 4.1  L∞ norm instead of element‑wise maximum.
//   Computing row sums requires many rational additions (LCM/GCD cascade).
//   Slower by factor 2–10; caused two log tests to exhaust max_scale.
//
// 4.2  Adaptive Padé order with “+4” safety margin.
//   Added 4–6 to m for typical ε=1e‑19, θ≈0.5.  Runtime penalty without
//   accuracy gain (error already below ε).
//
// 4.3  Paterson–Stockmeyer for P(A) and Q(A) evaluation.
//   Overhead exceeds saving for m ≤ 16, the regime we operate in.
//
// 4.4  Bareiss fraction‑free Gaussian elimination.
//   Clearing denominators produced gigantic integers (thousands of bits)
//   for 5×5 matrices, dwarfing plain LU solve.
//
// 4.5  Logarithm scaling by repeated square roots (Denman–Beavers).
//   Two inversions per step, plus catastrophic bit growth.  2500× slowdown.
//
// 4.6  Logarithm scaling by division by 2 for complex matrices.
//   Does not bring B closer to I; with non‑zero phase, repeated halving
//   drives B toward -I, not I.  Convergence fails.
//
//
// 5.  OPTIMISATIONS THAT WORKED
// ---------------------------------------------------------------------------
//
// 5.1  Replace explicit inverse with linear‑system solve in log.
//   Z = (X+I).partialPivLu().solve(X−I) instead of (X−I)*(X+I).inverse().
//   Avoids final matrix‑matrix product.  5–15% speed‑up on log tests.
//
// 5.2  Newton iteration for matrix square root.
//   Y' = (Y + A·Y⁻¹)/2 (one inversion per step) instead of Denman–Beavers
//   (two inversions).  ~40–50% improvement.
//
// 5.3  Padé order directly from Stirling’s formula, without safety margin.
//   For exp:      m = ⌈ ln(1/ε) / (2·ln(4/θ)) ⌉
//   For log:      m = ⌈ ln(1/ε) / (2·ln(1/θ)) ⌉   (different!)
//   Using the correct formula for each function eliminated the need for
//   empirical fudge factors.  For complex matrices, this gave 30–50% speed‑up.
//
// 5.4  Trace normalisation for complex matrix logarithm.
//   No iterative scaling.  B = M / (trace(M)/n) centres the spectrum at 1.
//   For typical matrices, ‖X‖ = ‖B−I‖ ≤ 1 (often ≤ 0.5).  Direct Padé for
//   log(I+X) with order m from the log formula yields 1e-19 accuracy in
//   hundreds of milliseconds for 5×5 matrices.  No bit growth explosion.
//
//
// 6.  BENCHMARK IMPACT (measured on 2026‑05‑12, 66 tests)
// ---------------------------------------------------------------------------
//   Version                                    Total time   Status
//   Before fixes (wrong formula, +4 fudge)      ~11.5 s      1 failure
//   After formula separation and sign fixes     ~11.5 s      66 PASSED
//
//   Key timings:
//     ComplexMatrixC_Log:    397 ms (no empirical margin)
//     ComplexMatrixC_Exp:    944 ms
//     ComplexMatrixC_Sin:   2042 ms
//     ComplexMatrixC_Cos:   1980 ms
//     ComplexMatrixC_Sqrt:   436 ms
//     SinSqPlusCosSq (GaussQi): 3403 ms
//
//   All Wolfram verification tests passed.
//
//
// 7.  KEY LESSONS FOR EXACT RATIONAL ARITHMETIC
// ---------------------------------------------------------------------------
//   • Integer length grows with every operation – minimise total arithmetic,
//     not just operation count.  GCD reduction is not free.
//   • Matrix size n ≤ 10; cubic LU cost is acceptable.
//   • Element‑wise maximum norm is cheap and sufficient.
//   • Padé order formulas differ between functions:
//        exp: m ≈ ln(1/ε) / (2·ln(4/θ))
//        log: m ≈ ln(1/ε) / (2·ln(1/θ))
//     Confusing them wastes performance or breaks accuracy.
//   • Complex matrix multiplication costs ~4× a real one.
//   • Always write the formula in a comment directly above the code that
//     implements it, then verify they match.  A sign error (ln(ε) vs ln(1/ε),
//     ln(θ/4) vs ln(4/θ)) can cost days of debugging.
//
//
// 8.  GUIDELINES FOR FUTURE DEVELOPMENT
// ---------------------------------------------------------------------------
//   • Keep norms simple – element‑wise maximum.
//   • No blind safety margins.  If a test fails, adjust the formula
//     or add a run‑time fallback (e.g., “if not converged, m+=2 and retry”).
//   • For complex log, trace normalisation + direct Padé suffices for most
//     matrices.  If a pathological case appears (eigenvalue spread > 10),
//     apply a single Cayley transform: compute Z = (B−I)(B+I)⁻¹, then
//     log(B) = 2·atanh(Z) using its own Padé table (not yet implemented).
//   • Precompute Padé coefficient tables for frequently used orders
//     (m = 4,6,8,…,20).  Already done for exp (c coefficients) and
//     partially for log (pade_log_table).
//   • Test honestly – use relative tolerance for large entries.
//   • Long‑term: consider Schur–Parlett methods if rational eigenvalue
//     problems become feasible for n ≈ 20.
//
//
// 9.  POST‑MORTEM: THE TWO DAYS THAT SHOOK THE LOGARITHM
// ---------------------------------------------------------------------------
// The complex matrix logarithm failed through several attempts:
//   • Division by 2 – no phase reduction, converges to -I.
//   • Repeated square roots – bit length explosion, effectively infinite.
//   • Diagonal shifts – unstable, potential division by zero.
//   • Trace normalisation + direct Padé – works.
//
// The final bug was trivial: the code formula did not match the comment.
// For exp, the code had log(θ/4) instead of log(4/θ).  For log, it used
// the exp formula.  Fixing the signs and splitting the functions took
// minutes once spotted.
//
// MORAL: If the algorithm does not converge, check the signs.  If it
// converges slowly, check whether you are using the right formula at all.
//
//
// ----------------------------------------------------------------------------
// SEMANTICS OF THE EPSILON PARAMETER FOR MATRIX FUNCTIONS
// ----------------------------------------------------------------------------
// The scalar `eps` is a **relative** accuracy requirement:
//   |computed(i,j) – exact(i,j)| ≤ C·eps·|exact(i,j)|
// where C is a small constant (≈1).  For near‑unity results, absolute
// tolerance is acceptable; for large entries (e.g., exp(A) ~ 10¹²), use
// relative tolerance: tol = eps·max(1, |reference|).
//
// • Identity‑based tests (sin²+cos²=I, exp(A)·exp(-A)=I) may use absolute
//   tolerance (expected values ≈1).
// • Wolfram‑comparison tests MUST use relative tolerance.  Our suite follows
//   this rule.
// ----------------------------------------------------------------------------
#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <type_traits>
#include <cmath>
#include <vector>
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
        // Minimal Padé order via Stirling’s formula (no safety margin)
        // --------------------------------------------------------------------
        inline int pade_order_stirling(const Rational& norm, const Rational& eps) {
            double theta = norm.to_double();
            double target = eps.to_double();
            if (target <= 0.0) target = 1e-19;
            if (theta < 1e-12) return 4;               // negligible norm
            // m ≈ ln(1/ε) / (2 * ln(4/θ)) = -1*ln(ε) / (2 * ln(4/θ))
            double m_approx = -1.0*std::log(target) / (2.0 * std::log(4.0/ theta));
            int m = static_cast<int>(std::ceil(m_approx));
            if (m < 4) m = 4;
            return m;
        }

        // ================================================================
        // FORWARD DECLARATIONS — all detail functions, to allow any
        // call ordering (mutual recursion, log→sqrt, sin/cos→exp, etc.)
        // ================================================================
        DynMatRational exp_rational(const DynMatRational& A, const Rational& eps);
        DynMatRational log_rational(const DynMatRational& A, const Rational& eps);
        DynMatRational sin_rational(const DynMatRational& A, const Rational& eps);
        DynMatRational cos_rational(const DynMatRational& A, const Rational& eps);
        DynMatRational sqrt_rational(const DynMatRational& A, const Rational& eps);

        DynMatGaussQi exp_gaussqi(const DynMatGaussQi& A, const Rational& eps);
        DynMatGaussQi log_gaussqi(const DynMatGaussQi& A, const Rational& eps);
        DynMatGaussQi sin_gaussqi(const DynMatGaussQi& A, const Rational& eps);
        DynMatGaussQi cos_gaussqi(const DynMatGaussQi& A, const Rational& eps);
        DynMatGaussQi sqrt_gaussqi(const DynMatGaussQi& A, const Rational& eps);
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
            Rational scaled_norm = normM / two_pow_k;  // <= 0.5

            int m = pade_order_stirling(scaled_norm, eps);
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

            // Check singularity
            if (M.determinant() == Rational{})
                throw std::domain_error("matrix::log: singular matrix");

            if (is_diagonal(M)) {
                DynMatRational R = DynMatRational::Zero(M.rows(), M.cols());
                for (int i = 0; i < M.rows(); ++i)
                    R(i, i) = delta::log(M(i, i), eps);
                return R;
            }

            // ----- Normalise by trace -----
            int n = M.rows();
            Rational trace = M.trace();
            Rational scale = trace / Rational(n);
            DynMatRational B = (M / scale).eval();

            // ----- Standard logarithm via division by 2 -----
            Rational log2 = delta::log(Rational(2), eps);
            DynMatRational X = B;
            int k = 0;
            const int max_scale = 100;
            DynMatRational I = DynMatRational::Identity(n, n);
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
            DynMatRational Z = X_plus_I.partialPivLu().solve(X_minus_I);

            DynMatRational Z2 = (Z * Z).eval();
            DynMatRational Z_pow = Z;
            DynMatRational sum = Z_pow;
            for (int n_iter = 1; n_iter <= 1000000; ++n_iter) {
                Z_pow = (Z_pow * Z2).eval();
                DynMatRational term = (Z_pow / Rational(2 * n_iter + 1)).eval();
                sum += term;
                Rational norm_term = matrix_max_norm(term);
                Rational norm_sum = matrix_max_norm(sum);
                if (!(norm_term > eps) && !(norm_term > eps * (norm_sum + 1)))
                    break;
            }
            sum = (sum * 2).eval();
            DynMatRational logB = (I * Rational(k) * log2 + sum).eval();

            // Reconstruct: log(M) = log(scale)*I + log(B)
            Rational log_scale = delta::log(scale, eps);
            return logB + I * log_scale;
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

            // Newton iteration: Y_{k+1} = 0.5 * (Y_k + A * Y_k^{-1})
            DynMatRational Y = M;
            const int max_iter = 1000;
            for (int iter = 0; iter < max_iter; ++iter) {
                DynMatRational Y_inv = Y.partialPivLu().inverse();
                DynMatRational Y_new = ((Y + M * Y_inv) / 2).eval();
                Rational diff = matrix_max_norm((Y_new - Y).eval());
                if (!(diff > eps))
                    return Y_new;
                Y = std::move(Y_new);
            }
            throw std::runtime_error("matrix::sqrt: Newton iteration did not converge");
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
            Rational scaled_norm = normM / two_pow_k;

            int m = pade_order_stirling(scaled_norm, eps);
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

        // Паде коэффициенты [m/m] для log(1+x)
// P_m(x) = sum p_j x^j, Q_m(x) = sum q_j x^j, q_0 = 1
        struct PadeLogCoeffs { std::vector<Rational> p, q; };

        static const std::map<int, PadeLogCoeffs> pade_log_table = {
            {4, {
                {0, Rational(1), Rational(1,2), Rational(1,15), Rational(1,140)},
                {1, Rational(4,5), Rational(3,5), Rational(4,35), Rational(1,70)}
            }},
            {6, {
                {0, Rational(1), Rational(1,2), Rational(1,9), Rational(1,72), Rational(1,525), Rational(1,3780)},
                {1, Rational(6,7), Rational(15,14), Rational(20,63), Rational(15,308), Rational(6,1617), Rational(1,3234)}
            }},
            {8, {
                {0, Rational(1), Rational(1,2), Rational(1,12), Rational(1,84), Rational(1,560), Rational(1,3150), Rational(1,16632), Rational(1,84084)},
                {1, Rational(8,9), Rational(14,9), Rational(56,99), Rational(70,429), Rational(56,2145), Rational(28,11583), Rational(8,64467), Rational(1,128934)}
            }},
            {10, {
                {0, Rational(1), Rational(1,2), Rational(1,15), Rational(1,90), Rational(1,525), Rational(1,2835), Rational(1,14580), Rational(1,72765), Rational(1,355300), Rational(1,1695100)},
                {1, Rational(10,11), Rational(27,11), Rational(360,407), Rational(1050,4477), Rational(252,4807), Rational(2100,218219), Rational(450,329891), Rational(75,502874), Rational(10,829477), Rational(1,1658954)}
            }},
            {12, {
                {0, Rational(1), Rational(1,2), Rational(1,18), Rational(1,96), Rational(1,525), Rational(1,2700), Rational(1,13230), Rational(1,62720), Rational(1,289800), Rational(1,1309770), Rational(1,5817420), Rational(1,25496328)},
                {1, Rational(12,13), Rational(33,13), Rational(440,247), Rational(1485,1482), Rational(1188,2431), Rational(2310,11063), Rational(1980,25051), Rational(1485,56258), Rational(440,56869), Rational(297,149006), Rational(12,268764), Rational(1,1343820)}
            }}
        };


        // --------------------------------------------------------------------
// Динамическое вычисление коэффициентов [m/m] Паде для log(1+x)
// --------------------------------------------------------------------
        inline void compute_pade_log_coeffs(int m,
            std::vector<Rational>& p,
            std::vector<Rational>& q) {
            // Коэффициенты ряда Меркатора: a_0 = 0, a_k = (-1)^{k+1} / k для k >= 1
            auto a = [](int k) -> Rational {
                if (k == 0) return Rational(0);
                return (k % 2 == 1) ? Rational(1, k) : Rational(-1, k);
                };

            // Строим систему для q_1...q_m (q_0 = 1)
            // Уравнения: Σ_{j=0}^m q_j a_{i-j} = 0 для i = m+1 ... 2m
            int n = m;
            std::vector<std::vector<Rational>> A(n, std::vector<Rational>(n));
            std::vector<Rational> rhs(n);

            for (int i = 0; i < n; ++i) {
                int idx = m + 1 + i;   // индекс в ряду, для которого сумма = 0
                for (int j = 0; j < n; ++j) {
                    A[i][j] = a(idx - (j + 1));   // q_j соответствует j+1
                }
                rhs[i] = -a(idx);   // перенос q_0 * a_idx
            }

            // Решаем систему A * q_vec = rhs методом Гаусса (без выбора главного)
            for (int col = 0; col < n; ++col) {
                // Находим ненулевой элемент в столбце
                int pivot = col;
                while (pivot < n && A[pivot][col] == Rational(0)) ++pivot;
                if (pivot == n) continue; // вырожденная система (не должна возникнуть)
                if (pivot != col) {
                    std::swap(A[col], A[pivot]);
                    std::swap(rhs[col], rhs[pivot]);
                }
                Rational inv = Rational(1) / A[col][col];
                for (int j = col; j < n; ++j) A[col][j] *= inv;
                rhs[col] *= inv;
                for (int row = 0; row < n; ++row) {
                    if (row != col && A[row][col] != Rational(0)) {
                        Rational factor = A[row][col];
                        for (int j = col; j < n; ++j) A[row][j] -= factor * A[col][j];
                        rhs[row] -= factor * rhs[col];
                    }
                }
            }

            // Формируем q: q_0 = 1, q_1..q_m из решения
            q.resize(m + 1);
            q[0] = Rational(1);
            for (int i = 0; i < n; ++i) q[i + 1] = rhs[i];

            // Вычисляем P_m: p_i = Σ_{j=0}^i q_j a_{i-j}
            p.resize(m + 1);
            for (int i = 0; i <= m; ++i) {
                Rational sum = 0;
                for (int j = 0; j <= i; ++j) {
                    sum += q[j] * a(i - j);
                }
                p[i] = sum;
            }
        }
        
        inline int pade_order_log(const Rational& normX, const Rational& eps) {
            double theta = normX.to_double();
            double target = eps.to_double();
            if (target <= 0.0) target = 1e-19;
            if (theta < 1e-12) return 4;
            // m ≈ ln(1/ε) / (2 * ln(1/θ)) = -ln(ε) / (2 * ln(1/θ))
            double m_approx = -1.0*std::log(target) / (2.0 * std::log(1.0 / theta));
            int m = static_cast<int>(std::ceil(m_approx));
            if (m < 4) m = 4;
            if (m > 100) m = 100; // или больше
            return m;
        }
        // --------------------------------------------------------------------
        // Оптимизированный log_gaussqi с Паде-аппроксимацией
        // --------------------------------------------------------------------
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

            // Нормировка следом
            int n = M.rows();
            GaussQi trace = M.trace();
            GaussQi scale = trace / GaussQi(Rational(n), Rational(0));
            DynMatGaussQi B = (M / scale).eval();

            DynMatGaussQi I = DynMatGaussQi::Identity(n, n);
            DynMatGaussQi X = (B - I).eval();
            Rational normX = matrix_max_norm(X);

            // Выбор порядка Паде
            int m = pade_order_log(normX, eps);
            if (m < 4) m = 4;
            //m += 4;//недостающий запас членов ряда для Логарифма.
            if (m > 100) m = 100;   // разумный верхний предел

            std::vector<Rational> p, q;
            // Используем таблицу для m <= 14, иначе вычисляем динамически
            if (pade_log_table.count(m)) {
                p = pade_log_table.at(m).p;
                q = pade_log_table.at(m).q;
            }
            else {
                compute_pade_log_coeffs(m, p, q);
            }

            // Вычисление P_m(X) и Q_m(X)
            DynMatGaussQi Xpow = I;
            DynMatGaussQi P = DynMatGaussQi::Zero(n, n);
            DynMatGaussQi Q = DynMatGaussQi::Zero(n, n);

            for (int j = 0; j <= m; ++j) {
                if (p[j] != 0) P += p[j] * Xpow;
                if (q[j] != 0) Q += q[j] * Xpow;
                if (j < m) Xpow = (Xpow * X).eval();
            }

            // logB = Q^{-1} * P
            DynMatGaussQi logB = Q.partialPivLu().solve(P);

            // Возвращаем масштаб
            GaussQi log_scale = delta::log(scale, eps);
            return logB + I * log_scale;
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

            // Newton iteration for complex matrices
            DynMatGaussQi Y = M;
            const int max_iter = 1000;
            for (int iter = 0; iter < max_iter; ++iter) {
                DynMatGaussQi Y_inv = Y.partialPivLu().inverse();
                DynMatGaussQi Y_new = ((Y + M * Y_inv) / GaussQi(2)).eval();
                Rational diff = matrix_max_norm((Y_new - Y).eval());
                if (!(diff > eps))
                    return Y_new;
                Y = std::move(Y_new);
            }
            throw std::runtime_error("matrix::sqrt: Newton iteration did not converge");
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