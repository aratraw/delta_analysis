// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// evaluation_core.h
// -----------------------------------------------------------------------------
// CORE IMPLEMENTATIONS OF TRANSCENDENTAL FUNCTIONS
// -----------------------------------------------------------------------------
// This file provides eager (immediate) implementations of elementary
// transcendental functions on the rational Value type.
//
// FUNCTIONS: sqrt, exp, log, sin, cos, acos, asin, atan, tan, pi, e, pow.
//
// DESIGN PHILOSOPHY:
//   - For each function we provide both a fast floating‑point path
//     (for coarse epsilon, using cpp_dec_float_100) and an exact rational
//     series path (for fine epsilon or when float is slower).
//   - The hybrid threshold HYBRID_THRESHOLD = 1e-35 decides which path to take.
//   - Series methods implement argument reduction to guarantee fast convergence.
//   - Binary splitting is used for π, sin, cos, atan to avoid rational swell.
//   - Epsilon scaling ensures absolute error meets the requested tolerance.
//
// -----------------------------------------------------------------------------
// IMPORTANT ENGINEERING DECISIONS AND LESSONS LEARNED (MUST READ)
// -----------------------------------------------------------------------------
//
// 1. Choice between fast (float) and accurate (series) paths.
// ------------------------------------------------------------
// Originally we used HYBRID_THRESHOLD = 1e-35 for ALL functions: if eps >= threshold,
// call float implementations based on cpp_dec_float_100.
//
// Benchmarks showed:
//   - For sin, cos, exp, pi, acos, asin, atan, tan the float path gives 2-3x speedup
//     for moderate precision (eps ~ 1e-21). This is justified.
//   - For sqrt, log, e the float path turned out to be SLOWER than pure rational
//     algorithms due to overhead of converting Value ↔ cpp_dec_float_100 and string
//     parsing in to_rational_with_eps. Therefore for these functions the float path
//     was removed – they always use rational (series) methods.
//   - For exp we additionally introduced EXP_FLOAT_ARG_THRESHOLD = 20.0.
//     When |x| > 20 the float path loses relative accuracy due to the limited mantissa
//     of cpp_dec_float_100; thus even for coarse eps we force series_exp.
//
// 2. Structure of series functions and argument reduction.
// ---------------------------------------------------------
// Each series function implements argument reduction for fast convergence:
//   - sin/cos: reduce to [-π, π] using series_pi.
//   - exp: divide by 2^k until |x| <= 2, then square the result k times.
//          internal_eps is scaled to account for both reduction and magnitude
//          of the final value, guaranteeing absolute error ≤ requested eps.
//   - log: reduce to [1/2, 2] via k * ln2.
//   - sqrt: scale by dividing/multiplying by 4 if x is outside [1e-8, 1e8].
//
// 3. Quadratic coefficient recomputation and a failed "optimisation" (LESSON).
// ----------------------------------------------------------------------------
// In one version we tried to accelerate series summation by generating all terms
// into a vector and using pyramidal compact reduction (PCR).
// Result: CATASTROPHIC SLOWDOWN (up to 5x slower than naive).
// Reasons:
//   - Coefficients (factorials) were recomputed from scratch for each term in O(i),
//     whereas the naive loop updates the term recur‑rently in O(1).
//   - Vector allocation and PCR for N ~ 200 incurred huge memory and copy overhead.
// CONCLUSION: recurrent term update in a simple loop is optimal.
// DO NOT attempt to "vectorise" Taylor series with rational numbers!
//
// 4. Nature of the eps parameter and testing of large arguments.
// ---------------------------------------------------------------
// The eps parameter specifies ABSOLUTE error: |f(x) - result| < eps.
// For exp(100) ~ 10^43, asking for absolute eps=1e-12 requires 55 correct significant
// digits, which demands enormous computational cost.
//
// In tests we adopted a pragmatic decision: for huge values we check relative
// closeness. If a user truly needs absolute eps=1e-12 for exp(1000), they can
// explicitly pass eps = 1e-12 / exp_est. The library does not do this automatically
// to keep performance predictable. If needed, uncomment the "ABSOLUTE PRECISION
// FOR LARGE X" block in series_exp (see below).
//
// 5. Handling of negative arguments.
// -----------------------------------
// For sin/cos/exp negative arguments are reduced to positive via parity properties
// or 1/exp(-x). This ensures working with positive series, avoiding alternating
// signs and loss of precision.
//
// 6. Safety and maximum iterations.
// ---------------------------------
// DEFAULT_MAX_ITER = 1'000'000 – protection against infinite loops.
// In practice for |x| <= 2 convergence occurs in 30-100 iterations.
//
// 7. Caching of π and acceleration of inverse trigonometric functions.
// --------------------------------------------------------------------
// - series_pi caches the result for each epsilon value.
// - series_acos uses std::acos for initial approximation (15 correct digits).
// - asin, atan, tan are implemented via identities, minimising new code.
//
// -----------------------------------------------------------------------------
// IF YOU WANT TO CHANGE SOMETHING – READ THE ABOVE.
// Particularly dangerous:
//   - adding vectorisation of series (kills performance);
//   - removing internal_eps scaling in series_exp (breaks accuracy);
//   - changing HYBRID_THRESHOLD without benchmarks.
// -----------------------------------------------------------------------------

#pragma once
#include "global_state.h"
#include "storage.h"
#include "utils.h"
#include <boost/math/constants/constants.hpp>

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <stack>
#include <string>
#include <vector>
#include <iostream>
#include <map>   // for π cache

namespace delta::internal {

    // Forward declarations
    Value eager_abs(const Value& a);
    Value eager_sqrt(const Value& x, const Value& eps);
    Value eager_exp(const Value& x, const Value& eps);
    Value eager_log(const Value& x, const Value& eps);
    Value eager_sin(const Value& x, const Value& eps);
    Value eager_cos(const Value& x, const Value& eps);
    Value eager_acos(const Value& x, const Value& eps);
    Value eager_asin(const Value& x, const Value& eps);
    Value eager_atan(const Value& x, const Value& eps);
    Value eager_tan(const Value& x, const Value& eps);
    Value eager_pi(const Value& eps);
    Value eager_e(const Value& eps);
    Value eager_pow(const Value& base, const Value& exp, const Value& eps);
    Value eager_pow_int(const Value& base, const dumb_int& exponent);

    // Series (rational) implementations
    Value series_sqrt(const Value& x, const Value& eps);
    Value series_exp(const Value& x, const Value& eps);
    Value series_log(const Value& x, const Value& eps);
    Value series_sin(const Value& x, const Value& eps);
    Value series_cos(const Value& x, const Value& eps);
    Value series_acos(const Value& x, const Value& eps);
    Value series_asin(const Value& x, const Value& eps);
    Value series_atan(const Value& x, const Value& eps);
    Value series_tan(const Value& x, const Value& eps);
    Value series_pi(const Value& eps);
    Value series_e(const Value& eps);
    Value series_ln2(const Value& eps);

    // ----------------------------------------------------------------------------
    // Helper predicates (using storage.h versions)
    // ----------------------------------------------------------------------------
    inline bool is_less(const Value& a, const Value& b) { return a < b; }
    inline bool is_greater(const Value& a, const Value& b) { return a > b; }

    // ----------------------------------------------------------------------------
    // GLOBAL THRESHOLD FOR CHOOSING FLOAT VS SERIES PATH.
    // Determined by benchmarks: for eps >= 1e-35 float paths via cpp_dec_float_100
    // are faster for sin, cos, exp, pi, acos, asin, atan, tan.
    // For sqrt, log, e float paths are removed because they are always slower.
    constexpr double HYBRID_THRESHOLD = 1e-35;

    // ============================================================================
    // Arithmetic operations now directly via Value operators
    // ============================================================================
    inline Value eager_abs(const Value& a) {
        return is_negative(a) ? -a : a;
    }

    // ============================================================================
    // High‑precision floating‑point helpers (float path)
    // ============================================================================

    using HighPrecFloat = boost::multiprecision::cpp_dec_float_100;

    inline HighPrecFloat to_high_prec(const Value& v) {
        return v.convert_to<HighPrecFloat>();
    }

    // ----------------------------------------------------------------------------
    // to_rational_with_eps: converts a high‑precision float to Value with
    // enough digits to guarantee error < eps.
    // Uses string representation – slow, but necessary to preserve rational accuracy.
    // Because of this cost, float paths are not beneficial for sqrt, log, e.
    // ----------------------------------------------------------------------------
    inline Value to_rational_with_eps(const HighPrecFloat& f, const Value& eps, int extra_digits = 2) {
        HighPrecFloat eps_f = to_high_prec(eps);
        if (eps_f <= 0) throw std::domain_error("Epsilon must be positive");

        // Determine how many decimal digits we need to represent to guarantee error < eps
        int digits_needed = static_cast<int>(-log10(eps_f.convert_to<double>())) + extra_digits;
        if (digits_needed < 1) digits_needed = 1;
        if (digits_needed > 100) digits_needed = 100;

        // Convert to fixed-point string with required digits
        std::string s = f.str(digits_needed, std::ios_base::fixed);
        size_t dot = s.find('.');
        std::string integer_part = s.substr(0, dot);
        std::string fractional_part = s.substr(dot + 1);
        if (fractional_part.size() > static_cast<size_t>(digits_needed))
            fractional_part = fractional_part.substr(0, digits_needed);

        // Handle sign
        bool negative = false;
        if (!integer_part.empty() && integer_part[0] == '-') {
            negative = true;
            integer_part = integer_part.substr(1);
        }

        // Strip leading zeros
        size_t non_zero = integer_part.find_first_not_of('0');
        if (non_zero != std::string::npos) integer_part = integer_part.substr(non_zero);
        else integer_part = "0";
        if (negative && integer_part != "0") integer_part = "-" + integer_part;

        // Build numerator string (integer_part + fractional_part)
        std::string num_str;
        if (integer_part == "0" || integer_part == "-0") {
            num_str = fractional_part;
            if (num_str.empty()) num_str = "0";
        }
        else {
            num_str = integer_part + fractional_part;
        }

        // Remove leading zeros from the combined number
        if (num_str.size() > 1 && num_str[0] == '0') {
            size_t first_nonzero = num_str.find_first_not_of('0');
            if (first_nonzero != std::string::npos) num_str = num_str.substr(first_nonzero);
            else num_str = "0";
        }

        // Create fraction: numerator = integer, denominator = 10^(fractional_part length)
        dumb_int num(num_str);
        dumb_int den(1);
        for (size_t i = 0; i < fractional_part.size(); ++i) den *= 10;
        dumb_int g = boost::multiprecision::gcd(num, den);
        num /= g; den /= g;
        return Value(num, den);
    }

    // ------------------------------------------------------------------------
    // Float implementations for functions where they give speedup for coarse eps.
    // IMPORTANT: for sin and cos we handle sign to guarantee odd/even symmetry
    // (otherwise negative‑argument tests would fail).
    // ------------------------------------------------------------------------
    inline Value float_exp(const Value& x, const Value& eps) {
        return to_rational_with_eps(exp(to_high_prec(x)), eps);
    }

    inline Value float_sin(const Value& x, const Value& eps) {
        // sin(-x) = -sin(x)
        if (is_negative(x)) return -float_sin(-x, eps);
        return to_rational_with_eps(sin(to_high_prec(x)), eps);
    }

    inline Value float_cos(const Value& x, const Value& eps) {
        // cos is even: cos(-x) = cos(x)
        Value positive_x = is_negative(x) ? -x : x;
        return to_rational_with_eps(cos(to_high_prec(positive_x)), eps);
    }

    inline Value float_acos(const Value& x, const Value& eps) {
        HighPrecFloat fx = to_high_prec(x);
        if (fx < -1 || fx > 1) throw std::domain_error("acos argument out of [-1,1]");
        return to_rational_with_eps(acos(fx), eps);
    }

    inline Value float_pi(const Value& eps) {
        HighPrecFloat pi_val = boost::math::constants::pi<HighPrecFloat>();
        return to_rational_with_eps(pi_val, eps);
    }

    inline Value float_asin(const Value& x, const Value& eps) {
        HighPrecFloat fx = to_high_prec(x);
        if (fx < -1 || fx > 1) throw std::domain_error("asin argument out of [-1,1]");
        return to_rational_with_eps(asin(fx), eps);
    }

    inline Value float_atan(const Value& x, const Value& eps) {
        return to_rational_with_eps(atan(to_high_prec(x)), eps);
    }

    inline Value float_tan(const Value& x, const Value& eps) {
        return to_rational_with_eps(tan(to_high_prec(x)), eps);
    }

    // ============================================================================
    // Exact integer roots – try to extract exact root before falling back to series
    // ============================================================================
    inline bool is_integer(const Value& v) {
        return denominator(v) == 1;
    }

    inline dumb_int get_integer(const Value& v) {
        return numerator(v);
    }

    // Binary search for integer nth root. Returns 0 if not an exact integer root.
    inline dumb_int integer_nth_root(const dumb_int& a, const dumb_int& n) {
        if (n == 0 || n == 1 || a == 0) return n == 0 ? 0 : a;
        if (a < 0) return 0;
        int n_int = n.convert_to<int>();
        if (n_int > 1000) return 0;
        size_t bits = boost::multiprecision::msb(a) + 1;
        dumb_int high = (dumb_int(1) << ((bits + n_int - 1) / n_int)) + 1;
        dumb_int low = 1;
        while (low <= high) {
            dumb_int mid = (low + high) / 2;
            dumb_int pow = boost::multiprecision::pow(mid, n_int);
            if (pow == a) return mid;
            if (pow < a) low = mid + 1;
            else high = mid - 1;
        }
        return 0;
    }

    inline std::optional<Value> try_exact_nth_root(const Value& base, const Value& n_val) {
        if (!is_integer(n_val)) return std::nullopt;
        dumb_int n = numerator(n_val);
        if (n <= 0 || n > 1000) return std::nullopt;

        if (is_zero(base)) return Value(0);
        bool negative = is_negative(base);
        if (negative && n % 2 == 0) return std::nullopt;  // even root of negative

        dumb_int num = numerator(base);
        dumb_int den = denominator(base);
        if (negative) num = -num;

        dumb_int root_num = integer_nth_root(num, n);
        dumb_int root_den = integer_nth_root(den, n);
        if (root_num != 0 && root_den != 0) {
            if (negative) root_num = -root_num;
            return Value(root_num) / Value(root_den);
        }
        return std::nullopt;
    }

    // ============================================================================
    // Configuration for series methods
    // ============================================================================
    constexpr size_t DEFAULT_MAX_ITER = 1000000;   // protection against infinite loops
    constexpr size_t NEWTON_MAX_ITER = 1000;
    constexpr size_t ACOS_MAX_ITER = 100;

    // ============================================================================
    // Series (rational) implementations of transcendentals
    // ============================================================================

    // ----------------------------------------------------------------------------
    // series_ln2: ln(2) using the series for arctanh(1/3).
    // arctanh(z) = z + z^3/3 + z^5/5 + ... converges faster than the Mercator series.
    // Used by series_log for reduction.
    // ----------------------------------------------------------------------------
    inline Value series_ln2(const Value& eps) {
        Value z = Value(1) / 3;
        Value z2 = z * z;
        Value term = z, sum = term;
        Value n = 1;
        size_t iter = 0;
        while (iter < DEFAULT_MAX_ITER) {
            term *= z2;
            n += 2;
            sum += term / n;
            ++iter;
            if (term < eps && term > -eps) break;
        }
        return sum * 2;  // ln(2) = 2 * arctanh(1/3)
    }

    // ----------------------------------------------------------------------------
    // series_sqrt: Newton's method with conditional argument reduction.
    // Reduction is applied only if x is outside [1e-8, 1e8] – for typical numbers
    // the overhead is unnecessary, while for extreme numbers it prevents hangs.
    // Initial guess via double accelerates convergence to 2-3 iterations.
    // ----------------------------------------------------------------------------
    inline Value series_sqrt(const Value& x, const Value& eps) {
        if (is_zero(x)) return Value(0);
        if (is_one(x)) return Value(1);
        if (is_negative(x)) throw std::domain_error("sqrt of negative number");

        double x_approx = to_double(x);
        const double SCALE_LOW = 1e-8;
        const double SCALE_HIGH = 1e8;
        bool need_scaling = (x_approx < SCALE_LOW || x_approx > SCALE_HIGH);

        Value m = x;
        int k = 0;
        if (need_scaling) {
            // Scale by 4^k to bring m into [1/4, 1]
            while (m > 1) {
                m /= 4;
                ++k;
            }
            while (m < Value(1) / 4) {
                m *= 4;
                --k;
            }
        }

        // Scale epsilon accordingly: each sqrt scale reduces precision by factor 2
        Value internal_eps = eps;
        if (need_scaling) {
            for (int i = 0; i < std::abs(k); ++i) {
                internal_eps /= 2;
            }
        }

        double m_approx = to_double(m);
        Value guess;
        guess.assign(std::sqrt(m_approx));  // double approximation as initial guess

        Value diff;
        size_t iter = 0;
        do {
            Value next = (guess + m / guess) / 2;  // Newton iteration
            diff = eager_abs(next - guess);
            guess = next;
            ++iter;
            if (iter > NEWTON_MAX_ITER) break;
        } while (diff > internal_eps);

        if (need_scaling) {
            Value result = guess;
            if (k > 0) {
                for (int i = 0; i < k; ++i) result *= 2;  // sqrt(x) = sqrt(m) * 2^k
            }
            else if (k < 0) {
                for (int i = 0; i < -k; ++i) result /= 2; // sqrt(x) = sqrt(m) / 2^{|k|}
            }
            return result;
        }
        return guess;
    }

    // ----------------------------------------------------------------------------
    // series_exp: exponential function.
    // Reduction: if |x| > SERIES_EXP_REDUCE_THRESHOLD (2.0), repeatedly divide by 2
    // until |x| <= 2, then series, then square k times.
    // Epsilon scaling: internal_eps is divided by 2^(exp_bits + k + 2) to guarantee
    // absolute error after squaring.
    // Negative arguments: use exp(x) = 1/exp(-x).
    // ----------------------------------------------------------------------------
    constexpr double SERIES_EXP_REDUCE_THRESHOLD = 2.0;

    inline Value series_exp(const Value& x, const Value& eps) {
        if (is_zero(x)) return Value(1);
        // For negative arguments: exp(x) = 1 / exp(-x)
        if (is_negative(x)) return Value(1) / series_exp(-x, eps);

        double x_d = to_double(x);

        // If argument is small, series converges quickly without reduction
        if (x_d <= SERIES_EXP_REDUCE_THRESHOLD) {
            Value sum = 1, term = 1;
            Value n = 1;
            size_t iter = 0;
            const size_t MAX_ITER = 1000;
            while (iter < MAX_ITER) {
                term *= x / n;
                sum += term;
                n += 1;
                ++iter;
                if (term < eps && term > -eps) break;
            }
            return sum;
        }

        // Argument reduction: exp(x) = (exp(x / 2^k))^(2^k)
        int k = 0;
        Value reduced = x;
        while (reduced > SERIES_EXP_REDUCE_THRESHOLD) {
            reduced /= 2;
            ++k;
        }

        // Estimate binary order of exp(x) via double (frexp gives exponent)
        double exp_est = std::exp(x_d);
        int exp_bits;
        std::frexp(exp_est, &exp_bits);

        // Scale epsilon: after squaring k times, error grows by factor 2^(exp_bits + k)
        Value internal_eps = eps;
        int total_shift = exp_bits + k + 2;   // +2 for safety margin
        for (int i = 0; i < total_shift; ++i) {
            internal_eps /= 2;
        }

        // Series for reduced argument
        Value sum = 1, term = 1;
        Value n = 1;
        size_t iter = 0;
        const size_t MAX_ITER = 1000;
        while (iter < MAX_ITER) {
            term *= reduced / n;
            sum += term;
            n += 1;
            ++iter;
            if (term < internal_eps && term > -internal_eps) break;
        }

        // Square k times (exact integer exponentiation)
        dumb_int exponent = dumb_int(1) << k;
        return eager_pow_int(sum, exponent);
    }

    // ----------------------------------------------------------------------------
    // series_log: natural logarithm.
    // Always series (float path removed). Reduction: scale argument to [1/2,2] via
    // k*ln2, then use fast series for ln((1+y)/(1-y)) with y = (m-1)/(m+1).
    // ----------------------------------------------------------------------------
    inline Value series_log(const Value& x, const Value& eps) {
        if (is_negative(x) || is_zero(x)) throw std::domain_error("log of non-positive");

        // Reduce argument to [1/2, 2]
        int k = 0;
        Value m = x;
        while (m > 2) {
            m /= 2;
            ++k;
        }
        while (m < Value(1) / 2) {
            m *= 2;
            --k;
        }

        Value ln2 = series_ln2(eps);
        // Use series: ln( (1+y)/(1-y) ) = 2*(y + y^3/3 + y^5/5 + ...)
        Value y = (m - 1) / (m + 1);
        Value y2 = y * y;
        Value term = y, sum = term;
        Value n = 1;
        size_t iter = 0;
        while (iter < DEFAULT_MAX_ITER) {
            term *= y2;
            n += 2;
            sum += term / n;
            ++iter;
            if (term < eps && term > -eps) break;
        }
        Value ln_m = sum * 2;
        return ln_m + Value(k) * ln2;
    }

    // ============================================================================
    // π via the Chudnovsky series using binary splitting.
    // NOTE: IF YOU THINK YOU WANT TO OPTIMIZE SOMETHING HERE - THINK AGAIN.
    // WE DEBUGGED THIS IMPLEMENTATION FOR HOURS. CHUDNOVSKY CONSISTS OF MAGIC
    // NUMBERS, AND IF YOU HANDLE A SINGLE ONE IMPROPERLY, THE RESULT IS GIBBERISH
    // ============================================================================

    // Constants of the Chudnovsky series.
    static const dumb_int CHUD_A = 545140134;
    static const dumb_int CHUD_B = 13591409;
    static const dumb_int CHUD_C = 640320;
    static const dumb_int CHUD_C3_OVER_24 = (dumb_int(CHUD_C) * CHUD_C * CHUD_C) / 24;
    static const dumb_int CHUD_D = 426880;

    struct ChudnovskyPQT {
        dumb_int P, Q, T;
    };

    // ----------------------------------------------------------------------------
    // Binary splitting for Chudnovsky.
    // Each term: a_k = (-1)^k * (6k)! * (A*k + B) / ((3k)! * (k!)^3 * C^(3k))
    // PQT = (P, Q, T) where T = Σ Q/P * a_k, but implemented recursively.
    // ----------------------------------------------------------------------------
    inline ChudnovskyPQT chudnovsky_bs(int64_t a, int64_t b) {
        if (b - a == 1) {
            dumb_int k(a);
            if (a == 0) {
                // Base term k=0: P=1, Q=1, T=B
                return { dumb_int(1), dumb_int(1), dumb_int(CHUD_B) };
            }
            else {
                // Base term k>=1: P(k) = (6k-5)(2k-1)(6k-1)
                dumb_int P = (6 * k - 5) * (2 * k - 1) * (6 * k - 1);
                dumb_int Q = k * k * k * CHUD_C3_OVER_24;
                // T(k) = A*k + B, sign flipped for odd k
                dumb_int T = k * CHUD_A + CHUD_B;
                if (a % 2 == 1) T = -T;
                return { P, Q, T };
            }
        }

        int64_t m = (a + b) / 2;
        auto L = chudnovsky_bs(a, m);
        auto R = chudnovsky_bs(m, b);

        // Merge formula:
        // P = P_L * P_R
        // Q = Q_L * Q_R
        // T = T_L * Q_R + P_L * T_R
        return {
            L.P * R.P,
            L.Q * R.Q,
            L.T * R.Q + L.P * R.T
        };
    }

    // ----------------------------------------------------------------------------
    // Recurrent version for small N (when binary splitting overhead not worth it)
    // ----------------------------------------------------------------------------
    inline Value pi_recurrent(int N, const Value& eps) {
        Value term(CHUD_B, 1);
        Value sum = term;

        for (int k = 0; k < N - 1; ++k) {
            dumb_int k1 = k + 1;
            // Transition factor from term_k to term_{k+1}
            dumb_int numer = (6 * k + 1) * (2 * k + 1) * (6 * k + 5);
            dumb_int denom_part = k1 * k1 * k1;

            Value factor = Value(-numer) * Value(CHUD_A * k1 + CHUD_B);
            Value denom = Value(denom_part) * Value(CHUD_C3_OVER_24) * Value(CHUD_A * k + CHUD_B);

            term = term * factor / denom;
            sum = sum + term;
        }

        Value sqrt_10005 = series_sqrt(Value(10005), eps / 1000);
        return (Value(CHUD_D) * sqrt_10005) / sum;
    }

    // ----------------------------------------------------------------------------
    // Main π function: switches between recurrent and binary splitting based on N,
    // caches results per epsilon.
    // ----------------------------------------------------------------------------
    inline Value series_pi(const Value& eps) {
        // Check cache first
        auto it = pi_cache.find(eps);
        if (it != pi_cache.end()) return it->second;

        // Each iteration gives ~14.18 decimal digits, +3 for safety margin
        double eps_d = std::abs(to_double(eps));
        int N = (eps_d <= 0) ? 10 : (int)std::max(2.0, std::ceil(-std::log10(eps_d) / 14.18) + 3);

        Value result;
        // Binary splitting becomes more efficient for N > 16 because it
        // prevents intermediate fraction growth
        if (N > 16) {
            auto res = chudnovsky_bs(0, N);
            Value S(res.T, res.Q);
            Value sqrt_10005 = series_sqrt(Value(10005), eps / 1000);
            result = (Value(CHUD_D) * sqrt_10005) / S;
        }
        else {
            result = pi_recurrent(N, eps);
        }

        pi_cache[eps] = result;
        return result;
    }

    // ============================================================================
    // Binary splitting for sin and cos
    // ============================================================================

    struct TrigPQT {
        dumb_int P;  // numerator (x2_num to the power)
        dumb_int Q;  // denominator (factorial * x2_den to the power)
        dumb_int T;  // accumulated sum (numerator, denominator will be Q of whole range)
    };

    // sin(x) = x * Σ_{k=0}∞ (-1)^k * (x^2)^k / (2k+1)!
    inline TrigPQT sin_bs_internal(int64_t a, int64_t b, const dumb_int& x2_num, const dumb_int& x2_den) {
        if (b - a == 1) {
            if (a == 0) {
                // k=0: term = 1
                return { x2_num, 1, 1 };
            }
            // k>=1: term = (-1)^k * x2_num^k / (x2_den^k * (2k+1)!)
            // Q accumulates denominator; T stores only sign for now
            dumb_int Q = x2_den * dumb_int(2 * a) * (2 * a + 1);
            dumb_int T = (a % 2 == 1) ? -1 : 1;
            return { x2_num, Q, T };
        }

        int64_t m = (a + b) / 2;
        auto L = sin_bs_internal(a, m, x2_num, x2_den);
        auto R = sin_bs_internal(m, b, x2_num, x2_den);

        // Merge: P = P_L * P_R, Q = Q_L * Q_R, T = T_L * Q_R + P_L * T_R
        return {
            L.P * R.P,
            L.Q * R.Q,
            L.T * R.Q + L.P * R.T
        };
    }

    // cos(x) = Σ_{k=0}∞ (-1)^k * (x^2)^k / (2k)!
    inline TrigPQT cos_bs_internal(int64_t a, int64_t b, const dumb_int& x2_num, const dumb_int& x2_den) {
        if (b - a == 1) {
            if (a == 0) {
                // k=0: term = 1
                return { x2_num, 1, 1 };
            }
            // k>=1: term = (-1)^k * x2_num^k / (x2_den^k * (2k)!)
            dumb_int Q = x2_den * dumb_int(2 * a - 1) * (2 * a);
            dumb_int T = (a % 2 == 1) ? -1 : 1;
            return { x2_num, Q, T };
        }

        int64_t m = (a + b) / 2;
        auto L = cos_bs_internal(a, m, x2_num, x2_den);
        auto R = cos_bs_internal(m, b, x2_num, x2_den);

        return {
            L.P * R.P,
            L.Q * R.Q,
            L.T * R.Q + L.P * R.T
        };
    }

    // ----------------------------------------------------------------------------
    // series_sin: reduces argument to [-π, π], then uses binary splitting.
    // Caches π, uses exact rational reduction without floating point.
    // ----------------------------------------------------------------------------
    inline Value series_sin(const Value& x, const Value& eps) {
        // sin(-x) = -sin(x)
        if (is_negative(x)) return -series_sin(-x, eps);

        Value pi_val = series_pi(eps);
        Value twopi = pi_val * 2;

        // Universal reduction without double: x = k*2π + reduced
        Value periods = x / twopi;
        dumb_int k_int = numerator(periods) / denominator(periods);
        Value reduced = x - Value(k_int) * twopi;

        // Reduce to [-π, π]
        if (reduced > pi_val) {
            reduced -= twopi;
        }
        else if (reduced < -pi_val) {
            reduced += twopi;
        }

        if (is_zero(reduced)) return Value(0);

        Value x2 = reduced * reduced;
        dumb_int x2_num = numerator(x2);
        dumb_int x2_den = denominator(x2);

        // Estimate number of terms needed based on epsilon
        double eps_d = to_double(eps);
        int64_t N = (eps_d <= 0) ? 10 : (int64_t)std::max(10.0, -std::log10(eps_d) * 0.8);
        if (N > 2000) N = 2000;  // safety cap

        auto res = sin_bs_internal(0, N, x2_num, x2_den);
        Value sum_series(res.T, res.Q);

        return reduced * sum_series;
    }

    // ----------------------------------------------------------------------------
    // series_cos: reduces argument to [0, π], then binary splitting.
    // ----------------------------------------------------------------------------
    inline Value series_cos(const Value& x, const Value& eps) {
        Value pi_val = series_pi(eps);
        Value twopi = pi_val * 2;

        // cos is even, work with absolute value
        Value abs_x = is_negative(x) ? -x : x;

        // Reduce to [0, π]
        Value periods = abs_x / twopi;
        dumb_int k_int = numerator(periods) / denominator(periods);
        Value reduced = abs_x - Value(k_int) * twopi;

        // Map to [0, π]
        if (reduced > pi_val) {
            reduced = twopi - reduced;
        }

        if (is_zero(reduced)) return Value(1);

        Value x2 = reduced * reduced;
        dumb_int x2_num = numerator(x2);
        dumb_int x2_den = denominator(x2);

        double eps_d = to_double(eps);
        int64_t N = (eps_d <= 0) ? 10 : (int64_t)std::max(10.0, -std::log10(eps_d) * 0.8);
        if (N > 2000) N = 2000;

        auto res = cos_bs_internal(0, N, x2_num, x2_den);
        Value sum_series(res.T, res.Q);

        return sum_series;
    }

    // ---------------------------------------------------------------
    // Helper structure for binary splitting (atan, asin)
    // ---------------------------------------------------------------
    struct BSResult {
        dumb_int P, Q, T;
    };

    // ---------------------------------------------------------------
    // series_atan: uses binary splitting, with reduction to small argument.
    // For |x| > 1: atan(x) = π/2 - atan(1/x)
    // For |x| > 0.5: atan(x) = π/4 + atan((x-1)/(x+1))
    // Then series for |x| ≤ 0.5: atan(x) = x * Σ (-x²)^k/(2k+1)
    // ---------------------------------------------------------------
    inline Value series_atan(const Value& x, const Value& eps) {
        if (is_zero(x)) return Value(0);

        bool negative = is_negative(x);
        Value xx = negative ? -x : x;

        // Reduction: |x| > 1 → π/2 - atan(1/x)
        if (xx > 1) {
            Value half_pi = series_pi(eps) / 2;
            Value inv = Value(1) / xx;
            Value atan_inv = series_atan(inv, eps);
            Value res = half_pi - atan_inv;
            return negative ? -res : res;
        }
        // Reduction: 0.5 < x ≤ 1 → π/4 + atan((x-1)/(x+1))
        if (xx > Value(1) / 2) {
            Value one(1);
            Value xp = (xx - one) / (xx + one);   // |xp| ≤ 1/3
            Value quarter_pi = series_pi(eps) / 4;
            Value atan_xp = series_atan(xp, eps);
            Value res = quarter_pi + atan_xp;
            return negative ? -res : res;
        }

        // Now xx ≤ 0.5, use series: atan(x) = x * Σ (-x²)^k / (2k+1)
        Value x2 = xx * xx;
        dumb_int x2_num = numerator(x2);
        dumb_int x2_den = denominator(x2);

        // Estimate number of terms N with safety margin
        double x2_d = to_double(x2);
        double eps_d = std::abs(to_double(eps));
        int N = 10;
        if (eps_d > 0) {
            while (std::pow(x2_d, N) / (2 * N + 1) > eps_d && N < 5000) ++N;
            N += 5;   // safety margin
        }
        else {
            N = 500;
        }

        // Binary splitting for S = Σ_{k=0}^{N-1} (-x²)^k / (2k+1)
        auto atan_bs = [&](auto&& self, int a, int b) -> BSResult {
            if (b - a == 1) {
                if (a == 0) {
                    // k=0: term = 1, multiplier for next terms = -x²
                    return { -x2_num, x2_den, dumb_int(1) };
                }
                else {
                    // k>=1: P = -x²_num, Q = x2_den * (2k+1), T = 1
                    dumb_int Q = x2_den * (2 * a + 1);
                    return { -x2_num, Q, dumb_int(1) };
                }
            }
            int m = (a + b) / 2;
            auto L = self(self, a, m);
            auto R = self(self, m, b);
            return {
                L.P * R.P,
                L.Q * R.Q,
                L.T * R.Q + L.P * R.T   // L.P already contains -x²_num
            };
            };

        auto res = atan_bs(atan_bs, 0, N);
        Value S(res.T, res.Q);   // S = T / Q
        Value result = xx * S;
        return negative ? -result : result;
    }

    // ---------------------------------------------------------------
    // series_asin: uses binary splitting.
    // For |x| close to 1 use asin(1)=π/2.
    // Series: asin(x) = x + Σ_{n=1}∞ ( (2n-1)!!/(2n)!! ) * x^(2n+1)/(2n+1)
    // Implemented via recurrence and binary splitting.
    // ---------------------------------------------------------------
    inline Value series_asin(const Value& x, const Value& eps) {
        if (x < -1 || x > 1)
            throw std::domain_error("asin argument out of [-1,1]");
        if (is_one(x))  return series_pi(eps) / 2;
        if (x == -1)    return -series_pi(eps) / 2;

        Value x2 = x * x;
        dumb_int x2_num = numerator(x2);
        dumb_int x2_den = denominator(x2);

        // Estimate number of terms N (starting from n=1)
        double x2_d = to_double(x2);
        double eps_d = std::abs(to_double(eps));
        int N = 10;
        if (eps_d > 0) {
            double x_d = to_double(eager_abs(x));
            while (x_d * std::pow(x2_d, N) / (2 * N + 1) > eps_d && N < 5000) ++N;
            N += 5;
        }
        else {
            N = 500;
        }

        // Binary splitting for S = Σ_{n=1}^{N-1} a_n,
        // where a_n = a_{n-1} * ( (2n-1)² x² ) / ( 2n (2n+1) ), a_0 = x
        auto asin_bs = [&](auto&& self, int a, int b) -> BSResult {
            if (b - a == 1) {
                // term with index n = a (n≥1)
                dumb_int P = (2 * a - 1) * (2 * a - 1) * x2_num;
                dumb_int Q = 2 * a * (2 * a + 1) * x2_den;
                return { P, Q, P };   // T = P (numerator of the term)
            }
            int m = (a + b) / 2;
            auto L = self(self, a, m);
            auto R = self(self, m, b);
            return {
                L.P * R.P,
                L.Q * R.Q,
                L.T * R.Q + L.P * R.T
            };
            };

        if (N <= 1) return x;

        auto res = asin_bs(asin_bs, 1, N);
        Value S(res.T, res.Q);
        return x + x * S;
    }

    // ---------------------------------------------------------------
    // series_acos: acos(x) = π/2 - asin(x)
    // ---------------------------------------------------------------
    inline Value series_acos(const Value& x, const Value& eps) {
        if (x < -1 || x > 1)
            throw std::domain_error("acos argument out of [-1,1]");

        Value clipped_x = x;
        if (clipped_x > Value(1)) clipped_x = Value(1);
        else if (clipped_x < Value(-1)) clipped_x = Value(-1);

        Value half_pi = series_pi(eps) / 2;
        return half_pi - series_asin(clipped_x, eps);
    }

    // ------------------------------------------------------------------------
    // series_tan: tan(x) = sin(x)/cos(x)
    // ------------------------------------------------------------------------
    inline Value series_tan(const Value& x, const Value& eps) {
        Value s = series_sin(x, eps);
        Value c = series_cos(x, eps);
        if (is_zero(c)) throw std::domain_error("tan: cos(x) is zero");
        return s / c;
    }

    // ------------------------------------------------------------------------
    // series_e: e = Σ 1/n! (float path removed – slower)
    // ------------------------------------------------------------------------
    inline Value series_e(const Value& eps) {
        Value sum = 1, term = 1;
        Value n = 1;
        size_t iter = 0;
        while (iter < DEFAULT_MAX_ITER) {
            term /= n;
            sum += term;
            n += 1;
            ++iter;
            if (term < eps) break;
        }
        return sum;
    }

    // ============================================================================
    // Integer exponentiation (binary exponentiation)
    // ============================================================================
    inline Value eager_pow_int(const Value& base, const dumb_int& exponent) {
        if (exponent == 0) return Value(1);
        if (exponent == 1) return base;
        bool negative = exponent < 0;
        dumb_int e = negative ? -exponent : exponent;
        Value result(1), b = base;
        while (e > 0) {
            if (e & 1) result *= b;
            e >>= 1;
            if (e > 0) b *= b;
        }
        return negative ? Value(1) / result : result;
    }

    // ============================================================================
    // Nth root (general rational exponent not integer)
    // ============================================================================
    inline int compute_extra_digits(const Value& eps, double operation_complexity = 1.0) {
        double eps_double = to_double(eps);
        if (eps_double <= 0) return 30;
        int digits_needed = static_cast<int>(std::ceil(-std::log10(eps_double))) + 2;
        int safety = static_cast<int>(std::ceil(10.0 * operation_complexity));
        return digits_needed + safety;
    }

    inline Value float_nth_root(const Value& x, const Value& n, const Value& eps) {
        bool x_neg = is_negative(x);
        if (x_neg) {
            bool n_even = false;
            if (is_integer(n)) {
                dumb_int n_int = numerator(n);
                if (n_int % 2 == 0) n_even = true;
            }
            if (n_even) throw std::domain_error("even root of negative number");
            return -float_nth_root(-x, n, eps);
        }
        if (is_zero(x)) return Value(0);
        double complexity = 1.0;
        if (is_integer(n)) {
            complexity = static_cast<double>(numerator(n));
        }
        int extra = compute_extra_digits(eps, complexity);
        HighPrecFloat fx = to_high_prec(x);
        HighPrecFloat fn = to_high_prec(n);
        HighPrecFloat res = pow(fx, 1.0 / fn);
        return to_rational_with_eps(res, eps, extra);
    }

    inline Value eager_nth_root(const Value& x, const Value& n, const Value& eps) {
        if (is_zero(n) || is_negative(n)) throw std::domain_error("nth_root: n must be positive");
        if (!is_integer(n)) throw std::domain_error("nth_root: n must be integer");
        dumb_int n_int = numerator(n);
        if (n_int == 0) throw std::domain_error("nth_root: n must be positive");
        if (n_int == 1) return x;
        if (n_int == 2) return eager_sqrt(x, eps);
        if (n_int % 2 == 0 && is_negative(x))
            throw std::domain_error("nth_root: even root of negative number");
        if (auto exact = try_exact_nth_root(x, n)) return *exact;
        if (n_int == 2 && to_double(eps) >= HYBRID_THRESHOLD)
            return float_nth_root(x, n, eps);
        Value guess = (x > 0) ? x / 2 : -eager_abs(x) / 2;
        Value n_val = n;
        Value n_minus_1 = n_val - 1;
        Value diff;
        size_t iter = 0;
        do {
            Value pow_n_minus_1 = eager_pow_int(guess, n_int - 1);
            Value next = (n_minus_1 * guess + x / pow_n_minus_1) / n_val;
            diff = eager_abs(next - guess);
            guess = next;
            ++iter;
            if (iter > NEWTON_MAX_ITER) break;
        } while (diff > eps);
        return guess;
    }

    // ============================================================================
    // General power with rational exponent: uses exp(log)
    // ============================================================================
    inline Value eager_pow(const Value& base, const Value& exp, const Value& eps) {
        if (is_zero(base)) {
            if (is_zero(exp)) throw std::domain_error("0^0 is undefined");
            if (is_negative(exp)) throw std::domain_error("0^negative is undefined");
            return base;
        }
        if (is_one(base)) return base;
        if (is_zero(exp)) return Value(1);

        bool exp_is_int = is_integer(exp);
        dumb_int exp_num = numerator(exp);
        dumb_int exp_den = denominator(exp);

        if (exp_is_int) {
            if (exp_num < 0) {
                Value base_recip = Value(1) / base;
                return eager_pow_int(base_recip, -exp_num);
            }
            return eager_pow_int(base, exp_num);
        }

        dumb_int p = exp_num, q = exp_den;
        bool negative = (p < 0);
        if (negative) p = -p;

        if (p == 1) {
            Value n_val = Value(q);
            if (q == 2) return eager_sqrt(base, eps);
            Value internal_eps = eps / 1000;
            return eager_nth_root(base, n_val, internal_eps);
        }

        Value internal_eps = (p == 0) ? eps : eps / Value(p * 1000);
        Value log_base = eager_log(base, internal_eps);
        Value p_val = negative ? Value(-p) : Value(p);
        Value p_log = p_val * log_base;
        Value q_val = Value(q);
        Value p_log_div_q = p_log / q_val;
        return eager_exp(p_log_div_q, internal_eps);
    }

    // ============================================================================
    // EAGER DISPATCHERS – entry points for user calls.
    // They choose between float and series paths based on eps and argument.
    // ============================================================================

    inline Value eager_sqrt(const Value& x, const Value& eps) {
        if (is_negative(x)) throw std::domain_error("called sqrt of negative - it's irrational");
        // First try exact integer square root
        if (auto exact = try_exact_nth_root(x, Value(2))) {
            return *exact;
        }
        // Float path removed – always series
        return series_sqrt(x, eps);
    }

    constexpr double EXP_FLOAT_ARG_THRESHOLD = 20.0;

    inline Value eager_exp(const Value& x, const Value& eps) {
        double eps_d = to_double(eps);
        double x_d = std::abs(to_double(x));
        // Float path is fast but loses relative accuracy for large arguments
        if (eps_d >= HYBRID_THRESHOLD && x_d <= EXP_FLOAT_ARG_THRESHOLD) {
            return float_exp(x, eps);
        }
        return series_exp(x, eps);
    }

    inline Value eager_log(const Value& x, const Value& eps) {
        // Float path removed – always series
        return series_log(x, eps);
    }

    inline Value eager_sin(const Value& x, const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? float_sin(x, eps) : series_sin(x, eps);
    }

    inline Value eager_cos(const Value& x, const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? float_cos(x, eps) : series_cos(x, eps);
    }

    inline Value eager_acos(const Value& x, const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? float_acos(x, eps) : series_acos(x, eps);
    }

    inline Value eager_asin(const Value& x, const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? float_asin(x, eps) : series_asin(x, eps);
    }

    inline Value eager_atan(const Value& x, const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? float_atan(x, eps) : series_atan(x, eps);
    }

    inline Value eager_tan(const Value& x, const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? float_tan(x, eps) : series_tan(x, eps);
    }

    inline Value eager_pi(const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? float_pi(eps) : series_pi(eps);
    }

    inline Value eager_e(const Value& eps) {
        // Float path for e gives no benefit – always series.
        return series_e(eps);
    }

} // namespace delta::internal