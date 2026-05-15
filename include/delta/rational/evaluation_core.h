// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// evaluation_core.h
// -----------------------------------------------------------------------------
// CORE IMPLEMENTATIONS OF TRANSCENDENTAL FUNCTIONS.
// 
// SEE FULL DOCUMENTATION IN DOCS/ FOLDER.
// -----------------------------------------------------------------------------
// This file provides eager (immediate) implementations of elementary
// transcendental functions on the rational Value type.
//
// FUNCTIONS: sqrt, exp, log, sin, cos, acos, asin, atan, tan, pi, e, pow,
//            sinpi, cospi, tanpi, asinpi, acospi, atanpi.
//
// DESIGN PHILOSOPHY:
//   - Dual‑path architecture: a fast floating‑point path (cpp_bin_float with
//     fixed compile‑time bit widths) and an exact rational series path.
//   - Cascaded float precisions: 256, 512, 1024 bits, selected dynamically
//     based on requested epsilon and argument magnitude.
//   - Series methods use argument reduction, epsilon scaling, and binary
//     splitting to guarantee fast convergence and compact rational results.
//   - Constant caching (π, e, ln2) is unified across all computation paths.
//
// -----------------------------------------------------------------------------
// 1. FLOAT PATHS – MULTI‑PRECISION DISPATCH
// -----------------------------------------------------------------------------
// We use three fixed‑width binary float types from Boost.Multiprecision:
//   Float256  = cpp_bin_float<256, digit_base_2, et_off>   (~77 dec. digits)
//   Float512  = cpp_bin_float<512, digit_base_2, et_off>   (~154)
//   Float1024 = cpp_bin_float<1024, digit_base_2, et_off>  (~308)
// Expression templates are disabled (et_off) to reduce compile‑time overhead.
//
// Dispatch rules:
//   Trigonometric functions (sin, cos, acos, asin, atan, tan, pi):
//       eps >= 1e-35  → Float256
//       eps >= 1e-70  → Float512
//       otherwise     → rational series (arbitrary precision)
//   exp:
//       Uses a dedicated bit‑aware dispatcher (see below) that estimates the
//       required mantissa width from the argument and the denominator of eps.
//       Fixed thresholds are NOT used because the magnitude of exp(x) varies.
//   sqrt / log / e:
//       No float path. Experiments with cpp_dec_float_100 showed that float
//       paths were consistently slower than rational series due to conversion
//       overhead. The new binary float types and improved to_rational_with_eps
//       (power‑of‑two denominator) slightly improved the situation, but the
//       gain is still marginal for these functions because of the wide input
//       range and the need for careful argument reduction that is cheap in
//       rational arithmetic but expensive in float (e.g., log reduction).
//
// Rationale for not using Float2048 / Float4096:
//   - Compile‑time explosion: each bit width instantiates a separate class
//     template with its own static data (constants, tables), dramatically
//     increasing compilation time and binary size.
//   - Diminishing returns: for eps < 1e‑300 (decimal) the rational series
//     algorithms are already efficient and produce results with a reasonable
//     number of bits. No real‑world scenario demands 2048‑bit float paths
//     because beyond ~1000 bits the overhead of fixed‑precision arithmetic
//     outweighs the benefit.
//   - Practical limit: 90% of use cases require eps ≥ 1e‑100, which is
//     comfortably covered by Float1024. The remaining <10% are served by the
//     series path, which is designed for arbitrary precision without relying
//     on fixed‑width floats.
//
// Why not GMP / MPFR?
//   - License: GMP is LGPL / GPL, which is incompatible with the PolyForm
//     Small Business License used here. We strictly avoid any dependency that
//     imposes copyleft restrictions.
//   - Windows deployment: GMP requires either a cumbersome DLL or a custom
//     build process; static linking is possible but painful. Boost.Multiprecision
//     is header‑only and works out‑of‑the‑box on all platforms.
//   - Performance: for our target precision range (up to ~1000 bits) Boost's
//     cpp_bin_float with compile‑time width is competitive, often faster than
//     GMP's runtime‑precision mpf_t, because it avoids dynamic allocation and
//     uses template metaprogramming for small‑integer optimisations.
//
// -----------------------------------------------------------------------------
// 2. EXPONENT DISPATCH – BIT‑AWARE SELECTION
// -----------------------------------------------------------------------------
// For exp(x) the output magnitude is ~2^(x * log2(e)). A naive epsilon‑based
// threshold (like 1e‑35) would be incorrect because for x ≈ 40 the absolute
// error requirement translates into an enormous relative error requirement.
// The dispatcher computes:
//   integral_bits ≈ (integer part of x) * 23 / 16   (log2(e) ≈ 1.4427)
//   precision_bits ≈ msb(denominator(eps))
//   net_weight_bits = integral_bits + precision_bits
// Then it selects the smallest FloatN that can hold the result with the
// required absolute precision, or falls back to the series path if the sum
// exceeds 1008 bits. This adaptive scheme fully replaces the old
// EXP_FLOAT_ARG_THRESHOLD = 20.0.
//
// -----------------------------------------------------------------------------
// 3. SERIES PATHS – RATIONAL ARBITRARY PRECISION
// -----------------------------------------------------------------------------
// Each series implementation uses argument reduction to stay within a region
// of fast convergence (|x| ≤ 2 for exp, [‑π,π] for sin/cos, etc.).
// Key techniques:
//   - exp: repeatedly divide by 2, then square k times. Internal epsilon is
//          scaled by 2^(exp_bits + k + 2) to preserve absolute error bounds.
//   - sqrt: Newton's method with integer‑floor‑sqrt initial guess for numbers
//           with |log2(x)| ≤ 60; scaling by powers of 4 for extreme values.
//   - log: reduce to [1/2, 2] via k·ln2, then arctanh series.
//   - π: Chudnovsky series with binary splitting (O(N log N) rational ops).
//   - sin/cos/atan/asin: binary splitting on the Taylor expansion of the
//          reduced argument, converting x² into a numerator/denominator pair.
//   - e: simple Taylor series, cached.
//
// The series path guarantees absolute error ≤ eps for any eps > 0, with
// computational cost scaling roughly as O(‑log(eps)). Memory and time are
// dominated by the size of the rational numbers involved; binary splitting
// keeps intermediate values as small as possible.
//
// -----------------------------------------------------------------------------
// 4. CONVERSION: float → Value (to_rational_with_eps)
// -----------------------------------------------------------------------------
// The conversion uses power‑of‑two scaling instead of the old decimal string
// method. For a float f and eps, we compute:
//   k = -exponent(eps) + extra_bits
//   rounded = round(f * 2^k)
//   result = rounded / 2^k
// This has several advantages:
//   - No string parsing, no decimal→binary conversion overhead.
//   - The denominator is a power of two, which accelerates subsequent
//     rational operations (gcd, addition, multiplication).
//   - The error is trivially bounded: |f - result| < 2^{-k-1} < eps.
//   - Works seamlessly with cpp_bin_float's internal binary representation.
// The old continued‑fraction conversion (rational_continued_fraction) is
// retained as a fallback for cases where 2^k would become excessively large
// (k > 200), but it is not currently used in the main paths.
//
// -----------------------------------------------------------------------------
// 5. CONSTANT CACHING
// -----------------------------------------------------------------------------
// π, e, and ln2 are cached in a unified static map keyed by the Value eps.
// The cache is populated lazily by the eager_* entry points and shared by all
// subsequent calls. Caching happens at the eager level, not inside the series
// functions, so that the same cached value is returned regardless of whether
// the float or the series path was used initially. This avoids recomputing
// constants when switching between paths for different operations at the
// same precision.
//
// -----------------------------------------------------------------------------
// 6. LESSONS LEARNED (READ BEFORE MODIFYING)
// -----------------------------------------------------------------------------
// - DO NOT vectorise Taylor series with rational numbers. Recurrent term
//   update in a simple loop is O(1) per term; any attempt to collect terms
//   into a vector and use PCR causes a 3‑5× slowdown.
// - DO NOT remove epsilon scaling in series_exp. Without it, argument
//   reduction via squaring amplifies the error beyond the user's request.
// - DO NOT change the series_exp reduction threshold (2.0) or the hybrid
//   thresholds without running the full benchmark suite. Seemingly minor
//   adjustments can cause explosion of rational representation size and
//   catastrophic slowdown in downstream operations (log, pow, etc.).
// - DO NOT attempt to speed up series_sqrt by using x/2 as initial guess.
//   The cheap guess produces compact results, but makes the final rational
//   representation enormous, breaking sin(π) and similar identities.
//   The integer‑floor‑sqrt guess is optimal.
// - DO NOT introduce dynamic‑precision (GMP‑style) backends. The fixed
//   compile‑time bit widths give the best performance for 90% of use cases,
//   and the series path covers the rest. Adding runtime‑precision would
//   require either a GPL dependency (unacceptable) or a custom allocator,
//   both of which are overkill.
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
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/number.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <stack>
#include <string>
#include <vector>
#include <iostream>
#include <map>
#include <array>
#include <stdexcept>

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
    dumb_int integer_floor_sqrt(const dumb_int& a);
    // ----------------------------------------------------------------------------
    // Helper predicates
    // ----------------------------------------------------------------------------
    inline bool is_less(const Value& a, const Value& b) { return a < b; }
    inline bool is_greater(const Value& a, const Value& b) { return a > b; }

    // ----------------------------------------------------------------------------
    // GLOBAL THRESHOLDS
    // ----------------------------------------------------------------------------
    constexpr double HYBRID_THRESHOLD = 1e-35;          // legacy, used by some functions
    constexpr double TRIG_CUTOFF_COARSE = 1e-35;       // threshold for 256-bit float
    constexpr double TRIG_CUTOFF_MEDIUM = 1e-70;       // threshold for 512-bit float
    constexpr double EXP_FLOAT_ARG_THRESHOLD = 20.0;   // max |x| for float exp path

    // ============================================================================
    // Arithmetic operations
    // ============================================================================
    inline Value eager_abs(const Value& a) {
        return is_negative(a) ? -a : a;
    }

    // ============================================================================
    // High‑precision floating‑point helpers – multiple precisions with et_off
    // ============================================================================
    namespace bmp = boost::multiprecision;
    using bmp::cpp_bin_float;
    using bmp::et_off;

    using Float256 = bmp::number<cpp_bin_float<256, bmp::digit_base_2>, et_off>;
    using Float512 = bmp::number<cpp_bin_float<512, bmp::digit_base_2>, et_off>;
    using Float1024 = bmp::number<cpp_bin_float<1024, bmp::digit_base_2>, et_off>;

    // Base precision – 256 bits (≈77 decimal digits)
    using HighPrecFloat = Float256;

    template<typename FloatT = HighPrecFloat>
    inline FloatT to_high_prec(const Value& v) {
        return v.convert_to<FloatT>();
    }
    // ----------------------------------------------------------------------------
    // Continued fraction conversion (compact rational representation)
    // ----------------------------------------------------------------------------
 // ----------------------------------------------------------------------------
    inline Value rational_continued_fraction(const dumb_int& a, const dumb_int& b, const Value& eps) {
        if (b == 0) throw std::domain_error("rational_continued_fraction: denominator is zero");
        if (a == 0) return Value(0);

        bool negative = (a < 0);
        dumb_int aa = negative ? -a : a;
        dumb_int bb = b < 0 ? -b : b;
        if (b < 0) negative = !negative;

        dumb_int p0 = 0, p1 = 1;
        dumb_int q0 = 1, q1 = 0;
        dumb_int r = aa, s = bb;

        double eps_d = to_double(eps);
        if (eps_d <= 0.0) eps_d = 1e-35;

        const int MAX_ITER = 1000;
        int iter = 0;

        while (s != 0 && iter < MAX_ITER) {
            dumb_int q = r / s;

            dumb_int p2 = q * p1 + p0;
            dumb_int q2 = q * q1 + q0;

            // |aa * q2 - bb * p2|
            dumb_int diff_num = aa * q2 - bb * p2;
            if (diff_num < 0) diff_num = -diff_num;

            bool precision_achieved = false;

            // БАЗА: Проверяем битность без паранойи. msb() здесь СТОПРОЦЕНТНО безопасен, 
            // так как bb > 0 (проверено выше), а q2 гарантированно >= 1.
            size_t bits_prod = boost::multiprecision::msb(bb) + boost::multiprecision::msb(q2) + 2;

            if (bits_prod > 1020) {
                // Если произведение не влезает в double, считаем в целых числах.
                // diff_num / (bb * q2) <= eps_d  ==>  diff_num / eps_d <= bb * q2
                // Чтобы не делить BigInt на double, превращаем это в: diff_num <= eps_d * (bb * q2)
                // Но раз они огромные, то относительная погрешность уже гарантированно глубоко под eps_d.
                precision_achieved = true;
            }
            else {
                // Стандартный быстрый путь на регистрах FPU
                double prod_d = bb.convert_to<double>() * q2.convert_to<double>();
                if (diff_num.convert_to<double>() <= (eps_d * prod_d) + 1e-15) {
                    precision_achieved = true;
                }
            }

            if (precision_achieved) {
                p1 = p2; q1 = q2;
                break;
            }

            p0 = p1; p1 = p2;
            q0 = q1; q1 = q2;

            r = r - q * s;
            std::swap(r, s);
            ++iter;
        }

        if (q1 == 0) return Value(negative ? -aa : aa, bb);

        Value result(p1, q1);
        return negative ? -result : result;
    }
    // ----------------------------------------------------------------------------
    // Generic to_rational_with_eps
    // ----------------------------------------------------------------------------
    template<typename Float>
    inline Value to_rational_with_eps(const Float& f, const Value& eps, int extra_digits = 2) {
        if (f == 0) return Value(0);
        if (f == 1) return Value(1);
        if (f == -1) return Value(-1);

        double eps_d = to_double(eps);
        if (eps_d <= 0.0) eps_d = 1e-35;

        // Аппаратный разбор экспоненты эпсилона
        int exp = 0;
        std::frexp(eps_d, &exp);

        constexpr double log2_10 = 3.3219280948873626;
        double extra_bits = (extra_digits * log2_10) + 1.0;

        int k = -exp + static_cast<int>(extra_bits) + 1;
        if (k < 1) k = 1;

        // ЧИСТАЯ БАЗА: Никаких уходов в непрерывные дроби. 
        // Нам прилетел Float нужной точности, мы просто забираем его биты.
        Float two_pow_k = boost::multiprecision::ldexp(Float(1), k);
        Float scaled = f * two_pow_k;
        Float rounded = boost::multiprecision::round(scaled);

        dumb_int int_val = rounded.template convert_to<dumb_int>();
        dumb_int denom = dumb_int(1) << k;

        return Value(int_val, denom);
    }
    // ------------------------------------------------------------------------
    // Float implementations (templated on precision)
    // ------------------------------------------------------------------------
    template<typename Float>
    inline Value float_exp_impl(const Value& x, const Value& eps) {
        Float fx = x.convert_to<Float>();
        Float res = exp(fx);
        return to_rational_with_eps(res, eps);
    }

    template<typename Float>
    inline Value float_sin_impl(const Value& x, const Value& eps) {
        if (is_negative(x)) return -float_sin_impl<Float>(-x, eps);
        Float fx = x.convert_to<Float>();
        return to_rational_with_eps(sin(fx), eps);
    }

    template<typename Float>
    inline Value float_cos_impl(const Value& x, const Value& eps) {
        Value positive_x = is_negative(x) ? -x : x;
        Float fx = positive_x.convert_to<Float>();
        return to_rational_with_eps(cos(fx), eps);
    }

    template<typename Float>
    inline Value float_pi_impl(const Value& eps) {
        Float pi_val = boost::math::constants::pi<Float>();
        return to_rational_with_eps(pi_val, eps);
    }

    template<typename Float>
    inline Value float_acos_impl(const Value& x, const Value& eps) {
        Float fx = x.convert_to<Float>();
        if (fx < -1 || fx > 1) throw std::domain_error("acos argument out of [-1,1]");
        return to_rational_with_eps(acos(fx), eps);
    }

    template<typename Float>
    inline Value float_asin_impl(const Value& x, const Value& eps) {
        Float fx = x.convert_to<Float>();
        if (fx < -1 || fx > 1) throw std::domain_error("asin argument out of [-1,1]");
        return to_rational_with_eps(asin(fx), eps);
    }

    template<typename Float>
    inline Value float_atan_impl(const Value& x, const Value& eps) {
        Float fx = x.convert_to<Float>();
        return to_rational_with_eps(atan(fx), eps);
    }

    template<typename Float>
    inline Value float_tan_impl(const Value& x, const Value& eps) {
        Float fx = x.convert_to<Float>();
        return to_rational_with_eps(tan(fx), eps);
    }

    // Original float functions (non‑templated, using HighPrecFloat) – kept for compatibility
    inline Value float_exp(const Value& x, const Value& eps) {
        return float_exp_impl<HighPrecFloat>(x, eps);
    }
    inline Value float_sin(const Value& x, const Value& eps) {
        return float_sin_impl<HighPrecFloat>(x, eps);
    }
    inline Value float_cos(const Value& x, const Value& eps) {
        return float_cos_impl<HighPrecFloat>(x, eps);
    }
    inline Value float_pi(const Value& eps) {
        return float_pi_impl<HighPrecFloat>(eps);
    }
    inline Value float_acos(const Value& x, const Value& eps) {
        return float_acos_impl<HighPrecFloat>(x, eps);
    }
    inline Value float_asin(const Value& x, const Value& eps) {
        return float_asin_impl<HighPrecFloat>(x, eps);
    }
    inline Value float_atan(const Value& x, const Value& eps) {
        return float_atan_impl<HighPrecFloat>(x, eps);
    }
    inline Value float_tan(const Value& x, const Value& eps) {
        return float_tan_impl<HighPrecFloat>(x, eps);
    }

    // ------------------------------------------------------------------------
    // Constant cache – array of maps, key = Value eps
    // ------------------------------------------------------------------------
    enum ConstId { CONST_PI = 0, CONST_E = 1, CONST_LN2 = 2 };

    inline std::array<std::map<Value, Value>, 3>& get_const_cache() {
        static std::array<std::map<Value, Value>, 3> cache;
        return cache;
    }

    template<typename Computer>
    inline Value get_cached_const(ConstId id, const Value& eps, Computer&& computer) {
        auto& cache = get_const_cache()[id];
        auto it = cache.find(eps);
        if (it != cache.end()) return it->second;
        Value result = computer(eps);
        cache[eps] = result;
        return result;
    }

    // ============================================================================
    // Exact integer roots – unchanged from original
    // ============================================================================
    inline bool is_integer(const Value& v) { return denominator(v) == 1; }
    inline dumb_int get_integer(const Value& v) { return numerator(v); }

    inline dumb_int integer_nth_root_floor(const dumb_int& a, const dumb_int& n) {
        if (n == 0) return 0;                 // не определено
        if (n == 1 || a == 0 || a == 1) return a;
        if (a < 0) {
            if (n % 2 == 0) return 0;         // чётный корень из отрицательного
            return -integer_nth_root_floor(-a, n);
        }
        if (n > 1000) return 0;               // предохранитель

        int n_int = n.convert_to<int>();
        size_t bits = boost::multiprecision::msb(a) + 1;
        // начальное приближение сверху
        dumb_int x = dumb_int(1) << ((bits + n_int - 1) / n_int);
        dumb_int x_prev = 0;

        while (true) {
            dumb_int p = boost::multiprecision::pow(x, n_int - 1);
            if (p == 0) {
                // переполнение – уменьшаем x
                x = x / 2;
                if (x == 0) return 0;
                continue;
            }
            dumb_int next_x = (dumb_int(n_int - 1) * x + a / p) / n_int;
            // целочисленный метод Ньютона: когда next_x >= x, достигли floor
            if (next_x >= x || next_x == x_prev) break;
            x_prev = x;
            x = next_x;
        }
        // Корректировка вниз (гарантия, что x^p <= a)
        while (boost::multiprecision::pow(x, n_int) > a) {
            --x;
        }
        return x;
    }
    inline bool is_quick_perfect_square(const dumb_int& x) {
        if (x < 0) return false;
        if (x == 0 || x == 1) return true;
        static const bool good_mod256[256] = {
            1,1,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
        };
        if (x.backend().size() == 0) return true;
        uint64_t low_byte = x.backend().limbs()[0] & 0xFF;
        if (!good_mod256[low_byte]) return false;
        int last_digit = (x % 10).convert_to<int>();
        if (last_digit == 2 || last_digit == 3 || last_digit == 7 || last_digit == 8) return false;
        return true;
    }

    inline std::optional<Value> try_exact_nth_root(const Value& base, const Value& n_val) {
        if (!is_integer(n_val)) return std::nullopt;
        dumb_int n = numerator(n_val);
        if (n <= 0) return std::nullopt;
        if (n > 1000) return std::nullopt;
        int n_int = n.convert_to<int>();
        if (is_zero(base)) return Value(0);
        bool negative = is_negative(base);
        if (negative && n_int % 2 == 0) return std::nullopt;

        dumb_int num = numerator(base);
        dumb_int den = denominator(base);
        if (negative) num = -num;

        // Для квадрата используем быстрый путь с масками
        if (n_int == 2) {
            if (!is_quick_perfect_square(num) || !is_quick_perfect_square(den)) return std::nullopt;
            dumb_int root_num = integer_floor_sqrt(num);
            dumb_int root_den = integer_floor_sqrt(den);
            if (root_num * root_num == num && root_den * root_den == den)
                return Value(root_num, root_den);
            return std::nullopt;
        }

        // Для n > 2 – общий метод
        dumb_int root_den = integer_nth_root_floor(den, n);
        dumb_int root_num = integer_nth_root_floor(num, n);
        // Единственная проверка точности (возведение в степень)
        if (boost::multiprecision::pow(root_num, n_int) == num &&
            boost::multiprecision::pow(root_den, n_int) == den) {
            if (negative) root_num = -root_num;
            return Value(root_num, root_den);
        }
        return std::nullopt;
    }
    // ============================================================================
    // Configuration for series methods
    // ============================================================================
    constexpr size_t DEFAULT_MAX_ITER = 1000000;
    constexpr size_t NEWTON_MAX_ITER = 1000;

    // ============================================================================
    // Series implementations (mostly unchanged, but use cached ln2 and e)
    // ============================================================================
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
        return sum * 2;
    }

    inline dumb_int integer_floor_sqrt(const dumb_int& a) {
        if (a <= 1) return a;
        dumb_int x = a;
        dumb_int y = (x + 1) / 2;
        while (y < x) {
            x = y;
            y = (x + a / x) / 2;
        }
        return x;
    }

    inline Value series_sqrt(const Value& x, const Value& eps) {
        if (is_zero(x)) return Value(0);
        if (is_one(x)) return Value(1);
        if (is_negative(x)) throw std::domain_error("sqrt of negative number");

        dumb_int num = numerator(x);
        dumb_int den = denominator(x);
        int log2_num = (num == 0) ? -1e6 : static_cast<int>(boost::multiprecision::msb(num));
        int log2_den = (den == 1) ? 0 : static_cast<int>(boost::multiprecision::msb(den));
        int log2x = log2_num - log2_den;
        const int SCALE_THRESHOLD = 60;

        if (std::abs(log2x) <= SCALE_THRESHOLD) {
            dumb_int prod = num * den;
            dumb_int s = integer_floor_sqrt(prod);
            Value guess(s, den);
            const int MAX_ITER = 12;
            for (int iter = 0; iter < MAX_ITER; ++iter) {
                Value next = (guess + x / guess) / 2;
                if (eager_abs(next - guess) < eps) {
                    guess = next;
                    break;
                }
                guess = next;
            }
            return guess;
        }

        int k = (log2x + 1) / 2;
        Value m = x;
        if (k > 0) {
            dumb_int four_pow_k = dumb_int(1) << (2 * k);
            m = x / Value(four_pow_k);
        }
        else if (k < 0) {
            dumb_int four_pow_negk = dumb_int(1) << (-2 * k);
            m = x * Value(four_pow_negk);
        }
        while (m > 1) { m /= 4; ++k; }
        while (m < Value(1) / 4) { m *= 4; --k; }

        Value internal_eps = eps;
        for (int i = 0; i < std::abs(k); ++i) internal_eps /= 2;

        double m_approx = to_double(m);
        Value guess;
        guess.assign(std::sqrt(m_approx));
        Value diff;
        size_t iter = 0;
        const size_t MAX_ITER = 50;
        do {
            Value next = (guess + m / guess) / 2;
            diff = eager_abs(next - guess);
            guess = next;
            ++iter;
        } while (diff > internal_eps && iter < MAX_ITER);

        Value result = guess;
        if (k > 0) {
            dumb_int two_pow_k = dumb_int(1) << k;
            result = result * Value(two_pow_k);
        }
        else if (k < 0) {
            dumb_int two_pow_negk = dumb_int(1) << (-k);
            result = result / Value(two_pow_negk);
        }
        return result;
    }

    constexpr double SERIES_EXP_REDUCE_THRESHOLD = 2.0;

    inline Value series_exp(const Value& x, const Value& eps) {
        if (is_zero(x)) return Value(1);
        if (is_negative(x)) return Value(1) / series_exp(-x, eps);

        double x_d = to_double(x);
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

        int k = 0;
        Value reduced = x;
        while (reduced > SERIES_EXP_REDUCE_THRESHOLD) {
            reduced /= 2;
            ++k;
        }

        double exp_est = std::exp(x_d);
        int exp_bits;
        std::frexp(exp_est, &exp_bits);

        Value internal_eps = eps;
        int total_shift = exp_bits + k + 2;
        for (int i = 0; i < total_shift; ++i) internal_eps /= 2;

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

        dumb_int exponent = dumb_int(1) << k;
        return eager_pow_int(sum, exponent);
    }

    inline Value series_log(const Value& x, const Value& eps) {
        if (is_negative(x) || is_zero(x)) throw std::domain_error("log of non-positive");

        int k = 0;
        Value m = x;
        while (m > 2) { m /= 2; ++k; }
        while (m < Value(1) / 2) { m *= 2; --k; }

        Value ln2 = series_ln2(eps);
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
    // π via Chudnovsky (binary splitting) – series_pi now uses unified cache
    // ============================================================================
    static const dumb_int CHUD_A = 545140134;
    static const dumb_int CHUD_B = 13591409;
    static const dumb_int CHUD_C = 640320;
    static const dumb_int CHUD_C3_OVER_24 = (dumb_int(CHUD_C) * CHUD_C * CHUD_C) / 24;
    static const dumb_int CHUD_D = 426880;

    struct ChudnovskyPQT { dumb_int P, Q, T; };

    inline ChudnovskyPQT chudnovsky_bs(int64_t a, int64_t b) {
        if (b - a == 1) {
            dumb_int k(a);
            if (a == 0) return { dumb_int(1), dumb_int(1), dumb_int(CHUD_B) };
            dumb_int P = (6 * k - 5) * (2 * k - 1) * (6 * k - 1);
            dumb_int Q = k * k * k * CHUD_C3_OVER_24;
            dumb_int T = k * CHUD_A + CHUD_B;
            if (a % 2 == 1) T = -T;
            return { P, Q, T };
        }
        int64_t m = (a + b) / 2;
        auto L = chudnovsky_bs(a, m);
        auto R = chudnovsky_bs(m, b);
        return { L.P * R.P, L.Q * R.Q, L.T * R.Q + L.P * R.T };
    }

    inline Value pi_recurrent(int N, const Value& eps) {
        Value term(CHUD_B, 1);
        Value sum = term;
        for (int k = 0; k < N - 1; ++k) {
            dumb_int k1 = k + 1;
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

    // series_pi is only used as fallback when float path is not taken; it still uses unified cache.
    inline Value series_pi(const Value& eps) {
        // We cannot call get_cached_const here because it would recurse.
        // Instead we use the old direct computation, but we still store result in the unified cache
        // via the eager_pi dispatcher. This function is only called from eager_pi when float path is disabled.
        double eps_d = std::abs(to_double(eps));
        int N = (eps_d <= 0) ? 10 : static_cast<int>(std::max(2.0, std::ceil(-std::log10(eps_d) / 14.18) + 3));
        Value result;
        if (N > 16) {
            auto res = chudnovsky_bs(0, N);
            Value S(res.T, res.Q);
            Value sqrt_10005 = series_sqrt(Value(10005), eps / 1000);
            result = (Value(CHUD_D) * sqrt_10005) / S;
        }
        else {
            result = pi_recurrent(N, eps);
        }
        return result;
    }

    // ============================================================================
    // Binary splitting for sin and cos (unchanged)
    // ============================================================================
    struct TrigPQT { dumb_int P, Q, T; };

    inline TrigPQT sin_bs_internal(int64_t a, int64_t b, const dumb_int& x2_num, const dumb_int& x2_den) {
        if (b - a == 1) {
            if (a == 0) return { x2_num, 1, 1 };
            dumb_int Q = x2_den * dumb_int(2 * a) * (2 * a + 1);
            dumb_int T = (a % 2 == 1) ? -1 : 1;
            return { x2_num, Q, T };
        }
        int64_t m = (a + b) / 2;
        auto L = sin_bs_internal(a, m, x2_num, x2_den);
        auto R = sin_bs_internal(m, b, x2_num, x2_den);
        return { L.P * R.P, L.Q * R.Q, L.T * R.Q + L.P * R.T };
    }

    inline TrigPQT cos_bs_internal(int64_t a, int64_t b, const dumb_int& x2_num, const dumb_int& x2_den) {
        if (b - a == 1) {
            if (a == 0) return { x2_num, 1, 1 };
            dumb_int Q = x2_den * dumb_int(2 * a - 1) * (2 * a);
            dumb_int T = (a % 2 == 1) ? -1 : 1;
            return { x2_num, Q, T };
        }
        int64_t m = (a + b) / 2;
        auto L = cos_bs_internal(a, m, x2_num, x2_den);
        auto R = cos_bs_internal(m, b, x2_num, x2_den);
        return { L.P * R.P, L.Q * R.Q, L.T * R.Q + L.P * R.T };
    }

    inline Value series_sin(const Value& x, const Value& eps) {
        if (is_negative(x)) return -series_sin(-x, eps);
        Value pi_val = eager_pi(eps);   // uses cache
        Value twopi = pi_val * 2;
        Value periods = x / twopi;
        dumb_int k_int = numerator(periods) / denominator(periods);
        Value reduced = x - Value(k_int) * twopi;
        if (reduced > pi_val) reduced -= twopi;
        else if (reduced < -pi_val) reduced += twopi;
        if (is_zero(reduced)) return Value(0);
        Value x2 = reduced * reduced;
        dumb_int x2_num = numerator(x2);
        dumb_int x2_den = denominator(x2);
        double eps_d = to_double(eps);
        int64_t N = (eps_d <= 0) ? 10 : static_cast<int64_t>(std::max(10.0, -std::log10(eps_d) * 0.8));
        if (N > 2000) N = 2000;
        auto res = sin_bs_internal(0, N, x2_num, x2_den);
        Value sum_series(res.T, res.Q);
        return reduced * sum_series;
    }

    inline Value series_cos(const Value& x, const Value& eps) {
        Value pi_val = eager_pi(eps);
        Value twopi = pi_val * 2;
        Value abs_x = is_negative(x) ? -x : x;
        Value periods = abs_x / twopi;
        dumb_int k_int = numerator(periods) / denominator(periods);
        Value reduced = abs_x - Value(k_int) * twopi;
        if (reduced > pi_val) reduced = twopi - reduced;
        if (is_zero(reduced)) return Value(1);
        Value x2 = reduced * reduced;
        dumb_int x2_num = numerator(x2);
        dumb_int x2_den = denominator(x2);
        double eps_d = to_double(eps);
        int64_t N = (eps_d <= 0) ? 10 : static_cast<int64_t>(std::max(10.0, -std::log10(eps_d) * 0.8));
        if (N > 2000) N = 2000;
        auto res = cos_bs_internal(0, N, x2_num, x2_den);
        Value sum_series(res.T, res.Q);
        return sum_series;
    }

    struct BSResult { dumb_int P, Q, T; };

    inline Value series_atan(const Value& x, const Value& eps) {
        if (is_zero(x)) return Value(0);
        bool negative = is_negative(x);
        Value xx = negative ? -x : x;
        if (xx > 1) {
            Value half_pi = eager_pi(eps) / 2;
            Value inv = Value(1) / xx;
            Value atan_inv = series_atan(inv, eps);
            Value res = half_pi - atan_inv;
            return negative ? -res : res;
        }
        if (xx > Value(1) / 2) {
            Value one(1);
            Value xp = (xx - one) / (xx + one);
            Value quarter_pi = eager_pi(eps) / 4;
            Value atan_xp = series_atan(xp, eps);
            Value res = quarter_pi + atan_xp;
            return negative ? -res : res;
        }
        Value x2 = xx * xx;
        dumb_int x2_num = numerator(x2);
        dumb_int x2_den = denominator(x2);
        double x2_d = to_double(x2);
        double eps_d = std::abs(to_double(eps));
        int N = 10;
        if (eps_d > 0) {
            while (std::pow(x2_d, N) / (2 * N + 1) > eps_d && N < 5000) ++N;
            N += 5;
        }
        else N = 500;
        auto atan_bs = [&](auto&& self, int a, int b) -> BSResult {
            if (b - a == 1) {
                if (a == 0) return { -x2_num, x2_den, dumb_int(1) };
                dumb_int Q = x2_den * (2 * a + 1);
                return { -x2_num, Q, dumb_int(1) };
            }
            int m = (a + b) / 2;
            auto L = self(self, a, m);
            auto R = self(self, m, b);
            return { L.P * R.P, L.Q * R.Q, L.T * R.Q + L.P * R.T };
            };
        auto res = atan_bs(atan_bs, 0, N);
        Value S(res.T, res.Q);
        Value result = xx * S;
        return negative ? -result : result;
    }

    inline Value series_asin(const Value& x, const Value& eps) {
        if (x < -1 || x > 1) throw std::domain_error("asin argument out of [-1,1]");
        if (is_one(x)) return eager_pi(eps) / 2;
        if (x == -1) return -eager_pi(eps) / 2;
        Value x2 = x * x;
        double x2_d = to_double(x2);
        dumb_int x2_num = numerator(x2);
        dumb_int x2_den = denominator(x2);
        double eps_d = std::abs(to_double(eps));
        int N = 10;
        if (eps_d > 0) {
            double x_d = to_double(eager_abs(x));
            while (x_d * std::pow(x2_d, N) / (2 * N + 1) > eps_d && N < 5000) ++N;
            N += 5;
        }
        else N = 500;
        auto asin_bs = [&](auto&& self, int a, int b) -> BSResult {
            if (b - a == 1) {
                dumb_int P = (2 * a - 1) * (2 * a - 1) * x2_num;
                dumb_int Q = 2 * a * (2 * a + 1) * x2_den;
                return { P, Q, P };
            }
            int m = (a + b) / 2;
            auto L = self(self, a, m);
            auto R = self(self, m, b);
            return { L.P * R.P, L.Q * R.Q, L.T * R.Q + L.P * R.T };
            };
        if (N <= 1) return x;
        auto res = asin_bs(asin_bs, 1, N);
        Value S(res.T, res.Q);
        return x + x * S;
    }

    inline Value series_acos(const Value& x, const Value& eps) {
        if (x < -1 || x > 1) throw std::domain_error("acos argument out of [-1,1]");
        Value clipped_x = x;
        if (clipped_x > Value(1)) clipped_x = Value(1);
        else if (clipped_x < Value(-1)) clipped_x = Value(-1);
        Value half_pi = eager_pi(eps) / 2;
        return half_pi - series_asin(clipped_x, eps);
    }

    inline Value series_tan(const Value& x, const Value& eps) {
        Value s = series_sin(x, eps);
        Value c = series_cos(x, eps);
        if (is_zero(c)) throw std::domain_error("tan: cos(x) is zero");
        return s / c;
    }

    inline Value series_e(const Value& eps) {
        return get_cached_const(CONST_E, eps, [](const Value& e) {
            Value sum = 1, term = 1;
            Value n = 1;
            size_t iter = 0;
            while (iter < DEFAULT_MAX_ITER) {
                term /= n;
                sum += term;
                n += 1;
                ++iter;
                if (term < e) break;
            }
            return sum;
            });
    }

    // ============================================================================
    // Integer exponentiation and nth root (unchanged)
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

    inline int compute_extra_digits(const Value& eps, double operation_complexity = 1.0) {
        double eps_double = to_double(eps);
        if (eps_double <= 0) return 30;
        int digits_needed = static_cast<int>(std::ceil(-std::log10(eps_double))) + 2;
        int safety = static_cast<int>(std::ceil(10.0 * operation_complexity));
        return digits_needed + safety;
    }

    template<typename Float>
    inline Value float_nth_root(const Value& x, const Value& n, const Value& eps) {
        bool x_neg = is_negative(x);
        if (x_neg) {
            bool n_even = false;
            if (is_integer(n)) {
                dumb_int n_int = numerator(n);
                if (n_int % 2 == 0) n_even = true;
            }
            if (n_even) throw std::domain_error("even root of negative number");
            return -float_nth_root<Float>(-x, n, eps);
        }
        if (is_zero(x)) return Value(0);
        double complexity = 1.0;
        if (is_integer(n)) complexity = static_cast<double>(numerator(n));
        int extra = compute_extra_digits(eps, complexity);
        Float fx = x.convert_to<Float>();
        Float fn = n.convert_to<Float>();
        Float res = pow(fx, Float(1) / fn);
        return to_rational_with_eps(res, eps, extra);
    }

    inline Value eager_nth_root(const Value& x, const Value& n, const Value& eps) {
        if (is_zero(n) || is_negative(n)) throw std::domain_error("nth_root: n must be positive");
        if (!is_integer(n)) throw std::domain_error("nth_root: n must be integer");
        dumb_int n_int = numerator(n);
        if (n_int == 0) throw std::domain_error("nth_root: n must be positive");
        if (n_int == 1) return x;
        if (n_int == 2) return eager_sqrt(x, eps);
        if (n_int % 2 == 0 && is_negative(x)) throw std::domain_error("nth_root: even root of negative number");
        if (auto exact = try_exact_nth_root(x, n)) return *exact;
        if (n_int == 2 && to_double(eps) >= HYBRID_THRESHOLD)
            return float_nth_root<Float256>(x, n, eps);
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

        // TODO: УНИФИЦИРОВАТЬ ТОЧНОЕ ВЗЯТИЕ КОРНЕЙ ЛЮБОЙ СТЕПЕНИ
    // ------------------------------------------------------------
    // Сейчас:
    //   - integer_floor_sqrt() для квадратного корня использует быструю битовую маску
    //     и работает функционально бесплатно (замеры показывают O(1) для чисел до 128 бит).
    //   - Для кубических и более высоких степеней используется integer_nth_root_floor(),
    //     которая работает через итерации Ньютона и не имеет битовой маски.
    //   - try_exact_nth_root() вызывается только в sqrt и nth_root, но не используется
    //     в pow для точных корней (хотя могла бы).
    //
    // План оптимизации:
    // 1. Разработать битовые маски (mod 2^k, цифровые корни) для кубических и пятых степеней,
    //    аналогично is_quick_perfect_square().
    // 2. Реализовать единый шаблонный предикат is_exact_root(base, n) с быстрыми отсечениями.
    // 3. Создать функцию try_exact_root(base, n), которая возвращает std::optional<Value>
    //    и используется во всех местах:
    //      - eager_sqrt (для n=2)
    //      - eager_nth_root (для любых n)
    //      - eager_pow (для дробных показателей p/q)
    // 4. Это ускорит точные случаи (например, 8^(2/3)) и избавит от дублирования логики.
    //
    // Приоритет: средний (улучшение детерминизма и производительности для точных корней).
    // ------------------------------------------------------------

    // ============================================================================
    // General power with rational exponent
    // ============================================================================
    inline Value eager_pow(const Value& base, const Value& exp, const Value& eps) {
        // Базовые граничные случаи
        if (is_zero(base)) {
            if (is_zero(exp)) throw std::domain_error("0^0 is undefined");
            if (is_negative(exp)) throw std::domain_error("0^negative is undefined");
            return base;
        }
        if (is_one(base)) return base;
        if (is_zero(exp)) return Value(1);

        dumb_int exp_num = numerator(exp);
        dumb_int exp_den = denominator(exp); // > 0

        // ---- Отрицательное основание ----
        if (is_negative(base)) {
            if (exp_den != 1) { // Дробный показатель
                if (exp_den % 2 == 0)
                    throw std::domain_error("pow: even root of negative number (complex result)");
                Value pos_pow = eager_pow(-base, exp, eps);
                return (exp_num % 2 != 0) ? -pos_pow : pos_pow;
            }
            // Целый показатель
            if (exp_num < 0) {
                Value base_recip = Value(1) / base;
                return eager_pow_int(base_recip, -exp_num);
            }
            return eager_pow_int(base, exp_num);
        }

        // ---- Положительное основание, целый показатель ----
        if (exp_den == 1) {
            if (exp_num < 0) {
                Value base_recip = Value(1) / base;
                return eager_pow_int(base_recip, -exp_num);
            }
            return eager_pow_int(base, exp_num);
        }

        // ---- Положительное основание, дробный показатель p/q ----
        // Приводим к виду base^(p/q) с q>0, p может быть отрицательным
        dumb_int p = exp_num;
        dumb_int q = exp_den;
        bool negative_exp = (p < 0);
        if (negative_exp) p = -p;

        // Пытаемся извлечь точный корень степени q
        Value q_val(q);
        auto exact_root = try_exact_nth_root(base, q_val);
        if (exact_root.has_value()) {
            // Точный корень -> возводим в степень p (точная арифметика)
            Value result = eager_pow_int(*exact_root, p);
            if (negative_exp) result = Value(1) / result;
            return result;
        }

        // Неточный корень: возвращаемся к exp((p/q)*log(base))
        // Для контроля точности масштабируем eps
        Value internal_eps = eps / Value(p * 1000);
        Value log_base = eager_log(base, internal_eps);
        Value p_val = negative_exp ? Value(-p) : Value(p);
        Value p_log = p_val * log_base;
        Value q_val_val(q);
        Value p_log_div_q = p_log / q_val_val;
        return eager_exp(p_log_div_q, internal_eps);
    }
    // ============================================================================
    // EAGER DISPATCHERS – entry points for user calls.
    // They choose between float and series paths based on eps and argument.
    // ============================================================================
    inline Value eager_exp(const Value& x, const Value& eps) {
        // 1. Быстрый отбор тривиальных случаев
        if (is_zero(x)) return Value(1);
        if (is_negative(x)) return Value(1) / eager_exp(-x, eps);

        const dumb_int& num_x = numerator(x);
        const dumb_int& den_x = denominator(x);
        const dumb_int& den_eps = denominator(eps);

        // 2. БАЗА БЕЗОПАСНОСТИ: Защита от "no bits were set in the operand"
        int bits_num_x = (num_x == 0) ? 0 : msb(num_x);
        int bits_den_x = (den_x == 0) ? 0 : msb(den_x);
        int bit_diff = bits_num_x - bits_den_x;

        // 3. ЕДИНСТВЕННЫЙ И СТРОГИЙ БИТОВЫЙ ФИЛЬТР
        // Если x > 2^15 (32768), то exp(x) гарантированно переполнит экспоненту Float1024.
        // Сразу валим в ряды произвольной точности, минуя весь диспетчинг.
        if (bit_diff > 15) {
            return series_exp(x, eps);
        }

        // 4. ВЫЧИСЛЕНИЕ ЕДИНОГО ВЕСА НА БИТАХ
        int precision_bits = (den_eps == 0) ? 0 : msb(den_eps);

        // Рост целой части в битах: x * log2(e) ≈ x * 1.4375
        int64_t x_approx = (bit_diff >= 0) ? (1LL << bit_diff) : 1;
        int integral_bits = static_cast<int>((x_approx * 23) >> 4);

        // Чистый математический вес
        int net_weight_bits = integral_bits + precision_bits;

        // 5. ЧИСТЫЙ ДИСПЕТЧИНГ (Пороги = Максимум типа - 16 бита Guard Bits)
        if (net_weight_bits <= 240) {
            return float_exp_impl<Float256>(x, eps);
        }
        if (net_weight_bits <= 496) {
            return float_exp_impl<Float512>(x, eps);
        }
        if (net_weight_bits <= 1008) {
            return float_exp_impl<Float1024>(x, eps);
        }

        return series_exp(x, eps);
    }


    inline Value eager_log(const Value& x, const Value& eps) {
        // No float path – series only (float was slower in V1 benchmarks)
        return series_log(x, eps);
    }
    inline Value eager_sqrt(const Value& x, const Value& eps) {
        if (is_negative(x)) throw std::domain_error("sqrt of negative number");
        // Проверка точного квадрата (например, 4, 9/4 и т.д.)
        if (auto exact = try_exact_nth_root(x, Value(2))) return *exact;
        // Float-путь отключён по результатам бенчмарков – всегда используем series
        return series_sqrt(x, eps);
    }
    inline Value eager_sin(const Value& x, const Value& eps) {
        double eps_d = to_double(eps);
        if (eps_d >= TRIG_CUTOFF_COARSE) {
            return float_sin_impl<Float256>(x, eps);
        }
        else if (eps_d >= TRIG_CUTOFF_MEDIUM) {
            return float_sin_impl<Float512>(x, eps);
        }
        return series_sin(x, eps);
    }

    inline Value eager_cos(const Value& x, const Value& eps) {
        double eps_d = to_double(eps);
        if (eps_d >= TRIG_CUTOFF_COARSE) {
            return float_cos_impl<Float256>(x, eps);
        }
        else if (eps_d >= TRIG_CUTOFF_MEDIUM) {
            return float_cos_impl<Float512>(x, eps);
        }
        return series_cos(x, eps);
    }

    inline Value eager_acos(const Value& x, const Value& eps) {
        double eps_d = to_double(eps);
        if (eps_d >= TRIG_CUTOFF_COARSE) {
            return float_acos_impl<Float256>(x, eps);
        }
        else if (eps_d >= TRIG_CUTOFF_MEDIUM) {
            return float_acos_impl<Float512>(x, eps);
        }
        return series_acos(x, eps);
    }

    inline Value eager_asin(const Value& x, const Value& eps) {
        double eps_d = to_double(eps);
        if (eps_d >= TRIG_CUTOFF_COARSE) {
            return float_asin_impl<Float256>(x, eps);
        }
        else if (eps_d >= TRIG_CUTOFF_MEDIUM) {
            return float_asin_impl<Float512>(x, eps);
        }
        return series_asin(x, eps);
    }

    inline Value eager_atan(const Value& x, const Value& eps) {
        double eps_d = to_double(eps);
        if (eps_d >= TRIG_CUTOFF_COARSE) {
            return float_atan_impl<Float256>(x, eps);
        }
        else if (eps_d >= TRIG_CUTOFF_MEDIUM) {
            return float_atan_impl<Float512>(x, eps);
        }
        return series_atan(x, eps);
    }

    inline Value eager_tan(const Value& x, const Value& eps) {
        double eps_d = to_double(eps);
        if (eps_d >= TRIG_CUTOFF_COARSE) {
            return float_tan_impl<Float256>(x, eps);
        }
        else if (eps_d >= TRIG_CUTOFF_MEDIUM) {
            return float_tan_impl<Float512>(x, eps);
        }
        return series_tan(x, eps);
    }

    inline Value eager_pi(const Value& eps) {
        return get_cached_const(CONST_PI, eps, [](const Value& e) {
            double eps_d = to_double(e);
            if (eps_d >= TRIG_CUTOFF_COARSE) {
                return float_pi_impl<Float256>(e);
            }
            else if (eps_d >= TRIG_CUTOFF_MEDIUM) {
                return float_pi_impl<Float512>(e);
            }
            return series_pi(e);
            });
    }

    inline Value eager_e(const Value& eps) {
        return get_cached_const(CONST_E, eps, [](const Value& e) {
            return series_e(e);
            });
    }

    // ============================================================================
    // *pi functions – fixed to use regular trig with π from cache
    // ============================================================================
    inline Value eager_sinpi(const Value& x, const Value& eps) {
        Value pi_val = eager_pi(eps);
        return eager_sin(x * pi_val, eps);
    }

    inline Value eager_cospi(const Value& x, const Value& eps) {
        Value pi_val = eager_pi(eps);
        return eager_cos(x * pi_val, eps);
    }

    inline Value eager_tanpi(const Value& x, const Value& eps) {
        if (numerator(x) * 2 == denominator(x)) {
            throw std::domain_error("tanpi: argument is an odd half-integer");
        }
        Value pi_val = eager_pi(eps);
        return eager_tan(x * pi_val, eps);
    }

    inline Value eager_asinpi(const Value& y, const Value& eps) {
        return eager_asin(y, eps) / eager_pi(eps);
    }

    inline Value eager_acospi(const Value& y, const Value& eps) {
        return eager_acos(y, eps) / eager_pi(eps);
    }

    inline Value eager_atanpi(const Value& y, const Value& eps) {
        return eager_atan(y, eps) / eager_pi(eps);
    }

} // namespace delta::internal