#pragma once

#include "expression_root.h"
#include "storage.h"
#include "utils.h"

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>
#include <stack>


namespace delta::internal {

    // ============================================================================
    // Forward declarations for eager arithmetic (defined later)
    // ============================================================================
    Value eager_add(const Value& a, const Value& b);
    Value eager_sub(const Value& a, const Value& b);
    Value eager_mul(const Value& a, const Value& b);
    Value eager_div(const Value& a, const Value& b);
    Value eager_neg(const Value& a);
    Value eager_abs(const Value& a);

    // ============================================================================
    // Forward declarations for slow transcendental functions
    // (defined after fast path and helper functions)
    // ============================================================================
    Value slow_sqrt(const Value& x, const Value& eps);
    Value slow_exp(const Value& x, const Value& eps);
    Value slow_log(const Value& x, const Value& eps);
    Value slow_sin(const Value& x, const Value& eps);
    Value slow_cos(const Value& x, const Value& eps);
    Value slow_acos(const Value& x, const Value& eps);
    Value slow_pi(const Value& eps);
    Value slow_e(const Value& eps);
    Value slow_ln2(const Value& eps);   // used internally

    // ============================================================================
    // Helper functions for Value
    // ============================================================================

    inline bool is_zero(const Value& v) {
        if (const auto* s = std::get_if<SmallStorage>(&v)) {
            return s->num == 0;
        }
        const auto& b = std::get<BigStorage>(v);
        return b.num() == 0;
    }

    inline bool is_one(const Value& v) {
        if (const auto* s = std::get_if<SmallStorage>(&v)) {
            SmallStorage norm = *s;
            norm.normalize();
            return norm.num == 1 && norm.den == 1;
        }
        const auto& b = std::get<BigStorage>(v);
        return b.num() == 1 && b.den() == 1;
    }

    inline bool is_positive(const Value& v) {
        if (const auto* s = std::get_if<SmallStorage>(&v)) {
            return s->num > 0;
        }
        const auto& b = std::get<BigStorage>(v);
        return b.num() > 0;
    }

    inline bool is_negative(const Value& v) {
        if (const auto* s = std::get_if<SmallStorage>(&v)) {
            return s->num < 0;
        }
        const auto& b = std::get<BigStorage>(v);
        return b.num() < 0;
    }

    inline bool is_less(const Value& a, const Value& b) {
        return a < b;  // uses operator< from storage.h
    }

    inline bool is_greater(const Value& a, const Value& b) {
        return a > b;
    }

    // Approximate Value as double (for hybrid threshold)
    inline double to_double(const Value& v) {
        if (const auto* s = std::get_if<SmallStorage>(&v)) {
            SmallStorage norm = *s;
            norm.normalize();
            return static_cast<double>(norm.num) / static_cast<double>(norm.den);
        }
        const auto& b = std::get<BigStorage>(v);
        boost::multiprecision::cpp_dec_float_100 f = static_cast<boost::multiprecision::cpp_dec_float_100>(b.num());
        f /= static_cast<boost::multiprecision::cpp_dec_float_100>(b.den());
        return f.convert_to<double>();
    }

    constexpr double HYBRID_THRESHOLD = 1e-50;   // Use fast path if eps >= 1e-50

    // ============================================================================
    // Eager arithmetic for Value (exact rational, with overflow handling)
    // ============================================================================

    inline Value eager_add(const Value& a, const Value& b) {
        // Both SmallStorage?
        if (const auto* sa = std::get_if<SmallStorage>(&a)) {
            if (const auto* sb = std::get_if<SmallStorage>(&b)) {
                SmallStorage sa_norm = *sa;
                SmallStorage sb_norm = *sb;
                sa_norm.normalize();
                sb_norm.normalize();

                // Compute common denominator
                absl::uint128 den = sa_norm.den;
                absl::uint128 den2 = sb_norm.den;
                if (den == den2) {
                    // Check overflow for numerator
                    absl::int128 num = sa_norm.num + sb_norm.num;
                    if (!would_overflow_add(sa_norm.num, sb_norm.num)) {
                        return SmallStorage(num, den);
                    }
                    // fallback to BigStorage
                }
                // Use cpp_int to avoid overflow
                boost::multiprecision::cpp_int num = to_cpp_int(sa_norm.num) * to_cpp_int(den2) +
                    to_cpp_int(sb_norm.num) * to_cpp_int(den);
                boost::multiprecision::cpp_int denom = to_cpp_int(den) * to_cpp_int(den2);
                // Reduce
                boost::multiprecision::cpp_int g = boost::multiprecision::gcd(num, denom);
                num /= g;
                denom /= g;
                // If numerator and denominator fit in 128 bits, store as SmallStorage
                if (num <= to_cpp_int((std::numeric_limits<absl::int128>::max)()) &&
                    denom <= to_cpp_int((std::numeric_limits<absl::uint128>::max)())) {
                    return SmallStorage(
                        int128_from_string(num.str()),
                        uint128_from_string(denom.str())
                    );
                }
                return BigStorage(num, denom);
            }
        }
        // If at least one is BigStorage, convert both to BigStorage and add
        boost::multiprecision::cpp_int num1, den1, num2, den2;
        if (const auto* sa = std::get_if<SmallStorage>(&a)) {
            SmallStorage s = *sa;
            s.normalize();
            num1 = to_cpp_int(s.num);
            den1 = to_cpp_int(s.den);
        }
        else {
            const auto& b = std::get<BigStorage>(a);
            num1 = b.num();
            den1 = b.den();
        }
        if (const auto* sb = std::get_if<SmallStorage>(&b)) {
            SmallStorage s = *sb;
            s.normalize();
            num2 = to_cpp_int(s.num);
            den2 = to_cpp_int(s.den);
        }
        else {
            const auto& big_b = std::get<BigStorage>(b);
            num2 = big_b.num();
            den2 = big_b.den();
        }
        boost::multiprecision::cpp_int num = num1 * den2 + num2 * den1;
        boost::multiprecision::cpp_int den = den1 * den2;
        boost::multiprecision::cpp_int g = boost::multiprecision::gcd(num, den);
        num /= g;
        den /= g;
        return BigStorage(num, den);
    }

    inline Value eager_sub(const Value& a, const Value& b) {
        return eager_add(a, eager_neg(b));
    }

    inline Value eager_mul(const Value& a, const Value& b) {
        // Both SmallStorage?
        if (const auto* sa = std::get_if<SmallStorage>(&a)) {
            if (const auto* sb = std::get_if<SmallStorage>(&b)) {
                SmallStorage sa_norm = *sa;
                SmallStorage sb_norm = *sb;
                sa_norm.normalize();
                sb_norm.normalize();

                // Check overflow
                if (!would_overflow_mul(sa_norm.num, sb_norm.num) &&
                    sa_norm.den <= (std::numeric_limits<absl::uint128>::max)() / sb_norm.den) {
                    absl::int128 num = sa_norm.num * sb_norm.num;
                    absl::uint128 den = sa_norm.den * sb_norm.den;
                    // Reduce (may need normalization)
                    SmallStorage result(num, den);
                    result.normalize();
                    return result;
                }
                // Fallback to BigStorage
            }
        }
        // Convert to BigStorage and multiply
        boost::multiprecision::cpp_int num1, den1, num2, den2;
        if (const auto* sa = std::get_if<SmallStorage>(&a)) {
            SmallStorage s = *sa;
            s.normalize();
            num1 = to_cpp_int(s.num);
            den1 = to_cpp_int(s.den);
        }
        else {
            const auto& b = std::get<BigStorage>(a);
            num1 = b.num();
            den1 = b.den();
        }
        if (const auto* sb = std::get_if<SmallStorage>(&b)) {
            SmallStorage s = *sb;
            s.normalize();
            num2 = to_cpp_int(s.num);
            den2 = to_cpp_int(s.den);
        }
        else {
            const auto& big_b = std::get<BigStorage>(b);
            num2 = big_b.num();
            den2 = big_b.den();
        }
        boost::multiprecision::cpp_int num = num1 * num2;
        boost::multiprecision::cpp_int den = den1 * den2;
        boost::multiprecision::cpp_int g = boost::multiprecision::gcd(num, den);
        num /= g;
        den /= g;
        return BigStorage(num, den);
    }

    inline Value eager_div(const Value& a, const Value& b) {
        if (is_zero(b)) {
            throw std::domain_error("Division by zero");
        }
        // a / b = a * (1/b)
        if (const auto* sb = std::get_if<SmallStorage>(&b)) {
            SmallStorage s = *sb;
            s.normalize();
            if (s.num == 0) throw std::domain_error("Division by zero");
            // reciprocal: den/num, with sign handling
            absl::int128 num_recip = static_cast<absl::int128>(s.den);  // явное приведение
            absl::uint128 den_recip = (s.num < 0) ? static_cast<absl::uint128>(-s.num) : static_cast<absl::uint128>(s.num);
            if (s.num < 0) num_recip = -num_recip;
            SmallStorage recip(num_recip, den_recip);
            recip.normalize();
            return eager_mul(a, recip);
        }
        else {
            const auto& bbig = std::get<BigStorage>(b);
            if (bbig.num() == 0) throw std::domain_error("Division by zero");
            boost::multiprecision::cpp_int num_recip = bbig.den();
            boost::multiprecision::cpp_int den_recip = bbig.num();
            if (den_recip < 0) {
                den_recip = -den_recip;
                num_recip = -num_recip;
            }
            BigStorage recip(num_recip, den_recip);
            return eager_mul(a, recip);
        }
    }

    inline Value eager_neg(const Value& a) {
        if (const auto* s = std::get_if<SmallStorage>(&a)) {
            return SmallStorage(-(s->num), s->den);
        }
        const auto& b = std::get<BigStorage>(a);
        return BigStorage(-b.num(), b.den());
    }

    inline Value eager_abs(const Value& a) {
        if (is_negative(a)) return eager_neg(a);
        return a;
    }

    // ============================================================================
    // High-precision floating-point helpers (fast path)
    // ============================================================================

    using HighPrecFloat = boost::multiprecision::cpp_dec_float_100;

    inline HighPrecFloat to_high_prec(const Value& v) {
        if (const auto* s = std::get_if<SmallStorage>(&v)) {
            SmallStorage norm = *s;
            norm.normalize();
            HighPrecFloat num = static_cast<HighPrecFloat>(to_cpp_int(norm.num));
            HighPrecFloat den = static_cast<HighPrecFloat>(to_cpp_int(norm.den));
            return num / den;
        }
        const auto& b = std::get<BigStorage>(v);
        HighPrecFloat num = static_cast<HighPrecFloat>(b.num());
        HighPrecFloat den = static_cast<HighPrecFloat>(b.den());
        return num / den;
    }

    inline Value to_rational_with_eps(const HighPrecFloat& f, const Value& eps) {
        HighPrecFloat eps_f = to_high_prec(eps);
        if (eps_f <= 0) throw std::domain_error("Epsilon must be positive");
        int digits_needed = static_cast<int>(-log10(eps_f.convert_to<double>())) + 2;
        if (digits_needed < 1) digits_needed = 1;
        if (digits_needed > 100) digits_needed = 100;

        std::string s = f.str(digits_needed, std::ios_base::fixed);
        size_t dot = s.find('.');
        std::string integer_part = s.substr(0, dot);
        std::string fractional_part = s.substr(dot + 1);
        boost::multiprecision::cpp_int num(integer_part + fractional_part);
        boost::multiprecision::cpp_int den(1);
        for (size_t i = 0; i < fractional_part.size(); ++i) den *= 10;
        boost::multiprecision::cpp_int g = boost::multiprecision::gcd(num, den);
        num /= g;
        den /= g;
        return BigStorage(num, den);
    }

    // Fast transcendental implementations (via high‑precision float)
    inline Value fast_sqrt(const Value& x, const Value& eps) {
        HighPrecFloat fx = to_high_prec(x);
        if (fx < 0) throw std::domain_error("sqrt of negative number");
        HighPrecFloat res = sqrt(fx);
        return to_rational_with_eps(res, eps);
    }

    inline Value fast_exp(const Value& x, const Value& eps) {
        HighPrecFloat fx = to_high_prec(x);
        HighPrecFloat res = exp(fx);
        return to_rational_with_eps(res, eps);
    }

    inline Value fast_log(const Value& x, const Value& eps) {
        HighPrecFloat fx = to_high_prec(x);
        if (fx <= 0) throw std::domain_error("log of non-positive number");
        HighPrecFloat res = log(fx);
        return to_rational_with_eps(res, eps);
    }

    inline Value fast_sin(const Value& x, const Value& eps) {
        HighPrecFloat fx = to_high_prec(x);
        HighPrecFloat res = sin(fx);
        return to_rational_with_eps(res, eps);
    }

    inline Value fast_cos(const Value& x, const Value& eps) {
        HighPrecFloat fx = to_high_prec(x);
        HighPrecFloat res = cos(fx);
        return to_rational_with_eps(res, eps);
    }

    inline Value fast_acos(const Value& x, const Value& eps) {
        HighPrecFloat fx = to_high_prec(x);
        if (fx < -1 || fx > 1) throw std::domain_error("acos argument out of [-1,1]");
        HighPrecFloat res = acos(fx);
        return to_rational_with_eps(res, eps);
    }

    inline Value fast_pi(const Value& eps) {
        HighPrecFloat pi_val("3.14159265358979323846264338327950288419716939937510");
        return to_rational_with_eps(pi_val, eps);
    }

    inline Value fast_e(const Value& eps) {
        HighPrecFloat e_val("2.71828182845904523536028747135266249775724709369995");
        return to_rational_with_eps(e_val, eps);
    }

    // ============================================================================
    // Slow path (exact rational series/Newton)
    // ============================================================================

    // Helper: compute ln(2) with given precision using arctanh(1/3) series
    inline Value slow_ln2(const Value& eps) {
        Value one = SmallStorage(absl::int128(1));
        Value three = SmallStorage(absl::int128(3));
        Value z = eager_div(one, three);   // 1/3
        Value z2 = eager_mul(z, z);
        Value term = z;
        Value sum = term;
        Value n = one;
        Value two = SmallStorage(absl::int128(2));
        size_t iter = 0;
        const size_t max_iter = 10000;
        while (iter < max_iter) {
            term = eager_mul(term, z2);
            n = eager_add(n, two);
            sum = eager_add(sum, eager_div(term, n));
            ++iter;
            if (is_less(eager_abs(term), eps)) break;
        }
        return eager_mul(two, sum);  // 2 * arctanh(1/3) = ln(2)
    }

    // Newton for sqrt
    inline Value slow_sqrt(const Value& x, const Value& eps) {
        if (is_zero(x)) return SmallStorage(absl::int128(0));
        Value one = SmallStorage(absl::int128(1));
        Value two = SmallStorage(absl::int128(2));
        Value guess = eager_div(x, two);
        Value diff;
        size_t iter = 0;
        const size_t max_iter = 1000;
        do {
            Value next = eager_div(eager_add(guess, eager_div(x, guess)), two);
            diff = eager_abs(eager_sub(next, guess));
            guess = next;
            ++iter;
            if (iter > max_iter) break;
        } while (is_greater(diff, eps));
        return guess;
    }

    // Taylor series for exp with argument reduction
    inline Value slow_exp(const Value& x, const Value& eps) {
        Value one = SmallStorage(absl::int128(1));
        Value two = SmallStorage(absl::int128(2));
        // Reduce: find k such that |x|/2^k <= 1
        int k = 0;
        Value reduced = x;
        while (is_greater(eager_abs(reduced), one)) {
            reduced = eager_div(reduced, two);
            ++k;
        }
        // Series for exp(reduced)
        Value sum = one;
        Value term = one;
        Value n = one;
        size_t iter = 0;
        const size_t max_iter = 10000;
        while (iter < max_iter) {
            term = eager_mul(term, eager_div(reduced, n));
            sum = eager_add(sum, term);
            n = eager_add(n, one);
            ++iter;
            if (is_less(eager_abs(term), eps)) break;
        }
        // Square k times
        Value result = sum;
        for (int i = 0; i < k; ++i) {
            result = eager_mul(result, result);
        }
        return result;
    }

    // Series for log (using reduction and arctanh)
    inline Value slow_log(const Value& x, const Value& eps) {
        if (is_negative(x) || is_zero(x)) throw std::domain_error("log of non-positive");
        Value one = SmallStorage(absl::int128(1));
        Value two = SmallStorage(absl::int128(2));
        Value half = SmallStorage(absl::int128(1), absl::uint128(2));
        // Reduce to [0.5, 2]
        int k = 0;
        Value m = x;
        while (is_greater(m, two)) {
            m = eager_div(m, two);
            ++k;
        }
        while (is_less(m, half)) {
            m = eager_mul(m, two);
            --k;
        }
        // Compute ln2 once
        Value ln2 = slow_ln2(eps);
        // Now m in [0.5, 2]; use arctanh((m-1)/(m+1))
        Value y = eager_div(eager_sub(m, one), eager_add(m, one));
        Value y2 = eager_mul(y, y);
        Value term = y;
        Value sum = term;
        Value n = one;
        size_t iter = 0;
        const size_t max_iter = 10000;
        while (iter < max_iter) {
            term = eager_mul(term, y2);
            n = eager_add(n, two);
            sum = eager_add(sum, eager_div(term, n));
            ++iter;
            if (is_less(eager_abs(term), eps)) break;
        }
        Value ln_m = eager_mul(two, sum);
        // result = ln_m + k * ln2
        return eager_add(ln_m, eager_mul(SmallStorage(absl::int128(k)), ln2));
    }

    // Sine via reduction and series
    inline Value slow_sin(const Value& x, const Value& eps) {
        Value one = SmallStorage(absl::int128(1));
        Value two = SmallStorage(absl::int128(2));
        Value pi_val = slow_pi(eps);
        Value twopi = eager_mul(pi_val, two);
        Value reduced = x;
        // Reduce to [-π, π]
        while (is_greater(eager_abs(reduced), pi_val)) {
            if (is_positive(reduced))
                reduced = eager_sub(reduced, twopi);
            else
                reduced = eager_add(reduced, twopi);
        }
        // Taylor series
        Value x2 = eager_mul(reduced, reduced);
        Value term = reduced;
        Value sum = term;
        Value k = one;
        size_t iter = 0;
        const size_t max_iter = 10000;
        while (iter < max_iter) {
            term = eager_mul(term, eager_neg(x2));
            term = eager_div(term, eager_mul(eager_mul(two, k), eager_add(eager_mul(two, k), one)));
            sum = eager_add(sum, term);
            k = eager_add(k, one);
            if (is_less(eager_abs(term), eps)) break;
            ++iter;
        }
        return sum;
    }

    // Cosine via reduction and series
    inline Value slow_cos(const Value& x, const Value& eps) {
        Value one = SmallStorage(absl::int128(1));
        Value two = SmallStorage(absl::int128(2));
        Value pi_val = slow_pi(eps);
        Value twopi = eager_mul(pi_val, two);
        Value reduced = x;
        while (is_greater(eager_abs(reduced), pi_val)) {
            if (is_positive(reduced))
                reduced = eager_sub(reduced, twopi);
            else
                reduced = eager_add(reduced, twopi);
        }
        Value x2 = eager_mul(reduced, reduced);
        Value term = one;
        Value sum = term;
        Value k = one;
        size_t iter = 0;
        const size_t max_iter = 10000;
        while (iter < max_iter) {
            term = eager_mul(term, eager_neg(x2));
            term = eager_div(term, eager_mul(eager_mul(two, k), eager_sub(eager_mul(two, k), one)));
            sum = eager_add(sum, term);
            k = eager_add(k, one);
            if (is_less(eager_abs(term), eps)) break;
            ++iter;
        }
        return sum;
    }

    // Arccos via Newton on cos
    inline Value slow_acos(const Value& x, const Value& eps) {
        Value one = SmallStorage(absl::int128(1));
        Value two = SmallStorage(absl::int128(2));
        Value pi_val = slow_pi(eps);
        Value half_pi = eager_div(pi_val, two);
        if (is_less(x, eager_neg(one)) || is_greater(x, one)) throw std::domain_error("acos argument out of [-1,1]");
        // Initial guess
        Value y;
        if (is_positive(x)) {
            y = eager_mul(half_pi, eager_sub(one, x));
        }
        else {
            y = eager_sub(pi_val, eager_mul(half_pi, eager_add(one, x)));
        }
        size_t iter = 0;
        const size_t max_iter = 100;
        while (iter < max_iter) {
            Value cos_y = slow_cos(y, eps);
            Value sin_y = slow_sin(y, eps);
            if (is_zero(sin_y)) break;
            Value delta = eager_div(eager_sub(cos_y, x), sin_y);
            y = eager_sub(y, delta);
            if (is_less(eager_abs(delta), eps)) break;
            ++iter;
        }
        return y;
    }

    // π using Machin-like formula: π = 16*arctan(1/5) - 4*arctan(1/239)
    inline Value slow_pi(const Value& eps) {
        Value one = SmallStorage(absl::int128(1));
        Value five = SmallStorage(absl::int128(5));
        Value two39 = SmallStorage(absl::int128(239));
        Value sixteen = SmallStorage(absl::int128(16));
        Value four = SmallStorage(absl::int128(4));
        Value two = SmallStorage(absl::int128(2));
        // arctan(1/5)
        Value a = eager_div(one, five);
        Value a2 = eager_mul(a, a);
        Value term = a;
        Value sum_atan5 = term;
        Value n = one;
        size_t iter = 0;
        const size_t max_iter = 10000;
        while (iter < max_iter) {
            term = eager_mul(term, eager_neg(a2));
            n = eager_add(n, two);
            sum_atan5 = eager_add(sum_atan5, eager_div(term, n));
            ++iter;
            if (is_less(eager_abs(term), eps)) break;
        }
        // arctan(1/239)
        Value b = eager_div(one, two39);
        Value b2 = eager_mul(b, b);
        term = b;
        Value sum_atan239 = term;
        n = one;
        iter = 0;
        while (iter < max_iter) {
            term = eager_mul(term, eager_neg(b2));
            n = eager_add(n, two);
            sum_atan239 = eager_add(sum_atan239, eager_div(term, n));
            ++iter;
            if (is_less(eager_abs(term), eps)) break;
        }
        Value pi_val = eager_sub(eager_mul(sixteen, sum_atan5), eager_mul(four, sum_atan239));
        return pi_val;
    }

    // e using series: e = sum 1/n!
    inline Value slow_e(const Value& eps) {
        Value one = SmallStorage(absl::int128(1));
        Value sum = one;
        Value term = one;
        Value n = one;
        size_t iter = 0;
        const size_t max_iter = 10000;
        while (iter < max_iter) {
            term = eager_div(term, n);
            sum = eager_add(sum, term);
            n = eager_add(n, one);
            ++iter;
            if (is_less(term, eps)) break;
        }
        return sum;
    }

    // ============================================================================
    // Eager dispatcher (choose fast or slow based on eps)
    // ============================================================================

    inline Value eager_sqrt(const Value& x, const Value& eps) {
        if (to_double(eps) >= HYBRID_THRESHOLD) {
            return fast_sqrt(x, eps);
        }
        else {
            return slow_sqrt(x, eps);
        }
    }

    inline Value eager_exp(const Value& x, const Value& eps) {
        if (to_double(eps) >= HYBRID_THRESHOLD) {
            return fast_exp(x, eps);
        }
        else {
            return slow_exp(x, eps);
        }
    }

    inline Value eager_log(const Value& x, const Value& eps) {
        if (to_double(eps) >= HYBRID_THRESHOLD) {
            return fast_log(x, eps);
        }
        else {
            return slow_log(x, eps);
        }
    }

    inline Value eager_sin(const Value& x, const Value& eps) {
        if (to_double(eps) >= HYBRID_THRESHOLD) {
            return fast_sin(x, eps);
        }
        else {
            return slow_sin(x, eps);
        }
    }

    inline Value eager_cos(const Value& x, const Value& eps) {
        if (to_double(eps) >= HYBRID_THRESHOLD) {
            return fast_cos(x, eps);
        }
        else {
            return slow_cos(x, eps);
        }
    }

    inline Value eager_acos(const Value& x, const Value& eps) {
        if (to_double(eps) >= HYBRID_THRESHOLD) {
            return fast_acos(x, eps);
        }
        else {
            return slow_acos(x, eps);
        }
    }

    inline Value eager_pi(const Value& eps) {
        if (to_double(eps) >= HYBRID_THRESHOLD) {
            return fast_pi(eps);
        }
        else {
            return slow_pi(eps);
        }
    }

    inline Value eager_e(const Value& eps) {
        if (to_double(eps) >= HYBRID_THRESHOLD) {
            return fast_e(eps);
        }
        else {
            return slow_e(eps);
        }
    }

} // namespace delta::internal