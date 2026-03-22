// include/delta/rational/evaluation.h
#pragma once

#include "delta/rational/rational_class.h"
#include "delta/rational/context.h"
#include "delta/rational/utils.h"
#include <stdexcept>
#include <cmath>

namespace delta::internal {

    // -------------------------------------------------------------------------
    // Eager arithmetic on non-lazy rationals
    // -------------------------------------------------------------------------

    // Normalise a rational number (small or big) to canonical form.
    inline void normalise(Rational& r) {
        if (r.is_small()) {
            const_cast<SmallStorage&>(*r.as_small()).normalize();
        }
        else if (r.is_big()) {
            // BigStorage is already normalised by construction, but we can call
            const_cast<BigStorage&>(*r.as_big()).normalize();
        }
    }

    inline Rational eager_add(const Rational& a, const Rational& b) {
        Rational ea = evaluate(a);
        Rational eb = evaluate(b);

        if (ea.is_small() && eb.is_small()) {
            const SmallStorage& sa = *ea.as_small();
            const SmallStorage& sb = *eb.as_small();
            SmallStorage ca = sa;
            SmallStorage cb = sb;
            ca.normalize();
            cb.normalize();

            // Compute (a_num * b_den + b_num * a_den) / (a_den * b_den)
            bool overflow_num = would_overflow_mul(ca.num, static_cast<absl::int128>(cb.den)) ||
                would_overflow_mul(cb.num, static_cast<absl::int128>(ca.den));
            bool overflow_den = would_overflow_mul(static_cast<absl::int128>(ca.den), static_cast<absl::int128>(cb.den));
            if (!overflow_num && !overflow_den) {
                absl::int128 num = ca.num * static_cast<absl::int128>(cb.den) +
                    cb.num * static_cast<absl::int128>(ca.den);
                absl::uint128 den = ca.den * cb.den;
                return Rational(num, den);
            }
            else {
                // Promote to big
                boost::multiprecision::cpp_int n1(ca.num), d1(ca.den), n2(cb.num), d2(cb.den);
                boost::multiprecision::cpp_int num = n1 * d2 + n2 * d1;
                boost::multiprecision::cpp_int den = d1 * d2;
                return Rational(num, den);
            }
        }
        // At least one is big: convert the small one to big if needed
        boost::multiprecision::cpp_int n1, d1, n2, d2;
        if (ea.is_small()) {
            const auto& s = *ea.as_small();
            n1 = s.num;
            d1 = s.den;
        }
        else {
            const auto& b = *ea.as_big();
            n1 = b.num;
            d1 = b.den;
        }
        if (eb.is_small()) {
            const auto& s = *eb.as_small();
            n2 = s.num;
            d2 = s.den;
        }
        else {
            const auto& b = *eb.as_big();
            n2 = b.num;
            d2 = b.den;
        }
        boost::multiprecision::cpp_int num = n1 * d2 + n2 * d1;
        boost::multiprecision::cpp_int den = d1 * d2;
        return Rational(num, den);
    }

    inline Rational eager_sub(const Rational& a, const Rational& b) {
        Rational ea = evaluate(a);
        Rational eb = evaluate(b);
        return eager_add(ea, eager_neg(eb));
    }

    inline Rational eager_mul(const Rational& a, const Rational& b) {
        Rational ea = evaluate(a);
        Rational eb = evaluate(b);
        if (ea.is_small() && eb.is_small()) {
            const SmallStorage& sa = *ea.as_small();
            const SmallStorage& sb = *eb.as_small();
            SmallStorage ca = sa;
            SmallStorage cb = sb;
            ca.normalize();
            cb.normalize();
            bool overflow_num = would_overflow_mul(ca.num, cb.num);
            bool overflow_den = would_overflow_mul(static_cast<absl::int128>(ca.den), static_cast<absl::int128>(cb.den));
            if (!overflow_num && !overflow_den) {
                absl::int128 num = ca.num * cb.num;
                absl::uint128 den = ca.den * cb.den;
                return Rational(num, den);
            }
            else {
                boost::multiprecision::cpp_int n1(ca.num), d1(ca.den), n2(cb.num), d2(cb.den);
                return Rational(n1 * n2, d1 * d2);
            }
        }
        // Mixed or big
        boost::multiprecision::cpp_int n1, d1, n2, d2;
        if (ea.is_small()) {
            const auto& s = *ea.as_small();
            n1 = s.num;
            d1 = s.den;
        }
        else {
            const auto& b = *ea.as_big();
            n1 = b.num;
            d1 = b.den;
        }
        if (eb.is_small()) {
            const auto& s = *eb.as_small();
            n2 = s.num;
            d2 = s.den;
        }
        else {
            const auto& b = *eb.as_big();
            n2 = b.num;
            d2 = b.den;
        }
        return Rational(n1 * n2, d1 * d2);
    }

    inline Rational eager_div(const Rational& a, const Rational& b) {
        Rational ea = evaluate(a);
        Rational eb = evaluate(b);
        // Check for zero denominator
        if (eb == Rational(0)) {
            throw std::domain_error("division by zero");
        }
        return eager_mul(ea, Rational(1) / eb);
    }

    inline Rational eager_neg(const Rational& a) {
        Rational ea = evaluate(a);
        if (ea.is_small()) {
            SmallStorage s = *ea.as_small();
            s.num = -s.num;
            return Rational(s.num, s.den);
        }
        else {
            BigStorage b = *ea.as_big();
            b.num = -b.num;
            return Rational(b.num, b.den);
        }
    }

    // -------------------------------------------------------------------------
    // Eager transcendental functions (exact with given precision)
    // -------------------------------------------------------------------------

    inline Rational eager_sqrt(const Rational& x, const Rational& eps) {
        Rational a = evaluate(x);
        if (a < Rational(0)) throw std::domain_error("sqrt of negative rational");
        if (a == Rational(0)) return Rational(0);
        Rational guess = a / Rational(2);
        Rational next;
        std::size_t iter = 0;
        do {
            next = (guess + a / guess) / Rational(2);
            if (delta::abs(next - guess) <= eps) break;
            guess = next;
            ++iter;
        } while (iter < 1000000);
        if (iter == 1000000) throw std::runtime_error("sqrt did not converge");
        return next;
    }

    inline Rational eager_exp(const Rational& x, const Rational& eps) {
        Rational a = evaluate(x);
        // Scale argument: find integer k such that |a/2^k| <= 1
        int k = 0;
        Rational scaled = a;
        while (delta::abs(scaled) > Rational(1)) {
            scaled /= Rational(2);
            ++k;
        }
        // Taylor series for exp(scaled)
        Rational sum = Rational(1);
        Rational term = Rational(1);
        std::size_t n = 0;
        while (true) {
            ++n;
            term = term * scaled / Rational(n);
            sum += term;
            if (delta::abs(term) <= eps) break;
            if (n > 1000000) throw std::runtime_error("exp did not converge");
        }
        // Square k times
        Rational result = sum;
        for (int i = 0; i < k; ++i) {
            result = result * result;
        }
        return result;
    }

    inline Rational eager_log(const Rational& x, const Rational& eps) {
        Rational a = evaluate(x);
        if (a <= Rational(0)) throw std::domain_error("log of non-positive rational");
        // Reduce argument to [0.5, 2] by extracting powers of two.
        int k = 0;
        Rational m = a;
        while (m >= Rational(2)) { m /= Rational(2); ++k; }
        while (m < Rational(1, 2)) { m *= Rational(2); --k; }

        // Compute ln(2) with required precision using series: ln(2) = 2 * (z + z^3/3 + z^5/5 + ...) with z = 1/3
        Rational ln2 = [&]() -> Rational {
            Rational z = Rational(1, 3);
            Rational z2 = z * z;
            Rational term = z;
            Rational sum = term;
            std::size_t n = 1;
            while (true) {
                term *= z2;
                n += 2;
                sum += term / Rational(n);
                if (delta::abs(term / Rational(n)) <= eps) break;
            }
            return Rational(2) * sum;
            }();

        // Compute ln(m) = 2 * arctanh((m-1)/(m+1))
        Rational y = (m - Rational(1)) / (m + Rational(1));
        Rational y2 = y * y;
        Rational term = y;
        Rational sum = term;
        std::size_t n = 1;
        while (true) {
            term *= y2;
            n += 2;
            sum += term / Rational(n);
            if (delta::abs(term / Rational(n)) <= eps) break;
        }
        Rational result = Rational(2) * sum + Rational(k) * ln2;
        return result;
    }

    inline Rational eager_sin(const Rational& x, const Rational& eps) {
        Rational a = evaluate(x);
        // Reduce argument to [-π, π] using π computed with same precision
        Rational pi_val = eager_pi(eps);
        Rational two_pi = pi_val * Rational(2);
        Rational reduced = a;
        while (reduced > pi_val) reduced -= two_pi;
        while (reduced < -pi_val) reduced += two_pi;

        // Taylor series: sin(y) = y - y^3/3! + y^5/5! - ...
        Rational y = reduced;
        Rational y2 = y * y;
        Rational term = y;
        Rational sum = term;
        std::size_t n = 1;
        while (true) {
            term = -term * y2 / Rational((2 * n) * (2 * n + 1));
            sum += term;
            if (delta::abs(term) <= eps) break;
            ++n;
            if (n > 1000000) throw std::runtime_error("sin did not converge");
        }
        return sum;
    }

    inline Rational eager_cos(const Rational& x, const Rational& eps) {
        Rational a = evaluate(x);
        Rational pi_val = eager_pi(eps);
        Rational two_pi = pi_val * Rational(2);
        Rational reduced = a;
        while (reduced > pi_val) reduced -= two_pi;
        while (reduced < -pi_val) reduced += two_pi;

        // Taylor series: cos(y) = 1 - y^2/2! + y^4/4! - ...
        Rational y2 = reduced * reduced;
        Rational term = Rational(1);
        Rational sum = term;
        std::size_t n = 0;
        while (true) {
            term = -term * y2 / Rational((2 * n + 1) * (2 * n + 2));
            sum += term;
            if (delta::abs(term) <= eps) break;
            ++n;
            if (n > 1000000) throw std::runtime_error("cos did not converge");
        }
        return sum;
    }

    inline Rational eager_acos(const Rational& x, const Rational& eps) {
        Rational a = evaluate(x);
        if (a < Rational(-1) || a > Rational(1)) throw std::domain_error("acos argument out of [-1,1]");
        Rational pi_val = eager_pi(eps);
        Rational y;
        if (a >= Rational(0)) {
            y = pi_val / Rational(2) * (Rational(1) - a);
        }
        else {
            y = pi_val - pi_val / Rational(2) * (Rational(1) + a);
        }
        // Newton's method: y_{n+1} = y_n - (cos(y_n) - a) / (-sin(y_n))
        Rational delta;
        std::size_t iter = 0;
        do {
            Rational cos_y = eager_cos(y, eps);
            Rational sin_y = eager_sin(y, eps);
            if (sin_y == Rational(0)) break;
            delta = (cos_y - a) / sin_y;
            y = y - delta;
            ++iter;
        } while (delta::abs(delta) > eps && iter < 1000000);
        if (iter == 1000000) throw std::runtime_error("acos did not converge");
        return y;
    }

    inline Rational eager_pi(const Rational& eps) {
        // π = 16 * arctan(1/5) - 4 * arctan(1/239)
        auto arctan = [&](const Rational& x) -> Rational {
            Rational x2 = x * x;
            Rational term = x;
            Rational sum = term;
            std::size_t n = 1;
            while (true) {
                term = -term * x2;
                n += 2;
                sum += term / Rational(n);
                if (delta::abs(term / Rational(n)) <= eps) break;
            }
            return sum;
            };
        return Rational(16) * arctan(Rational(1, 5)) - Rational(4) * arctan(Rational(1, 239));
    }

    inline Rational eager_e(const Rational& eps) {
        // e = Σ 1/n! 
        Rational sum = Rational(1);
        Rational term = Rational(1);
        std::size_t n = 1;
        while (true) {
            term /= Rational(n);
            sum += term;
            if (delta::abs(term) <= eps) break;
            ++n;
            if (n > 1000000) throw std::runtime_error("e did not converge");
        }
        return sum;
    }

} // namespace delta::internal