// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// rational_impl.h
// -----------------------------------------------------------------------------
// IMPLEMENTATION OF RATIONAL – DELEGATING TO BOOST.MULTIPRECISION
// -----------------------------------------------------------------------------
//
// This file contains the implementation of the Rational class defined in
// rational_class.h. All low‑level arithmetic (addition, subtraction,
// multiplication, division, comparison, normalisation) is delegated directly
// to the underlying internal::Value type, which is based on
// boost::multiprecision::rational_adaptor<...>.
//
// -----------------------------------------------------------------------------
// ARCHITECTURAL DECISION: WHY BOOST AND NOT CUSTOM SMALL‑BIG STORAGE?
// -----------------------------------------------------------------------------
//
// Earlier versions of the library had a custom SmallStorage class (using
// Abseil's inlined vectors and a separate heap‑allocated big integer path).
// The goal was to improve performance for small integers (fits in 64 bits)
// by avoiding heap allocations and using stack‑only storage.
//
// However, benchmarks showed that even the most optimised custom
// implementation was 12% SLOWER than a naive `boost::multiprecision::cpp_int`
// for typical rational arithmetic workloads. The reasons:
//   - Boost's backend uses extremely efficient limb operations, written in
//     assembly for common architectures.
//   - Custom allocators and branching between small/big paths introduced
//     overhead that outweighed the benefits.
//   - Modern compilers (GCC, Clang, MSVC) optimise Boost's expression
//     templates extremely well, even when `et_off` is disabled.
//
// Therefore, we made the pragmatic decision: **if you can't beat them, join
// them.** We abandoned the custom storage and now rely entirely on Boost.
//
// The rational_adaptor in storage.h (our Value type) is configured with
// fixed parameters (128-bit minimum, unlimited max, signed magnitude,
// unchecked, custom allocator). THESE PARAMETERS MUST NOT BE CHANGED.
// See storage.h for the detailed warning (the "sacred cow" comment).
//
// This decision has proven stable and performant across all use cases.
// -----------------------------------------------------------------------------
//
// Other implementation notes:
//   - Strings are parsed into fractions exactly (no floating‑point rounding).
//   - batch_add() uses a common denominator to minimise intermediate swell.
//   - All arithmetic operators normalise results (gcd reduction) as required
//     by the underlying rational_adaptor.
//   - Comparisons are exact, using the built‑in operators on Value.
//   - Inter‑type comparisons with LazyRational use interval arithmetic for
//     early termination, falling back to exact evaluation only when needed.
//
// -----------------------------------------------------------------------------

#pragma once

#include "storage.h"
#include "evaluation_core.h"
#include "lazy_rational.h"
#include "literals.h"
#include "interval.h"
#include <boost/multiprecision/cpp_int.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <functional>
#include <cmath> 

namespace delta {

    // ----------------------------------------------------------------------------
    // Eager wrappers (use internal::eager_* directly)
    // ----------------------------------------------------------------------------
    inline Rational eager_sqrt(const Rational& x, const Rational& eps) {
        return Rational(internal::eager_sqrt(x.value(), eps.value()));
    }
    inline Rational eager_exp(const Rational& x, const Rational& eps) {
        return Rational(internal::eager_exp(x.value(), eps.value()));
    }
    inline Rational eager_log(const Rational& x, const Rational& eps) {
        return Rational(internal::eager_log(x.value(), eps.value()));
    }
    inline Rational eager_sin(const Rational& x, const Rational& eps) {
        return Rational(internal::eager_sin(x.value(), eps.value()));
    }
    inline Rational eager_cos(const Rational& x, const Rational& eps) {
        return Rational(internal::eager_cos(x.value(), eps.value()));
    }
    inline Rational eager_acos(const Rational& x, const Rational& eps) {
        return Rational(internal::eager_acos(x.value(), eps.value()));
    }
    inline Rational eager_asin(const Rational& x, const Rational& eps) {
        return Rational(internal::eager_asin(x.value(), eps.value()));
    }
    inline Rational eager_atan(const Rational& x, const Rational& eps) {
        return Rational(internal::eager_atan(x.value(), eps.value()));
    }
    inline Rational eager_tan(const Rational& x, const Rational& eps) {
        return Rational(internal::eager_tan(x.value(), eps.value()));
    }
    inline Rational eager_pi(const Rational& eps) {
        return Rational(internal::eager_pi(eps.value()));
    }
    inline Rational eager_e(const Rational& eps) {
        return Rational(internal::eager_e(eps.value()));
    }
    inline Rational eager_pow(const Rational& base, const Rational& exp, const Rational& eps) {
        return Rational(internal::eager_pow(base.value(), exp.value(), eps.value()));
    }

    // ----------------------------------------------------------------------------
    // Constructors
    // ----------------------------------------------------------------------------
    inline Rational::Rational() noexcept : storage_(0) {}

    inline Rational::Rational(int num) : storage_(internal::dumb_int(num)) {}
    inline Rational::Rational(long long num) : storage_(internal::dumb_int(num)) {}
    inline Rational::Rational(unsigned long long num) : storage_(internal::dumb_int(num)) {}

    inline Rational::Rational(long long num, long long den) {
        if (den == 0) throw std::domain_error("Denominator cannot be zero");
        if (den < 0) { num = -num; den = -den; }
        internal::dumb_int n(num);
        internal::dumb_int d(den);
        internal::dumb_int g = boost::multiprecision::gcd(n, d);
        n /= g; d /= g;
        storage_ = internal::Value(n, d);
    }

    inline Rational::Rational(const internal::dumb_int& num)
        : storage_(num) {
    }

    inline Rational::Rational(const internal::dumb_int& num, const internal::dumb_int& den) {
        if (den == 0) throw std::domain_error("Denominator cannot be zero");
        internal::dumb_int n = num;
        internal::dumb_int d = den;
        if (d < 0) { d = -d; n = -n; }
        internal::dumb_int g = boost::multiprecision::gcd(n, d);
        n /= g; d /= g;
        storage_ = internal::Value(n, d);
    }

    inline Rational::Rational(const boost::multiprecision::cpp_int& num)
        : Rational(internal::dumb_int(num)) {
    }

    inline Rational::Rational(const boost::multiprecision::cpp_int& num,
        const boost::multiprecision::cpp_int& den)
        : Rational(internal::dumb_int(num), internal::dumb_int(den)) {
    }

    inline Rational::Rational(const std::string& s) {
        size_t slash = s.find('/');
        if (slash != std::string::npos) {
            // Format "a/b"
            std::string num_str = s.substr(0, slash);
            std::string den_str = s.substr(slash + 1);
            internal::dumb_int num(num_str);
            internal::dumb_int den(den_str);
            if (den == 0) throw std::domain_error("Denominator cannot be zero");
            if (den < 0) { den = -den; num = -num; }
            internal::dumb_int g = boost::multiprecision::gcd(num, den);
            num /= g; den /= g;
            storage_ = internal::Value(num, den);
        }
        else {
            size_t dot = s.find('.');
            if (dot == std::string::npos) {
                // Integer
                storage_ = internal::Value(internal::dumb_int(s));
            }
            else {
                // Decimal fraction
                std::string int_part = s.substr(0, dot);
                std::string frac_part = s.substr(dot + 1);
                if (frac_part.empty()) frac_part = "0";
                size_t decimal_places = frac_part.length();

                bool negative = false;
                if (!int_part.empty() && int_part[0] == '-') {
                    negative = true;
                    int_part = int_part.substr(1);
                }

                if (int_part.empty()) int_part = "0";
                size_t int_start = int_part.find_first_not_of('0');
                if (int_start != std::string::npos) int_part = int_part.substr(int_start);
                else int_part = "0";

                std::string numerator_str = int_part + frac_part;

                size_t num_start = numerator_str.find_first_not_of('0');
                if (num_start != std::string::npos) {
                    numerator_str = numerator_str.substr(num_start);
                }
                else {
                    numerator_str = "0";
                }

                if (negative && numerator_str != "0") {
                    numerator_str = "-" + numerator_str;
                }

                internal::dumb_int denominator = 1;
                for (size_t i = 0; i < decimal_places; ++i) denominator *= 10;

                internal::dumb_int numerator(numerator_str);
                internal::dumb_int g = boost::multiprecision::gcd(numerator, denominator);
                numerator /= g;
                denominator /= g;

                storage_ = internal::Value(numerator, denominator);
            }
        }
    }

    inline Rational::Rational(internal::Value val) : storage_(std::move(val)) {}
    // ---------------------------------------------------------------------------
    // Smart Conversion double -> Rational Value through continued fraction
    // ---------------------------------------------------------------------------
    // ---------------------------------------------------------------------------
    // Rational(double) – a pragmatic exception to the "no doubles" rule
    // ---------------------------------------------------------------------------
    //
    // The library's core philosophy is that exact rational arithmetic must never
    // be compromised by floating‑point noise.  Hence we deliberately avoid
    // implicit conversions from double in all user‑facing APIs.
    //
    // HOWEVER, deep Eigen integration (beyond simple matrix storage) occasionally
    // requires a scalar type to be constructible from double — some internal Eigen
    // routines initialise temporary values as 0.0 or 1.0, and certain solvers
    // (e.g. SparseLU) insist on this capability.  Without this constructor those
    // components fail to compile.
    //
    // The constructor uses a *continued‑fraction algorithm* with early stopping
    // at a deliberately coarse tolerance (~1e‑5).  Why so coarse?
    //
    //   • Eigen only ever feeds us trivial values — 0.0, 1.0, 0.5 — numbers that
    //     any sane person would instantly recognise as rational.
    //   • We TRUST that nobody in their right mind would use this constructor to
    //     convert 0.142857142857… into a rational.  If you possess a complicated
    //     decimal, use a string literal ("0.1"_r, "1/7"_r) — that guarantees
    //     exact representation and keeps your intentions clear.
    //   • Stopping early means we never chase the garbage bits that IEEE 754
    //     doubles inevitably accumulate for decimal fractions.  Those trailing
    //     digits are noise, not signal, and a faithful conversion would produce
    //     a hideous 10⁵⁸‑digit fraction that poisons every subsequent arithmetic
    //     operation.
    //
    // In short: the constructor is an emergency exit for Eigen.  It will cleanly
    // recognise 0.0, 1.0, 0.5, and equally simple values.  For everything else
    // there is the string literal.  Keep it that way.
    // ---------------------------------------------------------------------------
    inline internal::Value Rational::convert_smart(double val, double eps) {
        using internal::dumb_int;
        if (std::isnan(val) || std::isinf(val))
            return internal::Value(0);

        double x = val;
        double a = std::floor(x);

        dumb_int h_prev = 1, h_curr = static_cast<long long>(a);
        dumb_int k_prev = 0, k_curr = 1;

        while (std::abs(val - static_cast<double>(h_curr) / static_cast<double>(k_curr)) > eps) {
            double diff = x - a;
            if (std::abs(diff) < 1e-12) break;   // практически целое число

            x = 1.0 / diff;
            a = std::floor(x);

            // защита от переполнения разрядности long long
            if (a > 1e15) break;

            dumb_int h_next = static_cast<long long>(a) * h_curr + h_prev;
            dumb_int k_next = static_cast<long long>(a) * k_curr + k_prev;

            h_prev = h_curr;
            h_curr = h_next;
            k_prev = k_curr;
            k_curr = k_next;

            // дополнительная страховка от неограниченного роста
            if (k_curr > dumb_int("1000000000000000")) break;
        }

        // rational_adaptor сам сократит дробь
        return internal::Value(h_curr, k_curr);
    }
    inline Rational::Rational(double v) : storage_(convert_smart(v)) {}

    // ----------------------------------------------------------------------------
    // Copy, move, destructor
    // ----------------------------------------------------------------------------
    inline Rational::Rational(const Rational& other) : storage_(other.storage_) {}
    inline Rational::Rational(Rational&& other) noexcept : storage_(std::move(other.storage_)) {}
    inline Rational& Rational::operator=(const Rational& other) {
        storage_ = other.storage_;
        return *this;
    }
    inline Rational& Rational::operator=(Rational&& other) noexcept {
        storage_ = std::move(other.storage_);
        return *this;
    }
    inline Rational::~Rational() = default;

    // ----------------------------------------------------------------------------
    // as_lazy
    // ----------------------------------------------------------------------------
    inline LazyRational Rational::as_lazy() const {
        return LazyRational(*this);
    }

    // ----------------------------------------------------------------------------
    // numerator / denominator
    // ----------------------------------------------------------------------------
    inline Rational Rational::numerator() const {
        return Rational(internal::numerator(storage_));
    }

    inline Rational Rational::denominator() const {
        return Rational(internal::denominator(storage_));
    }

    // ----------------------------------------------------------------------------
    // to_double / to_string
    // ----------------------------------------------------------------------------
    inline double Rational::to_double() const {
        return internal::to_double(storage_);
    }

    inline std::string Rational::to_string() const {
        return internal::to_string(storage_);
    }

    // ----------------------------------------------------------------------------
    // Arithmetic operators
    // ----------------------------------------------------------------------------
    inline Rational operator+(const Rational& a, const Rational& b) {
        return Rational(a.value() + b.value());
    }
    inline Rational operator-(const Rational& a, const Rational& b) {
        return Rational(a.value() - b.value());
    }
    inline Rational operator*(const Rational& a, const Rational& b) {
        return Rational(a.value() * b.value());
    }
    inline Rational operator/(const Rational& a, const Rational& b) {
        if (internal::is_zero(b.value())) throw std::domain_error("Division by zero");
        return Rational(a.value() / b.value());
    }
    inline Rational operator-(const Rational& a) {
        return Rational(-a.value());
    }

    // ----------------------------------------------------------------------------
    // In‑place operations
    // ----------------------------------------------------------------------------
    inline void inplace_add(Rational& a, const Rational& b) {
        a.storage_ += b.storage_;
    }
    inline void inplace_mul(Rational& a, const Rational& b) {
        a.storage_ *= b.storage_;
    }

    inline Rational& operator+=(Rational& a, const Rational& b) {
        a.storage_ += b.storage_;
        return a;
    }
    inline Rational& operator-=(Rational& a, const Rational& b) {
        a.storage_ -= b.storage_;
        return a;
    }
    inline Rational& operator*=(Rational& a, const Rational& b) {
        a.storage_ *= b.storage_;
        return a;
    }
    inline Rational& operator/=(Rational& a, const Rational& b) {
        if (internal::is_zero(b.value())) throw std::domain_error("Division by zero");
        a.storage_ /= b.storage_;
        return a;
    }

    // ----------------------------------------------------------------------------
    // Comparisons
    // ----------------------------------------------------------------------------
    inline bool operator==(const Rational& a, const Rational& b) {
        return a.value() == b.value();
    }
    inline bool operator!=(const Rational& a, const Rational& b) {
        return !(a == b);
    }
    inline bool operator<(const Rational& a, const Rational& b) {
        return a.value() < b.value();
    }
    inline bool operator<=(const Rational& a, const Rational& b) {
        return a.value() <= b.value();
    }
    inline bool operator>(const Rational& a, const Rational& b) {
        return a.value() > b.value();
    }
    inline bool operator>=(const Rational& a, const Rational& b) {
        return a.value() >= b.value();
    }

    // ----------------------------------------------------------------------------
    // batch_add – efficient summation of a vector of Rationals
    // ----------------------------------------------------------------------------
    inline Rational batch_add(const std::vector<Rational>& terms) {
        if (terms.empty()) return Rational(0);

        using internal::dumb_int;
        dumb_int common_denom(1);
        std::vector<dumb_int> nums;
        nums.reserve(terms.size());

        for (const Rational& term : terms) {
            dumb_int num = internal::numerator(term.value());
            dumb_int den = internal::denominator(term.value());
            nums.push_back(num);
            common_denom = boost::multiprecision::lcm(common_denom, den);
        }

        dumb_int sum_num(0);
        for (size_t i = 0; i < terms.size(); ++i) {
            dumb_int den = internal::denominator(terms[i].value());
            dumb_int factor = common_denom / den;
            sum_num += nums[i] * factor;
        }

        // Reduce the result
        dumb_int g = boost::multiprecision::gcd(sum_num, common_denom);
        sum_num /= g;
        common_denom /= g;

        return Rational(sum_num, common_denom);
    }

    // ----------------------------------------------------------------------------
    // abs
    // ----------------------------------------------------------------------------
    inline Rational abs(const Rational& x) {
        return internal::is_negative(x.value()) ? Rational(-x.value()) : x;
    }

    // ----------------------------------------------------------------------------
    // convert_to<T>
    // ----------------------------------------------------------------------------
    template<typename T>
    inline T Rational::convert_to() const {
        if constexpr (std::is_same_v<T, double>) {
            return storage_.convert_to<double>();
        }
        else if constexpr (std::is_same_v<T, int>) {
            if (!internal::is_integer(storage_))
                throw std::domain_error("Rational::convert_to<int>: not an integer");
            internal::dumb_int num = internal::numerator(storage_);
            if (num < std::numeric_limits<int>::min() || num > std::numeric_limits<int>::max())
                throw std::overflow_error("Rational::convert_to<int>: value out of int range");
            return num.convert_to<int>();
        }
        else if constexpr (std::is_same_v<T, long long>) {
            if (!internal::is_integer(storage_))
                throw std::domain_error("Rational::convert_to<long long>: not an integer");
            internal::dumb_int num = internal::numerator(storage_);
            if (num < std::numeric_limits<long long>::min() || num > std::numeric_limits<long long>::max())
                throw std::overflow_error("Rational::convert_to<long long>: value out of long long range");
            return num.convert_to<long long>();
        }
        else if constexpr (std::is_same_v<T, internal::dumb_int>) {
            if (!internal::is_integer(storage_))
                throw std::domain_error("Rational::convert_to<dumb_int>: not an integer");
            return internal::numerator(storage_);
        }
        else {
            static_assert(sizeof(T) == 0, "convert_to not supported for this type");
        }
    }

    inline internal::Interval Rational::approx_interval() const {
        return internal::Interval(to_double());
    }

    // ----------------------------------------------------------------------------
    // Cross‑type comparisons between Rational and LazyRational
    // ----------------------------------------------------------------------------
    inline bool operator==(const Rational& a, const LazyRational& b) {
        internal::Interval ia = a.approx_interval();
        internal::Interval ib = b.approx_interval();
        if (!ia.overlaps(ib)) return false;
        return a == b.eval();
    }
    inline bool operator==(const LazyRational& a, const Rational& b) { return b == a; }

    inline bool operator!=(const Rational& a, const LazyRational& b) { return !(a == b); }
    inline bool operator!=(const LazyRational& a, const Rational& b) { return !(a == b); }

    inline bool operator<(const Rational& a, const LazyRational& b) {
        internal::Interval ia = a.approx_interval();
        internal::Interval ib = b.approx_interval();
        if (ia.upper() < ib.lower()) return true;
        if (ia.lower() >= ib.upper()) return false;
        return a < b.eval();
    }
    inline bool operator<(const LazyRational& a, const Rational& b) {
        internal::Interval ia = a.approx_interval();
        internal::Interval ib = b.approx_interval();
        if (ia.upper() < ib.lower()) return true;
        if (ia.lower() >= ib.upper()) return false;
        return a.eval() < b;
    }

    inline bool operator<=(const Rational& a, const LazyRational& b) { return !(b < a); }
    inline bool operator<=(const LazyRational& a, const Rational& b) { return !(b < a); }
    inline bool operator>(const Rational& a, const LazyRational& b) { return b < a; }
    inline bool operator>(const LazyRational& a, const Rational& b) { return b < a; }
    inline bool operator>=(const Rational& a, const LazyRational& b) { return !(a < b); }
    inline bool operator>=(const LazyRational& a, const Rational& b) { return !(a < b); }

    // ----------------------------------------------------------------------------
    // Output stream
    // ----------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const Rational& r) {
        os << r.to_string();
        return os;
    }

} // namespace delta