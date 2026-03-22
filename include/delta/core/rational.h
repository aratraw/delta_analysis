// include/delta/core/rational.h
/**
 * @file rational.h
 * @brief Exact arithmetic kernel for Δ‑analysis – High‑Performance Hybrid Rational.
 *
 * =============================================================================
 *                           CORE PRINCIPLES
 * =============================================================================
 * 0.  Do Not Disturb rational.h unless absolutely sure.
 *
 * 1.  No‑Double Policy (with one exception).
 *     Floating‑point types (float, long double) are strictly forbidden.
 *     In NATIVE_DOUBLE mode, double is allowed as the underlying type,
 *     but you still must use _r literals to create values; accidental mixing
 *     with bare double is prevented by the absence of implicit conversions.
 *
 * 2.  Three Precision Modes (select via CMake):
 *     - DELTA_RATIONAL_MODE_NATIVE_DOUBLE : use built‑in double (fastest, limited precision)
 *     - DELTA_RATIONAL_MODE_BIN_FLOAT     : binary float (default 100 digits, fast)
 *     - DELTA_RATIONAL_MODE_LAZY_HYBRID   : hybrid rational with lazy reduction (default)
 *         * SSO for small numbers (absl::int128 numerator/denominator)
 *         * Automatic promotion to unlimited cpp_int when overflow occurs
 *         * No GCD after each operation – only when needed (comparison, I/O, or manual reduce)
 *         * Up to 5x faster on heavy rational workloads
 *
 * 3.  Raw Literals _r.
 *     Literals like 0.1_r are parsed directly into the chosen Rational type,
 *     without any intermediate double conversion (except in NATIVE_DOUBLE mode,
 *     where it becomes a double literal). This guarantees exact initialisation
 *     according to the chosen backend.
 *
 * 4.  ⚠️⚠️⚠️ SCIENTIFIC NOTATION WITH _r IS NOT CURRENTLY SUPPORTED ⚠️⚠️⚠️
 *     Writing numbers like 1e-10_r WILL COMPILE BUT WILL CRASH AT RUNTIME.
 *     EVEN IF IT DOESN'T CRASH, IT WILL LEAD TO ERROR ACCUMULATION, AMBIGUOUS BEHAVIOR, ETC.
 *     This notation is NOT supported by the Rational backend.
 *
 *     Decimal notation like 0.0000001_r is valid ONLY as long as the number
 *     fits within double/long double (or whatever) precision limits during parsing.
 *
 *     For VERY BIG/SMALL NUMBERS, ALWAYS USE STRING LITERALS:
 *     "0.0000000000000000000000001"_r - this preserves full precision
 *     of the chosen Rational mode. If you chose double as backend - your loss.
 *
 * 5.  Unchecked Arithmetic.
 *     All rational modes use `unchecked` arithmetic – no overflow checks,
 *     for maximum performance and to avoid static initialization issues.
 *     In NATIVE_DOUBLE, standard IEEE behaviour applies.
 *     In LAZY_HYBRID, overflow detection is performed only for small storage.
 *
 * 6.  Precision‑controlled transcendental functions.
 *     In rational modes, functions accept an optional absolute precision `eps`.
 *     In NATIVE_DOUBLE and BIN_FLOAT, precision is fixed by the type and `eps` is ignored.
 *
 * =============================================================================
 *                           USAGE EXAMPLE
 * =============================================================================
 * using namespace delta;
 * auto a = 0.1_r;                    // becomes double, binary float, or rational depending on mode
 * auto b = sin(a, "0.0000000000000000000000000001"_r);  // epsilon is used in rational modes, ignored in double/float
 * auto c = exp(b, "0.00000000000000000001"_r);          // exponential with coarser accuracy
 * auto d = sqrt(2_r);                 // default precision (1e-30)
 *
 * // WRONG - WILL CRASH:
 * // auto wrong = sin(a, 1e-10_r);
 *
 * // CORRECT - use string for tiny numbers:
 * // auto correct = sin(a, "0.0000000001"_r);
 *
 * // No free lunch: you want precision, you pay in iterations.
 * =============================================================================
 */

#pragma once

#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/rational_adaptor.hpp>
#include <iostream>
#include <string>
#include <type_traits>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <charconv>
#include <system_error>
#include <variant>
#include <algorithm>
#include <cctype>

#ifndef DELTA_SERIES_MAX_ITER
#define DELTA_SERIES_MAX_ITER 1000000
#endif

 // Abseil for 128-bit integers and string conversion
#include <absl/numeric/int128.h>
#include <absl/strings/str_cat.h>

namespace delta {

    // -----------------------------------------------------------------------------
    // 1.  Type traits for forbidden types (float, double, long double)
    // -----------------------------------------------------------------------------

    template <typename T>
    struct is_forbidden_type : std::false_type {};

    template <>
    struct is_forbidden_type<float> : std::true_type {};

#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
    template <>
    struct is_forbidden_type<double> : std::false_type {};
#else
    template <>
    struct is_forbidden_type<double> : std::true_type {};
#endif

    template <>
    struct is_forbidden_type<long double> : std::true_type {};

} // namespace delta

// -----------------------------------------------------------------------------
// 2.  Define Rational according to the selected mode
// -----------------------------------------------------------------------------

#if defined(DELTA_RATIONAL_MODE_NATIVE_DOUBLE)

namespace delta {
    using Rational = double;
}

#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)

namespace delta {
    using Rational = boost::multiprecision::number<
        boost::multiprecision::cpp_bin_float<DELTA_BIN_FLOAT_DIGITS>,
        boost::multiprecision::et_off
    >;
}

#else
    // Default: LAZY_HYBRID mode
#define DELTA_RATIONAL_MODE_LAZY_HYBRID

namespace delta {
    namespace lazy_rational {

        using int128_t = absl::int128;
        using uint128_t = absl::uint128;

        // GCD for uint128_t (simple Euclidean)
        inline uint128_t gcd(uint128_t a, uint128_t b) {
            while (b != 0) {
                uint128_t t = b;
                b = a % b;
                a = t;
            }
            return a;
        }

        // Overflow‑safe multiplication for int128_t (uses Abseil's checked multiplication)
        inline bool mul_overflow(int128_t a, int128_t b, int128_t& result) {
            auto maybe = absl::MultiplyChecked(a, b);
            if (maybe.has_value()) {
                result = *maybe;
                return false;
            }
            return true;
        }

        // Overflow‑safe addition for int128_t
        inline bool add_overflow(int128_t a, int128_t b, int128_t& result) {
            auto maybe = absl::AddChecked(a, b);
            if (maybe.has_value()) {
                result = *maybe;
                return false;
            }
            return true;
        }

        struct SmallStorage {
            int128_t num;
            uint128_t den; // always positive

            SmallStorage() : num(0), den(1) {}
            SmallStorage(int128_t n, uint128_t d) : num(n), den(d) {
                if (den == 0) throw std::domain_error("denominator zero");
                if (den < 0) { num = -num; den = -den; }
            }
        };

        struct BigStorage {
            boost::multiprecision::cpp_int num;
            boost::multiprecision::cpp_int den; // always positive

            BigStorage() : num(0), den(1) {}
            BigStorage(const boost::multiprecision::cpp_int& n, const boost::multiprecision::cpp_int& d) : num(n), den(d) {
                if (den == 0) throw std::domain_error("denominator zero");
                if (den < 0) { num = -num; den = -den; }
            }
        };

        class Rational {
            std::variant<SmallStorage, BigStorage> m_data;
            bool m_normalized; // false means reduction is needed

        public:
            // Constructors
            Rational() : m_data(SmallStorage()), m_normalized(true) {}
            Rational(int n) : m_data(SmallStorage(n, 1)), m_normalized(true) {}
            Rational(unsigned long long n) : m_data(SmallStorage(static_cast<int128_t>(n), 1)), m_normalized(true) {}
            Rational(int128_t n) : m_data(SmallStorage(n, 1)), m_normalized(true) {}
            Rational(uint128_t n) : m_data(SmallStorage(static_cast<int128_t>(n), 1)), m_normalized(true) {}
            Rational(const char* s);
            Rational(const std::string& s) : Rational(s.c_str()) {}

            // Copy / move
            Rational(const Rational&) = default;
            Rational(Rational&&) = default;
            Rational& operator=(const Rational&) = default;
            Rational& operator=(Rational&&) = default;

            void reduce();
            explicit operator double() const;
            std::string to_string() const;

            friend Rational operator+(const Rational& a, const Rational& b);
            friend Rational operator-(const Rational& a, const Rational& b);
            friend Rational operator*(const Rational& a, const Rational& b);
            friend Rational operator/(const Rational& a, const Rational& b);
            Rational& operator+=(const Rational& other);
            Rational& operator-=(const Rational& other);
            Rational& operator*=(const Rational& other);
            Rational& operator/=(const Rational& other);
            Rational operator-() const;

            friend bool operator==(const Rational& a, const Rational& b);
            friend bool operator!=(const Rational& a, const Rational& b);
            friend bool operator<(const Rational& a, const Rational& b);
            friend bool operator>(const Rational& a, const Rational& b);
            friend bool operator<=(const Rational& a, const Rational& b);
            friend bool operator>=(const Rational& a, const Rational& b);

            void swap(Rational& other) noexcept {
                m_data.swap(other.m_data);
                std::swap(m_normalized, other.m_normalized);
            }

        private:
            static Rational parse_string(const char* s);
            void normalize_if_needed() { if (!m_normalized) reduce(); }
        };

        // -------------------------------------------------------------------------
        // Implementation of reduce()
        // -------------------------------------------------------------------------
        inline void Rational::reduce() {
            if (m_normalized) return;

            auto visitor = [&](auto& storage) {
                using T = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<T, SmallStorage>) {
                    uint128_t g = gcd(storage.num < 0 ? static_cast<uint128_t>(-storage.num) : static_cast<uint128_t>(storage.num),
                        storage.den);
                    if (g != 1) {
                        storage.num /= static_cast<int128_t>(g);
                        storage.den /= g;
                    }
                }
                else {
                    // BigStorage
                    boost::multiprecision::cpp_int g = boost::multiprecision::gcd(storage.num, storage.den);
                    if (g != 1) {
                        storage.num /= g;
                        storage.den /= g;
                    }
                    // Try to convert back to SmallStorage if possible
                    if (storage.num <= std::numeric_limits<int128_t>::max() &&
                        storage.num >= std::numeric_limits<int128_t>::min() &&
                        storage.den <= std::numeric_limits<uint128_t>::max()) {
                        int128_t n = static_cast<int128_t>(storage.num);
                        uint128_t d = static_cast<uint128_t>(storage.den);
                        m_data = SmallStorage(n, d);
                    }
                }
                };
            std::visit(visitor, m_data);
            m_normalized = true;
        }

        // -------------------------------------------------------------------------
        // Conversion to double
        // -------------------------------------------------------------------------
        inline Rational::operator double() const {
            Rational tmp = *this;
            tmp.reduce();
            auto visitor = [](const auto& storage) -> double {
                using T = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<T, SmallStorage>) {
                    return static_cast<double>(storage.num) / static_cast<double>(storage.den);
                }
                else {
                    return static_cast<double>(storage.num) / static_cast<double>(storage.den);
                }
                };
            return std::visit(visitor, tmp.m_data);
        }

        // -------------------------------------------------------------------------
        // String representation
        // -------------------------------------------------------------------------
        inline std::string Rational::to_string() const {
            Rational tmp = *this;
            tmp.reduce();
            auto visitor = [](const auto& storage) -> std::string {
                using T = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<T, SmallStorage>) {
                    if (storage.den == 1) return absl::StrCat(storage.num);
                    else return absl::StrCat(storage.num, "/", storage.den);
                }
                else {
                    std::string num_str = storage.num.str();
                    std::string den_str = storage.den.str();
                    if (den_str == "1") return num_str;
                    else return num_str + "/" + den_str;
                }
                };
            return std::visit(visitor, tmp.m_data);
        }

        // -------------------------------------------------------------------------
        // Parse string (supports "a/b" and "a.b" formats)
        // -------------------------------------------------------------------------
        inline Rational Rational::parse_string(const char* s) {
            std::string str(s);
            // trim whitespace
            str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
            if (str.empty()) throw std::runtime_error("Empty rational string");

            // Check for slash
            size_t slash = str.find('/');
            if (slash != std::string::npos) {
                std::string num_str = str.substr(0, slash);
                std::string den_str = str.substr(slash + 1);
                if (den_str.empty()) throw std::runtime_error("Missing denominator");
                boost::multiprecision::cpp_int num(num_str);
                boost::multiprecision::cpp_int den(den_str);
                if (den == 0) throw std::domain_error("Denominator zero");
                // Try small storage
                if (num <= std::numeric_limits<int128_t>::max() &&
                    num >= std::numeric_limits<int128_t>::min() &&
                    den <= std::numeric_limits<uint128_t>::max()) {
                    return Rational(SmallStorage(static_cast<int128_t>(num), static_cast<uint128_t>(den)));
                }
                else {
                    return Rational(BigStorage(num, den));
                }
            }

            // Check for decimal point
            size_t dot = str.find('.');
            if (dot != std::string::npos) {
                std::string int_part = str.substr(0, dot);
                std::string frac_part = str.substr(dot + 1);
                // Remove trailing zeros from fractional part for numerator, keep full length for denominator
                std::string frac_trimmed = frac_part;
                size_t last_nonzero = frac_trimmed.find_last_not_of('0');
                if (last_nonzero != std::string::npos) {
                    frac_trimmed = frac_trimmed.substr(0, last_nonzero + 1);
                }
                else {
                    frac_trimmed = "0";
                }

                std::string num_str = (int_part.empty() ? "0" : int_part) + frac_trimmed;
                // Remove leading zeros from numerator
                size_t first_nonzero = num_str.find_first_not_of('0');
                if (first_nonzero != std::string::npos) num_str = num_str.substr(first_nonzero);
                else num_str = "0";

                size_t denom_pow = frac_part.length();
                boost::multiprecision::cpp_int denom = 1;
                for (size_t i = 0; i < denom_pow; ++i) denom *= 10;

                boost::multiprecision::cpp_int num(num_str);
                if (denom == 0) throw std::domain_error("Denominator zero");
                if (num <= std::numeric_limits<int128_t>::max() &&
                    num >= std::numeric_limits<int128_t>::min() &&
                    denom <= std::numeric_limits<uint128_t>::max()) {
                    return Rational(SmallStorage(static_cast<int128_t>(num), static_cast<uint128_t>(denom)));
                }
                else {
                    return Rational(BigStorage(num, denom));
                }
            }

            // Plain integer
            boost::multiprecision::cpp_int num(str);
            if (num <= std::numeric_limits<int128_t>::max() &&
                num >= std::numeric_limits<int128_t>::min()) {
                return Rational(SmallStorage(static_cast<int128_t>(num), 1));
            }
            else {
                return Rational(BigStorage(num, 1));
            }
        }

        // -------------------------------------------------------------------------
        // Arithmetic helpers
        // -------------------------------------------------------------------------
        namespace detail {
            inline BigStorage promote_to_big(const SmallStorage& s) {
                return BigStorage(boost::multiprecision::cpp_int(s.num), boost::multiprecision::cpp_int(s.den));
            }

            inline BigStorage big_add(const BigStorage& a, const BigStorage& b) {
                boost::multiprecision::cpp_int num = a.num * b.den + b.num * a.den;
                boost::multiprecision::cpp_int den = a.den * b.den;
                return BigStorage(num, den);
            }

            inline BigStorage big_sub(const BigStorage& a, const BigStorage& b) {
                boost::multiprecision::cpp_int num = a.num * b.den - b.num * a.den;
                boost::multiprecision::cpp_int den = a.den * b.den;
                return BigStorage(num, den);
            }

            inline BigStorage big_mul(const BigStorage& a, const BigStorage& b) {
                return BigStorage(a.num * b.num, a.den * b.den);
            }

            inline BigStorage big_div(const BigStorage& a, const BigStorage& b) {
                return BigStorage(a.num * b.den, a.den * b.num);
            }
        }

        // -------------------------------------------------------------------------
        // Arithmetic operators
        // -------------------------------------------------------------------------
        inline Rational operator+(const Rational& a, const Rational& b) {
            // Try to stay in small storage if possible
            if (std::holds_alternative<SmallStorage>(a.m_data) && std::holds_alternative<SmallStorage>(b.m_data)) {
                const auto& sa = std::get<SmallStorage>(a.m_data);
                const auto& sb = std::get<SmallStorage>(b.m_data);
                int128_t num1 = sa.num;
                uint128_t den1 = sa.den;
                int128_t num2 = sb.num;
                uint128_t den2 = sb.den;

                // Check overflow for denominator multiplication
                int128_t den_tmp;
                bool den_overflow = mul_overflow(static_cast<int128_t>(den1), static_cast<int128_t>(den2), den_tmp);
                if (!den_overflow) {
                    uint128_t den = static_cast<uint128_t>(den_tmp);
                    // Compute term1 = num1 * den2, term2 = num2 * den1
                    int128_t term1, term2;
                    bool ov1 = mul_overflow(num1, static_cast<int128_t>(den2), term1);
                    bool ov2 = mul_overflow(num2, static_cast<int128_t>(den1), term2);
                    int128_t num;
                    bool ov_add = add_overflow(term1, term2, num);
                    if (!ov1 && !ov2 && !ov_add) {
                        // All fits
                        Rational res(SmallStorage(num, den));
                        res.m_normalized = false;
                        return res;
                    }
                }
                // Overflow, promote both to big
                Rational res(detail::big_add(detail::promote_to_big(sa), detail::promote_to_big(sb)));
                res.m_normalized = false;
                return res;
            }
            else {
                // At least one is big – convert both to big storage
                struct visitor {
                    const BigStorage* operator()(const SmallStorage& s) {
                        tmp = detail::promote_to_big(s);
                        return &tmp;
                    }
                    const BigStorage* operator()(const BigStorage& b) { return &b; }
                    BigStorage tmp;
                };
                visitor va, vb;
                const BigStorage* pa = std::visit([&va](const auto& v) -> const BigStorage* { return va(v); }, a.m_data);
                const BigStorage* pb = std::visit([&vb](const auto& v) -> const BigStorage* { return vb(v); }, b.m_data);
                Rational res(detail::big_add(*pa, *pb));
                res.m_normalized = false;
                return res;
            }
        }

        inline Rational operator-(const Rational& a, const Rational& b) {
            // similar to addition
            if (std::holds_alternative<SmallStorage>(a.m_data) && std::holds_alternative<SmallStorage>(b.m_data)) {
                const auto& sa = std::get<SmallStorage>(a.m_data);
                const auto& sb = std::get<SmallStorage>(b.m_data);
                int128_t num1 = sa.num;
                uint128_t den1 = sa.den;
                int128_t num2 = sb.num;
                uint128_t den2 = sb.den;

                int128_t den_tmp;
                bool den_overflow = mul_overflow(static_cast<int128_t>(den1), static_cast<int128_t>(den2), den_tmp);
                if (!den_overflow) {
                    uint128_t den = static_cast<uint128_t>(den_tmp);
                    int128_t term1, term2;
                    bool ov1 = mul_overflow(num1, static_cast<int128_t>(den2), term1);
                    bool ov2 = mul_overflow(num2, static_cast<int128_t>(den1), term2);
                    int128_t num;
                    bool ov_sub = add_overflow(term1, -term2, num);
                    if (!ov1 && !ov2 && !ov_sub) {
                        Rational res(SmallStorage(num, den));
                        res.m_normalized = false;
                        return res;
                    }
                }
                Rational res(detail::big_sub(detail::promote_to_big(sa), detail::promote_to_big(sb)));
                res.m_normalized = false;
                return res;
            }
            else {
                struct visitor {
                    const BigStorage* operator()(const SmallStorage& s) {
                        tmp = detail::promote_to_big(s);
                        return &tmp;
                    }
                    const BigStorage* operator()(const BigStorage& b) { return &b; }
                    BigStorage tmp;
                };
                visitor va, vb;
                const BigStorage* pa = std::visit([&va](const auto& v) -> const BigStorage* { return va(v); }, a.m_data);
                const BigStorage* pb = std::visit([&vb](const auto& v) -> const BigStorage* { return vb(v); }, b.m_data);
                Rational res(detail::big_sub(*pa, *pb));
                res.m_normalized = false;
                return res;
            }
        }

        inline Rational operator*(const Rational& a, const Rational& b) {
            if (std::holds_alternative<SmallStorage>(a.m_data) && std::holds_alternative<SmallStorage>(b.m_data)) {
                const auto& sa = std::get<SmallStorage>(a.m_data);
                const auto& sb = std::get<SmallStorage>(b.m_data);
                int128_t num1 = sa.num;
                uint128_t den1 = sa.den;
                int128_t num2 = sb.num;
                uint128_t den2 = sb.den;

                int128_t num_tmp, den_tmp;
                bool num_ov = mul_overflow(num1, num2, num_tmp);
                bool den_ov = mul_overflow(static_cast<int128_t>(den1), static_cast<int128_t>(den2), den_tmp);
                if (!num_ov && !den_ov) {
                    Rational res(SmallStorage(num_tmp, static_cast<uint128_t>(den_tmp)));
                    res.m_normalized = false;
                    return res;
                }
                Rational res(detail::big_mul(detail::promote_to_big(sa), detail::promote_to_big(sb)));
                res.m_normalized = false;
                return res;
            }
            else {
                struct visitor {
                    const BigStorage* operator()(const SmallStorage& s) {
                        tmp = detail::promote_to_big(s);
                        return &tmp;
                    }
                    const BigStorage* operator()(const BigStorage& b) { return &b; }
                    BigStorage tmp;
                };
                visitor va, vb;
                const BigStorage* pa = std::visit([&va](const auto& v) -> const BigStorage* { return va(v); }, a.m_data);
                const BigStorage* pb = std::visit([&vb](const auto& v) -> const BigStorage* { return vb(v); }, b.m_data);
                Rational res(detail::big_mul(*pa, *pb));
                res.m_normalized = false;
                return res;
            }
        }

        inline Rational operator/(const Rational& a, const Rational& b) {
            // Division by zero check
            auto is_zero = [](const Rational& r) -> bool {
                auto visitor = [](const auto& storage) -> bool {
                    using T = std::decay_t<decltype(storage)>;
                    if constexpr (std::is_same_v<T, SmallStorage>) return storage.num == 0;
                    else return storage.num == 0;
                    };
                return std::visit(visitor, r.m_data);
                };
            if (is_zero(b)) throw std::domain_error("division by zero");

            if (std::holds_alternative<SmallStorage>(a.m_data) && std::holds_alternative<SmallStorage>(b.m_data)) {
                const auto& sa = std::get<SmallStorage>(a.m_data);
                const auto& sb = std::get<SmallStorage>(b.m_data);
                int128_t num1 = sa.num;
                uint128_t den1 = sa.den;
                int128_t num2 = sb.num;
                uint128_t den2 = sb.den;

                int128_t num_tmp, den_tmp;
                bool num_ov = mul_overflow(num1, static_cast<int128_t>(den2), num_tmp);
                bool den_ov = mul_overflow(static_cast<int128_t>(den1), num2, den_tmp);
                if (!num_ov && !den_ov && den_tmp != 0) {
                    Rational res(SmallStorage(num_tmp, static_cast<uint128_t>(den_tmp)));
                    res.m_normalized = false;
                    return res;
                }
                Rational res(detail::big_div(detail::promote_to_big(sa), detail::promote_to_big(sb)));
                res.m_normalized = false;
                return res;
            }
            else {
                struct visitor {
                    const BigStorage* operator()(const SmallStorage& s) {
                        tmp = detail::promote_to_big(s);
                        return &tmp;
                    }
                    const BigStorage* operator()(const BigStorage& b) { return &b; }
                    BigStorage tmp;
                };
                visitor va, vb;
                const BigStorage* pa = std::visit([&va](const auto& v) -> const BigStorage* { return va(v); }, a.m_data);
                const BigStorage* pb = std::visit([&vb](const auto& v) -> const BigStorage* { return vb(v); }, b.m_data);
                Rational res(detail::big_div(*pa, *pb));
                res.m_normalized = false;
                return res;
            }
        }

        inline Rational& Rational::operator+=(const Rational& other) {
            *this = *this + other;
            return *this;
        }
        inline Rational& Rational::operator-=(const Rational& other) {
            *this = *this - other;
            return *this;
        }
        inline Rational& Rational::operator*=(const Rational& other) {
            *this = *this * other;
            return *this;
        }
        inline Rational& Rational::operator/=(const Rational& other) {
            *this = *this / other;
            return *this;
        }
        inline Rational Rational::operator-() const {
            Rational copy = *this;
            auto visitor = [](auto& storage) {
                using T = std::decay_t<decltype(storage)>;
                if constexpr (std::is_same_v<T, SmallStorage>) storage.num = -storage.num;
                else storage.num = -storage.num;
                };
            std::visit(visitor, copy.m_data);
            return copy;
        }

        // -------------------------------------------------------------------------
        // Comparison operators (always reduce first)
        // -------------------------------------------------------------------------
        inline bool operator==(const Rational& a, const Rational& b) {
            Rational aa = a, bb = b;
            aa.reduce();
            bb.reduce();
            auto visitor = [](const auto& sa, const auto& sb) -> bool {
                using T1 = std::decay_t<decltype(sa)>;
                using T2 = std::decay_t<decltype(sb)>;
                if constexpr (std::is_same_v<T1, SmallStorage> && std::is_same_v<T2, SmallStorage>) {
                    return sa.num == sb.num && sa.den == sb.den;
                }
                else if constexpr (std::is_same_v<T1, SmallStorage> && std::is_same_v<T2, BigStorage>) {
                    boost::multiprecision::cpp_int n1(sa.num), d1(sa.den);
                    return n1 * sb.den == sb.num * d1;
                }
                else if constexpr (std::is_same_v<T1, BigStorage> && std::is_same_v<T2, SmallStorage>) {
                    return sb.num * sa.den == sa.num * sb.den;
                }
                else {
                    return sa.num * sb.den == sb.num * sa.den;
                }
                };
            return std::visit(visitor, aa.m_data, bb.m_data);
        }

        inline bool operator!=(const Rational& a, const Rational& b) {
            return !(a == b);
        }

        inline bool operator<(const Rational& a, const Rational& b) {
            Rational aa = a, bb = b;
            aa.reduce();
            bb.reduce();
            auto visitor = [](const auto& sa, const auto& sb) -> bool {
                using T1 = std::decay_t<decltype(sa)>;
                using T2 = std::decay_t<decltype(sb)>;
                if constexpr (std::is_same_v<T1, SmallStorage> && std::is_same_v<T2, SmallStorage>) {
                    return static_cast<int128_t>(sa.num) * static_cast<int128_t>(sb.den) < static_cast<int128_t>(sb.num) * static_cast<int128_t>(sa.den);
                }
                else {
                    boost::multiprecision::cpp_int left, right;
                    if constexpr (std::is_same_v<T1, SmallStorage>) {
                        left = boost::multiprecision::cpp_int(sa.num) * sb.den;
                        right = sb.num * boost::multiprecision::cpp_int(sa.den);
                    }
                    else if constexpr (std::is_same_v<T2, SmallStorage>) {
                        left = sa.num * boost::multiprecision::cpp_int(sb.den);
                        right = boost::multiprecision::cpp_int(sb.num) * sa.den;
                    }
                    else {
                        left = sa.num * sb.den;
                        right = sb.num * sa.den;
                    }
                    return left < right;
                }
                };
            return std::visit(visitor, aa.m_data, bb.m_data);
        }

        inline bool operator>(const Rational& a, const Rational& b) { return b < a; }
        inline bool operator<=(const Rational& a, const Rational& b) { return !(b < a); }
        inline bool operator>=(const Rational& a, const Rational& b) { return !(a < b); }

    } // namespace lazy_rational

    namespace delta {
        using Rational = lazy_rational::Rational;
    }

#endif // DELTA_RATIONAL_MODE_LAZY_HYBRID

    // -----------------------------------------------------------------------------
    // 3.  Default epsilon (defined after Rational is known)
    // -----------------------------------------------------------------------------

    namespace delta {

        inline Rational& default_eps_value() {
            static Rational eps = [] {
#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
                return 1e-15;
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
                return Rational(1) / Rational("1000000000000000000000000000000");
#else
                // LAZY_HYBRID
                return Rational(1) / Rational("1000000000000000000000000000000");
#endif
                }();
            return eps;
        }

        inline const Rational& default_eps() {
            return default_eps_value();
        }

    } // namespace delta

#ifndef DELTA_DEFAULT_EPS
#define DELTA_DEFAULT_EPS delta::default_eps()
#endif

// -----------------------------------------------------------------------------
// 4.  User-defined literals _r (common for all modes)
// -----------------------------------------------------------------------------

    namespace delta {

        inline Rational operator""_r(unsigned long long num) {
#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
            return static_cast<double>(num);
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
            return Rational(num);
#else
            return Rational(static_cast<lazy_rational::int128_t>(num));
#endif
        }

        inline Rational operator""_r(const char* str, std::size_t len) {
#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
            double val;
            auto [ptr, ec] = std::from_chars(str, str + len, val);
            if (ec != std::errc()) {
                return std::stod(std::string(str, len));
            }
            return val;
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
            return Rational(std::string(str, len));
#else
            return Rational(std::string(str, len));
#endif
        }

#ifdef __cpp_user_defined_literals_floating_point
        template<char... digits>
        Rational operator""_r() {
            constexpr char str[] = { digits..., '\0' };
            return operator""_r(str, sizeof(str) - 1);
        }
#endif

    } // namespace delta

    // -----------------------------------------------------------------------------
    // 5.  Mathematical functions (common for all modes)
    // -----------------------------------------------------------------------------

    namespace delta {

        namespace detail {
            template<typename T>
            inline T abs(const T& x) {
                using std::abs;
                return abs(x);
            }
        }

        // ---------- pi ----------
        inline Rational pi(const Rational& eps = DELTA_DEFAULT_EPS) {
#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
            (void)eps;
            return 3.14159265358979323846264338327950288419716939937510;
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
            (void)eps;
            return boost::multiprecision::pi<Rational>();
#else
            // LAZY_HYBRID: Machin formula with series
            auto arctan = [&](const Rational& x) -> Rational {
                Rational x2 = x * x;
                Rational term = x;
                Rational sum = term;
                Rational n = 1;
                std::size_t iter = 0;
                while (detail::abs(term) > eps && iter < DELTA_SERIES_MAX_ITER) {
                    term *= -x2;
                    n += 2;
                    sum += term / n;
                    if (iter % 10 == 0) sum.reduce();
                    ++iter;
                }
                sum.reduce();
                return sum;
                };
            Rational pi_val = 16 * arctan(Rational(1, 5)) - 4 * arctan(Rational(1, 239));
            pi_val.reduce();
            return pi_val;
#endif
        }

        // ---------- e ----------
        inline Rational e(const Rational& eps = DELTA_DEFAULT_EPS) {
#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
            (void)eps;
            return 2.71828182845904523536028747135266249775724709369995;
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
            (void)eps;
            return boost::multiprecision::exp(Rational(1));
#else
            Rational sum = 0;
            Rational term = 1;
            std::size_t n = 0;
            while (detail::abs(term) > eps && n < DELTA_SERIES_MAX_ITER) {
                sum += term;
                ++n;
                term /= n;
                if (n % 10 == 0) sum.reduce();
            }
            sum.reduce();
            return sum;
#endif
        }

        // ---------- sqrt ----------
        template <typename T>
        inline Rational sqrt(const T& x, const Rational& eps = DELTA_DEFAULT_EPS) {
            static_assert(!is_forbidden_type<T>::value,
                "\n\n[DELTA ERROR]: sqrt() called with floating‑point type!\n"
                "All calculations must use delta::Rational. Use 0.1_r instead of 0.1.\n");
            static_assert(std::is_constructible_v<Rational, T>,
                "Argument must be convertible to delta::Rational.");

            Rational a(x);
            if (a < 0) throw std::domain_error("sqrt of negative rational");

#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
            (void)eps;
            return std::sqrt(a);
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
            (void)eps;
            return boost::multiprecision::sqrt(a);
#else
            if (a == 0) return 0;
            Rational guess = a / 2;
            Rational next;
            std::size_t iter = 0;
            do {
                next = (guess + a / guess) / 2;
                next.reduce();
                if (detail::abs(next - guess) <= eps) break;
                guess = next;
                ++iter;
            } while (iter < DELTA_SERIES_MAX_ITER);
            next.reduce();
            return next;
#endif
        }

        // ---------- cbrt ----------
        template <typename T>
        inline Rational cbrt(const T& x, const Rational& eps = DELTA_DEFAULT_EPS) {
            static_assert(!is_forbidden_type<T>::value,
                "[DELTA ERROR]: cbrt() called with floating‑point type!");
            Rational a(x);
            if (a == 0) return 0;
            if (a < 0) return -cbrt(-a, eps);

#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
            (void)eps;
            return std::cbrt(a);
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
            (void)eps;
            return boost::multiprecision::cbrt(a);
#else
            Rational guess = a / 3;
            Rational next;
            std::size_t iter = 0;
            do {
                next = (2 * guess + a / (guess * guess)) / 3;
                next.reduce();
                if (detail::abs(next - guess) <= eps) break;
                guess = next;
                ++iter;
            } while (iter < DELTA_SERIES_MAX_ITER);
            next.reduce();
            return next;
#endif
        }

        // ---------- pow (integer exponent) ----------
        template <typename T>
        inline Rational pow(const T& x, int exp) {
            static_assert(!is_forbidden_type<T>::value,
                "[DELTA ERROR]: pow() called with floating‑point type!");
            Rational base(x);
            if (exp == 0) return 1;
            Rational result = 1;
            int e = exp;
            if (e < 0) {
                base = 1 / base;
                e = -e;
            }
            while (e) {
                if (e & 1) result *= base;
                base *= base;
                e >>= 1;
            }
            return result;
        }

        // ---------- exp ----------
        template <typename T>
        inline Rational exp(const T& x, const Rational& eps = DELTA_DEFAULT_EPS) {
            static_assert(!is_forbidden_type<T>::value,
                "[DELTA ERROR]: exp() called with floating‑point type!");
            Rational a(x);

#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
            (void)eps;
            return std::exp(a);
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
            (void)eps;
            return boost::multiprecision::exp(a);
#else
            int k = 0;
            Rational scaled = a;
            while (detail::abs(scaled) > 1) {
                scaled /= 2;
                ++k;
            }
            Rational sum = 1;
            Rational term = 1;
            std::size_t n = 0;
            while (detail::abs(term) > eps && n < DELTA_SERIES_MAX_ITER) {
                ++n;
                term *= scaled / n;
                sum += term;
                if (n % 10 == 0) sum.reduce();
            }
            Rational result = sum;
            for (int i = 0; i < k; ++i) {
                result *= result;
                result.reduce();
            }
            result.reduce();
            return result;
#endif
        }

        // ---------- log ----------
        template <typename T>
        inline Rational log(const T& x, const Rational& eps = DELTA_DEFAULT_EPS) {
            static_assert(!is_forbidden_type<T>::value,
                "[DELTA ERROR]: log() called with floating‑point type!");
            Rational a(x);
            if (a <= 0) throw std::domain_error("log of non‑positive rational");

#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
            (void)eps;
            return std::log(a);
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
            (void)eps;
            return boost::multiprecision::log(a);
#else
            int k = 0;
            Rational m = a;
            while (m >= 2) { m /= 2; ++k; }
            while (m < 0.5) { m *= 2; --k; }

            Rational ln2 = 0;
            {
                Rational z = Rational(1, 3);
                Rational z2 = z * z;
                Rational term = z;
                Rational sum = term;
                Rational n = 1;
                std::size_t iter = 0;
                while (detail::abs(term) > eps && iter < DELTA_SERIES_MAX_ITER) {
                    term *= z2;
                    n += 2;
                    sum += term / n;
                    if (iter % 10 == 0) sum.reduce();
                    ++iter;
                }
                ln2 = 2 * sum;
                ln2.reduce();
            }

            Rational y = (m - 1) / (m + 1);
            Rational y2 = y * y;
            Rational term = y;
            Rational sum = term;
            Rational n = 1;
            std::size_t iter = 0;
            while (detail::abs(term) > eps && iter < DELTA_SERIES_MAX_ITER) {
                term *= y2;
                n += 2;
                sum += term / n;
                if (iter % 10 == 0) sum.reduce();
                ++iter;
            }
            Rational result = 2 * sum + k * ln2;
            result.reduce();
            return result;
#endif
        }

        // ---------- sin ----------
        template <typename T>
        inline Rational sin(const T& x, const Rational& eps = DELTA_DEFAULT_EPS) {
            static_assert(!is_forbidden_type<T>::value,
                "[DELTA ERROR]: sin() called with floating‑point type!");
            Rational a(x);

#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
            (void)eps;
            return std::sin(a);
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
            (void)eps;
            return boost::multiprecision::sin(a);
#else
            Rational twopi = 2 * pi(eps);
            Rational reduced = a;
            while (reduced > pi(eps)) reduced -= twopi;
            while (reduced < -pi(eps)) reduced += twopi;

            Rational y = reduced;
            Rational y2 = y * y;
            Rational term = y;
            Rational sum = term;
            Rational n = 1;
            std::size_t iter = 0;
            while (detail::abs(term) > eps && iter < DELTA_SERIES_MAX_ITER) {
                term *= -y2 / ((n + 1) * (n + 2));
                sum += term;
                if (iter % 10 == 0) sum.reduce();
                n += 2;
                ++iter;
            }
            sum.reduce();
            return sum;
#endif
        }

        // ---------- cos ----------
        template <typename T>
        inline Rational cos(const T& x, const Rational& eps = DELTA_DEFAULT_EPS) {
            static_assert(!is_forbidden_type<T>::value,
                "[DELTA ERROR]: cos() called with floating‑point type!");
            Rational a(x);

#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
            (void)eps;
            return std::cos(a);
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
            (void)eps;
            return boost::multiprecision::cos(a);
#else
            Rational twopi = 2 * pi(eps);
            Rational reduced = a;
            while (reduced > pi(eps)) reduced -= twopi;
            while (reduced < -pi(eps)) reduced += twopi;

            Rational y2 = reduced * reduced;
            Rational term = 1;
            Rational sum = term;
            Rational n = 0;
            std::size_t iter = 0;
            while (detail::abs(term) > eps && iter < DELTA_SERIES_MAX_ITER) {
                term *= -y2 / ((2 * n + 1) * (2 * n + 2));
                sum += term;
                if (iter % 10 == 0) sum.reduce();
                n += 2;
                ++iter;
            }
            sum.reduce();
            return sum;
#endif
        }

        // ---------- acos ----------
        template <typename T>
        inline Rational acos(const T& x, const Rational& eps = DELTA_DEFAULT_EPS) {
            static_assert(!is_forbidden_type<T>::value,
                "[DELTA ERROR]: acos() called with floating‑point type!");
            Rational a(x);
            if (a < -1_r || a > 1_r) throw std::domain_error("acos argument out of [-1,1]");

#ifdef DELTA_RATIONAL_MODE_NATIVE_DOUBLE
            (void)eps;
            return std::acos(a);
#elif defined(DELTA_RATIONAL_MODE_BIN_FLOAT)
            (void)eps;
            return boost::multiprecision::acos(a);
#else
            Rational y;
            if (a >= 0_r) {
                y = pi(eps) / 2_r * (1_r - a);
            }
            else {
                y = pi(eps) - pi(eps) / 2_r * (1_r + a);
            }
            Rational delta;
            std::size_t iter = 0;
            do {
                Rational cos_y = delta::cos(y, eps);
                Rational sin_y = delta::sin(y, eps);
                if (sin_y == 0_r) break;
                delta = (cos_y - a) / sin_y;
                y -= delta;
                y.reduce();
                ++iter;
            } while (detail::abs(delta) > eps && iter < DELTA_SERIES_MAX_ITER);
            y.reduce();
            return y;
#endif
        }

        // ---------- abs (convenience) ----------
        inline Rational abs(const Rational& x) {
            return (x < 0) ? -x : x;
        }

        // Bring detail::abs into delta namespace (as in old file)
        using detail::abs;

    } // namespace delta

    // -----------------------------------------------------------------------------
    // 6.  Eigen integration (only for LAZY_HYBRID mode)
    // -----------------------------------------------------------------------------

#ifdef DELTA_RATIONAL_MODE_LAZY_HYBRID

#include <Eigen/Core>

    namespace Eigen {
        namespace internal {
            template<>
            struct sqrt_impl<delta::Rational> {
                static inline delta::Rational run(const delta::Rational& x) {
                    return delta::sqrt(x);
                }
            };
        } // namespace internal
    } // namespace Eigen

#endif // DELTA_RATIONAL_MODE_LAZY_HYBRID