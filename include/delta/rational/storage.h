// storage.h
#pragma once

#include "rational_fwd.h"
#include "utils.h"

#include <absl/numeric/int128.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/number.hpp>
#include <boost/multiprecision/rational_adaptor.hpp>

#include <stdexcept>
#include <variant>

namespace delta::internal {

    // ============================================================================
    // SmallStorage – 128‑bit rational with lazy normalization
    // ============================================================================
    struct SmallStorage {
        absl::int128 num;
        absl::uint128 den;
        bool reduced;

        constexpr SmallStorage() noexcept : num(0), den(1), reduced(true) {}
        constexpr explicit SmallStorage(absl::int128 n) noexcept : num(n), den(1), reduced(true) {}
        constexpr SmallStorage(absl::int128 n, absl::uint128 d) noexcept : num(n), den(d), reduced(false) {}

        void normalize() {
            if (reduced) return;
            if (den == 0) throw std::domain_error("SmallStorage: denominator cannot be zero");
            if (num == 0) {
                den = 1;
                reduced = true;
                return;
            }
            if (den < 0) {
                den = -den;
                num = -num;
            }
            absl::uint128 abs_num = num < 0 ? static_cast<absl::uint128>(-num) : static_cast<absl::uint128>(num);
            absl::uint128 g = binary_gcd(abs_num, den);
            if (g > 1) {
                abs_num /= g;
                den /= g;
                num = (num < 0) ? -static_cast<absl::int128>(abs_num) : static_cast<absl::int128>(abs_num);
            }
            reduced = true;
        }
    };

    // ============================================================================
    // BigRationalType – canonical rational with expression templates disabled
    // ============================================================================
    using BigRationalType = boost::multiprecision::number<
        boost::multiprecision::rational_adaptor<
        boost::multiprecision::cpp_int_backend<>
        >,
        boost::multiprecision::et_off
    >;

    // ============================================================================
    // BigStorage – single‑field wrapper for arbitrary‑precision rationals
    // ============================================================================
    struct BigStorage {
        BigRationalType val;

        BigStorage() noexcept : val(0) {}
        explicit BigStorage(const dumb_int& n) : val(n) {}
        BigStorage(const dumb_int& n, const dumb_int& d) {
            if (d == 0) throw std::domain_error("BigStorage: denominator cannot be zero");
            BigRationalType tmp(n);
            tmp /= d;
            val = tmp;
        }
        explicit BigStorage(BigRationalType v) noexcept : val(std::move(v)) {}

        [[nodiscard]] dumb_int numerator() const {
            return boost::multiprecision::numerator(val);
        }
        [[nodiscard]] dumb_int denominator() const {
            return boost::multiprecision::denominator(val);
        }
        [[nodiscard]] bool is_normalized() const noexcept { return true; }
        void normalize() noexcept {}
    };

    // ============================================================================
    // Value – variant of immediate rationals
    // ============================================================================
    using Value = std::variant<SmallStorage, BigStorage>;

    // ============================================================================
    // Optimized comparisons (no normalize_to_dumb_int in hot path)
    // ============================================================================
    inline bool operator==(const Value& a, const Value& b) {
        const auto* sa = std::get_if<SmallStorage>(&a);
        const auto* sb = std::get_if<SmallStorage>(&b);
        if (sa && sb) {
            // Both SmallStorage: cross-multiplication with overflow check
            absl::int128 left_num = sa->num;
            absl::uint128 left_den = sa->den;
            absl::int128 right_num = sb->num;
            absl::uint128 right_den = sb->den;
            if (would_overflow_mul(left_num, static_cast<absl::int128>(right_den)) ||
                would_overflow_mul(right_num, static_cast<absl::int128>(left_den))) {
                dumb_int l = to_dumb_int(left_num) * to_dumb_int(right_den);
                dumb_int r = to_dumb_int(right_num) * to_dumb_int(left_den);
                return l == r;
            }
            return left_num * static_cast<absl::int128>(right_den) ==
                right_num * static_cast<absl::int128>(left_den);
        }
        // At least one BigStorage: convert both to BigRationalType and compare
        BigRationalType av, bv;
        if (const auto* ba = std::get_if<BigStorage>(&a)) {
            av = ba->val;
        }
        else {
            const auto& s = std::get<SmallStorage>(a);
            av = BigRationalType(to_dumb_int(s.num)) / to_dumb_int(s.den);
        }
        if (const auto* bb = std::get_if<BigStorage>(&b)) {
            bv = bb->val;
        }
        else {
            const auto& s = std::get<SmallStorage>(b);
            bv = BigRationalType(to_dumb_int(s.num)) / to_dumb_int(s.den);
        }
        return av == bv;
    }

    inline bool operator<(const Value& a, const Value& b) {
        const auto* sa = std::get_if<SmallStorage>(&a);
        const auto* sb = std::get_if<SmallStorage>(&b);
        if (sa && sb) {
            absl::int128 left_num = sa->num;
            absl::uint128 left_den = sa->den;
            absl::int128 right_num = sb->num;
            absl::uint128 right_den = sb->den;
            if (would_overflow_mul(left_num, static_cast<absl::int128>(right_den)) ||
                would_overflow_mul(right_num, static_cast<absl::int128>(left_den))) {
                dumb_int l = to_dumb_int(left_num) * to_dumb_int(right_den);
                dumb_int r = to_dumb_int(right_num) * to_dumb_int(left_den);
                return l < r;
            }
            return left_num * static_cast<absl::int128>(right_den) <
                right_num * static_cast<absl::int128>(left_den);
        }
        BigRationalType av, bv;
        if (const auto* ba = std::get_if<BigStorage>(&a)) {
            av = ba->val;
        }
        else {
            const auto& s = std::get<SmallStorage>(a);
            av = BigRationalType(to_dumb_int(s.num)) / to_dumb_int(s.den);
        }
        if (const auto* bb = std::get_if<BigStorage>(&b)) {
            bv = bb->val;
        }
        else {
            const auto& s = std::get<SmallStorage>(b);
            bv = BigRationalType(to_dumb_int(s.num)) / to_dumb_int(s.den);
        }
        return av < bv;
    }

    inline bool operator>(const Value& a, const Value& b) { return b < a; }
    inline bool operator<=(const Value& a, const Value& b) { return !(b < a); }
    inline bool operator>=(const Value& a, const Value& b) { return !(a < b); }

    // ============================================================================
    // normalize_to_dumb_int – only for batch addition, not for comparisons
    // ============================================================================
    inline std::pair<dumb_int, dumb_int> normalize_to_dumb_int(const Value& v) {
        if (const auto* small = std::get_if<SmallStorage>(&v)) {
            SmallStorage s = *small;
            s.normalize();
            return { to_dumb_int(s.num), to_dumb_int(s.den) };
        }
        else if (const auto* big = std::get_if<BigStorage>(&v)) {
            return { big->numerator(), big->denominator() };
        }
        throw std::invalid_argument("normalize_to_dumb_int: invalid variant");
    }

    // ============================================================================
    // String conversions (debugging only)
    // ============================================================================
    inline std::string to_string(const SmallStorage& s) {
        SmallStorage norm = s;
        norm.normalize();
        if (norm.den == 1) return int128_to_string(norm.num);
        return int128_to_string(norm.num) + "/" + uint128_to_string(norm.den);
    }

    inline std::string to_string(const BigStorage& b) {
        if (b.denominator() == 1) return b.numerator().str();
        return b.numerator().str() + "/" + b.denominator().str();
    }

    inline std::string to_string(const Value& v) {
        if (const auto* s = std::get_if<SmallStorage>(&v))
            return to_string(*s);
        else
            return to_string(std::get<BigStorage>(v));
    }

} // namespace delta::internal