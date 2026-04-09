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

    // SmallStorage (без изменений)
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

    // BigRationalType – без изменений
    using BigRationalType = boost::multiprecision::number<
        boost::multiprecision::rational_adaptor<
        boost::multiprecision::cpp_int_backend<>
        >,
        boost::multiprecision::et_off
    >;

    // BigStorage – без изменений (оставляем Приоритет 3 – работает везде)
    struct BigStorage {
        BigRationalType val;

        BigStorage() noexcept : val(0) {}
        explicit BigStorage(const dumb_int& n) : val(n) {}
        BigStorage(const dumb_int& n, const dumb_int& d) {
            if (d == 0) throw std::domain_error("BigStorage: denominator cannot be zero");
            val = BigRationalType(n) / d;
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

    using Value = std::variant<SmallStorage, BigStorage>;

    // ============================================================================
    // normalize_to_dumb_int – объявляем до использования в сравнениях
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
    // Оптимизированные сравнения – только cross‑multiplication, без создания BigRationalType
    // ============================================================================
    inline bool operator==(const Value& a, const Value& b) {
        const auto* sa = std::get_if<SmallStorage>(&a);
        const auto* sb = std::get_if<SmallStorage>(&b);
        if (sa && sb) {
            // SmallStorage + SmallStorage – кросс-умножение с проверкой переполнения
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
        // Общий случай: нормализуем через dumb_int и сравниваем крест-накрест
        auto [anum, aden] = normalize_to_dumb_int(a);
        auto [bnum, bden] = normalize_to_dumb_int(b);
        return anum * bden == bnum * aden;
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
        auto [anum, aden] = normalize_to_dumb_int(a);
        auto [bnum, bden] = normalize_to_dumb_int(b);
        return anum * bden < bnum * aden;
    }

    inline bool operator>(const Value& a, const Value& b) { return b < a; }
    inline bool operator<=(const Value& a, const Value& b) { return !(b < a); }
    inline bool operator>=(const Value& a, const Value& b) { return !(a < b); }

    // ============================================================================
    // Строковые функции (без изменений, для отладки)
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