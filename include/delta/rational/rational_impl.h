// rational_impl.h
#pragma once

#include "storage.h"
#include "evaluation_core.h"
#include "lazy_rational.h"
#include "literals.h"
#include "interval.h"
#include <absl/numeric/int128.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <functional>

namespace delta {

    // ----------------------------------------------------------------------------
    // Eager wrappers (используют internal::eager_* напрямую)
    // ----------------------------------------------------------------------------
    inline Rational eager_add(const Rational& a, const Rational& b) {
        return Rational(internal::eager_add(a.value(), b.value()));
    }
    inline Rational eager_sub(const Rational& a, const Rational& b) {
        return Rational(internal::eager_sub(a.value(), b.value()));
    }
    inline Rational eager_mul(const Rational& a, const Rational& b) {
        return Rational(internal::eager_mul(a.value(), b.value()));
    }
    inline Rational eager_div(const Rational& a, const Rational& b) {
        return Rational(internal::eager_div(a.value(), b.value()));
    }
    inline Rational eager_neg(const Rational& a) {
        return Rational(internal::eager_neg(a.value()));
    }
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
    // Конструкторы
    // ----------------------------------------------------------------------------
    inline Rational::Rational() noexcept : storage_(internal::SmallStorage{}) {}

    inline Rational::Rational(absl::int128 num) : storage_(internal::SmallStorage(num)) {}

    inline Rational::Rational(absl::int128 num, absl::uint128 den) {
        if (den == 0) throw std::domain_error("Denominator cannot be zero");
        internal::SmallStorage s(num, den);
        internal::Value v(s, false);
        v.normalize();   // при создании из пары – нормализуем
        storage_ = v;
    }

    inline Rational::Rational(int num) : Rational(static_cast<absl::int128>(num)) {}
    inline Rational::Rational(long long num) : Rational(static_cast<absl::int128>(num)) {}
    inline Rational::Rational(unsigned long long num) : Rational(static_cast<absl::int128>(num)) {}

    inline Rational::Rational(long long num, long long den) {
        if (den == 0) throw std::domain_error("Denominator cannot be zero");
        if (den < 0) { num = -num; den = -den; }

        constexpr absl::int128 max_i128 = (std::numeric_limits<absl::int128>::max)();
        constexpr absl::int128 min_i128 = (std::numeric_limits<absl::int128>::min)();
        constexpr absl::uint128 max_u128 = (std::numeric_limits<absl::uint128>::max)();

        if (num <= max_i128 && num >= min_i128 &&
            static_cast<absl::uint128>(den) <= max_u128) {
            *this = Rational(static_cast<absl::int128>(num),
                static_cast<absl::uint128>(den));
        }
        else {
            *this = Rational(internal::dumb_int(num), internal::dumb_int(den));
        }
    }

    inline Rational::Rational(const internal::dumb_int& num)
        : storage_(internal::BigStorage(num)) {
    }

    inline Rational::Rational(const internal::dumb_int& num, const internal::dumb_int& den) {
        if (den == 0) throw std::domain_error("Denominator cannot be zero");
        storage_ = internal::BigStorage(num, den);
        // BigStorage всегда нормализован (boost::rational_adaptor сокращает)
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
            std::string num_str = s.substr(0, slash);
            std::string den_str = s.substr(slash + 1);
            internal::dumb_int num(num_str);
            internal::dumb_int den(den_str);
            if (den == 0) throw std::domain_error("Denominator cannot be zero");
            if (den < 0) { den = -den; num = -num; }
            internal::dumb_int g = boost::multiprecision::gcd(num, den);
            num /= g; den /= g;
            if (internal::fits_in_int128(num) && internal::fits_in_uint128(den)) {
                internal::SmallStorage s_small(internal::dumb_int_to_int128(num),
                    internal::dumb_int_to_uint128(den));
                internal::Value v(s_small, false);
                v.normalize();
                storage_ = v;
            }
            else {
                storage_ = internal::Value(internal::BigStorage(num, den));
            }
        }
        else {
            size_t dot = s.find('.');
            if (dot == std::string::npos) {
                storage_ = internal::Value(internal::BigStorage(internal::dumb_int(s)));
            }
            else {
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

                if (internal::fits_in_int128(numerator) && internal::fits_in_uint128(denominator)) {
                    internal::SmallStorage s_small(internal::dumb_int_to_int128(numerator),
                        internal::dumb_int_to_uint128(denominator));
                    internal::Value v(s_small, false);
                    v.normalize();
                    storage_ = v;
                }
                else {
                    storage_ = internal::Value(internal::BigStorage(numerator, denominator));
                }
            }
        }
    }

    inline Rational::Rational(internal::Value val) : storage_(std::move(val)) {
        // Оставляем значение как есть, не форсируем нормализацию
    }

    // ----------------------------------------------------------------------------
    // Копирование, перемещение, деструктор
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
        internal::Value v = storage_;
        if (v.tag == internal::ValueType::Small) {
            internal::SmallStorage norm = v.storage.small;
            bool red = false;
            norm.normalize(red);   // нормализуем для получения числителя
            return Rational(internal::to_dumb_int(norm.num));
        }
        else {
            return Rational(v.storage.big.numerator());
        }
    }

    inline Rational Rational::denominator() const {
        internal::Value v = storage_;
        if (v.tag == internal::ValueType::Small) {
            internal::SmallStorage norm = v.storage.small;
            bool red = false;
            norm.normalize(red);
            return Rational(internal::to_dumb_int(norm.den));
        }
        else {
            return Rational(v.storage.big.denominator());
        }
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
    // Арифметические операторы
    // ----------------------------------------------------------------------------
    inline Rational operator+(const Rational& a, const Rational& b) {
        return eager_add(a, b);
    }
    inline Rational operator-(const Rational& a, const Rational& b) {
        return eager_sub(a, b);
    }
    inline Rational operator*(const Rational& a, const Rational& b) {
        return eager_mul(a, b);
    }
    inline Rational operator/(const Rational& a, const Rational& b) {
        return eager_div(a, b);
    }
    inline Rational operator-(const Rational& a) {
        return eager_neg(a);
    }
    // ----------------------------------------------------------------------------
// In‑place addition for immediate Rationals (always immediate now)
// ----------------------------------------------------------------------------
    inline void inplace_add(Rational& a, const Rational& b) {
        // Small + Small
        if (a.storage_.tag == internal::ValueType::Small && b.storage_.tag == internal::ValueType::Small) {
            auto& sa = a.storage_.storage.small;
            const auto& sb = b.storage_.storage.small;

            if (sb.is_zero()) return;
            if (sa.is_zero()) {
                sa = sb;
                a.storage_.small_reduced = false;
                return;
            }

            if (sa.den == sb.den) {
                if (internal::would_overflow_add(sa.num, sb.num)) {
                    a = a + b;
                    return;
                }
                sa.num += sb.num;
                if (sa.is_zero()) {
                    sa.den = 1;
                    a.storage_.small_reduced = true;
                }
                else {
                    a.storage_.small_reduced = false;
                }
                return;
            }

            bool denoms_small = (sa.den < (absl::uint128(1) << 62)) && (sb.den < (absl::uint128(1) << 62));
            if (denoms_small) {
                absl::int128 left = sa.num * static_cast<absl::int128>(sb.den);
                absl::int128 right = sb.num * static_cast<absl::int128>(sa.den);
                if (internal::would_overflow_add(left, right)) {
                    a = a + b;
                    return;
                }
                sa.num = left + right;
                sa.den = sa.den * sb.den;
                a.storage_.small_reduced = false;
                return;
            }

            // Большие знаменатели – проверки переполнения
            if (internal::would_overflow_mul(sa.num, static_cast<absl::int128>(sb.den)) ||
                internal::would_overflow_mul(sb.num, static_cast<absl::int128>(sa.den))) {
                a = a + b;
                return;
            }
            absl::int128 left = sa.num * static_cast<absl::int128>(sb.den);
            absl::int128 right = sb.num * static_cast<absl::int128>(sa.den);
            if (internal::would_overflow_add(left, right)) {
                a = a + b;
                return;
            }
            absl::int128 new_num = left + right;
            if (internal::would_overflow_mul(sa.den, sb.den)) {
                a = a + b;
                return;
            }
            sa.num = new_num;
            sa.den = sa.den * sb.den;
            a.storage_.small_reduced = false;
            return;
        }
        // Big + Big
        else if (a.storage_.tag == internal::ValueType::Big && b.storage_.tag == internal::ValueType::Big) {
            *a.storage_.storage.big.ptr += *b.storage_.storage.big.ptr;
            return;
        }
        // Mixed Big + Small (order independent)
        else if ((a.storage_.tag == internal::ValueType::Big && b.storage_.tag == internal::ValueType::Small) ||
            (a.storage_.tag == internal::ValueType::Small && b.storage_.tag == internal::ValueType::Big)) {
            // Ensure a becomes Big, b is Small
            if (a.storage_.tag == internal::ValueType::Small) {
                const auto& small = a.storage_.storage.small;
                const auto& big = b.storage_.storage.big;
                internal::BigRationalType res = *big.ptr +
                    (internal::BigRationalType(internal::to_dumb_int(small.num)) / internal::to_dumb_int(small.den));
                a.storage_ = internal::Value(internal::BigStorage(std::move(res)));
            }
            else {
                const auto& small = b.storage_.storage.small;
                *a.storage_.storage.big.ptr +=
                    internal::BigRationalType(internal::to_dumb_int(small.num)) / internal::to_dumb_int(small.den);
            }
            return;
        }
        // Fallback (не должно происходить)
        else {
            a = a + b;
        }
    }

    // ----------------------------------------------------------------------------
    // In‑place multiplication for immediate Rationals
    // ----------------------------------------------------------------------------
    inline void inplace_mul(Rational& a, const Rational& b) {
        // Small * Small
        if (a.storage_.tag == internal::ValueType::Small && b.storage_.tag == internal::ValueType::Small) {
            auto& sa = a.storage_.storage.small;
            const auto& sb = b.storage_.storage.small;

            if (sb.is_zero()) {
                sa.num = 0;
                sa.den = 1;
                a.storage_.small_reduced = true;
                return;
            }
            if (sa.is_zero()) return;

            bool denoms_small = (sa.den < (absl::uint128(1) << 62)) && (sb.den < (absl::uint128(1) << 62));
            if (denoms_small) {
                sa.num *= sb.num;
                sa.den *= sb.den;
                a.storage_.small_reduced = false;
                return;
            }

            if (internal::would_overflow_mul(sa.num, sb.num) ||
                internal::would_overflow_mul(sa.den, sb.den)) {
                a = a * b;
                return;
            }
            sa.num *= sb.num;
            sa.den *= sb.den;
            a.storage_.small_reduced = false;
            return;
        }
        // Big * Big
        else if (a.storage_.tag == internal::ValueType::Big && b.storage_.tag == internal::ValueType::Big) {
            *a.storage_.storage.big.ptr *= *b.storage_.storage.big.ptr;
            return;
        }
        // Mixed Big * Small (order independent)
        else if ((a.storage_.tag == internal::ValueType::Big && b.storage_.tag == internal::ValueType::Small) ||
            (a.storage_.tag == internal::ValueType::Small && b.storage_.tag == internal::ValueType::Big)) {
            if (a.storage_.tag == internal::ValueType::Small) {
                const auto& small = a.storage_.storage.small;
                const auto& big = b.storage_.storage.big;
                internal::BigRationalType res = *big.ptr *
                    (internal::BigRationalType(internal::to_dumb_int(small.num)) / internal::to_dumb_int(small.den));
                a.storage_ = internal::Value(internal::BigStorage(std::move(res)));
            }
            else {
                const auto& small = b.storage_.storage.small;
                *a.storage_.storage.big.ptr *=
                    internal::BigRationalType(internal::to_dumb_int(small.num)) / internal::to_dumb_int(small.den);
            }
            return;
        }
        // Fallback
        else {
            a = a * b;
        }
    }
    inline Rational& operator+=(Rational& a, const Rational& b) {
        inplace_add(a, b);
        return a;
    }
    inline Rational& operator-=(Rational& a, const Rational& b) {
        a = a - b;
        return a;
    }
    inline Rational& operator*=(Rational& a, const Rational& b) {
        inplace_mul(a, b);
        return a;
    }
    inline Rational& operator/=(Rational& a, const Rational& b) {
        a = a / b;
        return a;
    }

    // ----------------------------------------------------------------------------
    // Сравнения
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
        return !(b < a);
    }
    inline bool operator>(const Rational& a, const Rational& b) {
        return b < a;
    }
    inline bool operator>=(const Rational& a, const Rational& b) {
        return !(a < b);
    }

    // ----------------------------------------------------------------------------
    // batch_add
    // ----------------------------------------------------------------------------
    inline Rational batch_add(const std::vector<Rational>& terms) {
        if (terms.empty()) return Rational(0);

        // Проверяем, все ли слагаемые immediate
        bool all_immediate = true;
        for (const auto& t : terms) {
            if (t.value().tag == internal::ValueType::Lazy) {
                all_immediate = false;
                break;
            }
        }

        if (all_immediate) {
            using internal::dumb_int;
            dumb_int common_denom(1);
            std::vector<dumb_int> nums;
            nums.reserve(terms.size());

            for (const Rational& term : terms) {
                if (term.value().tag == internal::ValueType::Small) {
                    const internal::SmallStorage& s = term.value().storage.small;
                    // Не нормализуем – работаем с сырыми значениями
                    nums.push_back(internal::to_dumb_int(s.num));
                    dumb_int den = internal::to_dumb_int(s.den);
                    common_denom = boost::multiprecision::lcm(common_denom, den);
                }
                else { // Big
                    const internal::BigStorage& b = term.value().storage.big;
                    nums.push_back(b.numerator());
                    common_denom = boost::multiprecision::lcm(common_denom, b.denominator());
                }
            }

            dumb_int sum_num(0);
            for (size_t i = 0; i < terms.size(); ++i) {
                dumb_int factor = common_denom;
                if (terms[i].value().tag == internal::ValueType::Small) {
                    const internal::SmallStorage& s = terms[i].value().storage.small;
                    factor /= internal::to_dumb_int(s.den);
                    sum_num += nums[i] * factor;
                }
                else {
                    const internal::BigStorage& b = terms[i].value().storage.big;
                    factor /= b.denominator();
                    sum_num += b.numerator() * factor;
                }
            }
            dumb_int g = boost::multiprecision::gcd(sum_num, common_denom);
            sum_num /= g;
            common_denom /= g;

            if (internal::fits_in_int128(sum_num) && internal::fits_in_uint128(common_denom)) {
                return Rational(
                    internal::dumb_int_to_int128(sum_num),
                    internal::dumb_int_to_uint128(common_denom)
                );
            }
            else {
                return Rational(sum_num, common_denom);
            }
        }

        // Если есть ленивые слагаемые – такой сценарий не поддерживается в новой архитектуре,
        // т.к. Rational всегда immediate. Оставляем fallback с упрощённым вычислением.
        Rational sum = 0_r;
        for (const auto& t : terms) {
            sum = sum + t;
        }
        return sum;
    }

    // ----------------------------------------------------------------------------
    // abs
    // ----------------------------------------------------------------------------
    inline Rational abs(const Rational& x) {
        internal::Value v = x.value();
        if (internal::is_negative(v)) {
            return Rational(internal::eager_neg(v));
        }
        return x;
    }

    // ----------------------------------------------------------------------------
    // convert_to<T>
    // ----------------------------------------------------------------------------
    template<typename T>
    inline T Rational::convert_to() const {
        if constexpr (std::is_same_v<T, double>) {
            return to_double();
        }
        else if constexpr (std::is_same_v<T, int>) {
            if (denominator() != 1) {
                throw std::domain_error("Rational::convert_to<int>: not an integer");
            }
            if (storage_.tag == internal::ValueType::Small) {
                internal::SmallStorage norm = storage_.storage.small;
                bool red = false;
                norm.normalize(red);
                absl::int128 num = norm.num;
                if (num < std::numeric_limits<int>::min() || num > std::numeric_limits<int>::max()) {
                    throw std::overflow_error("Rational::convert_to<int>: value out of int range");
                }
                return static_cast<int>(num);
            }
            else {
                const auto& num = storage_.storage.big.numerator();
                if (num > std::numeric_limits<int>::max()) {
                    throw std::overflow_error("Rational::convert_to<int>: value out of range (positive)");
                }
                if (num < std::numeric_limits<int>::min()) {
                    throw std::overflow_error("Rational::convert_to<int>: value out of range (negative)");
                }
                return static_cast<int>(num.convert_to<long long>());
            }
        }
        else if constexpr (std::is_same_v<T, long long>) {
            if (denominator() != 1) {
                throw std::domain_error("Rational::convert_to<long long>: not an integer");
            }
            if (storage_.tag == internal::ValueType::Small) {
                internal::SmallStorage norm = storage_.storage.small;
                bool red = false;
                norm.normalize(red);
                absl::int128 num = norm.num;
                if (num < std::numeric_limits<long long>::min() || num > std::numeric_limits<long long>::max()) {
                    throw std::overflow_error("Rational::convert_to<long long>: value out of long long range");
                }
                return static_cast<long long>(num);
            }
            else {
                const auto& num = storage_.storage.big.numerator();
                if (num > std::numeric_limits<long long>::max()) {
                    throw std::overflow_error("Rational::convert_to<long long>: value out of range (positive)");
                }
                if (num < std::numeric_limits<long long>::min()) {
                    throw std::overflow_error("Rational::convert_to<long long>: value out of range (negative)");
                }
                return static_cast<long long>(num.convert_to<long long>());
            }
        }
        else if constexpr (std::is_same_v<T, internal::dumb_int>) {
            if (denominator() != 1) {
                throw std::domain_error("Rational::convert_to<dumb_int>: not an integer");
            }
            if (storage_.tag == internal::ValueType::Small) {
                internal::SmallStorage norm = storage_.storage.small;
                bool red = false;
                norm.normalize(red);
                return internal::to_dumb_int(norm.num);
            }
            else {
                return storage_.storage.big.numerator();
            }
        }
        else {
            static_assert(sizeof(T) == 0, "convert_to not supported for this type");
        }
    }
    inline internal::Interval Rational::approx_interval() const {
        return internal::Interval(to_double());
    }
    // ----------------------------------------------------------------------------
    // Межтиповые сравнения Rational с LazyRational
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

    inline std::ostream& operator<<(std::ostream& os, const Rational& r) {
        os << r.to_string();
        return os;
    }
} // namespace delta