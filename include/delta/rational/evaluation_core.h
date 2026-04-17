// evaluation_core.h
// Адаптирован под новый tagged union Value из storage.h
// Реализует оптимизации SmallStorage согласно спецификации: никакого GCD после арифметики,
// только при переполнении (try_reduce_to_small).

#pragma once

#include "storage.h"
#include "utils.h"

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

namespace delta::internal {

    // Forward declarations
    Value eager_add(const Value& a, const Value& b);
    Value eager_sub(const Value& a, const Value& b);
    Value eager_mul(const Value& a, const Value& b);
    Value eager_div(const Value& a, const Value& b);
    Value eager_neg(const Value& a);
    Value eager_abs(const Value& a);
    Value eager_sqrt(const Value& x, const Value& eps);
    Value eager_exp(const Value& x, const Value& eps);
    Value eager_log(const Value& x, const Value& eps);
    Value eager_sin(const Value& x, const Value& eps);
    Value eager_cos(const Value& x, const Value& eps);
    Value eager_acos(const Value& x, const Value& eps);
    Value eager_pi(const Value& eps);
    Value eager_e(const Value& eps);
    Value eager_pow(const Value& base, const Value& exp, const Value& eps);

    Value slow_sqrt(const Value& x, const Value& eps);
    Value slow_exp(const Value& x, const Value& eps);
    Value slow_log(const Value& x, const Value& eps);
    Value slow_sin(const Value& x, const Value& eps);
    Value slow_cos(const Value& x, const Value& eps);
    Value slow_acos(const Value& x, const Value& eps);
    Value slow_pi(const Value& eps);
    Value slow_e(const Value& eps);
    Value slow_ln2(const Value& eps);

    // ============================================================================
    // Fast predicates – просто обёртки над методами Value (без нормализации)
    // ============================================================================
    inline bool is_zero(const Value& v) { return v.is_zero(); }
    inline bool is_one(const Value& v) { return v.is_one(); }
    inline bool is_positive(const Value& v) { return v.is_positive(); }
    inline bool is_negative(const Value& v) { return v.is_negative(); }

    inline bool is_less(const Value& a, const Value& b) { return a < b; }
    inline bool is_greater(const Value& a, const Value& b) { return a > b; }

    inline double to_double(const Value& v) {
        if (v.tag == ValueType::Small) {
            SmallStorage norm = v.storage.small;
            bool red = false;
            norm.normalize(red);
            return static_cast<double>(norm.num) / static_cast<double>(norm.den);
        }
        else if (v.tag == ValueType::Big) {
            const auto& b = v.storage.big;
            boost::multiprecision::cpp_dec_float_100 f = static_cast<boost::multiprecision::cpp_dec_float_100>(b.numerator());
            f /= static_cast<boost::multiprecision::cpp_dec_float_100>(b.denominator());
            return f.convert_to<double>();
        }
        else {
            return 0.0;
        }
    }

    constexpr double HYBRID_THRESHOLD = 1e-35;

    // ============================================================================
    // Арифметические операции – прямые пути для всех комбинаций,
    // без нормализации и GCD (кроме fallback при переполнении)
    // ============================================================================

    inline Value eager_add(const Value& a, const Value& b) {
        // Small + Small
        if (a.tag == ValueType::Small && b.tag == ValueType::Small) {
            const SmallStorage& sa = a.storage.small;
            const SmallStorage& sb = b.storage.small;

            auto fallback = [&]() -> Value {
                auto [anum, aden] = normalize_to_dumb_int(a);
                auto [bnum, bden] = normalize_to_dumb_int(b);
                dumb_int num = anum * bden + bnum * aden;
                dumb_int den = aden * bden;
                if (auto small = try_reduce_to_small(num, den)) {
                    return Value(*small, true);
                }
                return Value(BigStorage(num, den));
                };

            if (sa.num == 0) return b;
            if (sb.num == 0) return a;

            // равные знаменатели
            if (sa.den == sb.den) {
                if (would_overflow_add(sa.num, sb.num)) {
                    return fallback();
                }
                absl::int128 new_num = sa.num + sb.num;
                if (new_num == 0) return Value(SmallStorage(0), true);
                return Value(SmallStorage(new_num, sa.den), false);
            }

            // общий случай
            bool denoms_small = (sa.den < (absl::uint128(1) << 62)) && (sb.den < (absl::uint128(1) << 62));
            if (denoms_small) {
                absl::int128 left = sa.num * static_cast<absl::int128>(sb.den);
                absl::int128 right = sb.num * static_cast<absl::int128>(sa.den);
                if (would_overflow_add(left, right)) {
                    return fallback();
                }
                absl::int128 new_num = left + right;
                absl::uint128 new_den = sa.den * sb.den;
                return Value(SmallStorage(new_num, new_den), false);
            }

            // большие знаменатели
            if (would_overflow_mul(sa.num, static_cast<absl::int128>(sb.den)) ||
                would_overflow_mul(sb.num, static_cast<absl::int128>(sa.den))) {
                return fallback();
            }
            absl::int128 left = sa.num * static_cast<absl::int128>(sb.den);
            absl::int128 right = sb.num * static_cast<absl::int128>(sa.den);
            if (would_overflow_add(left, right)) {
                return fallback();
            }
            absl::int128 new_num = left + right;
            if (would_overflow_mul(sa.den, sb.den)) {
                return fallback();
            }
            absl::uint128 new_den = sa.den * sb.den;
            return Value(SmallStorage(new_num, new_den), false);
        }
        // BigStorage + BigStorage
        else if (a.tag == ValueType::Big && b.tag == ValueType::Big) {
            BigRationalType res = *a.storage.big.ptr + *b.storage.big.ptr;
            return Value(BigStorage(std::move(res)));
        }
        // BigStorage + SmallStorage (в любом порядке)
        else if ((a.tag == ValueType::Big && b.tag == ValueType::Small) ||
            (a.tag == ValueType::Small && b.tag == ValueType::Big)) {
            const BigStorage* big = (a.tag == ValueType::Big) ? &a.storage.big : &b.storage.big;
            const SmallStorage& small = (a.tag == ValueType::Small) ? a.storage.small : b.storage.small;
            BigRationalType res = *big->ptr +
                (BigRationalType(to_dumb_int(small.num)) / to_dumb_int(small.den));
            return Value(BigStorage(std::move(res)));
        }
        // Fallback (не должен достигаться)
        auto [anum, aden] = normalize_to_dumb_int(a);
        auto [bnum, bden] = normalize_to_dumb_int(b);
        dumb_int num = anum * bden + bnum * aden;
        dumb_int den = aden * bden;
        if (auto small = try_reduce_to_small(num, den)) {
            return Value(*small, true);
        }
        return Value(BigStorage(num, den));
    }

    inline Value eager_sub(const Value& a, const Value& b) {
        return eager_add(a, eager_neg(b));
    }

    inline Value eager_mul(const Value& a, const Value& b) {
        // Small * Small
        if (a.tag == ValueType::Small && b.tag == ValueType::Small) {
            const SmallStorage& sa = a.storage.small;
            const SmallStorage& sb = b.storage.small;

            if (sa.num == 0 || sb.num == 0) return Value(SmallStorage(0), true);

            bool denoms_small = (sa.den < (absl::uint128(1) << 62)) && (sb.den < (absl::uint128(1) << 62));
            absl::int128 new_num;
            absl::uint128 new_den;

            if (denoms_small) {
                //checking overflow for cases of large integer arithmetic
                if (would_overflow_mul(sa.num, sb.num)) {
                    goto mul_fallback;
                }
                new_num = sa.num * sb.num;
                new_den = sa.den * sb.den;
                return Value(SmallStorage(new_num, new_den), false);
            }

            // Проверки переполнения
            if (would_overflow_mul(sa.num, sb.num)) goto mul_fallback;
            new_num = sa.num * sb.num;
            if (would_overflow_mul(sa.den, sb.den)) goto mul_fallback;
            new_den = sa.den * sb.den;
            return Value(SmallStorage(new_num, new_den), false);

        mul_fallback:
            auto [anum, aden] = normalize_to_dumb_int(a);
            auto [bnum, bden] = normalize_to_dumb_int(b);
            dumb_int num = anum * bnum;
            dumb_int den = aden * bden;
            if (auto small = try_reduce_to_small(num, den)) {
                return Value(*small, true);
            }
            return Value(BigStorage(num, den));
        }
        // BigStorage * BigStorage
        else if (a.tag == ValueType::Big && b.tag == ValueType::Big) {
            BigRationalType res = *a.storage.big.ptr * *b.storage.big.ptr;
            return Value(BigStorage(std::move(res)));
        }
        // BigStorage * SmallStorage (в любом порядке)
        else if ((a.tag == ValueType::Big && b.tag == ValueType::Small) ||
            (a.tag == ValueType::Small && b.tag == ValueType::Big)) {
            const BigStorage* big = (a.tag == ValueType::Big) ? &a.storage.big : &b.storage.big;
            const SmallStorage& small = (a.tag == ValueType::Small) ? a.storage.small : b.storage.small;
            BigRationalType res = *big->ptr *
                (BigRationalType(to_dumb_int(small.num)) / to_dumb_int(small.den));
            return Value(BigStorage(std::move(res)));
        }
        // Fallback
        auto [anum, aden] = normalize_to_dumb_int(a);
        auto [bnum, bden] = normalize_to_dumb_int(b);
        dumb_int num = anum * bnum;
        dumb_int den = aden * bden;
        if (auto small = try_reduce_to_small(num, den)) {
            return Value(*small, true);
        }
        return Value(BigStorage(num, den));
    }

    inline Value eager_div(const Value& a, const Value& b) {
        if (is_zero(b)) throw std::domain_error("Division by zero");

        // Small / Small – через умножение на обратное (без нормализации)
        if (a.tag == ValueType::Small && b.tag == ValueType::Small) {
            const SmallStorage& sb = b.storage.small;
            if (sb.num == 0) throw std::domain_error("Division by zero");

            // Обратное число: (den / num) со знаком
            absl::int128 num_recip = static_cast<absl::int128>(sb.den);
            absl::uint128 den_recip = (sb.num < 0) ? static_cast<absl::uint128>(-sb.num) : static_cast<absl::uint128>(sb.num);
            if (sb.num < 0) num_recip = -num_recip;
            SmallStorage recip(num_recip, den_recip);
            Value recip_val(recip, false);
            return eager_mul(a, recip_val);
        }
        // BigStorage / BigStorage
        else if (a.tag == ValueType::Big && b.tag == ValueType::Big) {
            if (b.storage.big.is_zero()) throw std::domain_error("Division by zero");
            BigRationalType res = *a.storage.big.ptr / *b.storage.big.ptr;
            return Value(BigStorage(std::move(res)));
        }
        // BigStorage / SmallStorage (в любом порядке)
        else if ((a.tag == ValueType::Big && b.tag == ValueType::Small) ||
            (a.tag == ValueType::Small && b.tag == ValueType::Big)) {
            // Определяем, кто Big, кто Small
            const BigStorage* big = (a.tag == ValueType::Big) ? &a.storage.big : &b.storage.big;
            const SmallStorage* small = (a.tag == ValueType::Small) ? &a.storage.small : &b.storage.small;

            // Делитель не может быть нулём
            if (small->num == 0) throw std::domain_error("Division by zero");

            // Вычисляем обратное для Small
            absl::int128 num_recip = static_cast<absl::int128>(small->den);
            absl::uint128 den_recip = (small->num < 0) ? static_cast<absl::uint128>(-small->num) : static_cast<absl::uint128>(small->num);
            if (small->num < 0) num_recip = -num_recip;
            SmallStorage recip(num_recip, den_recip);
            Value recip_val(recip, false);

            // Если a - Big, то big / small = big * recip
            // Если a - Small, то small / big = (small * recip) где recip = 1/big, но проще через eager_mul с Value(recip_big)
            if (a.tag == ValueType::Big && b.tag == ValueType::Small) {
                return eager_mul(Value(*big), recip_val);
            }
            else {
                // a - Small, b - Big: small / big = small * (1/big)
                dumb_int num = big->numerator();
                dumb_int den = big->denominator();
                dumb_int recip_num = den;
                dumb_int recip_den = num;
                if (recip_den < 0) { recip_den = -recip_den; recip_num = -recip_num; }
                BigStorage recip_big(recip_num, recip_den);
                return eager_mul(Value(*small), Value(recip_big));
            }
        }
        // Fallback
        auto [anum, aden] = normalize_to_dumb_int(a);
        auto [bnum, bden] = normalize_to_dumb_int(b);
        dumb_int num = anum * bden;
        dumb_int den = aden * bnum;
        if (den == 0) throw std::domain_error("Division by zero");
        if (den < 0) { den = -den; num = -num; }
        if (auto small = try_reduce_to_small(num, den)) {
            return Value(*small, true);
        }
        return Value(BigStorage(num, den));
    }

    inline Value eager_neg(const Value& a) {
        if (a.tag == ValueType::Small) {
            SmallStorage res(-a.storage.small.num, a.storage.small.den);
            return Value(res, false);
        }
        else if (a.tag == ValueType::Big) {
            return Value(BigStorage(-(*a.storage.big.ptr)));
        }
        else {
            throw std::logic_error("eager_neg called on lazy Value");
        }
    }

    inline Value eager_abs(const Value& a) {
        return is_negative(a) ? eager_neg(a) : a;
    }

    // ============================================================================
    // High‑precision floating‑point helpers (fast path)
    // ============================================================================

    using HighPrecFloat = boost::multiprecision::cpp_dec_float_100;

    inline HighPrecFloat to_high_prec(const Value& v) {
        if (v.tag == ValueType::Small) {
            SmallStorage norm = v.storage.small;
            bool red = false;
            norm.normalize(red);
            return static_cast<HighPrecFloat>(to_dumb_int(norm.num)) /
                static_cast<HighPrecFloat>(to_dumb_int(norm.den));
        }
        else if (v.tag == ValueType::Big) {
            const auto& b = v.storage.big;
            return static_cast<HighPrecFloat>(b.numerator()) /
                static_cast<HighPrecFloat>(b.denominator());
        }
        else {
            throw std::domain_error("to_high_prec on lazy Value");
        }
    }

    inline Value to_rational_with_eps(const HighPrecFloat& f, const Value& eps, int extra_digits = 2) {
        HighPrecFloat eps_f = to_high_prec(eps);
        if (eps_f <= 0) throw std::domain_error("Epsilon must be positive");
        int digits_needed = static_cast<int>(-log10(eps_f.convert_to<double>())) + extra_digits;
        if (digits_needed < 1) digits_needed = 1;
        if (digits_needed > 100) digits_needed = 100;

        std::string s = f.str(digits_needed, std::ios_base::fixed);
        size_t dot = s.find('.');
        std::string integer_part = s.substr(0, dot);
        std::string fractional_part = s.substr(dot + 1);
        if (fractional_part.size() > static_cast<size_t>(digits_needed))
            fractional_part = fractional_part.substr(0, digits_needed);

        bool negative = false;
        if (!integer_part.empty() && integer_part[0] == '-') {
            negative = true;
            integer_part = integer_part.substr(1);
        }
        size_t non_zero = integer_part.find_first_not_of('0');
        if (non_zero != std::string::npos) integer_part = integer_part.substr(non_zero);
        else integer_part = "0";
        if (negative && integer_part != "0") integer_part = "-" + integer_part;

        std::string num_str;
        if (integer_part == "0" || integer_part == "-0") {
            num_str = fractional_part;
            if (num_str.empty()) num_str = "0";
        }
        else {
            num_str = integer_part + fractional_part;
        }

        if (num_str.size() > 1 && num_str[0] == '0') {
            size_t first_nonzero = num_str.find_first_not_of('0');
            if (first_nonzero != std::string::npos) num_str = num_str.substr(first_nonzero);
            else num_str = "0";
        }

        dumb_int num(num_str);
        dumb_int den(1);
        for (size_t i = 0; i < fractional_part.size(); ++i) den *= 10;
        dumb_int g = boost::multiprecision::gcd(num, den);
        num /= g; den /= g;
        return Value(BigStorage(num, den));
    }

    // Быстрые трансцендентные функции
    inline Value fast_sqrt(const Value& x, const Value& eps) {
        HighPrecFloat fx = to_high_prec(x);
        if (fx < 0) throw std::domain_error("sqrt of negative number");
        return to_rational_with_eps(sqrt(fx), eps);
    }
    inline Value fast_exp(const Value& x, const Value& eps) { return to_rational_with_eps(exp(to_high_prec(x)), eps); }
    inline Value fast_log(const Value& x, const Value& eps) {
        HighPrecFloat fx = to_high_prec(x);
        if (fx <= 0) throw std::domain_error("log of non-positive number");
        return to_rational_with_eps(log(fx), eps);
    }
    inline Value fast_sin(const Value& x, const Value& eps) { return to_rational_with_eps(sin(to_high_prec(x)), eps); }
    inline Value fast_cos(const Value& x, const Value& eps) { return to_rational_with_eps(cos(to_high_prec(x)), eps); }
    inline Value fast_acos(const Value& x, const Value& eps) {
        HighPrecFloat fx = to_high_prec(x);
        if (fx < -1 || fx > 1) throw std::domain_error("acos argument out of [-1,1]");
        return to_rational_with_eps(acos(fx), eps);
    }
    inline Value fast_pi(const Value& eps) {
        HighPrecFloat pi_val("3.14159265358979323846264338327950288419716939937510");
        return to_rational_with_eps(pi_val, eps, 10);
    }
    inline Value fast_e(const Value& eps) {
        HighPrecFloat e_val("2.71828182845904523536028747135266249775724709369995");
        return to_rational_with_eps(e_val, eps, 10);
    }

    // ============================================================================
    // Точные корни (целочисленные)
    // ============================================================================
    inline bool is_integer(const Value& v) {
        if (v.tag == ValueType::Small) {
            SmallStorage norm = v.storage.small;
            bool red = false;
            norm.normalize(red);
            return norm.den == 1;
        }
        else if (v.tag == ValueType::Big) {
            return v.storage.big.denominator() == 1;
        }
        else {
            return false;
        }
    }

    inline dumb_int get_integer(const Value& v) {
        if (v.tag == ValueType::Small) {
            SmallStorage norm = v.storage.small;
            bool red = false;
            norm.normalize(red);
            return to_dumb_int(norm.num);
        }
        else if (v.tag == ValueType::Big) {
            return v.storage.big.numerator();
        }
        else {
            throw std::logic_error("get_integer on lazy Value");
        }
    }

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
        dumb_int n;
        if (n_val.tag == ValueType::Small) {
            SmallStorage norm = n_val.storage.small;
            bool red = false;
            norm.normalize(red);
            if (norm.den != 1) return std::nullopt;
            n = to_dumb_int(norm.num);
        }
        else if (n_val.tag == ValueType::Big) {
            if (n_val.storage.big.denominator() != 1) return std::nullopt;
            n = n_val.storage.big.numerator();
        }
        else {
            return std::nullopt;
        }
        if (n <= 0 || n > 1000) return std::nullopt;

        if (base.tag == ValueType::Small) {
            SmallStorage s_norm = base.storage.small;
            bool red = false;
            s_norm.normalize(red);
            if (s_norm.num == 0) return Value(SmallStorage(0));
            bool negative = s_norm.num < 0;
            if (negative && n % 2 == 0) return std::nullopt;
            dumb_int num = to_dumb_int(negative ? -s_norm.num : s_norm.num);
            dumb_int den = to_dumb_int(s_norm.den);
            dumb_int root_num = integer_nth_root(num, n);
            dumb_int root_den = integer_nth_root(den, n);
            if (root_num != 0 && root_den != 0) {
                if (negative) root_num = -root_num;
                return Value(BigStorage(root_num, root_den));
            }
        }
        else if (base.tag == ValueType::Big) {
            if (base.storage.big.is_zero()) return Value(SmallStorage(0));
            bool negative = base.storage.big.is_negative();
            if (negative && n % 2 == 0) return std::nullopt;
            dumb_int num = negative ? -base.storage.big.numerator() : base.storage.big.numerator();
            dumb_int den = base.storage.big.denominator();
            dumb_int root_num = integer_nth_root(num, n);
            dumb_int root_den = integer_nth_root(den, n);
            if (root_num != 0 && root_den != 0) {
                if (negative) root_num = -root_num;
                return Value(BigStorage(root_num, root_den));
            }
        }
        return std::nullopt;
    }

    // ============================================================================
    // Конфигурация медленных методов
    // ============================================================================
    constexpr size_t DEFAULT_MAX_ITER = 1000000;
    constexpr size_t NEWTON_MAX_ITER = 1000;
    constexpr size_t ACOS_MAX_ITER = 100;

    // ============================================================================
    // Медленные (точные) реализации трансцендентных функций
    // ============================================================================
    inline Value slow_ln2(const Value& eps) {
        Value one = Value(SmallStorage(1));
        Value three = Value(SmallStorage(3));
        Value z = eager_div(one, three);
        Value z2 = eager_mul(z, z);
        Value term = z, sum = term, n = one, two = Value(SmallStorage(2));
        size_t iter = 0;
        while (iter < DEFAULT_MAX_ITER) {
            term = eager_mul(term, z2);
            n = eager_add(n, two);
            sum = eager_add(sum, eager_div(term, n));
            ++iter;
            if (is_less(eager_abs(term), eps)) break;
        }
        return eager_mul(two, sum);
    }

    inline Value slow_sqrt(const Value& x, const Value& eps) {
        if (is_zero(x)) return Value(SmallStorage(0));
        if (is_negative(x)) throw std::domain_error("sqrt of negative number");
        Value one = Value(SmallStorage(1)), two = Value(SmallStorage(2));
        Value guess = eager_div(x, two), diff;
        size_t iter = 0;
        do {
            Value next = eager_div(eager_add(guess, eager_div(x, guess)), two);
            diff = eager_abs(eager_sub(next, guess));
            guess = next;
            ++iter;
            if (iter > NEWTON_MAX_ITER) break;
        } while (is_greater(diff, eps));
        return guess;
    }

    inline Value slow_exp(const Value& x, const Value& eps) {
        Value one = Value(SmallStorage(1)), two = Value(SmallStorage(2));
        int k = 0;
        Value reduced = x;
        while (is_greater(eager_abs(reduced), one)) { reduced = eager_div(reduced, two); ++k; }
        Value sum = one, term = one, n = one;
        size_t iter = 0;
        while (iter < DEFAULT_MAX_ITER) {
            term = eager_mul(term, eager_div(reduced, n));
            sum = eager_add(sum, term);
            n = eager_add(n, one);
            ++iter;
            if (is_less(eager_abs(term), eps)) break;
        }
        Value result = sum;
        for (int i = 0; i < k; ++i) result = eager_mul(result, result);
        return result;
    }

    inline Value slow_log(const Value& x, const Value& eps) {
        if (is_negative(x) || is_zero(x)) throw std::domain_error("log of non-positive");
        Value one = Value(SmallStorage(1)), two = Value(SmallStorage(2)), half = Value(SmallStorage(1, 2));
        int k = 0;
        Value m = x;
        while (is_greater(m, two)) { m = eager_div(m, two); ++k; }
        while (is_less(m, half)) { m = eager_mul(m, two); --k; }
        Value ln2 = slow_ln2(eps);
        Value y = eager_div(eager_sub(m, one), eager_add(m, one));
        Value y2 = eager_mul(y, y);
        Value term = y, sum = term, n = one;
        size_t iter = 0;
        while (iter < DEFAULT_MAX_ITER) {
            term = eager_mul(term, y2);
            n = eager_add(n, two);
            sum = eager_add(sum, eager_div(term, n));
            ++iter;
            if (is_less(eager_abs(term), eps)) break;
        }
        Value ln_m = eager_mul(two, sum);
        return eager_add(ln_m, eager_mul(Value(SmallStorage(absl::int128(k))), ln2));
    }

    inline Value slow_pi(const Value& eps) {
        Value one = Value(SmallStorage(1)), five = Value(SmallStorage(5)), two39 = Value(SmallStorage(239));
        Value sixteen = Value(SmallStorage(16)), four = Value(SmallStorage(4)), two = Value(SmallStorage(2));
        Value a = eager_div(one, five), a2 = eager_mul(a, a);
        Value term = a, sum_atan5 = term, n = one;
        size_t iter = 0;
        while (iter < DEFAULT_MAX_ITER) {
            term = eager_mul(term, eager_neg(a2));
            n = eager_add(n, two);
            sum_atan5 = eager_add(sum_atan5, eager_div(term, n));
            ++iter;
            if (is_less(eager_abs(term), eps)) break;
        }
        Value b = eager_div(one, two39), b2 = eager_mul(b, b);
        term = b;
        Value sum_atan239 = term;
        n = one;
        iter = 0;
        while (iter < DEFAULT_MAX_ITER) {
            term = eager_mul(term, eager_neg(b2));
            n = eager_add(n, two);
            sum_atan239 = eager_add(sum_atan239, eager_div(term, n));
            ++iter;
            if (is_less(eager_abs(term), eps)) break;
        }
        return eager_sub(eager_mul(sixteen, sum_atan5), eager_mul(four, sum_atan239));
    }

    inline Value slow_sin(const Value& x, const Value& eps) {
        Value one = Value(SmallStorage(1)), two = Value(SmallStorage(2));
        Value pi_val = slow_pi(eps), twopi = eager_mul(pi_val, two);
        Value reduced = x;
        while (is_greater(eager_abs(reduced), pi_val)) {
            if (is_positive(reduced)) reduced = eager_sub(reduced, twopi);
            else reduced = eager_add(reduced, twopi);
        }
        Value x2 = eager_mul(reduced, reduced);
        Value term = reduced, sum = term, k = one;
        size_t iter = 0;
        while (iter < DEFAULT_MAX_ITER) {
            term = eager_mul(term, eager_neg(x2));
            term = eager_div(term, eager_mul(eager_mul(two, k), eager_add(eager_mul(two, k), one)));
            sum = eager_add(sum, term);
            k = eager_add(k, one);
            if (is_less(eager_abs(term), eps)) break;
            ++iter;
        }
        return sum;
    }

    inline Value slow_cos(const Value& x, const Value& eps) {
        Value one = Value(SmallStorage(1)), two = Value(SmallStorage(2));
        Value pi_val = slow_pi(eps), twopi = eager_mul(pi_val, two);
        Value reduced = x;
        while (is_greater(eager_abs(reduced), pi_val)) {
            if (is_positive(reduced)) reduced = eager_sub(reduced, twopi);
            else reduced = eager_add(reduced, twopi);
        }
        Value x2 = eager_mul(reduced, reduced);
        Value term = one, sum = term, k = one;
        size_t iter = 0;
        while (iter < DEFAULT_MAX_ITER) {
            term = eager_mul(term, eager_neg(x2));
            term = eager_div(term, eager_mul(eager_mul(two, k), eager_sub(eager_mul(two, k), one)));
            sum = eager_add(sum, term);
            k = eager_add(k, one);
            if (is_less(eager_abs(term), eps)) break;
            ++iter;
        }
        return sum;
    }

    inline Value slow_acos(const Value& x, const Value& eps) {
        Value one = Value(SmallStorage(1)), two = Value(SmallStorage(2));
        Value pi_val = slow_pi(eps), half_pi = eager_div(pi_val, two);
        if (is_less(x, eager_neg(one)) || is_greater(x, one))
            throw std::domain_error("acos argument out of [-1,1]");
        Value y = is_positive(x) ? eager_mul(half_pi, eager_sub(one, x))
            : eager_sub(pi_val, eager_mul(half_pi, eager_add(one, x)));
        size_t iter = 0;
        while (iter < ACOS_MAX_ITER) {
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

    inline Value slow_e(const Value& eps) {
        Value one = Value(SmallStorage(1));
        Value sum = one, term = one, n = one;
        size_t iter = 0;
        while (iter < DEFAULT_MAX_ITER) {
            term = eager_div(term, n);
            sum = eager_add(sum, term);
            n = eager_add(n, one);
            ++iter;
            if (is_less(term, eps)) break;
        }
        return sum;
    }

    // ============================================================================
    // Целочисленное возведение в степень
    // ============================================================================
    inline Value eager_pow_int(const Value& base, const dumb_int& exponent) {
        if (exponent == 0) return Value(SmallStorage(1));
        if (exponent == 1) return base;
        bool negative = exponent < 0;
        dumb_int e = negative ? -exponent : exponent;
        Value result = Value(SmallStorage(1)), b = base;
        while (e > 0) {
            if (e & 1) result = eager_mul(result, b);
            e >>= 1;
            if (e > 0) b = eager_mul(b, b);
        }
        return negative ? eager_div(Value(SmallStorage(1)), result) : result;
    }

    // ============================================================================
    // Корень n-й степени
    // ============================================================================
    inline int compute_extra_digits(const Value& eps, double operation_complexity = 1.0) {
        double eps_double = to_double(eps);
        if (eps_double <= 0) return 30;
        int digits_needed = static_cast<int>(std::ceil(-std::log10(eps_double))) + 2;
        int safety = static_cast<int>(std::ceil(10.0 * operation_complexity));
        return digits_needed + safety;
    }

    inline Value fast_nth_root(const Value& x, const Value& n, const Value& eps) {
        bool x_neg = is_negative(x);
        if (x_neg) {
            bool n_even = false;
            if (n.tag == ValueType::Small) {
                SmallStorage norm = n.storage.small;
                bool red = false;
                norm.normalize(red);
                if (norm.den == 1 && (norm.num % 2 == 0)) n_even = true;
            }
            else if (n.tag == ValueType::Big) {
                if (n.storage.big.denominator() == 1 && (n.storage.big.numerator() % 2 == 0)) n_even = true;
            }
            if (n_even) throw std::domain_error("even root of negative number");
            return eager_neg(fast_nth_root(eager_neg(x), n, eps));
        }
        if (is_zero(x)) return Value(SmallStorage(0));
        double complexity = 1.0;
        if (n.tag == ValueType::Small) {
            SmallStorage norm = n.storage.small;
            bool red = false;
            norm.normalize(red);
            complexity = static_cast<double>(norm.num);
        }
        else if (n.tag == ValueType::Big) {
            complexity = n.storage.big.numerator().convert_to<double>();
        }
        int extra = compute_extra_digits(eps, complexity);
        HighPrecFloat fx = to_high_prec(x);
        HighPrecFloat fn = to_high_prec(n);
        HighPrecFloat res = pow(fx, 1.0 / fn);
        return to_rational_with_eps(res, eps, extra);
    }

    inline Value eager_nth_root(const Value& x, const Value& n, const Value& eps) {
        if (is_zero(n) || is_negative(n)) throw std::domain_error("nth_root: n must be positive");
        dumb_int n_int;
        if (n.tag == ValueType::Small) {
            SmallStorage norm = n.storage.small;
            bool red = false;
            norm.normalize(red);
            if (norm.den != 1) throw std::domain_error("nth_root: n must be integer");
            n_int = to_dumb_int(norm.num);
        }
        else if (n.tag == ValueType::Big) {
            if (n.storage.big.denominator() != 1) throw std::domain_error("nth_root: n must be integer");
            n_int = n.storage.big.numerator();
        }
        else {
            throw std::domain_error("nth_root: invalid n");
        }
        if (n_int == 0) throw std::domain_error("nth_root: n must be positive");
        if (n_int == 1) return x;
        if (n_int == 2) return eager_sqrt(x, eps);
        if (n_int % 2 == 0 && is_negative(x))
            throw std::domain_error("nth_root: even root of negative number");
        if (auto exact = try_exact_nth_root(x, n)) return *exact;
        if (n_int == 2 && to_double(eps) >= HYBRID_THRESHOLD)
            return fast_nth_root(x, n, eps);
        Value guess = is_positive(x) ? eager_div(x, Value(SmallStorage(2)))
            : eager_neg(eager_div(eager_abs(x), Value(SmallStorage(2))));
        Value n_val = n, n_minus_1 = eager_sub(n_val, Value(SmallStorage(1)));
        Value diff;
        size_t iter = 0;
        do {
            Value pow_n_minus_1 = eager_pow_int(guess, n_int - 1);
            Value next = eager_div(eager_add(eager_mul(n_minus_1, guess), eager_div(x, pow_n_minus_1)), n_val);
            diff = eager_abs(eager_sub(next, guess));
            guess = next;
            ++iter;
            if (iter > NEWTON_MAX_ITER) break;
        } while (is_greater(diff, eps));
        return guess;
    }

    // ============================================================================
    // Общая степень с рациональным показателем
    // ============================================================================
    inline Value eager_pow(const Value& base, const Value& exp, const Value& eps) {
        if (is_zero(base)) {
            if (is_zero(exp)) throw std::domain_error("0^0 is undefined");
            if (is_negative(exp)) throw std::domain_error("0^negative is undefined");
            return base;
        }
        if (is_one(base)) return base;
        if (is_zero(exp)) return Value(SmallStorage(1));

        bool exp_is_int = false;
        dumb_int exp_num, exp_den;
        if (exp.tag == ValueType::Small) {
            SmallStorage norm = exp.storage.small;
            bool red = false;
            norm.normalize(red);
            exp_num = to_dumb_int(norm.num);
            exp_den = to_dumb_int(norm.den);
            if (norm.den == 1) exp_is_int = true;
        }
        else if (exp.tag == ValueType::Big) {
            exp_num = exp.storage.big.numerator();
            exp_den = exp.storage.big.denominator();
            if (exp_den == 1) exp_is_int = true;
        }
        else {
            throw std::domain_error("eager_pow: exponent cannot be lazy");
        }

        if (exp_is_int) {
            if (exp_num < 0) {
                Value base_recip = eager_div(Value(SmallStorage(1)), base);
                return eager_pow_int(base_recip, -exp_num);
            }
            return eager_pow_int(base, exp_num);
        }

        dumb_int p = exp_num, q = exp_den;
        bool negative = (p < 0);
        if (negative) p = -p;

        if (p == 1) {
            Value n_val = Value(BigStorage(q));
            if (q == 2) return eager_sqrt(base, eps);
            Value internal_eps = eager_div(eps, Value(SmallStorage(1000)));
            return eager_nth_root(base, n_val, internal_eps);
        }

        Value internal_eps = (p == 0) ? eps : eager_div(eps, Value(BigStorage(p * 1000)));
        Value log_base = eager_log(base, internal_eps);
        Value p_val = Value(BigStorage(negative ? -p : p));
        Value p_log = eager_mul(p_val, log_base);
        Value q_val = Value(BigStorage(q));
        Value p_log_div_q = eager_div(p_log, q_val);
        return eager_exp(p_log_div_q, internal_eps);
    }

    // ============================================================================
    // Eager dispatchers (без изменений)
    // ============================================================================
    inline Value eager_sqrt(const Value& x, const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? fast_sqrt(x, eps) : slow_sqrt(x, eps);
    }
    inline Value eager_exp(const Value& x, const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? fast_exp(x, eps) : slow_exp(x, eps);
    }
    inline Value eager_log(const Value& x, const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? fast_log(x, eps) : slow_log(x, eps);
    }
    inline Value eager_sin(const Value& x, const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? fast_sin(x, eps) : slow_sin(x, eps);
    }
    inline Value eager_cos(const Value& x, const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? fast_cos(x, eps) : slow_cos(x, eps);
    }
    inline Value eager_acos(const Value& x, const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? fast_acos(x, eps) : slow_acos(x, eps);
    }
    inline Value eager_pi(const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? fast_pi(eps) : slow_pi(eps);
    }
    inline Value eager_e(const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? fast_e(eps) : slow_e(eps);
    }

} // namespace delta::internal