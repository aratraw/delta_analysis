// evaluation_core.h
// Смешанная версия: быстрые пути через cpp_dec_float_100 с контролируемой точностью,
// медленные пути для особых случаев (когда eps < порога). Интервалы остаются в double.

#pragma once

#include "expression_root.h"
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

namespace delta::internal {

    // ============================================================================
    // Forward declarations для eager-функций
    // ============================================================================
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

    // ============================================================================
    // Forward declarations для slow-функций (точные рациональные)
    // ============================================================================
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
    // Вспомогательные функции для Value
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
        return a < b;
    }

    inline bool is_greater(const Value& a, const Value& b) {
        return a > b;
    }

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

    // Порог для выбора быстрого пути – снижен до 1e-35,
    // чтобы для eps = 1e-30 использовался медленный (Newton) путь,
    // гарантирующий высокую точность.
    constexpr double HYBRID_THRESHOLD = 1e-35;

    // ============================================================================
    // Точные рациональные арифметические операции
    // ============================================================================

    inline Value eager_add(const Value& a, const Value& b) {
        if (const auto* sa = std::get_if<SmallStorage>(&a)) {
            if (const auto* sb = std::get_if<SmallStorage>(&b)) {
                SmallStorage sa_norm = *sa;
                SmallStorage sb_norm = *sb;
                sa_norm.normalize();
                sb_norm.normalize();

                absl::uint128 den = sa_norm.den;
                absl::uint128 den2 = sb_norm.den;
                if (den == den2) {
                    absl::int128 num = sa_norm.num + sb_norm.num;
                    if (!would_overflow_add(sa_norm.num, sb_norm.num)) {
                        SmallStorage res(num, den);
                        res.normalize();
                        return res;
                    }
                }
                boost::multiprecision::cpp_int num = to_cpp_int(sa_norm.num) * to_cpp_int(den2) +
                    to_cpp_int(sb_norm.num) * to_cpp_int(den);
                boost::multiprecision::cpp_int denom = to_cpp_int(den) * to_cpp_int(den2);
                boost::multiprecision::cpp_int g = boost::multiprecision::gcd(num, denom);
                num /= g;
                denom /= g;
                if (num <= to_cpp_int((std::numeric_limits<absl::int128>::max)()) &&
                    denom <= to_cpp_int((std::numeric_limits<absl::uint128>::max)())) {
                    return SmallStorage(int128_from_string(num.str()),
                        uint128_from_string(denom.str()));
                }
                return BigStorage(num, denom);
            }
        }
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
        if (const auto* sa = std::get_if<SmallStorage>(&a)) {
            if (const auto* sb = std::get_if<SmallStorage>(&b)) {
                SmallStorage sa_norm = *sa;
                SmallStorage sb_norm = *sb;
                sa_norm.normalize();
                sb_norm.normalize();

                if (!would_overflow_mul(sa_norm.num, sb_norm.num) &&
                    sa_norm.den <= (std::numeric_limits<absl::uint128>::max)() / sb_norm.den) {
                    absl::int128 num = sa_norm.num * sb_norm.num;
                    absl::uint128 den = sa_norm.den * sb_norm.den;
                    SmallStorage result(num, den);
                    result.normalize();
                    return result;
                }
            }
        }
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
        if (is_zero(b)) throw std::domain_error("Division by zero");
        if (const auto* sb = std::get_if<SmallStorage>(&b)) {
            SmallStorage s = *sb;
            s.normalize();
            if (s.num == 0) throw std::domain_error("Division by zero");
            absl::int128 num_recip = static_cast<absl::int128>(s.den);
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
            SmallStorage res(-(s->num), s->den);
            res.normalize();
            return res;
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

    // Преобразование high‑prec float в рациональное с заданным запасом точности
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

        if (fractional_part.size() > static_cast<size_t>(digits_needed)) {
            fractional_part = fractional_part.substr(0, digits_needed);
        }

        bool negative = false;
        if (!integer_part.empty() && integer_part[0] == '-') {
            negative = true;
            integer_part = integer_part.substr(1);
        }
        size_t non_zero = integer_part.find_first_not_of('0');
        if (non_zero != std::string::npos) {
            integer_part = integer_part.substr(non_zero);
        }
        else {
            integer_part = "0";
        }
        if (negative && integer_part != "0") {
            integer_part = "-" + integer_part;
        }

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
            if (first_nonzero != std::string::npos) {
                num_str = num_str.substr(first_nonzero);
            }
            else {
                num_str = "0";
            }
        }

        boost::multiprecision::cpp_int num(num_str);
        boost::multiprecision::cpp_int den(1);
        for (size_t i = 0; i < fractional_part.size(); ++i) den *= 10;
        boost::multiprecision::cpp_int g = boost::multiprecision::gcd(num, den);
        num /= g;
        den /= g;
        return BigStorage(num, den);
    }

    // Быстрые трансцендентные функции (через high‑precision float)
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
        if (const auto* s = std::get_if<SmallStorage>(&v)) {
            SmallStorage norm = *s;
            norm.normalize();
            return norm.den == 1;
        }
        if (const auto* b = std::get_if<BigStorage>(&v)) {
            return b->den() == 1;
        }
        return false;
    }

    inline boost::multiprecision::cpp_int get_integer(const Value& v) {
        if (const auto* s = std::get_if<SmallStorage>(&v)) {
            SmallStorage norm = *s;
            norm.normalize();
            return to_cpp_int(norm.num);
        }
        if (const auto* b = std::get_if<BigStorage>(&v)) {
            return b->num();
        }
        throw std::logic_error("get_integer: not an integer");
    }

    inline boost::multiprecision::cpp_int integer_nth_root(const boost::multiprecision::cpp_int& a,
        const boost::multiprecision::cpp_int& n) {
        if (n == 0) return 0;
        if (n == 1) return a;
        if (a == 0) return 0;
        if (a < 0) return 0;

        int n_int = n.convert_to<int>();
        if (n_int > 1000) return 0;

        size_t bits = boost::multiprecision::msb(a) + 1;
        boost::multiprecision::cpp_int high = boost::multiprecision::cpp_int(1) << ((bits + n_int - 1) / n_int);
        high += 1;
        boost::multiprecision::cpp_int low = 1;

        while (low <= high) {
            boost::multiprecision::cpp_int mid = (low + high) / 2;
            boost::multiprecision::cpp_int pow = boost::multiprecision::pow(mid, n_int);
            if (pow == a) return mid;
            if (pow < a) low = mid + 1;
            else high = mid - 1;
        }
        return 0;
    }

    inline std::optional<Value> try_exact_nth_root(const Value& base, const Value& n_val) {
        boost::multiprecision::cpp_int n;
        if (const auto* s = std::get_if<SmallStorage>(&n_val)) {
            SmallStorage norm = *s;
            norm.normalize();
            if (norm.den != 1) return std::nullopt;
            n = to_cpp_int(norm.num);
        }
        else if (const auto* b = std::get_if<BigStorage>(&n_val)) {
            if (b->den() != 1) return std::nullopt;
            n = b->num();
        }
        else {
            return std::nullopt;
        }
        if (n <= 0) return std::nullopt;
        if (n > 1000) return std::nullopt;

        if (const auto* s = std::get_if<SmallStorage>(&base)) {
            SmallStorage s_norm = *s;
            s_norm.normalize();
            if (s_norm.num == 0) return SmallStorage(0);
            bool negative = s_norm.num < 0;
            if (negative && n % 2 == 0) return std::nullopt;
            boost::multiprecision::cpp_int num = to_cpp_int(negative ? -s_norm.num : s_norm.num);
            boost::multiprecision::cpp_int den = to_cpp_int(s_norm.den);
            boost::multiprecision::cpp_int root_num = integer_nth_root(num, n);
            boost::multiprecision::cpp_int root_den = integer_nth_root(den, n);
            if (root_num != 0 && root_den != 0) {
                if (negative) root_num = -root_num;
                return BigStorage(root_num, root_den);
            }
        }
        else if (const auto* b = std::get_if<BigStorage>(&base)) {
            if (b->num() == 0) return SmallStorage(0);
            bool negative = b->num() < 0;
            if (negative && n % 2 == 0) return std::nullopt;
            boost::multiprecision::cpp_int num = negative ? -b->num() : b->num();
            boost::multiprecision::cpp_int den = b->den();
            boost::multiprecision::cpp_int root_num = integer_nth_root(num, n);
            boost::multiprecision::cpp_int root_den = integer_nth_root(den, n);
            if (root_num != 0 && root_den != 0) {
                if (negative) root_num = -root_num;
                return BigStorage(root_num, root_den);
            }
        }
        return std::nullopt;
    }


    // ============================================================================
// Конфигурация медленных методов
// ============================================================================

// Максимальное количество итераций для рядов (экспонента, логарифм, тригонометрия)
// Установлено с огромным запасом; реально для eps=1e-30 требуется ~200-300 итераций.
// Если ряд не сошёлся за это время – что-то пошло не так (например, расходится).
    inline constexpr size_t DEFAULT_MAX_ITER = 1000000;  // миллион

    // Для методов Ньютона (sqrt, nth_root) достаточно 1000, сходимость квадратичная
    inline constexpr size_t NEWTON_MAX_ITER = 1000;

    // Для acos (итерационный метод) тоже достаточно 100
    inline constexpr size_t ACOS_MAX_ITER = 100;
    // ============================================================================
    // Медленные (точные) реализации трансцендентных функций
    // ============================================================================

    inline Value slow_ln2(const Value& eps) {
        Value one = SmallStorage(absl::int128(1));
        Value three = SmallStorage(absl::int128(3));
        Value z = eager_div(one, three);
        Value z2 = eager_mul(z, z);
        Value term = z;
        Value sum = term;
        Value n = one;
        Value two = SmallStorage(absl::int128(2));
        size_t iter = 0;
        const size_t max_iter = DEFAULT_MAX_ITER;
        while (iter < max_iter) {
            term = eager_mul(term, z2);
            n = eager_add(n, two);
            sum = eager_add(sum, eager_div(term, n));
            ++iter;
            if (is_less(eager_abs(term), eps)) break;
        }
        return eager_mul(two, sum);
    }

    inline Value slow_sqrt(const Value& x, const Value& eps) {
        if (is_zero(x)) return SmallStorage(absl::int128(0));
        if (is_negative(x)) throw std::domain_error("sqrt of negative number");

        Value one = SmallStorage(absl::int128(1));
        Value two = SmallStorage(absl::int128(2));
        Value guess = eager_div(x, two);
        Value diff;
        size_t iter = 0;
        const size_t max_iter = NEWTON_MAX_ITER;
        do {
            Value next = eager_div(eager_add(guess, eager_div(x, guess)), two);
            diff = eager_abs(eager_sub(next, guess));
            guess = next;
            ++iter;
            if (iter > max_iter) break;
        } while (is_greater(diff, eps));
        return guess;
    }

    inline Value slow_exp(const Value& x, const Value& eps) {
        Value one = SmallStorage(absl::int128(1));
        Value two = SmallStorage(absl::int128(2));

        int k = 0;
        Value reduced = x;
        while (is_greater(eager_abs(reduced), one)) {
            reduced = eager_div(reduced, two);
            ++k;
        }

        Value sum = one;
        Value term = one;
        Value n = one;
        size_t iter = 0;
        const size_t max_iter = DEFAULT_MAX_ITER;
        while (iter < max_iter) {
            term = eager_mul(term, eager_div(reduced, n));
            sum = eager_add(sum, term);
            n = eager_add(n, one);
            ++iter;
            if (is_less(eager_abs(term), eps)) break;
        }

        Value result = sum;
        for (int i = 0; i < k; ++i) {
            result = eager_mul(result, result);
        }
        return result;
    }

    inline Value slow_log(const Value& x, const Value& eps) {
        if (is_negative(x) || is_zero(x)) throw std::domain_error("log of non-positive");
        Value one = SmallStorage(absl::int128(1));
        Value two = SmallStorage(absl::int128(2));
        Value half = SmallStorage(absl::int128(1), absl::uint128(2));

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

        Value ln2 = slow_ln2(eps);

        Value y = eager_div(eager_sub(m, one), eager_add(m, one));
        Value y2 = eager_mul(y, y);
        Value term = y;
        Value sum = term;
        Value n = one;
        size_t iter = 0;
        const size_t max_iter = DEFAULT_MAX_ITER;
        while (iter < max_iter) {
            term = eager_mul(term, y2);
            n = eager_add(n, two);
            sum = eager_add(sum, eager_div(term, n));
            ++iter;
            if (is_less(eager_abs(term), eps)) break;
        }
        Value ln_m = eager_mul(two, sum);
        return eager_add(ln_m, eager_mul(SmallStorage(absl::int128(k)), ln2));
    }

    inline Value slow_pi(const Value& eps) {
        Value one = SmallStorage(absl::int128(1));
        Value five = SmallStorage(absl::int128(5));
        Value two39 = SmallStorage(absl::int128(239));
        Value sixteen = SmallStorage(absl::int128(16));
        Value four = SmallStorage(absl::int128(4));
        Value two = SmallStorage(absl::int128(2));

        Value a = eager_div(one, five);
        Value a2 = eager_mul(a, a);
        Value term = a;
        Value sum_atan5 = term;
        Value n = one;
        size_t iter = 0;
        const size_t max_iter = DEFAULT_MAX_ITER;
        while (iter < max_iter) {
            term = eager_mul(term, eager_neg(a2));
            n = eager_add(n, two);
            sum_atan5 = eager_add(sum_atan5, eager_div(term, n));
            ++iter;
            if (is_less(eager_abs(term), eps)) break;
        }

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

        return eager_sub(eager_mul(sixteen, sum_atan5), eager_mul(four, sum_atan239));
    }

    inline Value slow_sin(const Value& x, const Value& eps) {
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
        Value term = reduced;
        Value sum = term;
        Value k = one;
        size_t iter = 0;
        const size_t max_iter = DEFAULT_MAX_ITER;
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
        const size_t max_iter = DEFAULT_MAX_ITER;
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

    inline Value slow_acos(const Value& x, const Value& eps) {
        Value one = SmallStorage(absl::int128(1));
        Value two = SmallStorage(absl::int128(2));
        Value pi_val = slow_pi(eps);
        Value half_pi = eager_div(pi_val, two);

        if (is_less(x, eager_neg(one)) || is_greater(x, one))
            throw std::domain_error("acos argument out of [-1,1]");

        Value y;
        if (is_positive(x)) {
            y = eager_mul(half_pi, eager_sub(one, x));
        }
        else {
            y = eager_sub(pi_val, eager_mul(half_pi, eager_add(one, x)));
        }

        size_t iter = 0;
        const size_t max_iter = ACOS_MAX_ITER;
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

    inline Value slow_e(const Value& eps) {
        Value one = SmallStorage(absl::int128(1));
        Value sum = one;
        Value term = one;
        Value n = one;
        size_t iter = 0;
        const size_t max_iter = DEFAULT_MAX_ITER;
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
    // Целочисленное возведение в степень
    // ============================================================================

    inline Value eager_pow_int(const Value& base, const boost::multiprecision::cpp_int& exponent) {
        if (exponent == 0) return SmallStorage(1);
        if (exponent == 1) return base;

        bool negative = exponent < 0;
        boost::multiprecision::cpp_int e = negative ? -exponent : exponent;

        Value result = SmallStorage(1);
        Value b = base;

        while (e > 0) {
            if (e & 1) result = eager_mul(result, b);
            e >>= 1;
            if (e > 0) b = eager_mul(b, b);
        }

        if (negative) {
            return eager_div(SmallStorage(1), result);
        }
        return result;
    }

    // ============================================================================
    // Корень n-й степени (с быстрым путём через high‑precision float)
    // ============================================================================

    inline int compute_extra_digits(const Value& eps, double operation_complexity = 1.0) {
        // operation_complexity: 1 для sqrt, 2 для cbrt, 3 для корня 4-й степени и т.д.
        double eps_double = to_double(eps);
        if (eps_double <= 0) return 30; // fallback
        // Желаемое количество значащих цифр после запятой
        int digits_needed = static_cast<int>(std::ceil(-std::log10(eps_double))) + 2;
        // Добавляем запас, пропорциональный сложности операции
        int safety = static_cast<int>(std::ceil(10.0 * operation_complexity));
        return digits_needed + safety;
    }


    inline Value fast_nth_root(const Value& x, const Value& n, const Value& eps) {
        // Обработка отрицательного основания
        bool x_neg = is_negative(x);
        if (x_neg) {
            // Определяем, является ли показатель чётным целым
            bool n_even = false;
            if (const auto* s = std::get_if<SmallStorage>(&n)) {
                SmallStorage norm = *s;
                norm.normalize();
                if (norm.den == 1 && (norm.num % 2 == 0)) n_even = true;
            }
            else if (const auto* b = std::get_if<BigStorage>(&n)) {
                if (b->den() == 1 && (b->num() % 2 == 0)) n_even = true;
            }
            if (n_even) {
                throw std::domain_error("even root of negative number");
            }
            // Нечётный корень: вычисляем корень от положительного числа и меняем знак
            Value pos_root = fast_nth_root(eager_neg(x), n, eps);
            return eager_neg(pos_root);
        }

        // Проверка на ноль (x >= 0)
        if (is_zero(x)) {
            return SmallStorage(0);
        }

        // Преобразуем n в double для вычисления сложности (только для расчёта extra_digits)
        double complexity = 1.0;
        if (const auto* s = std::get_if<SmallStorage>(&n)) {
            SmallStorage norm = *s;
            norm.normalize();
            if (norm.den == 1) {
                complexity = static_cast<double>(norm.num);
            }
            else {
                // Для нецелых показателей корня (например, n = 2/3) сложность оцениваем как числитель
                complexity = static_cast<double>(norm.num);
            }
        }
        else if (const auto* b = std::get_if<BigStorage>(&n)) {
            if (b->den() == 1) {
                complexity = b->num().convert_to<double>();
            }
            else {
                complexity = b->num().convert_to<double>();
            }
        }

        // Расчёт дополнительных цифр точности
        int extra = compute_extra_digits(eps, complexity);

        // Вычисление через высокоточную плавающую арифметику
        HighPrecFloat fx = to_high_prec(x);
        HighPrecFloat fn = to_high_prec(n);
        HighPrecFloat exponent = 1.0 / fn;
        HighPrecFloat res = pow(fx, exponent);

        // Преобразование обратно в рациональное с гарантированной точностью
        return to_rational_with_eps(res, eps, extra);
    }
    inline Value eager_nth_root(const Value& x, const Value& n, const Value& eps) {
        if (is_zero(n) || is_negative(n)) throw std::domain_error("nth_root: n must be positive");

        boost::multiprecision::cpp_int n_int;
        if (const auto* s = std::get_if<SmallStorage>(&n)) {
            SmallStorage norm = *s;
            norm.normalize();
            if (norm.den != 1) throw std::domain_error("nth_root: n must be integer");
            n_int = to_cpp_int(norm.num);
        }
        else if (const auto* b = std::get_if<BigStorage>(&n)) {
            if (b->den() != 1) throw std::domain_error("nth_root: n must be integer");
            n_int = b->num();
        }
        else {
            throw std::domain_error("nth_root: invalid n");
        }

        if (n_int == 0) throw std::domain_error("nth_root: n must be positive");
        if (n_int == 1) return x;
        if (n_int == 2) return eager_sqrt(x, eps);

        bool n_even = (n_int % 2 == 0);
        if (n_even && is_negative(x)) {
            throw std::domain_error("nth_root: even root of negative number");
        }

        // Точный корень, если есть
        if (auto exact = try_exact_nth_root(x, n)) {
            return *exact;
        }

        // Быстрый путь ТОЛЬКО для квадратного корня (n_int == 2)
        // Для n > 2 быстрый путь через cpp_dec_float_100 недостаточно точен
        if (n_int == 2 && to_double(eps) >= HYBRID_THRESHOLD) {
            return fast_nth_root(x, n, eps);
        }

        // Медленный метод Ньютона для всех n > 2 или маленьких eps
        Value guess;
        if (is_positive(x)) {
            guess = eager_div(x, SmallStorage(2));
        }
        else {
            guess = eager_neg(eager_div(eager_abs(x), SmallStorage(2)));
        }

        Value n_val = n;
        Value n_minus_1 = eager_sub(n_val, SmallStorage(1));

        Value diff;
        size_t iter = 0;
        const size_t max_iter = NEWTON_MAX_ITER;

        do {
            Value pow_n_minus_1 = eager_pow_int(guess, n_int - 1);
            Value next = eager_div(
                eager_add(
                    eager_mul(n_minus_1, guess),
                    eager_div(x, pow_n_minus_1)
                ),
                n_val
            );
            diff = eager_abs(eager_sub(next, guess));
            guess = next;
            ++iter;
            if (iter > max_iter) break;
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
            return base; // 0^positive = 0
        }
        if (is_one(base)) return base;
        if (is_zero(exp)) return SmallStorage(1);

        // Целочисленный показатель?
        bool exp_is_int = false;
        boost::multiprecision::cpp_int exp_num, exp_den;
        if (const auto* s = std::get_if<SmallStorage>(&exp)) {
            SmallStorage norm = *s;
            norm.normalize();
            exp_num = to_cpp_int(norm.num);
            exp_den = to_cpp_int(norm.den);
            if (norm.den == 1) exp_is_int = true;
        }
        else if (const auto* b = std::get_if<BigStorage>(&exp)) {
            exp_num = b->num();
            exp_den = b->den();
            if (exp_den == 1) exp_is_int = true;
        }

        if (exp_is_int) {
            if (exp_num < 0) {
                Value base_recip = eager_div(SmallStorage(1), base);
                return eager_pow_int(base_recip, -exp_num);
            }
            return eager_pow_int(base, exp_num);
        }

        // Рациональный показатель p/q
        boost::multiprecision::cpp_int p = exp_num;
        boost::multiprecision::cpp_int q = exp_den;
        bool negative = (p < 0);
        if (negative) p = -p;

        // Случай 1/n – используем nth_root с запасом точности
        if (p == 1) {
            Value n_val = BigStorage(q);
            // Специальный случай для квадратного корня
            if (q == 2) {
                return eager_sqrt(base, eps);
            }
            // Для остальных корней – запас 1000
            Value internal_eps = eager_div(eps, SmallStorage(1000));
            return eager_nth_root(base, n_val, internal_eps);
        }

        // Общий случай: p > 1, используем exp(p * log(base) / q)
        // Увеличенный запас точности: eps / (p * 1000)
        Value internal_eps;
        if (p == 0) {
            internal_eps = eps;
        }
        else {
            boost::multiprecision::cpp_int divisor = p * 1000;
            Value divisor_val = BigStorage(divisor);
            internal_eps = eager_div(eps, divisor_val);
        }

        Value log_base = eager_log(base, internal_eps);
        Value p_val = BigStorage(negative ? -p : p);
        Value p_log = eager_mul(p_val, log_base);
        Value q_val = BigStorage(q);
        Value p_log_div_q = eager_div(p_log, q_val);
        Value result = eager_exp(p_log_div_q, internal_eps);

        return result;
    }

    // ============================================================================
    // Eager dispatchers (выбирают быстрый или медленный путь)
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