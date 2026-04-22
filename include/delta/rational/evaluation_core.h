// evaluation_core.h
// Адаптирован под единый тип Value (boost::multiprecision::number<rational_adaptor<...>>)
// Все операции выполняются напрямую через операторы Value, без ветвлений Small/Big.
// ----------------------------------------------------------------------------
// Версия 2.1 – устранены явные локальные переменные для констант (1, 2, 3...),
// используется непосредственное преобразование целых литералов в Value.
// ----------------------------------------------------------------------------

#pragma once

#include "storage.h"
#include "utils.h"
#include <boost/math/constants/constants.hpp>

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
    Value eager_pow_int(const Value& base, const dumb_int& exponent);
    // Series (rational) implementations
    Value series_sqrt(const Value& x, const Value& eps);
    Value series_exp(const Value& x, const Value& eps);
    Value series_log(const Value& x, const Value& eps);
    Value series_sin(const Value& x, const Value& eps);
    Value series_cos(const Value& x, const Value& eps);
    Value series_acos(const Value& x, const Value& eps);
    Value series_pi(const Value& eps);
    Value series_e(const Value& eps);
    Value series_ln2(const Value& eps);

    // ----------------------------------------------------------------------------
    // Вспомогательные предикаты (используют версии из storage.h)
    // ----------------------------------------------------------------------------
    inline bool is_less(const Value& a, const Value& b) { return a < b; }
    inline bool is_greater(const Value& a, const Value& b) { return a > b; }

    constexpr double HYBRID_THRESHOLD = 1e-35;

    // ============================================================================
    // Арифметические операции теперь напрямую через операторы Value
    // ============================================================================
    inline Value eager_abs(const Value& a) {
        return is_negative(a) ? -a : a;
    }

    // ============================================================================
    // High‑precision floating‑point helpers (float path)
    // ============================================================================

    using HighPrecFloat = boost::multiprecision::cpp_dec_float_100;

    inline HighPrecFloat to_high_prec(const Value& v) {
        return v.convert_to<HighPrecFloat>();
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
        return Value(num, den);
    }

    // Float-реализации для тех функций, где они дают выигрыш при грубых eps
    inline Value float_exp(const Value& x, const Value& eps) {
        return to_rational_with_eps(exp(to_high_prec(x)), eps);
    }
    inline Value float_sin(const Value& x, const Value& eps) {
        if (is_negative(x)) return -float_sin(-x, eps);
        return to_rational_with_eps(sin(to_high_prec(x)), eps);
    }
    inline Value float_cos(const Value& x, const Value& eps) {
        Value positive_x = is_negative(x) ? -x : x;
        return to_rational_with_eps(cos(to_high_prec(positive_x)), eps);
    }
    inline Value float_acos(const Value& x, const Value& eps) {
        HighPrecFloat fx = to_high_prec(x);
        if (fx < -1 || fx > 1) throw std::domain_error("acos argument out of [-1,1]");
        return to_rational_with_eps(acos(fx), eps);
    }
    inline Value float_pi(const Value& eps) {
        HighPrecFloat pi_val = boost::math::constants::pi<HighPrecFloat>();
        return to_rational_with_eps(pi_val, eps);
    }

    // ============================================================================
    // Точные корни (целочисленные)
    // ============================================================================
    inline bool is_integer(const Value& v) {
        return denominator(v) == 1;
    }

    inline dumb_int get_integer(const Value& v) {
        return numerator(v);
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
        if (!is_integer(n_val)) return std::nullopt;
        dumb_int n = numerator(n_val);
        if (n <= 0 || n > 1000) return std::nullopt;

        if (is_zero(base)) return Value(0);
        bool negative = is_negative(base);
        if (negative && n % 2 == 0) return std::nullopt;

        dumb_int num = numerator(base);
        dumb_int den = denominator(base);
        if (negative) num = -num;

        dumb_int root_num = integer_nth_root(num, n);
        dumb_int root_den = integer_nth_root(den, n);
        if (root_num != 0 && root_den != 0) {
            if (negative) root_num = -root_num;
            return Value(root_num) / Value(root_den);
        }
        return std::nullopt;
    }

    // ============================================================================
    // Конфигурация series-методов
    // ============================================================================
    constexpr size_t DEFAULT_MAX_ITER = 1000000;
    constexpr size_t NEWTON_MAX_ITER = 1000;
    constexpr size_t ACOS_MAX_ITER = 100;

    // ============================================================================
    // Series (рациональные) реализации трансцендентных функций
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

    inline Value series_sqrt(const Value& x, const Value& eps) {
        if (is_zero(x)) return Value(0);
        if (is_one(x)) return Value(1);
        if (is_negative(x)) throw std::domain_error("sqrt of negative number");

        // Быстрая оценка через double — нужна ли редукция?
        double x_approx = to_double(x);
        const double SCALE_LOW = 1e-8;
        const double SCALE_HIGH = 1e8;
        bool need_scaling = (x_approx < SCALE_LOW || x_approx > SCALE_HIGH);

        Value m = x;
        int k = 0;
        if (need_scaling) {
            while (m > 1) {
                m /= 4;
                ++k;
            }
            while (m < Value(1) / 4) {
                m *= 4;
                --k;
            }
        }

        Value internal_eps = eps;
        if (need_scaling) {
            for (int i = 0; i < std::abs(k); ++i) {
                internal_eps /= 2;
            }
        }

        // Начальное приближение через double для ускорения сходимости
        double m_approx = to_double(m);
        Value guess = Value(std::sqrt(m_approx));

        Value diff;
        size_t iter = 0;
        do {
            Value next = (guess + m / guess) / 2;
            diff = eager_abs(next - guess);
            guess = next;
            ++iter;
            if (iter > NEWTON_MAX_ITER) break;
        } while (diff > internal_eps);

        if (need_scaling) {
            Value result = guess;
            if (k > 0) {
                for (int i = 0; i < k; ++i) result *= 2;
            }
            else if (k < 0) {
                for (int i = 0; i < -k; ++i) result /= 2;
            }
            return result;
        }
        return guess;
    }
    // ----------------------------------------------------------------------------
    // Series (рациональная) реализация exp
    // ----------------------------------------------------------------------------
    // Порог для редукции в series_exp: если |x| > 2, применяем редукцию
    constexpr double SERIES_EXP_REDUCE_THRESHOLD = 2.0;
    inline Value series_exp(const Value& x, const Value& eps) {
        if (is_zero(x)) return Value(1);
        // Для отрицательных аргументов: exp(x) = 1 / exp(-x)
        if (is_negative(x)) return Value(1) / series_exp(-x, eps);

        double x_d = to_double(x);

        // Если аргумент мал, ряд сходится быстро без редукции
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

        // Редукция аргумента: exp(x) = (exp(x / 2^k))^{2^k}
        int k = 0;
        Value reduced = x;
        while (reduced > SERIES_EXP_REDUCE_THRESHOLD) {
            reduced /= 2;
            ++k;
        }

        // Оценка двоичного порядка величины exp(x) через double
        double exp_est = std::exp(x_d);
        int exp_bits;
        std::frexp(exp_est, &exp_bits);   // exp_bits — двоичный порядок (смещённый)

        // Масштабирование eps: нужно разделить на 2^{exp_bits + k + запас}
        Value internal_eps = eps;
        int total_shift = exp_bits + k + 2;   // +2 для надёжности
        for (int i = 0; i < total_shift; ++i) {
            internal_eps /= 2;
        }

        // Вычисление ряда для reduced
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

        // Возведение в степень 2^k (рациональное, целочисленное)
        dumb_int exponent = dumb_int(1) << k;
        return eager_pow_int(sum, exponent);
    }
    inline Value series_log(const Value& x, const Value& eps) {
        if (is_negative(x) || is_zero(x)) throw std::domain_error("log of non-positive");
        int k = 0;
        Value m = x;
        while (m > 2) {
            m /= 2;
            ++k;
        }
        while (m < Value(1) / 2) {
            m *= 2;
            --k;
        }
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

    inline Value series_pi(const Value& eps) {
        Value a = Value(1) / 5, a2 = a * a;
        Value term = a, sum_atan5 = term;
        Value n = 1;
        size_t iter = 0;
        while (iter < DEFAULT_MAX_ITER) {
            term *= -a2;
            n += 2;
            sum_atan5 += term / n;
            ++iter;
            if (term < eps && term > -eps) break;
        }
        Value b = Value(1) / 239, b2 = b * b;
        term = b;
        Value sum_atan239 = term;
        n = 1;
        iter = 0;
        while (iter < DEFAULT_MAX_ITER) {
            term *= -b2;
            n += 2;
            sum_atan239 += term / n;
            ++iter;
            if (term < eps && term > -eps) break;
        }
        return 16 * sum_atan5 - 4 * sum_atan239;
    }

    inline Value series_sin(const Value& x, const Value& eps) {
        if (is_negative(x)) return -series_sin(-x, eps);
        Value pi_val = series_pi(eps);
        Value twopi = pi_val * 2;
        Value reduced = x;
        while (eager_abs(reduced) > pi_val) {
            if (reduced > 0) reduced -= twopi;
            else reduced += twopi;
        }
        Value x2 = reduced * reduced;
        Value term = reduced, sum = term;
        Value k = 1;
        size_t iter = 0;
        while (iter < DEFAULT_MAX_ITER) {
            term *= -x2;
            term /= (2 * k) * (2 * k + 1);
            sum += term;
            k += 1;
            if (term < eps && term > -eps) break;
            ++iter;
        }
        return sum;
    }

    inline Value series_cos(const Value& x, const Value& eps) {
        Value pi_val = series_pi(eps);
        Value twopi = pi_val * 2;
        Value reduced = is_negative(x) ? -x : x;
        while (eager_abs(reduced) > pi_val) {
            if (reduced > 0) reduced -= twopi;
            else reduced += twopi;
        }
        Value x2 = reduced * reduced;
        Value term = 1, sum = term;
        Value k = 1;
        size_t iter = 0;
        while (iter < DEFAULT_MAX_ITER) {
            term *= -x2;
            term /= (2 * k) * (2 * k - 1);
            sum += term;
            k += 1;
            if (term < eps && term > -eps) break;
            ++iter;
        }
        return sum;
    }

    inline Value series_acos(const Value& x, const Value& eps) {
        Value pi_val = series_pi(eps);
        Value half_pi = pi_val / 2;
        if (x < -1 || x > 1)
            throw std::domain_error("acos argument out of [-1,1]");
        Value y = (x > 0) ? half_pi * (1 - x) : pi_val - half_pi * (1 + x);
        size_t iter = 0;
        while (iter < ACOS_MAX_ITER) {
            Value cos_y = series_cos(y, eps);
            Value sin_y = series_sin(y, eps);
            if (is_zero(sin_y)) break;
            Value delta = (cos_y - x) / sin_y;
            y -= delta;
            if (eager_abs(delta) < eps) break;
            ++iter;
        }
        return y;
    }

    inline Value series_e(const Value& eps) {
        Value sum = 1, term = 1;
        Value n = 1;
        size_t iter = 0;
        while (iter < DEFAULT_MAX_ITER) {
            term /= n;
            sum += term;
            n += 1;
            ++iter;
            if (term < eps) break;
        }
        return sum;
    }

    // ============================================================================
    // Целочисленное возведение в степень
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

    inline Value float_nth_root(const Value& x, const Value& n, const Value& eps) {
        bool x_neg = is_negative(x);
        if (x_neg) {
            bool n_even = false;
            if (is_integer(n)) {
                dumb_int n_int = numerator(n);
                if (n_int % 2 == 0) n_even = true;
            }
            if (n_even) throw std::domain_error("even root of negative number");
            return -float_nth_root(-x, n, eps);
        }
        if (is_zero(x)) return Value(0);
        double complexity = 1.0;
        if (is_integer(n)) {
            complexity = static_cast<double>(numerator(n));
        }
        int extra = compute_extra_digits(eps, complexity);
        HighPrecFloat fx = to_high_prec(x);
        HighPrecFloat fn = to_high_prec(n);
        HighPrecFloat res = pow(fx, 1.0 / fn);
        return to_rational_with_eps(res, eps, extra);
    }

    inline Value eager_nth_root(const Value& x, const Value& n, const Value& eps) {
        if (is_zero(n) || is_negative(n)) throw std::domain_error("nth_root: n must be positive");
        if (!is_integer(n)) throw std::domain_error("nth_root: n must be integer");
        dumb_int n_int = numerator(n);
        if (n_int == 0) throw std::domain_error("nth_root: n must be positive");
        if (n_int == 1) return x;
        if (n_int == 2) return eager_sqrt(x, eps);
        if (n_int % 2 == 0 && is_negative(x))
            throw std::domain_error("nth_root: even root of negative number");
        if (auto exact = try_exact_nth_root(x, n)) return *exact;
        if (n_int == 2 && to_double(eps) >= HYBRID_THRESHOLD)
            return float_nth_root(x, n, eps);
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
        if (is_zero(exp)) return Value(1);

        bool exp_is_int = is_integer(exp);
        dumb_int exp_num = numerator(exp);
        dumb_int exp_den = denominator(exp);

        if (exp_is_int) {
            if (exp_num < 0) {
                Value base_recip = Value(1) / base;
                return eager_pow_int(base_recip, -exp_num);
            }
            return eager_pow_int(base, exp_num);
        }

        dumb_int p = exp_num, q = exp_den;
        bool negative = (p < 0);
        if (negative) p = -p;

        if (p == 1) {
            Value n_val = Value(q);
            if (q == 2) return eager_sqrt(base, eps);
            Value internal_eps = eps / 1000;
            return eager_nth_root(base, n_val, internal_eps);
        }

        Value internal_eps = (p == 0) ? eps : eps / Value(p * 1000);
        Value log_base = eager_log(base, internal_eps);
        Value p_val = negative ? Value(-p) : Value(p);
        Value p_log = p_val * log_base;
        Value q_val = Value(q);
        Value p_log_div_q = p_log / q_val;
        return eager_exp(p_log_div_q, internal_eps);
    }

    // ============================================================================
    // Eager dispatchers
    // ============================================================================
    inline Value eager_sqrt(const Value& x, const Value& eps) {
        // Сначала пробуем извлечь точный квадратный корень
        if (auto exact = try_exact_nth_root(x, Value(2))) {
            return *exact;
        }
        // Float-путь для sqrt удалён, т.к. бенчмарки показали,
        // что конвертация в cpp_dec_float_100 и обратно медленнее
        // чистого рационального метода Ньютона при любых точностях.
        return series_sqrt(x, eps);
    }

    // Порог аргумента для exp, после которого float-путь теряет точность
    constexpr double EXP_FLOAT_ARG_THRESHOLD = 20.0;
    inline Value eager_exp(const Value& x, const Value& eps) {
        double eps_d = to_double(eps);
        double x_d = std::abs(to_double(x));
        // float-путь быстр, но при больших аргументах теряет относительную точность.
        // Поэтому используем его только если и точность не слишком высока, и аргумент не слишком велик.
        if (eps_d >= HYBRID_THRESHOLD && x_d <= EXP_FLOAT_ARG_THRESHOLD) {
            return float_exp(x, eps);
        }
        return series_exp(x, eps);
    }

    inline Value eager_log(const Value& x, const Value& eps) {
        // Float-путь для log удалён по той же причине, что и для sqrt.
        return series_log(x, eps);
    }

    inline Value eager_sin(const Value& x, const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? float_sin(x, eps) : series_sin(x, eps);
    }

    inline Value eager_cos(const Value& x, const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? float_cos(x, eps) : series_cos(x, eps);
    }

    inline Value eager_acos(const Value& x, const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? float_acos(x, eps) : series_acos(x, eps);
    }

    inline Value eager_pi(const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? float_pi(eps) : series_pi(eps);
    }

    inline Value eager_e(const Value& eps) {
        // Float-путь для e также не даёт выигрыша, всегда используем series.
        return series_e(eps);
    }

} // namespace delta::internal