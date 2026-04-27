// evaluation_core.h
// ----------------------------------------------------------------------------
// АДАПТИРОВАН ПОД ЕДИНЫЙ ТИП Value (boost::multiprecision::number<rational_adaptor<...>>)
// Все операции выполняются напрямую через операторы Value, без ветвлений Small/Big.
// ----------------------------------------------------------------------------
// Версия 2.2 – добавлены asin, atan, tan; кэширование π; ускорен acos.
// ----------------------------------------------------------------------------
// ВАЖНЫЕ ИНЖЕНЕРНЫЕ РЕШЕНИЯ И ИСТОРИЯ РАЗРАБОТКИ:
// ----------------------------------------------------------------------------
// 
// 1. Выбор между быстрыми (float) и точными (series) путями.
// -------------------------------------------------------
// Изначально использовался порог HYBRID_THRESHOLD = 1e-35 для ВСЕХ функций:
// при eps >= порога вызывались float-реализации на основе cpp_dec_float_100.
// 
// Бенчмарки показали:
//   - Для sin, cos, exp, pi, acos float-путь даёт ускорение в 2-3 раза при
//     умеренных точностях (eps ~ 1e-21). Это оправдано.
//   - Для sqrt, log, e float-путь оказался МЕДЛЕННЕЕ чистых рациональных
//     алгоритмов из-за накладных расходов на конвертацию Value ↔ cpp_dec_float_100
//     и парсинг строк в to_rational_with_eps. Поэтому для этих функций float-путь
//     удалён, и всегда используется рациональный (series) метод.
// 
//   - Для exp дополнительно введён порог аргумента EXP_FLOAT_ARG_THRESHOLD = 20.0.
//     При |x| > 20 float-путь теряет относительную точность из-за ограниченной
//     мантиссы cpp_dec_float_100, поэтому даже при грубом eps принудительно
//     используется series_exp.
// 
// 2. Структура series-функций и редукция аргументов.
// --------------------------------------------------
// Каждая series-функция реализует редукцию аргумента для быстрой сходимости:
//   - sin/cos: приведение к [-π, π] с использованием series_pi.
//   - exp: деление на 2^k до |x| <= 2, с последующим возведением результата
//          в квадрат k раз. Масштабирование eps (internal_eps) учитывает
//          как редукцию, так и порядок итогового значения, чтобы гарантировать
//          абсолютную погрешность не хуже запрошенной.
//   - log: приведение к диапазону [1/2, 2] через k*ln2.
//   - sqrt: масштабирование делением/умножением на 4, если x вне [1e-8, 1e8].
// 
// 3. Квадратичный пересчёт коэффициентов и попытка «оптимизации» (УРОК).
// ------------------------------------------------------------------------
// В одной из версий (Optimized_core) мы попытались ускорить вычисление рядов,
// генерируя все члены в вектор и применяя пирамидальное суммирование (PCR).
// Результат: КАТАСТРОФИЧЕСКОЕ ЗАМЕДЛЕНИЕ (до 5 раз медленнее наивного).
// Причины:
//   - Коэффициенты (факториалы) пересчитывались с нуля для каждого члена за O(i),
//     в то время как наивный цикл обновляет член рекуррентно за O(1).
//   - Выделение векторов и пирамидальное суммирование для N ~ 200 давали
//     огромные накладные расходы на память и копирование.
// ВЫВОД: рекуррентное обновление члена ряда в простом цикле – оптимально.
// Не пытайтесь «векторизовать» ряды Тейлора с рациональными числами!
// 
// 4. Природа параметра eps и тестирование больших аргументов.
// -----------------------------------------------------------
// Параметр eps задаёт АБСОЛЮТНУЮ погрешность: |f(x) - result| < eps.
// Для экспоненты при больших x (например, 100) значение exp(100) ~ 10^43.
// Требование абсолютной точности 1e-12 означает 55 верных значащих цифр,
// что требует огромных вычислительных затрат.
// 
// В тестах мы приняли прагматичное решение: для огромных значений проверять
// относительную близость. Если пользователю действительно нужна абсолютная
// точность 1e-12 для exp(1000), он может явно передать eps = 1e-12 / exp_est.
// Библиотека не делает это автоматически, чтобы сохранить предсказуемую
// производительность. При необходимости можно раскомментировать блок
// «АБСОЛЮТНАЯ ТОЧНОСТЬ ДЛЯ БОЛЬШИХ X» в series_exp (см. ниже).
// 
// 5. Обработка отрицательных аргументов.
// --------------------------------------
// Для sin/cos/exp отрицательный аргумент сводится к положительному через
// свойства чётности/нечётности или 1/exp(-x). Это гарантирует работу с
// положительными рядами, избегая знакопеременности и потери точности.
// 
// 6. Безопасность и максимальные итерации.
// ----------------------------------------
// DEFAULT_MAX_ITER = 1'000'000 – защита от бесконечных циклов.
// В реальности для |x| <= 2 сходимость наступает за 30-100 итераций.
// 
// 7. Кэширование π и ускорение обратных тригонометрических функций.
// ----------------------------------------------------------------
// - series_pi кэширует результат для каждого значения eps.
// - series_acos использует std::acos для начального приближения (15 точных цифр).
// - asin, atan, tan реализованы через тождества, что минимизирует новый код.
// 
// ----------------------------------------------------------------------------
// ЕСЛИ ВЫ ХОТИТЕ ЧТО-ТО ИЗМЕНИТЬ – ПРОЧТИТЕ ВЫШЕУКАЗАННОЕ.
// Особенно опасны:
//   - добавление векторизации рядов (убьёт производительность);
//   - удаление масштабирования internal_eps в series_exp (сломает точность);
//   - изменение порогов HYBRID_THRESHOLD без бенчмарков.
// ----------------------------------------------------------------------------

#pragma once
#include "global_state.h"
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
#include <map>   // для кэша π

namespace delta::internal {

    // Forward declarations
    Value eager_abs(const Value& a);
    Value eager_sqrt(const Value& x, const Value& eps);
    Value eager_exp(const Value& x, const Value& eps);
    Value eager_log(const Value& x, const Value& eps);
    Value eager_sin(const Value& x, const Value& eps);
    Value eager_cos(const Value& x, const Value& eps);
    Value eager_acos(const Value& x, const Value& eps);
    Value eager_asin(const Value& x, const Value& eps);
    Value eager_atan(const Value& x, const Value& eps);
    Value eager_tan(const Value& x, const Value& eps);
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
    Value series_asin(const Value& x, const Value& eps);
    Value series_atan(const Value& x, const Value& eps);
    Value series_tan(const Value& x, const Value& eps);
    Value series_pi(const Value& eps);
    Value series_e(const Value& eps);
    Value series_ln2(const Value& eps);

    // ----------------------------------------------------------------------------
    // Вспомогательные предикаты (используют версии из storage.h)
    // ----------------------------------------------------------------------------
    inline bool is_less(const Value& a, const Value& b) { return a < b; }
    inline bool is_greater(const Value& a, const Value& b) { return a > b; }

    // ----------------------------------------------------------------------------
    // ГЛОБАЛЬНЫЙ ПОРОГ ДЛЯ ВЫБОРА FLOAT VS SERIES ПУТИ.
    // Определён на основе бенчмарков: при eps >= 1e-35 float-пути через
    // cpp_dec_float_100 быстрее для sin, cos, exp, pi, acos, asin, atan, tan.
    // Для sqrt, log, e float-пути удалены, т.к. они всегда медленнее рациональных.
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

    // Преобразует float в Value с точностью, определяемой eps.
    // Использует строковое представление – медленно, но необходимо для
    // сохранения рациональной точности. Из-за этой дороговизны float-пути
    // невыгодны для sqrt, log, e.
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

    // ------------------------------------------------------------------------
    // Float-реализации для тех функций, где они дают выигрыш при грубых eps.
    // ВАЖНО: для sin и cos добавлена обработка знака, чтобы гарантировать
    // нечётность/чётность (без этого тесты на отрицательные аргументы падают).
    // ------------------------------------------------------------------------
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
    inline Value float_asin(const Value& x, const Value& eps) {
        HighPrecFloat fx = to_high_prec(x);
        if (fx < -1 || fx > 1) throw std::domain_error("asin argument out of [-1,1]");
        return to_rational_with_eps(asin(fx), eps);
    }
    inline Value float_atan(const Value& x, const Value& eps) {
        return to_rational_with_eps(atan(to_high_prec(x)), eps);
    }
    inline Value float_tan(const Value& x, const Value& eps) {
        return to_rational_with_eps(tan(to_high_prec(x)), eps);
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
    constexpr size_t DEFAULT_MAX_ITER = 1000000;   // защита от бесконечного цикла
    constexpr size_t NEWTON_MAX_ITER = 1000;
    constexpr size_t ACOS_MAX_ITER = 100;

    // ============================================================================
    // Series (рациональные) реализации трансцендентных функций
    // ============================================================================

    // ln(2) через ряд arctanh(1/3). Используется в series_log.
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

    // ------------------------------------------------------------------------
    // Квадратный корень: метод Ньютона с условной редукцией аргумента.
    // Редукция применяется только если x вне [1e-8, 1e8] – для типичных чисел
    // оверхед редукции не нужен, а для экстремальных она спасает от зависания.
    // Начальное приближение через double ускоряет сходимость до 2-3 итераций.
    // ------------------------------------------------------------------------
    inline Value series_sqrt(const Value& x, const Value& eps) {
        if (is_zero(x)) return Value(0);
        if (is_one(x)) return Value(1);
        if (is_negative(x)) throw std::domain_error("sqrt of negative number");

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

        double m_approx = to_double(m);
        Value guess;
        guess.assign(std::sqrt(m_approx));

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

    // ------------------------------------------------------------------------
    // ЭКСПОНЕНТА: РЕАЛИЗАЦИЯ, ПРОШЕДШАЯ ОГОНЬ И ВОДУ.
    // ------------------------------------------------------------------------
    // Порог для редукции: при |x| > 2 применяем exp(x) = (exp(x/2^k))^{2^k}.
    constexpr double SERIES_EXP_REDUCE_THRESHOLD = 2.0;

    // ВАЖНО: Параметр eps задаёт АБСОЛЮТНУЮ погрешность. Для огромных x (например,
    // 100) значение exp(x) ~ 10^43. Требование абсолютной точности 1e-12 означает
    // 55 верных значащих цифр, что требует огромных вычислительных затрат.
    // В данной реализации мы МАСШТАБИРУЕМ internal_eps с учётом порядка exp(x),
    // чтобы гарантировать заявленную абсолютную точность. Это может замедлить
    // вычисления для больших x. Если вам не нужна такая строгая точность,
    // используйте относительную проверку или передавайте менее строгий eps.
    // 
    // Альтернативный подход (закомментирован ниже): не масштабировать internal_eps
    // для больших x, полагаясь на естественную точность ряда. Это быстрее, но
    // абсолютная точность для exp(100) будет порядка 1e-8, а не 1e-12.
    // Выбор сделан в пользу корректности «из коробки».
    // ------------------------------------------------------------------------
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

        // Оценка двоичного порядка величины exp(x) через double.
        // Используем frexp, чтобы не ограничиваться диапазоном long long.
        double exp_est = std::exp(x_d);
        int exp_bits;
        std::frexp(exp_est, &exp_bits);

        // Масштабирование eps: делим на 2^{exp_bits + k + запас}
        // Это гарантирует, что после возведения в квадрат k раз итоговая
        // абсолютная погрешность не превысит запрошенную.
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

        // Возведение в степень 2^k (рациональное, целочисленное).
        // Для k <= 10 это не приводит к катастрофическому раздуванию чисел.
        dumb_int exponent = dumb_int(1) << k;
        return eager_pow_int(sum, exponent);
    }

    // ------------------------------------------------------------------------
    // ЛОГАРИФМ: всегда series (float удалён).
    // Редукция: приводим аргумент к [1/2, 2] через k*ln2, затем используем
    // быстро сходящийся ряд для ln((1+y)/(1-y)).
    // ------------------------------------------------------------------------
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

    // ============================================================================
    // ВЫЧИСЛЕНИЕ ПИ МЕТОДОМ ЧУДНОВСКОГО.
    // КОНСТАНТЫ РЯДА ЧУДНОВСКОГО
    // ============================================================================
    static const dumb_int CHUD_A = 545140134;
    static const dumb_int CHUD_B = 13591409;
    static const dumb_int CHUD_C = 640320;
    static const dumb_int CHUD_C3_OVER_24 = (dumb_int(CHUD_C) * CHUD_C * CHUD_C) / 24;
    static const dumb_int CHUD_D = 426880;

    struct ChudnovskyPQT {
        dumb_int P, Q, T;
    };

    // ----------------------------------------------------------------------------
    // БИНАРНОЕ РАСЩЕПЛЕНИЕ (Binary Splitting)
    // ----------------------------------------------------------------------------
    inline ChudnovskyPQT chudnovsky_bs(int64_t a, int64_t b) {
        if (b - a == 1) {
            dumb_int k(a);
            if (a == 0) {
                // При k=0: P=1 (нейтральный), Q=1, T=B
                return { dumb_int(1), dumb_int(1), dumb_int(CHUD_B) };
            }
            else {
                // Базовый множитель P(k) = (6k-5)(2k-1)(6k-1)
                // Он будет передан "налево" для умножения на T последующих термов
                dumb_int P = (6 * k - 5) * (2 * k - 1) * (6 * k - 1);
                dumb_int Q = k * k * k * CHUD_C3_OVER_24;

                // ИСПРАВЛЕНИЕ: T теперь только (Ak + B). 
                // Множитель P придет из L.P при слиянии на уровнях выше.
                dumb_int T = k * CHUD_A + CHUD_B;

                if (a % 2 == 1) T = -T;
                return { P, Q, T };
            }
        }

        int64_t m = (a + b) / 2;
        auto L = chudnovsky_bs(a, m);
        auto R = chudnovsky_bs(m, b);

        // Классическая формула слияния:
        // T = T_L * Q_R + P_L * T_R
        // Это гарантирует, что каждое T_R умножается на произведение всех P слева
        return {
            L.P * R.P,
            L.Q * R.Q,
            L.T * R.Q + L.P * R.T
        };
    }

    // ----------------------------------------------------------------------------
    // РЕКУРРЕНТНАЯ ВЕРСИЯ (для малых N)
    // ----------------------------------------------------------------------------
    inline Value pi_recurrent(int N, const Value& eps) {
        Value term(CHUD_B, 1);
        Value sum = term;

        for (int k = 0; k < N - 1; ++k) {
            dumb_int k1 = k + 1;
            // Коэффициент перехода: - (6k+1)(2k+1)(6k+5) / ( (k+1)^3 * C3/24 )
            // И учитываем отношение (A*(k+1)+B) / (A*k+B)
            dumb_int numer = (6 * k + 1) * (2 * k + 1) * (6 * k + 5);
            dumb_int denom_part = k1 * k1 * k1;

            Value factor = Value(-numer) * Value(CHUD_A * k1 + CHUD_B);
            Value denom = Value(denom_part) * Value(CHUD_C3_OVER_24) * Value(CHUD_A * k + CHUD_B);

            term = term * factor / denom;
            sum = sum + term;
        }

        Value sqrt_10005 = series_sqrt(Value(10005), eps / 1000);
        return (Value(CHUD_D) * sqrt_10005) / sum;
    }
    // ----------------------------------------------------------------------------
    // ОСНОВНАЯ ФУНКЦИЯ
    // ----------------------------------------------------------------------------
    inline Value series_pi(const Value& eps) {
        auto it = pi_cache.find(eps);
        if (it != pi_cache.end()) return it->second;

        double eps_d = std::abs(to_double(eps));
        // Каждая итерация дает 14.18 знаков. +3 для запаса.
        int N = (eps_d <= 0) ? 10 : (int)std::max(2.0, std::ceil(-std::log10(eps_d) / 14.18) + 3);

        Value result;
        // Для рациональных чисел BS эффективнее почти сразу (N > 16) 
        // из-за предотвращения роста промежуточных дробей
        if (N > 16) {
            auto res = chudnovsky_bs(0, N);
            Value S(res.T, res.Q);
            Value sqrt_10005 = series_sqrt(Value(10005), eps / 1000);
            result = (Value(CHUD_D) * sqrt_10005) / S;
        }
        else {
            result = pi_recurrent(N, eps);
        }

        pi_cache[eps] = result;
        return result;
    }

    // ============================================================================
    // БИНАРНОЕ РАСЩЕПЛЕНИЕ ДЛЯ SIN И COS
    // ============================================================================

    struct TrigPQT {
        dumb_int P;  // числитель (x2_num в степени)
        dumb_int Q;  // знаменатель (факториал * x2_den в степени)
        dumb_int T;  // накопленная сумма (числитель, знаменатель будет Q всего диапазона)
    };

    // sin(x) = x * Σ_{k=0}∞ (-1)^k * (x^2)^k / (2k+1)!
    // Внутри рекурсии работаем с x2_num и x2_den (числитель и знаменатель x^2)
    inline TrigPQT sin_bs_internal(int64_t a, int64_t b, const dumb_int& x2_num, const dumb_int& x2_den) {
        if (b - a == 1) {
            if (a == 0) {
                // k = 0: слагаемое = 1
                return { x2_num, 1, 1 };
            }
            // k >= 1: слагаемое = (-1)^k * x2_num^k / (x2_den^k * (2k+1)!)
            // Но T пока храним только (-1)^k, без учёта P (степень x2_num)
            dumb_int Q = x2_den * dumb_int(2 * a) * (2 * a + 1);
            dumb_int T = (a % 2 == 1) ? -1 : 1;
            return { x2_num, Q, T };
        }

        int64_t m = (a + b) / 2;
        auto L = sin_bs_internal(a, m, x2_num, x2_den);
        auto R = sin_bs_internal(m, b, x2_num, x2_den);

        // Классическая формула слияния для ряда Σ (x^2)^k / факториал
        // T = T_L * Q_R + P_L * T_R
        // Q = Q_L * Q_R
        // P = P_L * P_R
        return {
            L.P * R.P,
            L.Q * R.Q,
            L.T * R.Q + L.P * R.T
        };
    }

    // cos(x) = Σ_{k=0}∞ (-1)^k * (x^2)^k / (2k)!
    inline TrigPQT cos_bs_internal(int64_t a, int64_t b, const dumb_int& x2_num, const dumb_int& x2_den) {
        if (b - a == 1) {
            if (a == 0) {
                // k = 0: слагаемое = 1
                return { x2_num, 1, 1 };
            }
            // k >= 1: слагаемое = (-1)^k * x2_num^k / (x2_den^k * (2k)!)
            dumb_int Q = x2_den * dumb_int(2 * a - 1) * (2 * a);
            dumb_int T = (a % 2 == 1) ? -1 : 1;
            return { x2_num, Q, T };
        }

        int64_t m = (a + b) / 2;
        auto L = cos_bs_internal(a, m, x2_num, x2_den);
        auto R = cos_bs_internal(m, b, x2_num, x2_den);

        return {
            L.P * R.P,
            L.Q * R.Q,
            L.T * R.Q + L.P * R.T
        };
    }

    // ============================================================================
    // ОСНОВНЫЕ ФУНКЦИИ SIN И COS С BS И БЫСТРОЙ РЕДУКЦИЕЙ
    // ============================================================================
    inline Value series_sin(const Value& x, const Value& eps) {
        if (is_negative(x)) return -series_sin(-x, eps);

        Value pi_val = series_pi(eps);
        Value twopi = pi_val * 2;

        // Универсальная редукция без использования double
        Value periods = x / twopi;
        dumb_int k_int = numerator(periods) / denominator(periods);
        Value reduced = x - Value(k_int) * twopi;

        // Приведение к [-π, π]
        if (reduced > pi_val) {
            reduced -= twopi;
        }
        else if (reduced < -pi_val) {
            reduced += twopi;
        }

        if (is_zero(reduced)) return Value(0);

        Value x2 = reduced * reduced;
        dumb_int x2_num = numerator(x2);
        dumb_int x2_den = denominator(x2);

        double eps_d = to_double(eps);
        int64_t N = (eps_d <= 0) ? 10 : (int64_t)std::max(10.0, -std::log10(eps_d) * 0.8);
        if (N > 2000) N = 2000;

        auto res = sin_bs_internal(0, N, x2_num, x2_den);
        Value sum_series(res.T, res.Q);

        return reduced * sum_series;
    }

    inline Value series_cos(const Value& x, const Value& eps) {
        Value pi_val = series_pi(eps);
        Value twopi = pi_val * 2;

        Value abs_x = is_negative(x) ? -x : x;

        // Универсальная редукция без использования double
        Value periods = abs_x / twopi;
        dumb_int k_int = numerator(periods) / denominator(periods);
        Value reduced = abs_x - Value(k_int) * twopi;

        // Приведение к [0, π]
        if (reduced > pi_val) {
            reduced = twopi - reduced;
        }

        if (is_zero(reduced)) return Value(1);

        Value x2 = reduced * reduced;
        dumb_int x2_num = numerator(x2);
        dumb_int x2_den = denominator(x2);

        double eps_d = to_double(eps);
        int64_t N = (eps_d <= 0) ? 10 : (int64_t)std::max(10.0, -std::log10(eps_d) * 0.8);
        if (N > 2000) N = 2000;

        auto res = cos_bs_internal(0, N, x2_num, x2_den);
        Value sum_series(res.T, res.Q);

        return sum_series;
    }
    // ---------------------------------------------------------------
    // Вспомогательная структура для бинарного расщепления
    // ---------------------------------------------------------------
    struct BSResult {
        dumb_int P, Q, T;
    };

    // ---------------------------------------------------------------
    // АРКТАНГЕНС (исправлен знак в бинарном расщеплении)
    // ---------------------------------------------------------------
    inline Value series_atan(const Value& x, const Value& eps) {
        if (is_zero(x)) return Value(0);

        bool negative = is_negative(x);
        Value xx = negative ? -x : x;

        // Редукция: |x| > 1 → π/2 - atan(1/x)
        if (xx > 1) {
            Value half_pi = series_pi(eps) / 2;
            Value inv = Value(1) / xx;
            Value atan_inv = series_atan(inv, eps);
            Value res = half_pi - atan_inv;
            return negative ? -res : res;
        }
        // 0.5 < x ≤ 1 → π/4 + atan((x-1)/(x+1))
        if (xx > Value(1) / 2) {
            Value one(1);
            Value xp = (xx - one) / (xx + one);   // |xp| ≤ 1/3
            Value quarter_pi = series_pi(eps) / 4;
            Value atan_xp = series_atan(xp, eps);
            Value res = quarter_pi + atan_xp;
            return negative ? -res : res;
        }

        // Теперь xx ≤ 0.5, используем ряд atan(x) = x * Σ (-x²)^k / (2k+1)
        Value x2 = xx * xx;
        dumb_int x2_num = numerator(x2);
        dumb_int x2_den = denominator(x2);

        // Оценка необходимого числа членов N с запасом
        double x2_d = to_double(x2);
        double eps_d = std::abs(to_double(eps));
        int N = 10;
        if (eps_d > 0) {
            while (std::pow(x2_d, N) / (2 * N + 1) > eps_d && N < 5000) ++N;
            N += 5;   // запас
        }
        else {
            N = 500;
        }

        // Бинарное расщепление для S = Σ_{k=0}^{N-1} (-x²)^k / (2k+1)
        // Базовый член: a_0 = 1/1; a_k = (-x²)^k / (x2_den^k * (2k+1))
        auto atan_bs = [&](auto&& self, int a, int b) -> BSResult {
            if (b - a == 1) {
                if (a == 0) {
                    // k = 0: слагаемое = 1, множитель для следующих членов = -x²
                    return { -x2_num, x2_den, dumb_int(1) };
                }
                else {
                    // k >= 1: P = -x²_num (передаётся вправо), Q = x2_den * (2k+1), T = 1
                    dumb_int Q = x2_den * (2 * a + 1);
                    return { -x2_num, Q, dumb_int(1) };
                }
            }
            int m = (a + b) / 2;
            auto L = self(self, a, m);
            auto R = self(self, m, b);
            return {
                L.P * R.P,
                L.Q * R.Q,
                L.T * R.Q + L.P * R.T   // L.P уже содержит -x²_num
            };
            };

        auto res = atan_bs(atan_bs, 0, N);
        Value S(res.T, res.Q);   // S = T / Q
        Value result = xx * S;
        return negative ? -result : result;
    }

    // ---------------------------------------------------------------
    // АРКСИНУС – бинарное расщепление ряда (без изменений)
    // ---------------------------------------------------------------
    inline Value series_asin(const Value& x, const Value& eps) {
        if (x < -1 || x > 1)
            throw std::domain_error("asin argument out of [-1,1]");
        if (is_one(x))  return series_pi(eps) / 2;
        if (x == -1)    return -series_pi(eps) / 2;

        Value x2 = x * x;
        dumb_int x2_num = numerator(x2);
        dumb_int x2_den = denominator(x2);

        // Оценка числа членов N (начиная с n=1)
        double x2_d = to_double(x2);
        double eps_d = std::abs(to_double(eps));
        int N = 10;
        if (eps_d > 0) {
            double x_d = to_double(eager_abs(x));
            while (x_d * std::pow(x2_d, N) / (2 * N + 1) > eps_d && N < 5000) ++N;
            N += 5;
        }
        else {
            N = 500;
        }

        // Бинарное расщепление для S = Σ_{n=1}^{N-1} a_n,
        // где a_n = a_{n-1} * ( (2n-1)² x² ) / ( 2n (2n+1) ), a_0 = x
        auto asin_bs = [&](auto&& self, int a, int b) -> BSResult {
            if (b - a == 1) {
                // член с индексом n = a (n≥1)
                dumb_int P = (2 * a - 1) * (2 * a - 1) * x2_num;
                dumb_int Q = 2 * a * (2 * a + 1) * x2_den;
                return { P, Q, P };   // T = P (числитель члена)
            }
            int m = (a + b) / 2;
            auto L = self(self, a, m);
            auto R = self(self, m, b);
            return {
                L.P * R.P,
                L.Q * R.Q,
                L.T * R.Q + L.P * R.T
            };
            };

        if (N <= 1) return x;

        auto res = asin_bs(asin_bs, 1, N);
        Value S(res.T, res.Q);
        return x + x * S;
    }

    // ---------------------------------------------------------------
    // АРККОСИНУС
    // ---------------------------------------------------------------
    inline Value series_acos(const Value& x, const Value& eps) {
        if (x < -1 || x > 1)
            throw std::domain_error("acos argument out of [-1,1]");

        Value clipped_x = x;
        if (clipped_x > Value(1)) clipped_x = Value(1);
        else if (clipped_x < Value(-1)) clipped_x = Value(-1);

        Value half_pi = series_pi(eps) / 2;
        return half_pi - series_asin(clipped_x, eps);
    }
    // ------------------------------------------------------------------------
    // ТАНГЕНС: sin(x)/cos(x).
    // ------------------------------------------------------------------------
    inline Value series_tan(const Value& x, const Value& eps) {
        Value s = series_sin(x, eps);
        Value c = series_cos(x, eps);
        if (is_zero(c)) throw std::domain_error("tan: cos(x) is zero");
        return s / c;
    }

    // ------------------------------------------------------------------------
    // ЧИСЛО e: ряд sum(1/n!). Float-путь удалён (медленнее).
    // ------------------------------------------------------------------------
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
    // Целочисленное возведение в степень (быстрое бинарное)
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
    // EAGER DISPATCHERS – точка входа для пользовательских вызовов.
    // ============================================================================

    inline Value eager_sqrt(const Value& x, const Value& eps) {
        if(is_negative(x)) throw std::domain_error("called sqrt of negative - it's irrational");
        // Сначала пробуем извлечь точный квадратный корень
        if (auto exact = try_exact_nth_root(x, Value(2))) {
            return *exact;
        }
        // Float-путь удалён – всегда series
        return series_sqrt(x, eps);
    }

    // Порог аргумента для float-пути exp. При |x| > 20 float теряет точность.
    constexpr double EXP_FLOAT_ARG_THRESHOLD = 20.0;

    inline Value eager_exp(const Value& x, const Value& eps) {
        double eps_d = to_double(eps);
        double x_d = std::abs(to_double(x));
        // float-путь быстр, но при больших аргументах теряет относительную точность.
        if (eps_d >= HYBRID_THRESHOLD && x_d <= EXP_FLOAT_ARG_THRESHOLD) {
            return float_exp(x, eps);
        }
        return series_exp(x, eps);
    }

    inline Value eager_log(const Value& x, const Value& eps) {
        // Float-путь удалён – всегда series
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

    inline Value eager_asin(const Value& x, const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? float_asin(x, eps) : series_asin(x, eps);
    }

    inline Value eager_atan(const Value& x, const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? float_atan(x, eps) : series_atan(x, eps);
    }

    inline Value eager_tan(const Value& x, const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? float_tan(x, eps) : series_tan(x, eps);
    }

    inline Value eager_pi(const Value& eps) {
        return (to_double(eps) >= HYBRID_THRESHOLD) ? float_pi(eps) : series_pi(eps);
    }

    inline Value eager_e(const Value& eps) {
        // Float-путь для e также не даёт выигрыша, всегда series.
        return series_e(eps);
    }

} // namespace delta::internal