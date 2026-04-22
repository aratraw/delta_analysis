// transcendentals_comparative.cpp
// Сравнительные тесты производительности: Delta vs наивные (naive) реализации.
// Оцениваются три режима точности:
//   - HIGH_PRECISION_EPS = 1e-21 (в Delta используются float-пути для sin/cos/exp/pi)
//   - ULTRA_HIGH_PRECISION_EPS = 1e-40 (в Delta везде используются series-пути)
//   - EXTREME_PRECISION_EPS = 1e-80 (в Delta везде series-пути, проверка масштабирования)
// Вывод оформлен в виде компактной таблицы с медианными временами и сравнением.

#include <gtest/gtest.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <functional>

#include "delta/core/rational.h"
#include "test_utils.h"
#include "lazy_rational_test_fixture.h"

namespace delta::testing {

    // -----------------------------------------------------------------------------
    // Наивные (baseline) реализации через последовательные ряды / итерации.
    // -----------------------------------------------------------------------------
    namespace {

        Rational naive_series_ln2(const Rational& eps) {
            Rational one(1);
            Rational three(3);
            Rational z = one / three;
            Rational z2 = z * z;
            Rational term = z, sum = term, n = one, two(2);
            while (true) {
                term = term * z2;
                n = n + two;
                sum = sum + term / n;
                if (abs(term) < eps) break;
            }
            return two * sum;
        }

        Rational naive_series_pi(const Rational& eps) {
            Rational one(1), five(5), two39(239);
            Rational sixteen(16), four(4), two(2);
            Rational a = one / five, a2 = a * a;
            Rational term = a, sum_atan5 = term, n = one;
            while (true) {
                term = term * (-a2);
                n = n + two;
                sum_atan5 = sum_atan5 + term / n;
                if (abs(term) < eps) break;
            }
            Rational b = one / two39, b2 = b * b;
            term = b;
            Rational sum_atan239 = term;
            n = one;
            while (true) {
                term = term * (-b2);
                n = n + two;
                sum_atan239 = sum_atan239 + term / n;
                if (abs(term) < eps) break;
            }
            return sixteen * sum_atan5 - four * sum_atan239;
        }

        Rational naive_series_sin(const Rational& x, const Rational& eps) {
            Rational one(1), two(2);
            Rational pi_val = naive_series_pi(eps), twopi = pi_val * two;
            Rational reduced = x;
            while (abs(reduced) > pi_val) {
                if (reduced > 0) reduced = reduced - twopi;
                else reduced = reduced + twopi;
            }
            Rational x2 = reduced * reduced;
            Rational term = reduced, sum = term, k = one;
            while (true) {
                term = term * (-x2);
                term = term / (two * k * (two * k + one));
                sum = sum + term;
                k = k + one;
                if (abs(term) < eps) break;
            }
            return sum;
        }

        Rational naive_series_cos(const Rational& x, const Rational& eps) {
            Rational one(1), two(2);
            Rational pi_val = naive_series_pi(eps), twopi = pi_val * two;
            Rational reduced = x;
            while (abs(reduced) > pi_val) {
                if (reduced > 0) reduced = reduced - twopi;
                else reduced = reduced + twopi;
            }
            Rational x2 = reduced * reduced;
            Rational term = one, sum = term, k = one;
            while (true) {
                term = term * (-x2);
                term = term / ((two * k - one) * (two * k));
                sum = sum + term;
                k = k + one;
                if (abs(term) < eps) break;
            }
            return sum;
        }

        Rational naive_series_exp(const Rational& x, const Rational& eps) {
            Rational one(1), two(2);
            int k = 0;
            Rational reduced = x;
            while (abs(reduced) > one) {
                reduced = reduced / two;
                ++k;
            }
            Rational sum = one, term = one, n = one;
            while (true) {
                term = term * reduced / n;
                sum = sum + term;
                n = n + one;
                if (abs(term) < eps) break;
            }
            Rational result = sum;
            for (int i = 0; i < k; ++i) result = result * result;
            return result;
        }

        Rational naive_series_log(const Rational& x, const Rational& eps) {
            Rational one(1), two(2), half = one / two;
            int k = 0;
            Rational m = x;
            while (m > two) { m = m / two; ++k; }
            while (m < half) { m = m * two; --k; }
            Rational ln2 = naive_series_ln2(eps);
            Rational y = (m - one) / (m + one);
            Rational y2 = y * y;
            Rational term = y, sum = term, n = one;
            while (true) {
                term = term * y2;
                n = n + two;
                sum = sum + term / n;
                if (abs(term) < eps) break;
            }
            Rational ln_m = two * sum;
            return ln_m + Rational(k) * ln2;
        }

        Rational naive_series_sqrt(const Rational& x, const Rational& eps) {
            if (x == 0_r) return 0_r;
            if (x < 0_r) throw std::domain_error("sqrt of negative number");
            Rational one(1), two(2);
            Rational guess = x / two, diff;
            size_t iter = 0;
            const size_t max_iter = 1000;
            do {
                Rational next = (guess + x / guess) / two;
                diff = abs(next - guess);
                guess = next;
                ++iter;
                if (iter > max_iter) break;
            } while (diff > eps);
            return guess;
        }

        Rational naive_series_e(const Rational& eps) {
            Rational one(1);
            Rational sum = one, term = one, n = one;
            while (true) {
                term = term / n;
                sum = sum + term;
                n = n + one;
                if (term < eps) break;
            }
            return sum;
        }

        // -------------------------------------------------------------------------
        // Определение того, какой путь использует Delta для заданной точности
        // -------------------------------------------------------------------------
        constexpr double HYBRID_THRESHOLD = 1e-35;

        std::string delta_path_for_eps(const std::string& func_name, const Rational& eps) {
            double eps_d = eps.to_double();
            if (func_name == "log" || func_name == "sqrt" || func_name == "e") {
                return "series";   // всегда series
            }
            // sin, cos, exp, pi, acos
            return (eps_d >= HYBRID_THRESHOLD) ? "float" : "series";
        }

    } // namespace

    // -----------------------------------------------------------------------------
    // Фикстура для сравнительных тестов производительности
    // -----------------------------------------------------------------------------
    class TranscendentalPerformanceTest : public LazyRationalTestFixture {
    protected:
        const Rational HIGH_PRECISION_EPS = "1/1000000000000000000000"_r;   // 1e-21
        const Rational ULTRA_HIGH_PRECISION_EPS = "1/10000000000000000000000000000000000000000"_r; // 1e-40
        const Rational EXTREME_PRECISION_EPS = "1/10000000000000000000000000000000000000000000000000000000000000000000000000000000"_r; // 1e-80
        const int RUNS = 15;

        // Вспомогательная функция для медианного бенчмарка
        template<typename F>
        long long benchmark_median(F&& func) {
            std::vector<long long> times;
            times.reserve(RUNS);
            for (int i = 0; i < RUNS; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                func();
                auto end = std::chrono::high_resolution_clock::now();
                times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
            }
            std::sort(times.begin(), times.end());
            return times[times.size() / 2];
        }

        // Структура для описания тестируемой функции
        struct FuncConfig {
            std::string name;
            std::function<Rational(const Rational&)> arg_gen;   // возвращает аргумент (для pi/e - фиктивный)
            std::function<Rational(const Rational&, const Rational&)> delta_func;
            std::function<Rational(const Rational&, const Rational&)> naive_func;
        };

        // Форматирование сравнения: "Delta X.xx times faster/slower (diff us)"
        std::string format_comparison(long long delta_us, long long ref_us) {
            std::ostringstream oss;
            if (delta_us < ref_us) {
                double ratio = static_cast<double>(ref_us) / delta_us;
                long long diff = ref_us - delta_us;
                oss << std::fixed << std::setprecision(2) << ratio << "x faster (" << diff << " us)";
            }
            else if (delta_us > ref_us) {
                double ratio = static_cast<double>(delta_us) / ref_us;
                long long diff = delta_us - ref_us;
                oss << std::fixed << std::setprecision(2) << ratio << "x slower (" << diff << " us)";
            }
            else {
                oss << "equal (0 us)";
            }
            return oss.str();
        }

        // Выполнить замеры для одной функции и одного eps, вернуть медианные времена и путь
        struct Measurement {
            long long delta_us;
            long long naive_us;
            std::string delta_path;
        };

        Measurement measure(const FuncConfig& cfg, const Rational& eps) {
            // Прогрев
            for (int i = 0; i < 3; ++i) {
                Rational arg = cfg.arg_gen(eps);  // для pi/e arg игнорируется
                cfg.delta_func(arg, eps);
                cfg.naive_func(arg, eps);
            }

            auto delta_lambda = [&]() {
                Rational arg = cfg.arg_gen(eps);
                volatile auto res = cfg.delta_func(arg, eps);
                (void)res;
                };
            auto naive_lambda = [&]() {
                Rational arg = cfg.arg_gen(eps);
                volatile auto res = cfg.naive_func(arg, eps);
                (void)res;
                };

            Measurement m;
            m.delta_us = benchmark_median(delta_lambda);
            m.naive_us = benchmark_median(naive_lambda);
            m.delta_path = delta_path_for_eps(cfg.name, eps);
            return m;
        }
    };

    // -----------------------------------------------------------------------------
    // Единый тест, собирающий все сравнения и выводящий таблицу
    // -----------------------------------------------------------------------------
    TEST_F(TranscendentalPerformanceTest, FullComparisonTable) {
        // Конфигурации всех функций
        std::vector<FuncConfig> configs = {
            {"sin",
             [](const Rational&) { return "1.23456789"_r; },
             [](const Rational& x, const Rational& eps) { return delta::sin(x, eps); },
             [](const Rational& x, const Rational& eps) { return naive_series_sin(x, eps); }},
            {"cos",
             [](const Rational&) { return "1.23456789"_r; },
             [](const Rational& x, const Rational& eps) { return delta::cos(x, eps); },
             [](const Rational& x, const Rational& eps) { return naive_series_cos(x, eps); }},
            {"exp",
             [](const Rational&) { return "2.3456789"_r; },
             [](const Rational& x, const Rational& eps) { return delta::exp(x, eps); },
             [](const Rational& x, const Rational& eps) { return naive_series_exp(x, eps); }},
            {"log",
             [](const Rational&) { return "2.718281828"_r; },
             [](const Rational& x, const Rational& eps) { return delta::log(x, eps); },
             [](const Rational& x, const Rational& eps) { return naive_series_log(x, eps); }},
            {"sqrt",
             [](const Rational&) { return 2_r; },
             [](const Rational& x, const Rational& eps) { return delta::sqrt(x, eps); },
             [](const Rational& x, const Rational& eps) { return naive_series_sqrt(x, eps); }},
            {"pi",
             [](const Rational&) { return 0_r; }, // аргумент не используется
             [](const Rational&, const Rational& eps) { return delta::pi(eps); },
             [](const Rational&, const Rational& eps) { return naive_series_pi(eps); }},
            {"e",
             [](const Rational&) { return 0_r; },
             [](const Rational&, const Rational& eps) { return delta::e(eps); },
             [](const Rational&, const Rational& eps) { return naive_series_e(eps); }}
        };

        // Заголовок таблицы
        std::cout << "\n=== Transcendental Performance Comparison (median of " << RUNS << " runs) ===\n";
        std::cout << std::left
            << std::setw(8) << "Func"
            << std::setw(10) << "Eps"
            << std::setw(10) << "Path"
            << std::setw(12) << "Delta(us)"
            << std::setw(12) << "Naive(us)"
            << "Comparison\n";
        std::cout << std::string(85, '-') << "\n";

        // Для каждого eps собираем и выводим строки
        std::vector<Rational> epsilons = {
            HIGH_PRECISION_EPS,        // 1e-21
            ULTRA_HIGH_PRECISION_EPS,  // 1e-40
            EXTREME_PRECISION_EPS      // 1e-80
        };
        std::vector<std::string> eps_labels = { "1e-21", "1e-40", "1e-80" };

        for (size_t i = 0; i < epsilons.size(); ++i) {
            const Rational& eps = epsilons[i];
            const std::string& eps_label = eps_labels[i];

            for (const auto& cfg : configs) {
                Measurement m = measure(cfg, eps);

                // Проверка корректности (последний вызов)
                Rational arg = cfg.arg_gen(eps);
                Rational res_delta = cfg.delta_func(arg, eps);
                Rational res_naive = cfg.naive_func(arg, eps);
                EXPECT_RATIONAL_NEAR(res_delta, res_naive, eps);

                std::cout << std::left
                    << std::setw(8) << cfg.name
                    << std::setw(10) << eps_label
                    << std::setw(10) << m.delta_path
                    << std::setw(12) << m.delta_us
                    << std::setw(12) << m.naive_us
                    << format_comparison(m.delta_us, m.naive_us) << "\n";
            }
            // Разделитель между блоками eps (кроме последнего)
            if (i < epsilons.size() - 1) {
                std::cout << std::string(85, '-') << "\n";
            }
        }
        std::cout << std::string(85, '=') << "\n";
    }

} // namespace delta::testing