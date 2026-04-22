// transcendentals_canonicalization_benchmark.cpp
// Бенчмарки канонизации и интернирования подвыражений в LazyRational.

#include <gtest/gtest.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

#include "delta/core/rational.h"
#include "test_utils.h"
#include "lazy_rational_test_fixture.h"

namespace delta::testing {

    // -----------------------------------------------------------------------------
    // Генераторы выражений для бенчмарков
    // -----------------------------------------------------------------------------
    namespace {

        // Генерирует сложное выражение с вложенными трансцендентными функциями.
        // expr_i = sin(expr_{i-1}) + cos(expr_{i-1}) * sqrt(expr_{i-1} + 1)
        LazyRational generate_complex_expression(int depth, const Rational& seed = 2_r) {
            LazyRational current = seed.as_lazy();
            for (int i = 0; i < depth; ++i) {
                current = Sin(current) + Cos(current) * Sqrt(current + 1_r);
            }
            return current;
        }

        // Генерирует выражение с повторяющимися подграфами (для проверки интернирования).
        // expr_i = sin(expr_{i-1}) * cos(expr_{i-1})
        LazyRational generate_repeating_expression(int repeats, const Rational& base = 3_r) {
            LazyRational current = base.as_lazy();
            for (int i = 0; i < repeats; ++i) {
                current = Sin(current) * Cos(current);
            }
            return current;
        }

    } // namespace

    // -----------------------------------------------------------------------------
    // Фикстура для бенчмарков канонизации
    // -----------------------------------------------------------------------------
    class CanonicalizationBenchmark : public LazyRationalTestFixture {
    public:
        static void SetUpTestSuite() {
            std::cout << "\n=== Canonicalization Impact Benchmark ===\n";
            std::cout << "Depth | With Canon (ms) | Without Canon (ms) | Result\n";
            std::cout << std::string(70, '-') << "\n";
        }

    protected:
        void SetUp() override {
            internal::reset_pool();   // сброс пула перед каждым тестом
        }

        // Вспомогательная функция для медианного бенчмарка
        template<typename F>
        long long benchmark_median(F&& func, int runs) {
            std::vector<long long> times;
            times.reserve(runs);
            for (int i = 0; i < runs; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                func();
                auto end = std::chrono::high_resolution_clock::now();
                times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
            }
            std::sort(times.begin(), times.end());
            return times[times.size() / 2];
        }
    };

    // -----------------------------------------------------------------------------
    // Бенчмарк: канонизация vs без канонизации для сложного выражения
    // -----------------------------------------------------------------------------
    TEST_F(CanonicalizationBenchmark, DISABLED_ComplexExpressionCanonVsNoCanon) {
        const std::vector<int> depths = { 10, 20, 30, 40, 50 };
        const int runs = 15;  // количество прогонов для медианы

        for (int depth : depths) {
            internal::reset_pool();
            LazyRational expr_with = generate_complex_expression(depth, 2_r);
            LazyRational expr_without = expr_with.clone();

            // Прогрев (3 итерации)
            for (int i = 0; i < 3; ++i) {
                LazyRational copy1 = expr_with.clone();
                copy1.eval();
                LazyRational copy2 = expr_without.clone();
                copy2.eval_inplace(true);
                volatile auto res = copy2.eval();
                (void)res;
            }

            auto func_with = [&]() {
                LazyRational copy = expr_with.clone();
                copy.eval();
                };

            auto func_without = [&]() {
                LazyRational copy = expr_without.clone();
                copy.eval_inplace(true);
                volatile auto res = copy.eval();
                (void)res;
                };

            long long med_with = benchmark_median(func_with, runs);
            long long med_without = benchmark_median(func_without, runs);

            std::string result;
            if (med_with < med_without) {
                double speedup = static_cast<double>(med_without) / med_with;
                std::ostringstream oss;
                oss << "simplify " << std::fixed << std::setprecision(2) << speedup << "x faster";
                result = oss.str();
            }
            else {
                double slowdown = static_cast<double>(med_with) / med_without;
                std::ostringstream oss;
                oss << "no_simplify " << std::fixed << std::setprecision(2) << slowdown << "x faster";
                result = oss.str();
            }

            std::cout << std::setw(5) << depth << " | "
                << std::setw(15) << med_with / 1000 << " | "
                << std::setw(18) << med_without / 1000 << " | "
                << result << "\n";

            internal::reset_pool();
        }
    }

    // -----------------------------------------------------------------------------
    // Бенчмарк интернирования подвыражений (повторяющиеся подграфы)
    // -----------------------------------------------------------------------------
    TEST_F(CanonicalizationBenchmark, DISABLED_RepeatingSubexpressionsCanonVsNoCanon) {
        const std::vector<int> repeats = { 50, 100, 200 };
        const int runs = 15;

        for (int rep : repeats) {
            internal::reset_pool();
            LazyRational expr_with = generate_repeating_expression(rep, 3_r);
            LazyRational expr_without = expr_with.clone();

            // Прогрев
            for (int i = 0; i < 3; ++i) {
                LazyRational copy1 = expr_with.clone();
                copy1.eval();
                LazyRational copy2 = expr_without.clone();
                copy2.eval_inplace(true);
                volatile auto res = copy2.eval();
                (void)res;
            }

            auto func_with = [&]() {
                LazyRational copy = expr_with.clone();
                copy.eval();
                };

            auto func_without = [&]() {
                LazyRational copy = expr_without.clone();
                copy.eval_inplace(true);
                volatile auto res = copy.eval();
                (void)res;
                };

            long long med_with = benchmark_median(func_with, runs);
            long long med_without = benchmark_median(func_without, runs);

            std::string result;
            if (med_with < med_without) {
                double speedup = static_cast<double>(med_without) / med_with;
                std::ostringstream oss;
                oss << "simplify " << std::fixed << std::setprecision(2) << speedup << "x faster";
                result = oss.str();
            }
            else {
                double slowdown = static_cast<double>(med_with) / med_without;
                std::ostringstream oss;
                oss << "no_simplify " << std::fixed << std::setprecision(2) << slowdown << "x faster";
                result = oss.str();
            }

            std::cout << std::setw(5) << rep << " | "
                << std::setw(15) << med_with / 1000 << " | "
                << std::setw(18) << med_without / 1000 << " | "
                << result << "\n";

            internal::reset_pool();
        }
    }

} // namespace delta::testing