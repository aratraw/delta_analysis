// transcendentals_canonicalization_benchmark.cpp
// Benchmarks for canonicalization and subexpression interning in LazyRational.
//
// Three scenarios where canonicalization provides a measurable benefit:
//   1. Exp-Log chain: algebraic simplification Exp(Log(x)) -> x
//   2. Repeating term: interning of identical constant subgraphs in a sum
//   3. Zeros in sum: removal of neutral elements (0 in SUM)
//
// Each benchmark compares clone()+eval() (with canonicalization)
// against clone()+eval_inplace(true) (without canonicalization).
//
// IMPORTANT: All transcendental functions in generators use eager Rational
// arguments to avoid lazy explosion of nested symbolic nodes. The benchmarks
// measure canonicalization cost, not transcendental function performance.

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
    // Expression generators for benchmarks
    // -----------------------------------------------------------------------------
    namespace {

        // expr = Exp(Log(Exp(Log(...Exp(Log(seed))...))))  — depth pairs
        // Canonicalization collapses all Exp(Log(x)) -> x down to seed.
        LazyRational generate_exp_log_chain(int depth, const Rational& seed = 2_r) {
            LazyRational current = seed.as_lazy();
            for (int i = 0; i < depth; ++i) {
                current = Exp(Log(current));
            }
            return current;
        }

        // expr = term_val + term_val + ... + term_val (repeats times)
        // term_val = sin(x) * cos(x) computed eagerly as a single Rational.
        // Canonicalization interns the repeated identical constants in the pool.
        LazyRational generate_repeating_term(int repeats, const Rational& x_val = "0.5"_r) {
            LazyRational term_val = Sin(x_val) * Cos(x_val);

            LazyRational acc;
            for (int i = 0; i < repeats; ++i) {
                acc + term_val;
            }
            return acc;
        }

        // expr = val + 0 + val + 0 + val + 0 + ... (total_terms, half are zeros)
        // Canonicalization removes zero leaf_values from the SUM node.
        LazyRational generate_sum_with_zeros(int total_terms, const Rational& val = "0.5"_r) {
            LazyRational acc;
            for (int i = 0; i < total_terms; ++i) {
                if (i % 2 == 0) {
                    acc + val;
                }
                else {
                    acc + 0_r;
                }
            }
            return acc;
        }

    } // namespace

    // -----------------------------------------------------------------------------
    // Benchmark fixture
    // -----------------------------------------------------------------------------
    class CanonicalizationBenchmark : public LazyRationalTestFixture {
    protected:
        void SetUp() override {
            internal::reset_pool();
            reset_default_eps();
        }

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

        void run_comparison(int param,
            const LazyRational& expr_with,
            const LazyRational& expr_without,
            int runs,
            const char* param_name)
        {
            auto func_with = [&]() {
                LazyRational copy = expr_with.clone();
                copy.eval();
                };

            auto func_without = [&]() {
                LazyRational copy = expr_without.clone();
                copy.eval_inplace(true);
                (void)copy.eval();   // CONST node after eval_inplace, nearly free
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

            std::cout << "  " << std::setw(5) << param << "  | "
                << std::setw(15) << med_with / 1000 << " | "
                << std::setw(18) << med_without / 1000 << " | "
                << result << "\n";
        }
    };

    // =========================================================================
    // Benchmark 1: Exp-Log chain (algebraic simplification)
    // =========================================================================
    TEST_F(CanonicalizationBenchmark, ExpLogChainSimplification) {
        const std::vector<int> depths = { 1, 2, 3, 4, 5, 6, 8, 10 };
        const int runs = 4;

        std::cout << "\n==============================================================\n";
        std::cout << "  CANONICALIZATION BENCHMARK 1: Exp-Log Chain Simplification\n";
        std::cout << "==============================================================\n";
        std::cout << "  expr = Exp(Log(Exp(Log(...Exp(Log(seed))...))))\n";
        std::cout << "  seed = 2,  depth = number of Exp-Log pairs\n";
        std::cout << "  " << runs << " runs per depth, median reported\n";
        std::cout << "\n";
        std::cout << "  WITH CANON:    Exp(Log(x)) -> x  (algebraic identity)\n";
        std::cout << "                 Entire chain collapses to seed in one pass.\n";
        std::cout << "  WITHOUT CANON: Each Exp and Log is computed numerically.\n";
        std::cout << "                 2*depth transcendental evaluations.\n";
        std::cout << "\n";
        std::cout << "  EXPECTED: canonicalization should be faster by ~2*depth.\n";
        std::cout << "--------------------------------------------------------------\n";
        std::cout << " Depth | With Canon (ms) | Without Canon (ms) | Result\n";
        std::cout << "--------------------------------------------------------------\n";

        for (int depth : depths) {
            internal::reset_pool();
            LazyRational expr_with = generate_exp_log_chain(depth, 2_r);
            LazyRational expr_without = expr_with.clone();

            run_comparison(depth, expr_with, expr_without, runs, "depth");

            internal::reset_pool();
        }

        std::cout << "--------------------------------------------------------------\n";
        std::cout << "  NOTE: " << runs << " runs is the minimum for a meaningful median.\n";
        std::cout << "  This measures the cost of algebraic simplification.\n";
        std::cout << "==============================================================\n\n";
    }

    // =========================================================================
    // Benchmark 2: Repeating constants (interning)
    // =========================================================================

    // Временный диагностический тест (можно закомментировать после выяснения причины)
// Временный диагностический тест (можно закомментировать после выяснения причины)
    TEST_F(CanonicalizationBenchmark, DISABLED_DiagnoseRepeatingTerm) {
        const int repeats = 10;   // начнём с малого
        const int runs = 1;       // один прогон, чтобы увидеть всё

        std::cout << "\n==============================================================\n";
        std::cout << "DIAGNOSTIC: RepeatingSubgraphInterning for repeats=" << repeats << "\n";
        std::cout << "==============================================================\n";

        // 1. Создаём выражение
        std::cout << "\n--- Generating expression ---\n";
        LazyRational expr = generate_repeating_term(repeats, "0.5"_r);
        std::cout << "Expression generated.\n";
        print_lazy(expr, "expr (dirty)");
        print_pool("Pool before any eval");

        // 2. Тест с канонизацией (eval)
        std::cout << "\n--- Testing WITH canonicalization (eval) ---\n";
        LazyRational expr_with = generate_repeating_term(repeats, "0.5"_r);
        std::cout << "Created fresh expr_with (dirty).\n";
        print_lazy(expr_with, "expr_with (dirty)");

        auto start = std::chrono::high_resolution_clock::now();
        Rational result_with = expr_with.eval();
        auto end = std::chrono::high_resolution_clock::now();
        auto time_with = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "eval took " << time_with / 1000.0 << " ms\n";
        std::cout << "Result: " << result_with << "\n";
        print_lazy(expr_with, "expr_with after eval (should be Clean)");
        print_pool("Pool after eval with canonicalization");

        // 3. Тест без канонизации (eval_inplace(true))
        std::cout << "\n--- Testing WITHOUT canonicalization (eval_inplace(true)) ---\n";
        internal::reset_pool();  // начинаем с чистого пула для чистоты замера
        LazyRational expr_without = generate_repeating_term(repeats, "0.5"_r);
        std::cout << "Created fresh expr_without (dirty).\n";
        print_lazy(expr_without, "expr_without (dirty)");

        start = std::chrono::high_resolution_clock::now();
        expr_without.eval_inplace(true);   // <-- ИСПРАВЛЕНО: void, не присваиваем
        Rational result_without = expr_without.eval();  // после eval_inplace дерево стало CONST, можно взять значение
        end = std::chrono::high_resolution_clock::now();
        auto time_without = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "eval_inplace(true) took " << time_without / 1000.0 << " ms\n";
        std::cout << "Result: " << result_without << "\n";
        print_lazy(expr_without, "expr_without after eval_inplace (should be Clean CONST)");
        print_pool("Pool after eval_inplace (no canonicalization)");

        // 4. Сравнение
        std::cout << "\n--- Comparison ---\n";
        double speedup = static_cast<double>(time_without) / time_with;
        std::cout << "With canonicalization: " << time_with / 1000.0 << " ms\n";
        std::cout << "Without canonicalization: " << time_without / 1000.0 << " ms\n";
        std::cout << "Speedup: " << speedup << "x\n";

        // 5. Проверка соответствия значений
        EXPECT_EQ(result_with, result_without);
        std::cout << "==============================================================\n";
    }
    TEST_F(CanonicalizationBenchmark, RepeatingSubgraphInterning) {
        const std::vector<int> repeats = { 10, 50, 100, 200, 500 };
        const int runs = 4;

        std::cout << "--------------------------------------------------------------\n";
        std::cout << " Repeats | With Canon (ms) | Without Canon (ms) | Result\n";
        std::cout << "--------------------------------------------------------------\n";

        for (int rep : repeats) {
            auto func_with = [rep]() {
                LazyRational expr = generate_repeating_term(rep, "0.5"_r);
                expr.eval();   // упростится и вычислится
                };
            auto func_without = [rep]() {
                LazyRational expr = generate_repeating_term(rep, "0.5"_r);
                expr.eval_inplace(true);   // вычисление без упрощения
                (void)expr.eval();         // уже CONST, почти бесплатно
                };

            long long med_with = benchmark_median(func_with, runs);
            long long med_without = benchmark_median(func_without, runs);

            double speedup = static_cast<double>(med_without) / med_with;
            std::cout << "  " << std::setw(5) << rep << "  | "
                << std::setw(15) << med_with / 1000 << " | "
                << std::setw(18) << med_without / 1000 << " | "
                << "simplify " << std::fixed << std::setprecision(2) << speedup << "x faster\n";
        }
        std::cout << "--------------------------------------------------------------\n";
    }
    // =========================================================================
    // Benchmark 3: Zero removal in SUM (neutral element elimination)
    // =========================================================================
    TEST_F(CanonicalizationBenchmark, ZeroRemovalInSum) {
        const std::vector<int> total_terms = { 100, 500, 1000, 5000 };
        const int runs = 4;

        std::cout << "\n==============================================================\n";
        std::cout << "  CANONICALIZATION BENCHMARK 3: Zero Removal in SUM\n";
        std::cout << "==============================================================\n";
        std::cout << "  expr = val + 0 + val + 0 + ...   (N terms, half are zeros)\n";
        std::cout << "  val = 0.5,  " << runs << " runs per term count, median reported\n";
        std::cout << "\n";
        std::cout << "  WITH CANON:    Zeros are removed from leaf_values during\n";
        std::cout << "                 flattening. The SUM node has N/2 leaves.\n";
        std::cout << "  WITHOUT CANON: All N terms (including zeros) are processed.\n";
        std::cout << "                 The SUM node has N leaves.\n";
        std::cout << "\n";
        std::cout << "  EXPECTED: canonicalization should be ~2x faster\n";
        std::cout << "            (tree is half the size).\n";
        std::cout << "--------------------------------------------------------------\n";
        std::cout << "  N     | With Canon (ms) | Without Canon (ms) | Result\n";
        std::cout << "--------------------------------------------------------------\n";

        for (int n : total_terms) {
            internal::reset_pool();
            LazyRational expr_with = generate_sum_with_zeros(n, "0.5"_r);
            LazyRational expr_without = expr_with.clone();

            run_comparison(n, expr_with, expr_without, runs, "N");

            internal::reset_pool();
        }

        std::cout << "--------------------------------------------------------------\n";
        std::cout << "  NOTE: " << runs << " runs is the minimum for a meaningful median.\n";
        std::cout << "  This measures the benefit of neutral element elimination.\n";
        std::cout << "==============================================================\n\n";
    }

} // namespace delta::testing