// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/rational/transcendentals_comparative.cpp
// ============================================================================
// COMPARATIVE PERFORMANCE TESTS: Delta vs. NAIVE (SERIES) IMPLEMENTATIONS
// ============================================================================
//
// This file benchmarks the transcendental functions of Delta::Rational against
// naive (reference) series implementations (sin, cos, exp, log, sqrt, pi, e).
// Three precision regimes are tested:
//   - HIGH_PRECISION_EPS    = 1e-21   (Delta uses float‑path for sin/cos/exp/pi)
//   - ULTRA_HIGH_PRECISION_EPS = 1e-40 (Delta uses series‑path for all functions)
//   - EXTREME_PRECISION_EPS  = 1e-80   (Delta uses series‑path, tests scaling)
//
// The output is a compact table with median times (microseconds) and a
// comparison (faster / slower) against the naive implementation.
// ============================================================================

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
    // Naive (baseline) implementations using iterative series / Newton method.
    // -----------------------------------------------------------------------------
    namespace {

        /**
         * @brief Naive series for ln(2) using arctanh(1/3).
         */
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

        /**
         * @brief Naive series for π using Machin's formula: π/4 = 4*atan(1/5) - atan(1/239).
         */
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

        /**
         * @brief Naive series for sin(x).
         */
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

        /**
         * @brief Naive series for cos(x).
         */
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

        /**
         * @brief Naive series for exp(x) (scaling‑and‑squaring).
         */
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

        /**
         * @brief Naive series for log(x) (argument reduction + arctanh series).
         */
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

        /**
         * @brief Naive sqrt using Newton's method.
         */
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

        /**
         * @brief Naive series for e (base of natural logarithm).
         */
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
        // Determine which path Delta uses for a given epsilon
        // -------------------------------------------------------------------------
        constexpr double HYBRID_THRESHOLD = 1e-35;

        std::string delta_path_for_eps(const std::string& func_name, const Rational& eps) {
            double eps_d = eps.to_double();
            // These functions always use series (float path removed)
            if (func_name == "log" || func_name == "sqrt" || func_name == "e") {
                return "series";
            }
            // For sin, cos, exp, pi, acos: float path is used when eps >= threshold
            return (eps_d >= HYBRID_THRESHOLD) ? "float" : "series";
        }

    } // namespace

    // -----------------------------------------------------------------------------
    // Test fixture for comparative performance tests
    // -----------------------------------------------------------------------------
    class TranscendentalPerformanceTest : public LazyRationalTestFixture {
    protected:
        const Rational HIGH_PRECISION_EPS = "1/1000000000000000000000"_r;   // 1e-21
        const Rational ULTRA_HIGH_PRECISION_EPS = "1/10000000000000000000000000000000000000000"_r; // 1e-40
        const Rational EXTREME_PRECISION_EPS = "1/10000000000000000000000000000000000000000000000000000000000000000000000000000000"_r; // 1e-80
        const int RUNS = 15;

        /**
         * @brief Runs a function many times and returns the median execution time in microseconds.
         * @tparam F Callable (lambda) with no arguments.
         */
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

        /**
         * @brief Configuration for a single function to be tested.
         */
        struct FuncConfig {
            std::string name;                                                     // function name (sin, cos, ...)
            std::function<Rational(const Rational&)> arg_gen;                    // generates the argument (ignored for pi/e)
            std::function<Rational(const Rational&, const Rational&)> delta_func; // Delta implementation
            std::function<Rational(const Rational&, const Rational&)> naive_func; // Naive (baseline) implementation
        };

        /**
         * @brief Format a comparison string: "X.xx times faster/slower (diff us)".
         */
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

        /**
         * @brief Measure execution times for a single function and epsilon.
         */
        struct Measurement {
            long long delta_us;
            long long naive_us;
            std::string delta_path;
        };

        Measurement measure(const FuncConfig& cfg, const Rational& eps) {
            // Warm‑up (3 runs)
            for (int i = 0; i < 3; ++i) {
                Rational arg = cfg.arg_gen(eps);  // for pi/e, arg is ignored
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
    // Single test that produces the full comparison table
    // -----------------------------------------------------------------------------
    TEST_F(TranscendentalPerformanceTest, FullComparisonTable) {
        // List of functions to test
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
             [](const Rational&) { return 0_r; }, // argument not used
             [](const Rational&, const Rational& eps) { return delta::pi(eps); },
             [](const Rational&, const Rational& eps) { return naive_series_pi(eps); }},
            {"e",
             [](const Rational&) { return 0_r; },
             [](const Rational&, const Rational& eps) { return delta::e(eps); },
             [](const Rational&, const Rational& eps) { return naive_series_e(eps); }}
        };

        // Table header
        std::cout << "\n=== Transcendental Performance Comparison (median of " << RUNS << " runs) ===\n";
        std::cout << std::left
            << std::setw(8) << "Func"
            << std::setw(10) << "Eps"
            << std::setw(10) << "Path"
            << std::setw(12) << "Delta(us)"
            << std::setw(12) << "Naive(us)"
            << "Comparison\n";
        std::cout << std::string(85, '-') << "\n";

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

                // Correctness check (one extra run)
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
            // Separator between epsilon blocks (except last)
            if (i < epsilons.size() - 1) {
                std::cout << std::string(85, '-') << "\n";
            }
        }
        std::cout << std::string(85, '=') << "\n";
    }

} // namespace delta::testing