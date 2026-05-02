// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/rational/performance_compare_test.cpp
// ============================================================================
// PERFORMANCE COMPARISON BETWEEN DELTA::RATIONAL AND BOOST.MULTIPRECISION
// ============================================================================
//
// This file benchmarks the performance of Delta::Rational (eager and lazy)
// against Boost.Multiprecision with expression templates both disabled (et_off)
// and enabled (et_on). The benchmarks measure:
//   - Immediate (eager) summation using Delta::Rational.
//   - Lazy summation using LazyRational (build + evaluation).
//   - Boost et_off (immediate style) summation.
//   - Boost et_on (lazy expression templates) summation.
//
// The test uses three types of input data:
//   - Random rationals (uniform numerator/denominator).
//   - Fast rationals (denominators are powers of two, for faster arithmetic).
//   - Harmonic series (1/i).
//
// Before running benchmarks, a correctness check verifies that Delta's sums
// match Boost's sums for each data type (N = 50 000). All timings are median
// values over TRIAL_RUNS (excluding the first warm‑up run). The results are
// printed in a human‑readable table with comparisons (faster/slower).
// ============================================================================

#pragma once
#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <algorithm>
#include "delta/core/rational.h"
#include "test_utils.h"

// Boost with expression templates disabled (immediate style)
using BoostRational = boost::multiprecision::number<
    boost::multiprecision::rational_adaptor<
    boost::multiprecision::cpp_int_backend<>
    >,
    boost::multiprecision::et_off
>;

// Boost with expression templates enabled (lazy style)
using BoostRationalEtOn = boost::multiprecision::number<
    boost::multiprecision::rational_adaptor<
    boost::multiprecision::cpp_int_backend<>
    >,
    boost::multiprecision::et_on
>;

namespace delta::testing {

    constexpr int TRIAL_RUNS = 15;
    constexpr int CORRECTNESS_CHECK_N = 50000;

    // -------------------------------------------------------------------------
    // Structure Pools now stores both Boost variants
    // -------------------------------------------------------------------------
    struct Pools {
        std::vector<BoostRational>      boost_et_off_pool;
        std::vector<BoostRationalEtOn>  boost_et_on_pool;
        std::vector<Rational>           delta_pool;
    };

    Pools generate_random_pools(size_t N);
    Pools generate_fast_pools(size_t N);
    Pools generate_harmonic_pools(int N);

    // -------------------------------------------------------------------------
    // Test fixture
    // -------------------------------------------------------------------------
    class RationalPerformanceCompareTest : public RationalTest {
    public:
        static void SetUpTestSuite() {
            // Perform correctness check before benchmarking
            PerformCorrectnessCheck();
            std::cout << "\n=== Performance benchmark with " << TRIAL_RUNS
                << " trial runs per N (first run excluded) ===\n";
            std::cout << "    Timings are median values.\n";
            std::cout << "    Delta::Rational compared against Boost et_off (immediate).\n";
            std::cout << "    Delta::LazyRational compared against Boost et_on (lazy).\n\n";
        }

        static void PerformCorrectnessCheck() {
            std::cout << "[----------] Performing correctness check (N = " << CORRECTNESS_CHECK_N << ")\n";

            {
                Pools pools = generate_random_pools(CORRECTNESS_CHECK_N);
                CheckSumsEqual(pools, "Random rationals");
            }
            {
                Pools pools = generate_fast_pools(CORRECTNESS_CHECK_N);
                CheckSumsEqual(pools, "Fast rationals (powers of two)");
            }
            {
                Pools pools = generate_harmonic_pools(CORRECTNESS_CHECK_N);
                CheckSumsEqual(pools, "Harmonic series");
            }
            std::cout << "[----------] Correctness check for delta::Rational & delta::LazyRational passed on random Rationals, powers of two, harmonic series \n";
        }

        static void CheckSumsEqual(const Pools& pools, const std::string& scenario) {
            // Immediate (eager) Delta sum
            Rational sum_imm = 0_r;
            for (const auto& t : pools.delta_pool) sum_imm += t;

            // Lazy Delta sum
            internal::reset_pool();
            LazyRational sum_lazy;
            for (const auto& t : pools.delta_pool) sum_lazy += t;
            sum_lazy.eval_inplace(true);
            Rational sum_lazy_eval = sum_lazy.eval();

            // Boost et_off sum (for correctness verification)
            BoostRational sum_boost_off = 0;
            for (const auto& t : pools.boost_et_off_pool) sum_boost_off += t;

            // Boost et_on sum (for correctness verification)
            BoostRationalEtOn sum_boost_on = 0;
            for (const auto& t : pools.boost_et_on_pool) sum_boost_on += t;

            auto to_string = [](const auto& val) {
                std::ostringstream oss;
                oss << val;
                return oss.str();
                };

            std::string s_imm = to_string(sum_imm);
            std::string s_lazy = to_string(sum_lazy_eval);
            std::string s_boost_off = to_string(sum_boost_off);
            std::string s_boost_on = to_string(sum_boost_on);

            if (s_imm != s_boost_off) {
                std::cerr << "\n[ERROR] Delta immediate differs from Boost et_off in " << scenario << "\n";
                FAIL();
            }
            if (s_lazy != s_boost_on) {
                std::cerr << "\n[ERROR] Delta lazy differs from Boost et_on in " << scenario << "\n";
                FAIL();
            }
        }
    };

    // -------------------------------------------------------------------------
    // Data generation
    // -------------------------------------------------------------------------
    Pools generate_random_pools(size_t N) {
        Pools pools;
        pools.boost_et_off_pool.reserve(N);
        pools.boost_et_on_pool.reserve(N);
        pools.delta_pool.reserve(N);

        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> num_dist(-1000, 1000);
        std::uniform_int_distribution<int> den_dist(1, 1000);

        for (size_t i = 0; i < N; ++i) {
            int num = num_dist(rng);
            int den = den_dist(rng);
            pools.boost_et_off_pool.emplace_back(num, den);
            pools.boost_et_on_pool.emplace_back(num, den);
            pools.delta_pool.emplace_back(num, den);
        }
        return pools;
    }

    Pools generate_fast_pools(size_t N) {
        Pools pools;
        pools.boost_et_off_pool.reserve(N);
        pools.boost_et_on_pool.reserve(N);
        pools.delta_pool.reserve(N);

        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> num_dist(-1000, 1000);
        std::uniform_int_distribution<int> exp_dist(0, 20);

        for (size_t i = 0; i < N; ++i) {
            int num = num_dist(rng);
            int den = 1 << exp_dist(rng);
            pools.boost_et_off_pool.emplace_back(num, den);
            pools.boost_et_on_pool.emplace_back(num, den);
            pools.delta_pool.emplace_back(num, den);
        }
        return pools;
    }

    Pools generate_harmonic_pools(int N) {
        Pools pools;
        pools.boost_et_off_pool.reserve(N);
        pools.boost_et_on_pool.reserve(N);
        pools.delta_pool.reserve(N);

        for (int i = 1; i <= N; ++i) {
            pools.boost_et_off_pool.emplace_back(1, i);
            pools.boost_et_on_pool.emplace_back(1, i);
            pools.delta_pool.emplace_back(1, i);
        }
        return pools;
    }

    // -------------------------------------------------------------------------
    // Time measurement functions
    // -------------------------------------------------------------------------
    template<typename TimeUnit = std::chrono::milliseconds>
    long long measure_delta_immediate_sum(const std::vector<Rational>& terms) {
        auto start = std::chrono::high_resolution_clock::now();
        Rational sum = 0_r;
        for (const auto& t : terms) sum += t;
        volatile Rational dummy = sum;
        (void)dummy;
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<TimeUnit>(end - start).count();
    }

    template<typename BoostType, typename TimeUnit = std::chrono::milliseconds>
    long long measure_boost_sum(const std::vector<BoostType>& terms) {
        auto start = std::chrono::high_resolution_clock::now();
        BoostType sum = 0;
        for (const auto& t : terms) sum += t;
        volatile BoostType dummy = sum;
        (void)dummy;
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<TimeUnit>(end - start).count();
    }

    template<typename TimeUnit = std::chrono::milliseconds>
    std::tuple<long long, long long, long long> measure_delta_lazy_sum(const std::vector<Rational>& terms) {
        internal::reset_pool();
        auto start_build = std::chrono::high_resolution_clock::now();
        LazyRational sum;
        for (const auto& t : terms) sum += t;
        auto end_build = std::chrono::high_resolution_clock::now();

        auto start_eval = std::chrono::high_resolution_clock::now();
        sum.eval_inplace(true);
        Rational result = sum.eval();
        auto end_eval = std::chrono::high_resolution_clock::now();
        volatile Rational dummy = result;
        (void)dummy;

        long long build_ms = std::chrono::duration_cast<TimeUnit>(end_build - start_build).count();
        long long eval_ms = std::chrono::duration_cast<TimeUnit>(end_eval - start_eval).count();
        return { build_ms, eval_ms, build_ms + eval_ms };
    }

    // -------------------------------------------------------------------------
    // Benchmark runner
    // -------------------------------------------------------------------------
    void run_benchmark_extended(const std::string& test_name,
        const std::vector<int>& sizes,
        std::function<Pools(int)> generator) {
        std::cout << "\n=== " << test_name << " ===\n";
        // Column width increased for clearer separation
        std::cout << std::left << std::setw(8) << "N"
            << std::setw(30) << "Delta mode (ms)"
            << std::setw(18) << "Boost ref (ms)"
            << "Comparison\n";
        std::cout << std::string(90, '-') << "\n";

        for (int N : sizes) {
            Pools pools = generator(N);

            std::vector<long long> imm_times, boost_off_times, boost_on_times;
            std::vector<long long> lazy_total_times, lazy_build_times, lazy_eval_times;

            for (int rep = 0; rep < TRIAL_RUNS; ++rep) {
                long long imm = measure_delta_immediate_sum(pools.delta_pool);
                long long boost_off = measure_boost_sum<BoostRational>(pools.boost_et_off_pool);
                long long boost_on = measure_boost_sum<BoostRationalEtOn>(pools.boost_et_on_pool);
                auto [build, eval, total] = measure_delta_lazy_sum(pools.delta_pool);

                if (rep == 0) continue; // warm‑up
                imm_times.push_back(imm);
                boost_off_times.push_back(boost_off);
                boost_on_times.push_back(boost_on);
                lazy_total_times.push_back(total);
                lazy_build_times.push_back(build);
                lazy_eval_times.push_back(eval);
            }

            if (imm_times.empty()) {
                std::cout << std::setw(8) << N << "  insufficient data\n";
                continue;
            }

            auto median = [](std::vector<long long>& v) {
                std::sort(v.begin(), v.end());
                return v[v.size() / 2];
                };

            long long med_imm = median(imm_times);
            long long med_boost_off = median(boost_off_times);
            long long med_boost_on = median(boost_on_times);
            long long med_lazy_total = median(lazy_total_times);
            long long med_lazy_build = median(lazy_build_times);
            long long med_lazy_eval = median(lazy_eval_times);

            auto format_comparison = [](long long delta_time, long long ref_time) -> std::string {
                if (delta_time == 0 && ref_time == 0) {
                    return "delta equal (0 ms)";
                }
                if (delta_time == 0) {
                    return "delta infinitely faster (" + std::to_string(ref_time) + " ms)";
                }
                if (ref_time == 0) {
                    return "delta infinitely slower (" + std::to_string(delta_time) + " ms)";
                }
                std::ostringstream oss;
                if (delta_time < ref_time) {
                    double ratio = static_cast<double>(ref_time) / delta_time;
                    long long diff = ref_time - delta_time;
                    oss << "delta " << std::fixed << std::setprecision(2)
                        << ratio << " times faster (" << diff << " ms)";
                }
                else if (delta_time > ref_time) {
                    double ratio = static_cast<double>(delta_time) / ref_time;
                    long long diff = delta_time - ref_time;
                    oss << "delta " << std::fixed << std::setprecision(2)
                        << ratio << " times slower (" << diff << " ms)";
                }
                else {
                    oss << "delta equal (0 ms)";
                }
                return oss.str();
                };

            // Immediate (eager) vs Boost et_off
            std::cout << std::left << std::setw(8) << N
                << std::setw(30) << ("immediate: " + std::to_string(med_imm))
                << std::setw(18) << med_boost_off
                << format_comparison(med_imm, med_boost_off) << "\n";

            // Lazy vs Boost et_on
            std::cout << std::left << std::setw(8) << ""
                << std::setw(30) << ("lazy: " + std::to_string(med_lazy_total) + " ("
                    + std::to_string(med_lazy_build) + " build, "
                    + std::to_string(med_lazy_eval) + " eval)")
                << std::setw(18) << med_boost_on
                << format_comparison(med_lazy_total, med_boost_on) << "\n";
        }
    }

    // -------------------------------------------------------------------------
    // Test cases
    // -------------------------------------------------------------------------

    /**
     * @test RandomRationalsCompare
     * @brief Benchmarks summation of random rationals (uniform distribution).
     */
    TEST_F(RationalPerformanceCompareTest, RandomRationalsCompare) {
        std::vector<int> sizes = { 100, 500, 1000, 5000, 10000, 20000, 50000, 250000, 500000 };
        run_benchmark_extended("Random rationals (uniform)", sizes, generate_random_pools);
    }

    /**
     * @test FastRationalsCompare
     * @brief Benchmarks summation of rationals where denominators are powers of two
     *        (leading to faster arithmetic).
     */
    TEST_F(RationalPerformanceCompareTest, FastRationalsCompare) {
        std::vector<int> sizes = { 100, 500, 1000, 5000, 10000, 20000, 50000 };
        run_benchmark_extended("Fast rationals (denominators powers of two)", sizes, generate_fast_pools);
    }

    /**
     * @test HarmonicSeriesCompare
     * @brief Benchmarks summation of the harmonic series (1 + 1/2 + ... + 1/N).
     */
    TEST_F(RationalPerformanceCompareTest, HarmonicSeriesCompare) {
        std::vector<int> sizes = { 100, 500, 1000, 5000, 10000, 20000, 50000 };
        run_benchmark_extended("Harmonic series (1 + 1/2 + ... + 1/N)", sizes, generate_harmonic_pools);
    }

} // namespace delta::testing