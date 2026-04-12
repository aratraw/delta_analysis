// tests/rational/performance_compare_test.cpp
#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>
#include <functional>
#include "delta/core/rational.h"
#include "delta/rational/gc.h"
#include "test_utils.h"

using BoostRational = boost::multiprecision::number<
    boost::multiprecision::rational_adaptor<
    boost::multiprecision::cpp_int_backend<>
    >,
    boost::multiprecision::et_off
>;

namespace delta::testing {

    constexpr int TRIAL_RUNS = 15;

    class RationalPerformanceCompareTest : public RationalTest {
    public:
        static void SetUpTestSuite() {
            std::cout << "\n=== Performance benchmark with " << TRIAL_RUNS
                << " trial runs per N (first run excluded) ===\n";
        }
    };

    // -------------------------------------------------------------------------
    // 1. Генерация пулов
    // -------------------------------------------------------------------------
    struct Pools {
        std::vector<BoostRational> boost_pool;
        std::vector<Rational> delta_pool;
    };

    Pools generate_random_pools(size_t N) {
        Pools pools;
        pools.boost_pool.reserve(N);
        pools.delta_pool.reserve(N);
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> num_dist(-1000, 1000);
        std::uniform_int_distribution<int> den_dist(1, 1000);

        for (size_t i = 0; i < N; ++i) {
            int num = num_dist(rng);
            int den = den_dist(rng);
            pools.boost_pool.emplace_back(num, den);
            pools.delta_pool.emplace_back(static_cast<absl::int128>(num), static_cast<absl::uint128>(den));
        }
        return pools;
    }

    Pools generate_fast_pools(size_t N) {
        Pools pools;
        pools.boost_pool.reserve(N);
        pools.delta_pool.reserve(N);
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> num_dist(-1000, 1000);
        std::uniform_int_distribution<int> exp_dist(0, 20);

        for (size_t i = 0; i < N; ++i) {
            int num = num_dist(rng);
            int den = 1 << exp_dist(rng);
            pools.boost_pool.emplace_back(num, den);
            pools.delta_pool.emplace_back(static_cast<absl::int128>(num), static_cast<absl::uint128>(den));
        }
        return pools;
    }

    Pools generate_harmonic_pools(int N) {
        Pools pools;
        pools.boost_pool.reserve(N);
        pools.delta_pool.reserve(N);
        for (int i = 1; i <= N; ++i) {
            pools.boost_pool.emplace_back(1, i);
            pools.delta_pool.emplace_back(static_cast<absl::int128>(1), static_cast<absl::uint128>(i));
        }
        return pools;
    }

    // -------------------------------------------------------------------------
    // 2. Функции замера
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

    template<typename TimeUnit = std::chrono::milliseconds>
    long long measure_boost_sum(const std::vector<BoostRational>& terms) {
        auto start = std::chrono::high_resolution_clock::now();
        BoostRational sum = 0;
        for (const auto& t : terms) sum += t;
        volatile BoostRational dummy = sum;
        (void)dummy;
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<TimeUnit>(end - start).count();
    }

    // Возвращает (build_ms, eval_ms, total_ms)
    template<typename TimeUnit = std::chrono::milliseconds>
    std::tuple<long long, long long, long long> measure_delta_lazy_sum(const std::vector<Rational>& terms) {
        internal::reset_pool();   // каждый замер стартует с чистого пула
        auto start_build = std::chrono::high_resolution_clock::now();
        Rational sum = 0_r.lazy();
        for (const auto& t : terms) sum += t;
        auto end_build = std::chrono::high_resolution_clock::now();

        auto start_eval = std::chrono::high_resolution_clock::now();
        Rational result = sum.eval(true);
        auto end_eval = std::chrono::high_resolution_clock::now();
        volatile Rational dummy = result;
        (void)dummy;

        long long build_ms = std::chrono::duration_cast<TimeUnit>(end_build - start_build).count();
        long long eval_ms = std::chrono::duration_cast<TimeUnit>(end_eval - start_eval).count();
        return { build_ms, eval_ms, build_ms + eval_ms };
    }

    // -------------------------------------------------------------------------
    // 3. Бенчмарк с выводом двух строк (immediate и lazy)
    // -------------------------------------------------------------------------
    void run_benchmark_extended(const std::string& test_name,
        const std::vector<int>& sizes,
        std::function<Pools(int)> generator) {
        std::cout << "\n=== " << test_name << " ===\n";
        std::cout << std::left << std::setw(8) << "N"
            << std::setw(24) << "Delta mode (ms)"
            << std::setw(12) << "Boost (ms)"
            << "Delta vs Boost\n";
        std::cout << std::string(80, '-') << "\n";

        for (int N : sizes) {
            Pools pools = generator(N);

            std::vector<long long> imm_times, boost_times;
            std::vector<long long> lazy_total_times, lazy_build_times, lazy_eval_times;

            imm_times.reserve(TRIAL_RUNS);
            boost_times.reserve(TRIAL_RUNS);
            lazy_total_times.reserve(TRIAL_RUNS);
            lazy_build_times.reserve(TRIAL_RUNS);
            lazy_eval_times.reserve(TRIAL_RUNS);

            for (int rep = 0; rep < TRIAL_RUNS; ++rep) {
                long long imm = measure_delta_immediate_sum(pools.delta_pool);
                auto [build, eval, total] = measure_delta_lazy_sum(pools.delta_pool);
                long long boost = measure_boost_sum(pools.boost_pool);

                if (rep == 0) continue;   // прогревочный замер
                imm_times.push_back(imm);
                lazy_total_times.push_back(total);
                lazy_build_times.push_back(build);
                lazy_eval_times.push_back(eval);
                boost_times.push_back(boost);
            }

            if (imm_times.empty()) {
                std::cout << std::setw(8) << N << "  insufficient data\n";
                continue;
            }

            std::sort(imm_times.begin(), imm_times.end());
            std::sort(boost_times.begin(), boost_times.end());
            std::sort(lazy_total_times.begin(), lazy_total_times.end());
            std::sort(lazy_build_times.begin(), lazy_build_times.end());
            std::sort(lazy_eval_times.begin(), lazy_eval_times.end());

            long long med_imm = imm_times[imm_times.size() / 2];
            long long med_boost = boost_times[boost_times.size() / 2];
            long long med_lazy_total = lazy_total_times[lazy_total_times.size() / 2];
            long long med_lazy_build = lazy_build_times[lazy_build_times.size() / 2];
            long long med_lazy_eval = lazy_eval_times[lazy_eval_times.size() / 2];

            // Строка для immediate
            {
                long long diff = med_imm - med_boost;
                double percent = 0.0;
                std::string cmp;
                if (med_imm < med_boost) {
                    percent = 100.0 * (med_boost - med_imm) / med_boost;
                    cmp = "faster";
                }
                else if (med_imm > med_boost) {
                    percent = 100.0 * (med_imm - med_boost) / med_boost;
                    cmp = "slower";
                }
                else {
                    cmp = "equal";
                }
                std::cout << std::left << std::setw(8) << N
                    << std::setw(24) << ("immediate: " + std::to_string(med_imm))
                    << std::setw(12) << med_boost
                    << "Delta " << cmp << " by " << std::fixed << std::setprecision(2) << percent
                    << "% (" << std::abs(diff) << " ms)\n";
            }

            // Строка для lazy
            {
                long long diff = med_lazy_total - med_boost;
                double percent = 0.0;
                std::string cmp;
                if (med_lazy_total < med_boost) {
                    percent = 100.0 * (med_boost - med_lazy_total) / med_boost;
                    cmp = "faster";
                }
                else if (med_lazy_total > med_boost) {
                    percent = 100.0 * (med_lazy_total - med_boost) / med_boost;
                    cmp = "slower";
                }
                else {
                    cmp = "equal";
                }
                std::cout << std::left << std::setw(8) << ""
                    << std::setw(24) << ("lazy: " + std::to_string(med_lazy_total) + " (" +
                        std::to_string(med_lazy_build) + " build, " +
                        std::to_string(med_lazy_eval) + " eval)")
                    << std::setw(12) << ""
                    << "Delta " << cmp << " by " << std::fixed << std::setprecision(2) << percent
                    << "% (" << std::abs(diff) << " ms)\n";
            }
        }
    }

    // -------------------------------------------------------------------------
    // 4. Тесты
    // -------------------------------------------------------------------------
    TEST_F(RationalPerformanceCompareTest, RandomRationalsCompare) {
        std::vector<int> sizes = { 100, 500, 1000, 5000, 10000, 20000 };
        run_benchmark_extended("Random rationals (uniform)", sizes, generate_random_pools);
    }

    TEST_F(RationalPerformanceCompareTest, FastRationalsCompare) {
        std::vector<int> sizes = { 100, 500, 1000, 5000, 10000, 20000, 50000 };
        run_benchmark_extended("Fast rationals (denominators powers of two)", sizes, generate_fast_pools);
    }

    TEST_F(RationalPerformanceCompareTest, HarmonicSeriesCompare) {
        std::vector<int> sizes = { 100, 500, 1000, 5000, 10000, 20000, 50000 };
        run_benchmark_extended("Harmonic series (1 + 1/2 + ... + 1/N)", sizes, generate_harmonic_pools);
    }

    // Проверка корректности
    TEST_F(RationalPerformanceCompareTest, CorrectnessCheck) {
        for (int N : {10, 100, 500, 10000, 20000}) {
            Pools pools = generate_random_pools(N);
            Rational delta_imm = 0_r;
            for (const auto& t : pools.delta_pool) delta_imm += t;
            Rational delta_lazy = 0_r.lazy();
            for (const auto& t : pools.delta_pool) delta_lazy += t;
            BoostRational boost_sum = 0;
            for (const auto& t : pools.boost_pool) boost_sum += t;
            EXPECT_EQ(delta_imm, delta_lazy.eval(true));
            EXPECT_EQ(to_string(delta_imm), boost_sum.str());
        }
    }

} // namespace delta::testing