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
#include "test_utils.h"


// ЧТОБЫ УВИДЕТЬ КОНКРЕТЫЙ ВКЛАД ОПТИМИЗАЦИИ РАБОТЫ DELTA СО SMALLSTORAGE - ЗАМЕНИ sum+= term на sum = sum + term;
// ЧТОБЫ УВИДЕТЬ КАК DELTA::RATIONAL ОБХОДИТ БУСТ НА ПИКЕ В 10 РАЗ ПО МЕДИАННОМУ ВРЕМЕНИ БЕЗ ВИДИМОЙ ПРИЧИНЫ - ЗАМЕНИТЕ sum = sum+term на sum+=term;
// Контрольная группа: чистый boost::multiprecision::rational_adaptor с et_off
using BoostRational = boost::multiprecision::number<
    boost::multiprecision::rational_adaptor<
    boost::multiprecision::cpp_int_backend<>
    >,
    boost::multiprecision::et_off
>;

namespace delta::testing {

    // Количество повторных запусков для каждого N (первый отбрасывается)
    constexpr int TRIAL_RUNS = 15;

    class RationalPerformanceCompareTest : public RationalTest {
    public:
        static void SetUpTestSuite() {
            std::cout << "\n=== Performance benchmark with " << TRIAL_RUNS
                << " trial runs per N (first run excluded) ===\n";
        }
    };

    // -------------------------------------------------------------------------
    // 1. Генерация пулов с фиксированным seed (один раз для всех повторений)
    // -------------------------------------------------------------------------
    struct Pools {
        std::vector<BoostRational> boost_pool;
        std::vector<Rational> delta_pool;
    };

    // Генерация случайных дробей (числа в [-1000,1000], знаменатели [1,1000])
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

    // Генерация "быстрых" дробей (знаменатели – степени двойки, числители небольшие)
    Pools generate_fast_pools(size_t N) {
        Pools pools;
        pools.boost_pool.reserve(N);
        pools.delta_pool.reserve(N);
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> num_dist(-1000, 1000);
        std::uniform_int_distribution<int> exp_dist(0, 20); // 2^20 ~ 1e6

        for (size_t i = 0; i < N; ++i) {
            int num = num_dist(rng);
            int den = 1 << exp_dist(rng); // степень двойки, влезает в int
            pools.boost_pool.emplace_back(num, den);
            pools.delta_pool.emplace_back(static_cast<absl::int128>(num), static_cast<absl::uint128>(den));
        }
        return pools;
    }

    // Генератор гармонического ряда: 1, 1/2, 1/3, ..., 1/N
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
    // 2. Функции замера (только время, никакого вывода рациональных)
    // -------------------------------------------------------------------------
    template<typename TimeUnit = std::chrono::milliseconds>
    long long measure_delta_sum(const std::vector<Rational>& terms) {
        set_eager_mode(true);
        auto start = std::chrono::high_resolution_clock::now();
        Rational sum = 0_r;
        for (const auto& t : terms) {
            sum += t;
        }
        volatile Rational dummy = sum;
        (void)dummy;
        auto end = std::chrono::high_resolution_clock::now();
        set_eager_mode(false);
        return std::chrono::duration_cast<TimeUnit>(end - start).count();
    }

    template<typename TimeUnit = std::chrono::milliseconds>
    long long measure_boost_sum(const std::vector<BoostRational>& terms) {
        auto start = std::chrono::high_resolution_clock::now();
        BoostRational sum = 0;
        for (const auto& t : terms) {
            sum += t;
        }
        volatile BoostRational dummy = sum;
        (void)dummy;
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<TimeUnit>(end - start).count();
    }

    // -------------------------------------------------------------------------
    // 3. Запуск тестов для заданного пула и N
    // -------------------------------------------------------------------------
    void run_benchmark(const std::string& test_name,
        const std::vector<int>& sizes,
        std::function<Pools(int)> generator) {
        std::cout << "\n=== " << test_name << " ===\n";
        std::cout << std::left << std::setw(10) << "N"
            << std::setw(20) << "Delta median (ms)"
            << std::setw(20) << "Boost median (ms)"
            << "Delta vs Boost (%, abs diff ms)\n";
        std::cout << std::string(90, '-') << "\n";

        for (int N : sizes) {
            Pools pools = generator(N);

            std::vector<long long> delta_times, boost_times;
            delta_times.reserve(TRIAL_RUNS);
            boost_times.reserve(TRIAL_RUNS);

            for (int rep = 0; rep < TRIAL_RUNS; ++rep) {
                long long dt = measure_delta_sum(pools.delta_pool);
                long long bt = measure_boost_sum(pools.boost_pool);
                if (rep == 0) continue; // прогревочный замер отбрасываем
                delta_times.push_back(dt);
                boost_times.push_back(bt);
            }

            if (delta_times.empty()) {
                std::cout << std::setw(10) << N << "  недостаточно данных\n";
                continue;
            }

            std::sort(delta_times.begin(), delta_times.end());
            std::sort(boost_times.begin(), boost_times.end());
            long long med_delta = delta_times[delta_times.size() / 2];
            long long med_boost = boost_times[boost_times.size() / 2];

            long long abs_diff = med_delta - med_boost; // Delta - Boost
            double percent = 0.0;
            std::string cmp;
            if (med_delta < med_boost) {
                percent = 100.0 * (med_boost - med_delta) / med_boost;
                cmp = "faster";
            }
            else if (med_delta > med_boost) {
                percent = 100.0 * (med_delta - med_boost) / med_boost;
                cmp = "slower";
            }
            else {
                cmp = "equal";
            }

            std::cout << std::left << std::setw(10) << N
                << std::setw(20) << med_delta
                << std::setw(20) << med_boost
                << "Delta " << cmp << " by " << std::fixed << std::setprecision(2) << percent << "%"
                << " (Delta - Boost: " << abs_diff << " ms)\n";
        }
    }

    // -------------------------------------------------------------------------
    // 4. Тесты
    // -------------------------------------------------------------------------
    TEST_F(RationalPerformanceCompareTest, RandomRationalsCompare) {
        std::vector<int> sizes = { 100, 500, 1000, 5000, 10000, 20000 };
        run_benchmark("Random rationals (uniform)", sizes, generate_random_pools);
    }

    TEST_F(RationalPerformanceCompareTest, FastRationalsCompare) {
        std::vector<int> sizes = { 100, 500, 1000, 5000, 10000, 20000, 50000 };
        run_benchmark("Fast rationals (denominators powers of two)", sizes, generate_fast_pools);
    }

    TEST_F(RationalPerformanceCompareTest, HarmonicSeriesCompare) {
        std::vector<int> sizes = { 100, 500, 1000, 5000, 10000, 20000, 50000 };
        run_benchmark("Harmonic series (1 + 1/2 + ... + 1/N)", sizes, generate_harmonic_pools);
    }

    // Проверка корректности (чтобы суммы совпадали)
    TEST_F(RationalPerformanceCompareTest, CorrectnessCheck) {
        for (int N : {10, 100, 500,10000,20000}) {
            Pools pools = generate_random_pools(N);
            set_eager_mode(true);
            Rational delta_sum = 0_r;
            for (const auto& t : pools.delta_pool) delta_sum = delta_sum + t;
            BoostRational boost_sum = 0;
            for (const auto& t : pools.boost_pool) boost_sum = boost_sum + t;
            EXPECT_EQ(to_string(delta_sum), boost_sum.str());
        }
    }

} // namespace delta::testing