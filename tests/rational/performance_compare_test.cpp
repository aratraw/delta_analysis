// tests/rational/performance_compare_test.cpp
#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include "delta/core/rational.h"
#include "test_utils.h"

// Контрольная группа: чистый boost::multiprecision::rational_adaptor с et_off
using ControlRational = boost::multiprecision::number<
    boost::multiprecision::rational_adaptor<
    boost::multiprecision::cpp_int_backend<>
    >,
    boost::multiprecision::et_off
>;

namespace delta::testing {

    class RationalPerformanceCompareTest : public RationalTest {};

    // Генерация пула случайных дробей для заданного N (фиксированный seed)
    std::vector<ControlRational> generate_control_pool(size_t N) {
        std::vector<ControlRational> pool;
        pool.reserve(N);
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> num_dist(-1000, 1000);
        std::uniform_int_distribution<int> den_dist(1, 1000);
        for (size_t i = 0; i < N; ++i) {
            int num = num_dist(rng);
            int den = den_dist(rng);
            pool.emplace_back(num) / den;
        }
        return pool;
    }

    std::vector<Rational> convert_pool_to_delta(const std::vector<ControlRational>& pool) {
        std::vector<Rational> result;
        result.reserve(pool.size());
        for (const auto& cr : pool) {
            std::string s = cr.str();
            result.emplace_back(s);
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Тест 1: Гармонический ряд (сумма 1/i)
    // -------------------------------------------------------------------------
    void test_harmonic_series(int N) {
        std::cout << "\n=== Harmonic series N = " << N << " ===\n";

        // 1. Immediate (eager) delta::Rational
        auto start = std::chrono::high_resolution_clock::now();
        {
            ScopedEagerEval eager;
            Rational sum = 0_r;
            for (int i = 1; i <= N; ++i) {
                sum = sum + Rational(1, i);
            }
            volatile Rational dummy = sum;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_immediate = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // 2. Lazy delta::Rational – строим дерево и затем eval
        //    Сначала создаём ленивый объект через .lazy() (или изначально lazy-конструктор)
        //set_eager_mode(false); // отключаем eager режим для новых операций
        //Rational lazy_sum = 0_r.lazy(); // делаем начальный ноль ленивым
        //auto build_start = std::chrono::high_resolution_clock::now();
        //for (int i = 1; i <= N; ++i) {
        //    lazy_sum = lazy_sum + Rational(1, i).lazy(); // все слагаемые тоже ленивые
        //}
        //auto build_end = std::chrono::high_resolution_clock::now();
        //auto elapsed_lazy_build = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();

        //auto eval_start = std::chrono::high_resolution_clock::now();
        //Rational result = lazy_sum.eval(true); // eval(true) – без упрощения, только вычисление
        //auto eval_end = std::chrono::high_resolution_clock::now();
        //auto elapsed_lazy_eval = std::chrono::duration_cast<std::chrono::milliseconds>(eval_end - eval_start).count();
        //(void)result;

        // 3. Контрольная группа (чистый Boost)
        start = std::chrono::high_resolution_clock::now();
        {
            ControlRational sum = 0;
            for (int i = 1; i <= N; ++i) {
                sum = sum + ControlRational(1) / i;
            }
            volatile ControlRational dummy = sum;
        }
        end = std::chrono::high_resolution_clock::now();
        auto elapsed_control = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // Вывод таблицы
        std::cout << std::left << std::setw(25) << "Implementation"
            << std::setw(15) << "Time (ms)" << "\n";
        std::cout << std::string(40, '-') << "\n";
        std::cout << std::left << std::setw(25) << "Delta immediate (eager)"
            << std::setw(15) << elapsed_immediate << "\n";
        //std::cout << std::left << std::setw(25) << "Delta lazy (build tree)"
        //    << std::setw(15) << elapsed_lazy_build << "\n";
        //std::cout << std::left << std::setw(25) << "Delta lazy (eval only)"
        //    << std::setw(15) << elapsed_lazy_eval << "\n";
        std::cout << std::left << std::setw(25) << "Boost control (et_off)"
            << std::setw(15) << elapsed_control << "\n";
    }

    // -------------------------------------------------------------------------
    // Тест 2: Сумма случайных рациональных дробей (фиксированный пул)
    // -------------------------------------------------------------------------
    void test_random_sum(int N) {
        std::cout << "\n=== Random rationals sum N = " << N << " ===\n";
        auto control_pool = generate_control_pool(N);
        auto delta_pool = convert_pool_to_delta(control_pool);

        // 1. Immediate delta
        auto start = std::chrono::high_resolution_clock::now();
        {
            ScopedEagerEval eager;
            Rational sum = 0_r;
            for (const auto& term : delta_pool) {
                sum = sum + term;
            }
            volatile Rational dummy = sum;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_immediate = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // 2. Lazy delta
        set_eager_mode(false);
        Rational lazy_sum = 0_r.lazy();
        auto build_start = std::chrono::high_resolution_clock::now();
        for (const auto& term : delta_pool) {
            lazy_sum = lazy_sum + term;
        }
        auto build_end = std::chrono::high_resolution_clock::now();
        auto elapsed_lazy_build = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();

        auto eval_start = std::chrono::high_resolution_clock::now();
        Rational result = lazy_sum.eval(true);
        auto eval_end = std::chrono::high_resolution_clock::now();
        auto elapsed_lazy_eval = std::chrono::duration_cast<std::chrono::milliseconds>(eval_end - eval_start).count();
        (void)result;

        // 3. Контрольная группа (чистый Boost)
        start = std::chrono::high_resolution_clock::now();
        {
            ControlRational sum = 0;
            for (const auto& term : control_pool) {
                sum = sum + term;
            }
            volatile ControlRational dummy = sum;
        }
        end = std::chrono::high_resolution_clock::now();
        auto elapsed_control = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << std::left << std::setw(25) << "Implementation"
            << std::setw(15) << "Time (ms)" << "\n";
        std::cout << std::string(40, '-') << "\n";
        std::cout << std::left << std::setw(25) << "Delta immediate (eager)"
            << std::setw(15) << elapsed_immediate << "\n";
        std::cout << std::left << std::setw(25) << "Delta lazy (build tree)"
            << std::setw(15) << elapsed_lazy_build << "\n";
        std::cout << std::left << std::setw(25) << "Delta lazy (eval only)"
            << std::setw(15) << elapsed_lazy_eval << "\n";
        std::cout << std::left << std::setw(25) << "Boost control (et_off)"
            << std::setw(15) << elapsed_control << "\n";
    }

    // -------------------------------------------------------------------------
    // Запуск всех тестов с разными N
    // -------------------------------------------------------------------------
    TEST_F(RationalPerformanceCompareTest, HarmonicSeriesCompare) {
        std::vector<int> sizes = { 100, 1000, 10000, 20000,50000,100000 };
        for (int N : sizes) {
            test_harmonic_series(N);
        }
    }

    TEST_F(RationalPerformanceCompareTest, RandomRationalsCompare) {
        std::vector<int> sizes = { 100, 1000, 10000, 20000 };
        for (int N : sizes) {
            test_random_sum(N);
        }
    }

} // namespace delta::testing