// tests/rational/performance_test.cpp
#pragma once
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include "delta/core/rational.h"
#include "test_utils.h"

namespace delta::testing {

    class RationalPerformanceTest : public RationalTest {};

    // -------------------------------------------------------------------------
    // 1. Harmonic series (eager mode) – immediate Rational
    // -------------------------------------------------------------------------
    TEST_F(RationalPerformanceTest, HarmonicSeries10000EagerMode) {
        const int N = 10000;
        Rational sum = 0_r;
        for (int i = 1; i <= N; ++i) {
            sum = sum + Rational(1, i);
        }

        using boost::multiprecision::cpp_dec_float_100;
        cpp_dec_float_100 ref = 0;
        for (int i = 1; i <= N; ++i) {
            ref += cpp_dec_float_100(1) / i;
        }
        std::string ref_str = ref.str(60, std::ios_base::fixed);
        Rational expected(ref_str);

        Rational eps = Rational("1/1000000000000000000000000000000");
        EXPECT_RATIONAL_NEAR(sum, expected, eps);
    }

    // -------------------------------------------------------------------------
    // 2. Harmonic series (lazy mode) – build SUM tree, then evaluate once
    // -------------------------------------------------------------------------
    TEST_F(RationalPerformanceTest, HarmonicSeries10000LazyMode) {
        const int N = 10000;
        LazyRational sum;
        for (int i = 1; i <= N; ++i) {
            sum += Rational(1, i);
        }
        Rational result = sum.eval();

        using boost::multiprecision::cpp_dec_float_100;
        cpp_dec_float_100 ref = 0;
        for (int i = 1; i <= N; ++i) {
            ref += cpp_dec_float_100(1) / i;
        }
        std::string ref_str = ref.str(60, std::ios_base::fixed);
        Rational expected(ref_str);

        Rational eps = Rational("1/1000000000000000000000000000000");
        EXPECT_RATIONAL_NEAR(result, expected, eps);
    }

    // -------------------------------------------------------------------------
    // 3. Deep tree (lazy mode) – build a chain of additions, then evaluate
    // -------------------------------------------------------------------------
    TEST_F(RationalPerformanceTest, DeepTree10000LazyMode) {
        const int DEPTH = 10000;
        LazyRational tree;
        for (int i = 0; i < DEPTH; ++i) {
            tree += 1_r;
        }
        Rational result = tree.eval();
        EXPECT_EQ(result, Rational(DEPTH));
    }

    // -------------------------------------------------------------------------
    // 4. Large product (factorial) – immediate Rational
    // -------------------------------------------------------------------------
    TEST_F(RationalPerformanceTest, LargeProduct500) {
        const int N = 500;
        Rational prod = 1_r;
        for (int i = 1; i <= N; ++i) {
            prod = prod * Rational(i);
        }
        EXPECT_GT(prod, 0_r);
    }

    // -------------------------------------------------------------------------
    // 5. Nested transcendental functions with simplification (lazy)
    // -------------------------------------------------------------------------
    TEST_F(RationalPerformanceTest, DISABLED_NestedTranscendentalsLazy) {
        LazyRational x = Rational(2).as_lazy();
        LazyRational expr = delta::lazy_sin(delta::lazy_cos(delta::lazy_exp(delta::lazy_log(x))));
        expr.simplify_inplace();
        Rational expected = delta::sin(delta::cos(2_r));
        EXPECT_RATIONAL_NEAR(expr.eval(), expected, default_eps());
    }

    // -------------------------------------------------------------------------
    // 6. Huge random lazy additions (500k) – stress test for batching
    // -------------------------------------------------------------------------
    TEST_F(RationalPerformanceTest, DISABLED_HugeRandomLazyAdditions500k) {
        std::cout << "--- IT'S GONNA TAKE A WHILE (around 5 minutes in debug or 30 seconds in release) - TIME TO THINK ABOUT ETERNITY. ---" << std::endl;
        const size_t N = 500000;
        std::vector<Rational> terms;
        terms.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            int num = rand() % 2000 - 1000;
            int den = rand() % 999 + 1;
            terms.push_back(Rational(num, den));
        }

        LazyRational sum;
        for (const auto& t : terms) {
            sum += t;
        }

        Rational result = sum.eval();
        EXPECT_GT(result, 0_r);
        std::cout << "Sum of 500k random rationals ≈ " << result.to_double() << std::endl;
    }

} // namespace delta::testing