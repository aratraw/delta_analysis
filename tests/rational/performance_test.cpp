// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/rational/performance_test.cpp
// ============================================================================
// PERFORMANCE TESTS FOR RATIONAL AND LAZYRATIONAL
// ============================================================================
//
// This file contains performance‑oriented tests that do not measure exact
// timings but verify that certain operations complete within reasonable
// resource limits (no stack overflow, no excessive runtime). The tests are:
//   - Harmonic series summation (eager and lazy modes) up to 10 000 terms.
//   - Deep tree of additions (10 000 nested additions) – ensures iterative
//     evaluation does not overflow the stack.
//   - Large factorial product (500 terms) – checks multiplication of big integers.
//   - Nested transcendental functions with simplification.
//   - Huge random lazy addition (500 000 terms) – stresses the batching
//     mechanism in the SUM node and the evaluation engine.
//
// All tests are deterministic; the random test uses a fixed seed.
// ============================================================================

#pragma once
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include "delta/core/rational.h"
#include <boost/multiprecision/cpp_dec_float.hpp>
#include "test_utils.h"

namespace delta::testing {

    class RationalPerformanceTest : public RationalTest {};

    // -------------------------------------------------------------------------
    // 1. Harmonic series (eager mode) – immediate Rational
    // -------------------------------------------------------------------------
    /**
     * @test HarmonicSeries10000EagerMode
     * @brief Sums the harmonic series 1 + 1/2 + ... + 1/10000 eagerly
     *        and compares with a high‑precision reference (cpp_dec_float_100).
     */
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
    /**
     * @test HarmonicSeries10000LazyMode
     * @brief Builds a lazy SUM tree for the harmonic series (10 000 terms),
     *        then evaluates it once. Verifies correctness against the same reference.
     */
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
    /**
     * @test DeepTree10000LazyMode
     * @brief Creates a deeply nested tree of 10 000 additions (right‑associative)
     *        by repeatedly adding 1 to an accumulator. This tests that evaluation
     *        does not cause a stack overflow (iterative traversal is used).
     */
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
    /**
     * @test LargeProduct500
     * @brief Computes 500! and checks that the result is positive (no overflow).
     */
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
    /**
     * @test NestedTranscendentalsLazy
     * @brief Builds a deep expression sin(cos(exp(log(x)))) with x = 2,
     *        simplifies it, and compares with the eagerly computed value.
     */
    TEST_F(RationalPerformanceTest, NestedTranscendentalsLazy) {
        LazyRational x = Rational(2).as_lazy();
        LazyRational expr = delta::lazy_sin(delta::lazy_cos(delta::lazy_exp(delta::lazy_log(x))));
        expr.simplify_inplace();
        Rational expected = delta::sin(delta::cos(2_r));
        EXPECT_RATIONAL_NEAR(expr.eval(), expected, default_eps());
    }

    // -------------------------------------------------------------------------
    // 6. Huge random lazy additions (500k) – stress test for batching
    // -------------------------------------------------------------------------
    /**
     * @test HugeRandomLazyAdditions500k
     * @brief Adds 500 000 random rational numbers using the lazy mechanism.
     *        This stresses the batching in the SUM node and the pyramidal
     *        reduction. The test only checks that the result is finite and
     *        non‑zero (it is extremely unlikely to be zero).
     */
    TEST_F(RationalPerformanceTest, HugeRandomLazyAdditions500k) {
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
        // The sum of 500 000 random rationals is extremely unlikely to be exactly zero.
        // Moreover, if it were zero, the test would still pass, but we require at least
        // that the evaluation completes without exceptions and produces a finite value.
        // We check non‑zero to ensure something meaningful was computed.
        EXPECT_NE(result, 0_r);
    }

} // namespace delta::testing