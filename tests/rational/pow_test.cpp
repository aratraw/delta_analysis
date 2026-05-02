// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/rational/pow_test.cpp
// ============================================================================
// TESTS FOR POWER FUNCTION (EAGER AND LAZY)
// ============================================================================
//
// This file tests the power (exponentiation) functionality:
//   - Eager pow with integer exponent (including negative and zero).
//   - Eager pow with rational exponent (e.g., square root, cube root).
//   - Lazy pow (returns LazyRational) with integer and rational exponents.
//   - Simplification of lazy pow expressions (x^1 → x, x^0 → 1, 1^y → 1).
//   - Structural equality (hash‑consing) of lazy pow nodes.
//
// All tests use the global default epsilon for transcendental computations
// (sqrt, exp, log) that may be required for rational exponents.
// ============================================================================

#pragma once
#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "test_utils.h"
#include "lazy_rational_test_fixture.h"

namespace delta::testing {

    class RationalPowTest : public LazyRationalTestFixture {
    protected:
        void SetUp() override {
            internal::reset_pool();
            reset_default_eps();
        }
        void TearDown() override {
            internal::reset_pool();
        }
    };

    // -------------------------------------------------------------------------
    // 1. Eager pow with integer exponent (returns Rational)
    // -------------------------------------------------------------------------
    /**
     * @test EagerPowIntegerExponent
     * @brief Checks integer exponentiation: positive, zero, negative,
     *        including error cases (0 raised to negative power).
     */
    TEST_F(RationalPowTest, EagerPowIntegerExponent) {
        EXPECT_EQ(delta::pow(2_r, 3), 8_r);
        EXPECT_EQ(delta::pow("2/3"_r, 2), "4/9"_r);
        EXPECT_EQ(delta::pow(0_r, 5), 0_r);
        EXPECT_EQ(delta::pow(2_r, 0), 1_r);
        EXPECT_EQ(delta::pow(2_r, -2), "1/4"_r);
        EXPECT_THROW(delta::pow(0_r, -1), std::domain_error);
    }

    /**
     * @test DISABLED_EagerPowRationalExponent_Debug
     * @brief Debug version of rational exponentiation (disabled by default).
     *        Outputs timing information; kept for manual debugging.
     */
    TEST_F(RationalPowTest, DISABLED_EagerPowRationalExponent_Debug) {
        Rational eps = default_eps();
        internal::reset_pool();

        auto start = std::chrono::high_resolution_clock::now();
        auto log = [&](const char* msg) {
            auto now = std::chrono::high_resolution_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
            std::cerr << "[" << us << " us] " << msg << std::endl;
            };

        log("Testing pow(4, 1/2)");
        Rational p1 = delta::pow(4_r, "1/2"_r, eps);
        log("pow(4, 1/2) done");
        EXPECT_EQ(p1, 2_r);

        log("Testing pow(8, 1/3)");
        Rational p2 = delta::pow(8_r, "1/3"_r, eps);
        log("pow(8, 1/3) done");
        EXPECT_EQ(p2, 2_r);

        log("Testing pow(2, 1/2)");
        Rational p3 = delta::pow(2_r, "1/2"_r, eps);
        log("pow(2, 1/2) done");
        Rational expected_sqrt2 = Rational("14142135623730950488/10000000000000000000");
        EXPECT_RATIONAL_NEAR(p3, expected_sqrt2, "1/1000000000000"_r);

        log("Testing pow(0,0) exception");
        EXPECT_THROW(delta::pow(0_r, 0_r, eps), std::domain_error);
        log("pow(0,0) ok");

        log("Testing pow(0,-1) exception");
        EXPECT_THROW(delta::pow(0_r, -1_r, eps), std::domain_error);
        log("pow(0,-1) ok");

        log("Test finished");
    }

    // -------------------------------------------------------------------------
    // 2. Eager pow with rational exponent (returns Rational)
    // -------------------------------------------------------------------------
    /**
     * @test EagerPowRationalExponent
     * @brief Tests exponentiation with rational exponents (1/2, 1/3, etc.)
     *        using the general formula `exp(log(base) * exp)`.
     */
    TEST_F(RationalPowTest, EagerPowRationalExponent) {
        Rational eps = default_eps();
        Rational p = delta::pow(4_r, "1/2"_r, eps);
        EXPECT_EQ(p, 2_r);

        p = delta::pow(8_r, "1/3"_r, eps);
        EXPECT_EQ(p, 2_r);

        p = delta::pow(2_r, "1/2"_r, eps);
        Rational expected_sqrt2 = Rational("14142135623730950488/10000000000000000000");
        EXPECT_RATIONAL_NEAR(p, expected_sqrt2, "1/1000000000000"_r);

        EXPECT_THROW(delta::pow(0_r, 0_r, eps), std::domain_error);
        EXPECT_THROW(delta::pow(0_r, -1_r, eps), std::domain_error);
    }

    // -------------------------------------------------------------------------
    // 3. Lazy pow with integer exponent (returns LazyRational)
    // -------------------------------------------------------------------------
    /**
     * @test LazyPowIntegerExponent
     * @brief Checks lazy power with integer exponents (positive and negative).
     */
    TEST_F(RationalPowTest, LazyPowIntegerExponent) {
        LazyRational base = Rational(2).as_lazy();
        auto res = delta::lazy_pow(base, 3);
        static_assert(std::is_same_v<decltype(res), LazyRational>);
        EXPECT_EQ(res.eval(), 8_r);

        auto res2 = delta::lazy_pow(base, -2);
        EXPECT_EQ(res2.eval(), "1/4"_r);
    }

    // -------------------------------------------------------------------------
    // 4. Lazy pow with rational exponent (returns LazyRational)
    // -------------------------------------------------------------------------
    /**
     * @test LazyPowRationalExponent
     * @brief Checks lazy power with a rational exponent (e.g., 1/2).
     */
    TEST_F(RationalPowTest, LazyPowRationalExponent) {
        LazyRational base = Rational(2).as_lazy();
        LazyRational exp = Rational(1, 2).as_lazy();
        auto res = delta::lazy_pow(base, exp);
        static_assert(std::is_same_v<decltype(res), LazyRational>);
        Rational expected_sqrt2 = Rational("14142135623730950488/10000000000000000000");
        EXPECT_RATIONAL_NEAR(res.eval(), expected_sqrt2, "1/1000000000000"_r);
    }

    // -------------------------------------------------------------------------
    // 5. Simplification of lazy pow
    // -------------------------------------------------------------------------
    /**
     * @test LazyPowSimplify
     * @brief Verifies algebraic simplification rules for power nodes:
     *        - x^1 → x
     *        - x^0 → 1
     *        - 1^y → 1
     */
    TEST_F(RationalPowTest, LazyPowSimplify) {
        LazyRational base = Rational(2).as_lazy();

        // x^1 -> x
        LazyRational p1 = delta::lazy_pow(base, 1);
        p1.simplify_inplace();
        EXPECT_EQ(p1.eval(), 2_r);

        // x^0 -> 1
        LazyRational p0 = delta::lazy_pow(base, 0);
        p0.simplify_inplace();
        EXPECT_EQ(p0.eval(), 1_r);

        // 1^y -> 1
        LazyRational one = Rational(1).as_lazy();
        LazyRational one_pow = delta::lazy_pow(one, Rational(1, 2).as_lazy());
        one_pow.simplify_inplace();
        EXPECT_EQ(one_pow.eval(), 1_r);
    }

    // -------------------------------------------------------------------------
    // 6. Structural equality for lazy pow
    // -------------------------------------------------------------------------
    /**
     * @test LazyPowStructuralEquality
     * @brief Checks that identical lazy pow expressions share the same
     *        clean node after simplification (hash‑consing).
     */
    TEST_F(RationalPowTest, LazyPowStructuralEquality) {
        LazyRational a = delta::lazy_pow(Rational(2).as_lazy(), Rational(1, 2).as_lazy());
        LazyRational b = delta::lazy_pow(Rational(2).as_lazy(), Rational(1, 2).as_lazy());
        EXPECT_TRUE(a == b);

        LazyRational c = delta::lazy_pow(Rational(2).as_lazy(), Rational(1, 3).as_lazy());
        EXPECT_FALSE(a == c);
    }

} // namespace delta::testing