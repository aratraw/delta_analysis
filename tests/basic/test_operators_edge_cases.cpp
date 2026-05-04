// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

//test_operators_edge_cases.cpp
#include <gtest/gtest.h>
#include "../test_fixtures.h"

namespace delta::testing {

    /**
     * @class MidpointOperatorTest
     * @brief Tests for the MidpointOperator.
     *
     * MidpointOperator always returns the arithmetic mean of the two endpoints,
     * regardless of the context provided in IntervalInfo.
     */
    class MidpointOperatorTest : public DeltaTest {};

    /**
     * @test Verify that the midpoint operator always returns the midpoint,
     *       independent of the interval endpoints.
     */
    TEST_F(MidpointOperatorTest, AlwaysReturnsMidpoint) {
        MidpointOperator op;
        auto info = make_info(0_r, 1_r, 0_r, 0_r, 1_r);
        Addr result = op(0_r, 1_r, info);
        EXPECT_EQ(result, 1_r / 2_r);

        result = op(2_r, 5_r, info);
        EXPECT_EQ(result, 7_r / 2_r);
    }

    /**
     * @class FixedLambdaOperatorTest
     * @brief Tests for FixedLambdaOperator.
     *
     * FixedLambdaOperator places a new point at a fixed fraction λ of the interval.
     * If λ is outside (0,1) it falls back to the midpoint.
     */
    class FixedLambdaOperatorTest : public DeltaTest {};

    /**
     * @test With λ in (0,1), the operator should return the point at that fraction.
     */
    TEST_F(FixedLambdaOperatorTest, LambdaInRange) {
        FixedLambdaOperator op(1_r / 3_r);
        auto info = make_info(0_r, 1_r, 0_r, 0_r, 1_r);
        Addr result = op(0_r, 1_r, info);
        EXPECT_EQ(result, 1_r / 3_r);
    }

    /**
     * @test λ = 0 is out of range; the operator should fall back to the midpoint.
     */
    TEST_F(FixedLambdaOperatorTest, LambdaZero) {
        FixedLambdaOperator op(0_r);
        auto info = make_info(0_r, 1_r, 0_r, 0_r, 1_r);
        Addr result = op(0_r, 1_r, info);
        EXPECT_EQ(result, 1_r / 2_r);
    }

    /**
     * @test λ = 1 is out of range; the operator should fall back to the midpoint.
     */
    TEST_F(FixedLambdaOperatorTest, LambdaOne) {
        FixedLambdaOperator op(1_r);
        auto info = make_info(0_r, 1_r, 0_r, 0_r, 1_r);
        Addr result = op(0_r, 1_r, info);
        EXPECT_EQ(result, 1_r / 2_r);
    }

    /**
     * @test Negative λ is out of range; the operator should fall back to the midpoint.
     */
    TEST_F(FixedLambdaOperatorTest, LambdaNegative) {
        FixedLambdaOperator op(-1_r / 2_r);
        auto info = make_info(0_r, 1_r, 0_r, 0_r, 1_r);
        Addr result = op(0_r, 1_r, info);
        EXPECT_EQ(result, 1_r / 2_r);
    }

    /**
     * @class DynamicLambdaOperatorTest
     * @brief Tests for DynamicLambdaOperator.
     *
     * DynamicLambdaOperator uses a level‑dependent function to determine the fraction λ.
     * For levels where the generated λ lies in (0,1) it returns that fraction;
     * otherwise it falls back to the midpoint.
     */
    class DynamicLambdaOperatorTest : public DeltaTest {};

    /**
     * @test For level 0, the generator returns 1/2, so the result should be the midpoint.
     *       For level 1, the generator returns 1/3, so the result should be at 1/3.
     */
    TEST_F(DynamicLambdaOperatorTest, LevelDependent) {
        auto gen = [](std::size_t level) { return Rational(1) / Rational(level + 2); };
        DynamicLambdaOperator op(gen);
        auto info0 = make_info(0_r, 1_r, 0_r, 0_r, 1_r, 0);
        auto info1 = make_info(0_r, 1_r, 0_r, 0_r, 1_r, 1);

        Addr result0 = op(0_r, 1_r, info0);
        Addr result1 = op(0_r, 1_r, info1);

        EXPECT_RATIONAL_NEAR(result0, 1_r / 2_r, Rational(1, 1000000));
        EXPECT_RATIONAL_NEAR(result1, 1_r / 3_r, Rational(1, 1000000));
    }

} // namespace delta::testing