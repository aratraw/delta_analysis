// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

#include <gtest/gtest.h>
#include "../test_fixtures.h"

namespace delta::testing {

    /**
     * @class AdaptiveOperatorTest
     * @brief Tests for the AdaptiveOperator class.
     *
     * AdaptiveOperator selects a point inside an interval based on the
     * deviation from linearity and the current maximum oscillation.
     * The operator is designed to cluster points in regions where the function
     * changes rapidly.
     */
    class AdaptiveOperatorTest : public DeltaTest {};

    /**
     * @test Verifies that when the oscillation (max_oscillation) is zero,
     * the operator returns the midpoint regardless of function values.
     */
    TEST_F(AdaptiveOperatorTest, ReturnsMidpointWhenOscillationZero) {
        AdaptiveOperator op(1_r / 10_r, 1_r / 10_r);
        auto info = make_info(0_r, 1_r, 5_r, 5_r, 0_r);
        Addr result = op(0_r, 1_r, info);
        EXPECT_EQ(result, 1_r / 2_r);
    }

    /**
     * @test Verifies that if the difference between endpoint values is below
     * the threshold, the operator returns the midpoint (no adaptive shift).
     */
    TEST_F(AdaptiveOperatorTest, ReturnsMidpointWhenDiffBelowThreshold) {
        AdaptiveOperator op(1_r / 10_r, 1_r / 10_r);
        auto info = make_info(0_r, 1_r, 5_r, 5_r + 1_r / 20_r, 1_r);
        Addr result = op(0_r, 1_r, info);
        EXPECT_EQ(result, 1_r / 2_r);
    }

    /**
     * @test Checks the calculation of the alpha parameter when the deviation
     * is exactly half the maximum oscillation.
     */
    TEST_F(AdaptiveOperatorTest, AlphaCalculationWorks) {
        AdaptiveOperator op(1_r / 10_r, 1_r / 10_r);
        auto info = make_info(0_r, 1_r, 0_r, 1_r / 2_r, 1_r);
        Addr result = op(0_r, 1_r, info);
        EXPECT_EQ(result, 1_r / 2_r);
    }

    /**
     * @test Ensures that alpha is clamped to the lower epsilon bound
     * when the computed alpha falls below epsilon.
     */
    TEST_F(AdaptiveOperatorTest, AlphaClampedToEpsilon) {
        AdaptiveOperator op(1_r / 10_r, 1_r / 5_r);
        auto info = make_info(0_r, 1_r, 0_r, 15_r / 100_r, 1_r);
        Addr result = op(0_r, 1_r, info);
        EXPECT_EQ(result, 1_r / 5_r);
    }

    /**
     * @test Ensures that alpha is clamped to (1 - epsilon) when the computed
     * alpha exceeds that upper bound.
     */
    TEST_F(AdaptiveOperatorTest, AlphaClampedToOneMinusEpsilon) {
        AdaptiveOperator op(1_r / 10_r, 1_r / 5_r);
        auto info = make_info(0_r, 1_r, 0_r, 95_r / 100_r, 1_r);
        Addr result = op(0_r, 1_r, info);
        EXPECT_EQ(result, 4_r / 5_r);
    }

    /**
     * @test When the difference equals the threshold exactly, the operator
     * still returns the midpoint (strict inequality is used for the check).
     */
    TEST_F(AdaptiveOperatorTest, ExactThresholdUsesMidpoint) {
        AdaptiveOperator op(1_r / 10_r, 1_r / 10_r);
        auto info = make_info(0_r, 1_r, 0_r, 1_r / 10_r, 1_r);
        Addr result = op(0_r, 1_r, info);
        EXPECT_EQ(result, 1_r / 2_r);
    }

    /**
     * @test Verifies that the operator works correctly with very large
     * rational numbers and that the returned point lies strictly inside the interval.
     */
    TEST_F(AdaptiveOperatorTest, LargeNumbers) {
        AdaptiveOperator op(1_r / 100_r, 1_r / 100_r);
        // left ≈ 0.990..., right = 1
        Addr left = "4478508612376765966049"_r / "4521910375044022450050"_r;
        Addr right = 1_r;
        // max_oscillation ≈ 0.94148, df ≈ 0.1883
        Dist max_osc = 941480149401_r / 1000000000000_r;
        Dist df = max_osc * 2_r / 10_r;
        Val f_left = 0_r;
        Val f_right = df;
        auto info = make_info(left, right, f_left, f_right, max_osc);
        Addr result = op(left, right, info);
        EXPECT_TRUE(left < result && result < right);
    }

    /**
     * @test Randomized test that ensures the operator never returns a point
     * outside the interval for a wide range of inputs.
     */
    TEST_F(AdaptiveOperatorTest, NeverReturnsOutside) {
        AdaptiveOperator op(1_r / 100_r, 1_r / 100_r);
        const int iterations = 1000;
        for (int i = 0; i < iterations; ++i) {
            // Generate random left and right in [0,1] with left < right
            double a = static_cast<double>(rand()) / RAND_MAX;
            double b = a + static_cast<double>(rand()) / RAND_MAX * (1.0 - a);
            Addr left = Rational(static_cast<int>(a * 10000), 10000);
            Addr right = Rational(static_cast<int>(b * 10000), 10000);
            if (left >= right) std::swap(left, right);

            Dist max_osc = Rational(rand() % 100, 100);
            Dist df = Rational(rand() % 100, 100);
            Val f_left = 0_r;
            Val f_right = df;

            auto info = make_info(left, right, f_left, f_right, max_osc);
            Addr mid = op(left, right, info);
            EXPECT_TRUE(left < mid && mid < right);
        }
    }

} // namespace delta::testing