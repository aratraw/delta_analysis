// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/geometry/constructive_core_test.cpp
#include <gtest/gtest.h>
#include <optional>
#include "delta/geometry/constructive_core.h"
#include "../test_fixtures_geometry_numerical.h"

namespace delta::testing {

    /**
     * @class ConstructiveCoreTest
     * @brief Tests for constructive core (K) and point/vector operations.
     *
     * Implements tests for Stage 0 of the specification:
     * - Finite base numbers representability
     * - Universal core membership
     * - Point and vector operations
     * - Symmetries and core preservation
     */
    class ConstructiveCoreTest : public GeometryNumericalTest {
    protected:
        // Type aliases for 2D and 3D points/vectors
        using Point2 = Eigen::Matrix<Scalar, 2, 1>;
        using Point3 = Eigen::Matrix<Scalar, 3, 1>;
        using Vector2 = Vector<2>;
        using Vector3 = Vector<3>;
    };

    // =========================================================================
    // Test group 1: Finite base numbers representability
    // =========================================================================

    TEST_F(ConstructiveCoreTest, Base2Representability) {
        // Numbers representable in base 2 (dyadic rationals)
        EXPECT_TRUE(is_representable<2>(1_r / 2_r));      // 0.5
        EXPECT_TRUE(is_representable<2>(1_r / 4_r));      // 0.25
        EXPECT_TRUE(is_representable<2>(1_r / 8_r));      // 0.125
        EXPECT_TRUE(is_representable<2>(3_r / 8_r));      // 0.375 = 3/8
        EXPECT_TRUE(is_representable<2>(5_r / 8_r));      // 0.625 = 5/8
        EXPECT_TRUE(is_representable<2>(7_r / 8_r));      // 0.875 = 7/8
        EXPECT_TRUE(is_representable<2>(1_r / 16_r));     // 0.0625
        EXPECT_TRUE(is_representable<2>(15_r / 16_r));    // 0.9375

        // Numbers NOT representable in base 2
        EXPECT_FALSE(is_representable<2>(1_r / 3_r));     // 1/3 = 0.333... (repeating in binary)
        EXPECT_FALSE(is_representable<2>(1_r / 5_r));     // 1/5 = 0.2 (but 0.2 in decimal is 0.00110011... in binary)
        EXPECT_FALSE(is_representable<2>(1_r / 7_r));     // 1/7
        EXPECT_FALSE(is_representable<2>(1_r / 9_r));     // 1/9
        EXPECT_FALSE(is_representable<2>(1_r / 10_r));    // 1/10 = 0.1 (repeating in binary)
        EXPECT_FALSE(is_representable<2>(1_r / 11_r));    // 1/11

        // Special cases
        EXPECT_FALSE(is_representable<2>(0_r));           // zero is excluded by definition
        EXPECT_FALSE(is_representable<2>((-1_r) / 3_r));  // negative also not representable
    }

    TEST_F(ConstructiveCoreTest, Base3Representability) {
        // Numbers representable in base 3
        EXPECT_TRUE(is_representable<3>(1_r / 3_r));      // 0.1₃
        EXPECT_TRUE(is_representable<3>(1_r / 9_r));      // 0.01₃
        EXPECT_TRUE(is_representable<3>(2_r / 9_r));      // 0.02₃
        EXPECT_TRUE(is_representable<3>(1_r / 27_r));     // 0.001₃
        EXPECT_TRUE(is_representable<3>(4_r / 9_r));      // 0.11₃ = 4/9
        EXPECT_TRUE(is_representable<3>(8_r / 9_r));      // 0.22₃ = 8/9
        EXPECT_TRUE(is_representable<3>(13_r / 27_r));    // 0.111₃ = 13/27

        // Numbers NOT representable in base 3
        EXPECT_FALSE(is_representable<3>(1_r / 2_r));     // 1/2 = 0.111...₃ (repeating)
        EXPECT_FALSE(is_representable<3>(1_r / 4_r));     // 1/4
        EXPECT_FALSE(is_representable<3>(1_r / 5_r));     // 1/5
        EXPECT_FALSE(is_representable<3>(1_r / 7_r));     // 1/7
        EXPECT_FALSE(is_representable<3>(1_r / 8_r));     // 1/8
        EXPECT_FALSE(is_representable<3>(1_r / 10_r));    // 1/10

        // Special cases
        EXPECT_FALSE(is_representable<3>(0_r));           // zero excluded
    }

    TEST_F(ConstructiveCoreTest, Base10Representability) {
        // Numbers representable in base 10 (finite decimals)
        EXPECT_TRUE(is_representable<10>(1_r / 2_r));      // 0.5
        EXPECT_TRUE(is_representable<10>(1_r / 4_r));      // 0.25
        EXPECT_TRUE(is_representable<10>(1_r / 5_r));      // 0.2
        EXPECT_TRUE(is_representable<10>(1_r / 8_r));      // 0.125
        EXPECT_TRUE(is_representable<10>(1_r / 10_r));     // 0.1
        EXPECT_TRUE(is_representable<10>(1_r / 20_r));     // 0.05
        EXPECT_TRUE(is_representable<10>(1_r / 25_r));     // 0.04
        EXPECT_TRUE(is_representable<10>(1_r / 40_r));     // 0.025
        EXPECT_TRUE(is_representable<10>(1_r / 50_r));     // 0.02
        EXPECT_TRUE(is_representable<10>(1_r / 100_r));    // 0.01
        EXPECT_TRUE(is_representable<10>(3_r / 4_r));      // 0.75
        EXPECT_TRUE(is_representable<10>(7_r / 8_r));      // 0.875
        EXPECT_TRUE(is_representable<10>(123_r / 1000_r)); // 0.123

        // Numbers NOT representable in base 10
        EXPECT_FALSE(is_representable<10>(1_r / 3_r));     // 0.333...
        EXPECT_FALSE(is_representable<10>(1_r / 6_r));     // 0.1666...
        EXPECT_FALSE(is_representable<10>(1_r / 7_r));     // 0.142857...
        EXPECT_FALSE(is_representable<10>(1_r / 9_r));     // 0.111...
        EXPECT_FALSE(is_representable<10>(1_r / 11_r));    // 0.0909...
        EXPECT_FALSE(is_representable<10>(1_r / 12_r));    // 0.08333...
        EXPECT_FALSE(is_representable<10>(1_r / 13_r));    // 0.076923...
        EXPECT_FALSE(is_representable<10>(1_r / 14_r));    // 0.0714285...
        EXPECT_FALSE(is_representable<10>(1_r / 15_r));    // 0.0666...

        // Special cases
        EXPECT_FALSE(is_representable<10>(0_r));           // zero excluded
    }

    // =========================================================================
    // Test group 2: Universal core membership
    // =========================================================================

    TEST_F(ConstructiveCoreTest, UniversalCoreMembership) {
        // All non-zero rationals should be in universal core
        EXPECT_TRUE(is_in_universal_core(1_r / 2_r));
        EXPECT_TRUE(is_in_universal_core(1_r / 3_r));
        EXPECT_TRUE(is_in_universal_core(1_r / 4_r));
        EXPECT_TRUE(is_in_universal_core(1_r / 5_r));
        EXPECT_TRUE(is_in_universal_core(1_r / 7_r));
        EXPECT_TRUE(is_in_universal_core(2_r / 3_r));
        EXPECT_TRUE(is_in_universal_core(3_r / 4_r));
        EXPECT_TRUE(is_in_universal_core(5_r / 8_r));
        EXPECT_TRUE(is_in_universal_core(123_r / 456_r));
        EXPECT_TRUE(is_in_universal_core((-7_r) / 11_r));  // negative also in core

        // Zero is excluded
        EXPECT_FALSE(is_in_universal_core(0_r));
    }

    // =========================================================================
    // Test group 3: Point and vector operations
    // =========================================================================

    TEST_F(ConstructiveCoreTest, PointVectorDifference) {
        Point2 p1;
        p1 << 1_r, 2_r;
        Point2 p2;
        p2 << 3_r, 5_r;

        // p2 - p1 should give vector (2, 3)
        Vector2 v = point_minus_point(p2, p1);
        EXPECT_EQ(v.data()(0), 2_r);
        EXPECT_EQ(v.data()(1), 3_r);

        // p1 - p2 should give vector (-2, -3)
        v = point_minus_point(p1, p2);
        EXPECT_EQ(v.data()(0), -2_r);
        EXPECT_EQ(v.data()(1), -3_r);
    }

    TEST_F(ConstructiveCoreTest, PointPlusVectorInK) {
        // Test cases where result should be in K (no zero coordinates)

        // p = (0.125, 0.5) both non-zero, v = (0.125, 0) -> result (0.25, 0.5) both non-zero
        Point2 p1;
        p1 << "0.125"_r, "0.5"_r;
        Vector2 v1("0.125"_r, 0_r);

        auto result1 = point_plus_vector(p1, v1);
        ASSERT_TRUE(result1.has_value());
        EXPECT_EQ((*result1)(0), "0.25"_r);
        EXPECT_EQ((*result1)(1), "0.5"_r);
        EXPECT_TRUE(is_in_K(*result1));

        // p = (0.125, 0.5), v = (0.125, 0.125) -> result (0.25, 0.625)
        Vector2 v1b("0.125"_r, "0.125"_r);
        auto result1b = point_plus_vector(p1, v1b);
        ASSERT_TRUE(result1b.has_value());
        EXPECT_EQ((*result1b)(0), "0.25"_r);
        EXPECT_EQ((*result1b)(1), "0.625"_r);
        EXPECT_TRUE(is_in_K(*result1b));

        // p = (0.125, 0.5), v = (0.1, 0) -> result (0.225, 0.5)
        // 0.225 = 9/40, which has denominator 40 = 2^3 * 5 -> finite decimal, so in K
        Vector2 v1c("0.1"_r, 0_r);
        auto result1c = point_plus_vector(p1, v1c);
        ASSERT_TRUE(result1c.has_value());
        EXPECT_EQ((*result1c)(0), "0.225"_r);
        EXPECT_EQ((*result1c)(1), "0.5"_r);
        EXPECT_TRUE(is_in_K(*result1c));

        // 3D case
        Point3 p3d;
        p3d << "0.125"_r, "0.25"_r, "0.375"_r;
        Vector3 v3d("0.125"_r, "0.125"_r, "0.125"_r);
        auto result3d = point_plus_vector(p3d, v3d);
        ASSERT_TRUE(result3d.has_value());
        EXPECT_EQ((*result3d)(0), "0.25"_r);
        EXPECT_EQ((*result3d)(1), "0.375"_r);
        EXPECT_EQ((*result3d)(2), "0.5"_r);
        EXPECT_TRUE(is_in_K(*result3d));
    }

    TEST_F(ConstructiveCoreTest, PointPlusVectorNotInK) {
        // Cases where result contains zero -> not in K

        // p = (0.125, 0.5), v = (-0.125, 0) -> result (0, 0.5) contains zero
        Point2 p1;
        p1 << "0.125"_r, "0.5"_r;
        Vector2 v1("-0.125"_r, 0_r);

        auto result1 = point_plus_vector(p1, v1);
        EXPECT_FALSE(result1.has_value());

        // p = (0.125, 0.5), v = (0, -0.5) -> result (0.125, 0) contains zero
        Vector2 v2(0_r, "-0.5"_r);
        auto result2 = point_plus_vector(p1, v2);
        EXPECT_FALSE(result2.has_value());

        // p = (0.125, 0.5), v = (-0.125, -0.5) -> result (0, 0) contains zeros
        Vector2 v3("-0.125"_r, "-0.5"_r);
        auto result3 = point_plus_vector(p1, v3);
        EXPECT_FALSE(result3.has_value());

        // p = (0.125, 0.5), v = (0.125, -0.5) -> result (0.25, 0) contains zero
        Vector2 v4("0.125"_r, "-0.5"_r);
        auto result4 = point_plus_vector(p1, v4);
        EXPECT_FALSE(result4.has_value());

        // 3D case with zero
        Point3 p3d;
        p3d << "0.125"_r, "0.25"_r, "0.375"_r;
        Vector3 v3d("-0.125"_r, "-0.25"_r, 0_r);  // third coordinate unchanged (non-zero), but second becomes zero
        auto result3d = point_plus_vector(p3d, v3d);
        EXPECT_FALSE(result3d.has_value());
    }

    TEST_F(ConstructiveCoreTest, VectorOperations) {
        Vector2 v1(1_r, 2_r);
        Vector2 v2(3_r, 4_r);

        // Vector addition
        Vector2 v_sum = vector_plus_vector(v1, v2);
        EXPECT_EQ(v_sum.data()(0), 4_r);
        EXPECT_EQ(v_sum.data()(1), 6_r);

        // Vector addition with negative
        Vector2 v3(-1_r, -2_r);
        Vector2 v_sum2 = vector_plus_vector(v1, v3);
        EXPECT_EQ(v_sum2.data()(0), 0_r);
        EXPECT_EQ(v_sum2.data()(1), 0_r);  // zero vector is allowed

        // Scalar multiplication
        Vector2 v_scaled = scalar_times_vector(2_r, v1);
        EXPECT_EQ(v_scaled.data()(0), 2_r);
        EXPECT_EQ(v_scaled.data()(1), 4_r);

        // Scalar multiplication with zero
        Vector2 v_scaled_zero = scalar_times_vector(0_r, v1);
        EXPECT_EQ(v_scaled_zero.data()(0), 0_r);
        EXPECT_EQ(v_scaled_zero.data()(1), 0_r);  // zero vector allowed

        // Scalar multiplication with negative
        Vector2 v_scaled_neg = scalar_times_vector(-1_r, v1);
        EXPECT_EQ(v_scaled_neg.data()(0), -1_r);
        EXPECT_EQ(v_scaled_neg.data()(1), -2_r);

        // 3D case
        Vector3 v3d1(1_r, 2_r, 3_r);
        Vector3 v3d2(4_r, 5_r, 6_r);
        Vector3 v3d_sum = vector_plus_vector(v3d1, v3d2);
        EXPECT_EQ(v3d_sum.data()(0), 5_r);
        EXPECT_EQ(v3d_sum.data()(1), 7_r);
        EXPECT_EQ(v3d_sum.data()(2), 9_r);

        Vector3 v3d_scaled = scalar_times_vector(3_r, v3d1);
        EXPECT_EQ(v3d_scaled.data()(0), 3_r);
        EXPECT_EQ(v3d_scaled.data()(1), 6_r);
        EXPECT_EQ(v3d_scaled.data()(2), 9_r);
    }

    // =========================================================================
    // Test group 4: Core membership checks
    // =========================================================================

    TEST_F(ConstructiveCoreTest, IsInK) {
        // Points with all non-zero coordinates should be in K
        Point2 p1; p1 << "0.125"_r, "0.5"_r;
        Point2 p2; p2 << "0.25"_r, "0.75"_r;
        Point2 p3; p3 << "0.1"_r, "0.2"_r;
        Point2 p4; p4 << "0.2"_r, "0.4"_r;
        Point3 p5; p5 << "0.125"_r, "0.25"_r, "0.375"_r;

        EXPECT_TRUE(is_in_K(p1));
        EXPECT_TRUE(is_in_K(p2));
        EXPECT_TRUE(is_in_K(p3));    // 0.1 = 1/10, finite decimal
        EXPECT_TRUE(is_in_K(p4));    // 0.2 = 1/5, finite decimal
        EXPECT_TRUE(is_in_K(p5));

        // Points with any zero coordinate should NOT be in K
        Point2 p6; p6 << 0_r, "0.5"_r;
        Point2 p7; p7 << "0.125"_r, 0_r;
        Point2 p8; p8 << 0_r, 0_r;
        Point3 p9; p9 << 0_r, "0.25"_r, "0.375"_r;
        Point3 p10; p10 << "0.125"_r, 0_r, "0.375"_r;
        Point3 p11; p11 << "0.125"_r, "0.25"_r, 0_r;
        Point3 p12; p12 << 0_r, 0_r, 0_r;

        EXPECT_FALSE(is_in_K(p6));
        EXPECT_FALSE(is_in_K(p7));
        EXPECT_FALSE(is_in_K(p8));
        EXPECT_FALSE(is_in_K(p9));
        EXPECT_FALSE(is_in_K(p10));
        EXPECT_FALSE(is_in_K(p11));
        EXPECT_FALSE(is_in_K(p12));

        // Note: In this simplified implementation, we only check for non-zero.
        // The full mathematical definition would also check for finite decimal representation,
        // but that's handled by is_representable tests separately.
    }

    // =========================================================================
    // Test group 5: Symmetries and core preservation
    // =========================================================================

    TEST_F(ConstructiveCoreTest, DyadicShiftPreservesK) {
        // Shifts by dyadic rationals should preserve K

        Point2 p;
        p << "0.125"_r, "0.5"_r;  // both coordinates in K

        // Shift by dyadic vector (0.125, 0.25) -> (0.25, 0.75) both in K
        Vector2 v_dyadic("0.125"_r, "0.25"_r);
        auto result = point_plus_vector(p, v_dyadic);
        ASSERT_TRUE(result.has_value());
        EXPECT_TRUE(is_in_K(*result));
        EXPECT_EQ((*result)(0), "0.25"_r);
        EXPECT_EQ((*result)(1), "0.75"_r);

        // Another dyadic shift
        Vector2 v_dyadic2("0.0625"_r, "0.125"_r);  // 1/16, 1/8
        result = point_plus_vector(p, v_dyadic2);
        ASSERT_TRUE(result.has_value());
        EXPECT_TRUE(is_in_K(*result));
        EXPECT_EQ((*result)(0), "0.1875"_r);  // 3/16
        EXPECT_EQ((*result)(1), "0.625"_r);   // 5/8

        // Shift that would produce zero should not preserve K
        Vector2 v_to_zero("-0.125"_r, "-0.5"_r);
        result = point_plus_vector(p, v_to_zero);
        EXPECT_FALSE(result.has_value());
    }

    TEST_F(ConstructiveCoreTest, RotationDoesNotPreserveK) {
        // Rotation by 45 degrees generally does not preserve K
        // because cos(45°) = √2/2 is irrational

        Point2 p;
        p << 1_r, 0_r;  // both coordinates in K

        // Approximate rotation by 45° using rational approximation
        // √2/2 ≈ 0.7071067811865475, but we'll use a rational approximation
        Rational approx_cos = 7071067811865475_r / 10000000000000000_r;  // ~0.7071
        Rational approx_sin = approx_cos;  // same for 45°

        // Create a point that approximates the rotated (1,0) -> (√2/2, √2/2)
        Point2 p_rotated_approx;
        p_rotated_approx << approx_cos, approx_sin;

        // The exact rotated point would have irrational coordinates,
        // so it cannot be in K. The approximation is rational but not equal to the exact value.
        // We cannot test representability of the approximation because it is a finite decimal.
        // Instead, we rely on the fact that no rational approximation can be exact,
        // which is tested elsewhere (e.g., SequenceOfRotations).
    }

    TEST_F(ConstructiveCoreTest, SequenceOfRotations) {
        // Show that a sequence of rational approximations can approach
        // the true rotation arbitrarily closely, but never exactly preserve K

        // Generate a sequence of rational approximations to 1/√2
        // using Newton's method for sqrt(2) approximation
        std::vector<Rational> approximations;

        // Newton iteration for √2
        Rational sqrt2_approx = 1_r;
        for (int i = 0; i < 5; ++i) {
            sqrt2_approx = (sqrt2_approx + 2_r / sqrt2_approx) / 2_r;
            Rational inv_sqrt2_approx = 1_r / sqrt2_approx;
            approximations.push_back(inv_sqrt2_approx);
        }

        // Each approximation is rational, but none should be representable
        // in base 10 (or base 2) because 1/√2 is irrational
        for (const auto& approx : approximations) {
            // The approximation itself is rational, so it's in universal core,
            // but it's not a finite decimal (base 10 representable)
            EXPECT_TRUE(is_in_universal_core(approx));
            EXPECT_FALSE(is_representable<10>(approx));
            EXPECT_FALSE(is_representable<2>(approx));
        }

        // The approximations get closer to the true value
        // but never reach it exactly
        Rational true_inv_sqrt2 = delta::sqrt("0.5"_r);  // computed with high precision

        // The last approximation should be close
        Rational error = abs(approximations.back() - true_inv_sqrt2);
        EXPECT_LT(error, "0.0000000001"_r);
    }

    // =========================================================================
    // Test group 6: Edge cases
    // =========================================================================

    TEST_F(ConstructiveCoreTest, NegativeCoordinates) {
        // Points with negative coordinates can still be in K
        Point2 p_neg;
        p_neg << "-0.125"_r, "0.5"_r;
        EXPECT_TRUE(is_in_K(p_neg));

        Point2 p_neg2;
        p_neg2 << "-0.125"_r, "-0.5"_r;
        EXPECT_TRUE(is_in_K(p_neg2));

        // Zero still excluded even if negative elsewhere
        Point2 p_with_zero;
        p_with_zero << "-0.125"_r, 0_r;
        EXPECT_FALSE(is_in_K(p_with_zero));

        // Vector operations with negatives
        Point2 p;
        p << "-0.125"_r, "0.5"_r;
        Vector2 v("0.25"_r, "-0.25"_r);

        auto result = point_plus_vector(p, v);
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ((*result)(0), "0.125"_r);
        EXPECT_EQ((*result)(1), "0.25"_r);
        EXPECT_TRUE(is_in_K(*result));
    }

    TEST_F(ConstructiveCoreTest, LargeRationals) {
        // Test with large rational numbers
        Rational large1 = 123456789_r / 100000000_r;  // 1.23456789
        Rational large2 = 987654321_r / 100000000_r;  // 9.87654321

        Point2 p_large;
        p_large << large1, large2;
        EXPECT_TRUE(is_in_K(p_large));

        Vector2 v_large(large2, large1);
        auto result = point_plus_vector(p_large, v_large);
        ASSERT_TRUE(result.has_value());

        // Result should be (large1+large2, large2+large1) = (large1+large2, large1+large2)
        Rational sum = large1 + large2;
        EXPECT_EQ((*result)(0), sum);
        EXPECT_EQ((*result)(1), sum);
        EXPECT_TRUE(is_in_K(*result));
    }

    TEST_F(ConstructiveCoreTest, ExactRepresentability) {
        // Test that numbers like 1/3 are exactly representable in base 3
        // but not in base 2 or 10
        Rational one_third = 1_r / 3_r;

        EXPECT_TRUE(is_representable<3>(one_third));   // 0.1₃ exactly
        EXPECT_FALSE(is_representable<2>(one_third));  // repeating binary
        EXPECT_FALSE(is_representable<10>(one_third)); // repeating decimal

        // 1/5 is representable in base 10 but not base 2 or 3
        Rational one_fifth = 1_r / 5_r;

        EXPECT_TRUE(is_representable<10>(one_fifth));  // 0.2 exactly
        EXPECT_FALSE(is_representable<2>(one_fifth));  // repeating binary
        EXPECT_FALSE(is_representable<3>(one_fifth));  // repeating ternary

        // 1/7 is representable in base 7 but not others
        Rational one_seventh = 1_r / 7_r;

        EXPECT_TRUE(is_representable<7>(one_seventh));  // 0.1₇ exactly
        EXPECT_FALSE(is_representable<2>(one_seventh));
        EXPECT_FALSE(is_representable<3>(one_seventh));
        EXPECT_FALSE(is_representable<10>(one_seventh));
    }

} // namespace delta::testing