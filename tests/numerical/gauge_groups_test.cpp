// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/numerical/gauge_groups_test.cpp
// ============================================================================
// MATHEMATICAL VERIFICATION OF GAUGE GROUPS
// (fully rational, no double anywhere)
// ============================================================================
//
// This test suite validates the basic group axioms and exponential / logarithm
// maps for U(1) = SO(2), SU(2) and SU(3).  All computations use exact
// rational arithmetic (Rational / GaussQi).  The only approximations are
// those inherent in the delta:: transcendental functions; their tolerance is
// compared with EXPECT_RATIONAL_NEAR.
//
// IF ANY TEST FAILS, THE BUG IS IN THE GROUP IMPLEMENTATION,
// NOT IN THE TESTS.
// ============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <complex>
#include "delta/core/rational.h"
#include "delta/rational/gauss_qi.h"           // GaussQi
#include "delta/rational/transcendentals.h"    // delta::cos, delta::sin, delta::acos, delta::abs
#include "delta/numerical/gauge_groups.h"      // U1, SU2, SU3
#include "../test_fixtures_geometry_numerical.h"

namespace delta::testing {

    class GaugeGroupsTest : public GeometryNumericalTest {};
    // =========================================================================
    // U(1) = SO(2) – exact rational group axioms
    // =========================================================================
    TEST_F(GaugeGroupsTest, U1RationalGroupAxioms) {
        using Scalar = Rational;
        using Matrix2 = Eigen::Matrix<Scalar, 2, 2>;

        Matrix2 I = Matrix2::Identity();
        // Rotation by +90°  [[0, -1], [1, 0]]
        Matrix2 R90;   R90 << 0_r, -1_r, 1_r, 0_r;
        // Rotation by -90°  [[0, 1], [-1, 0]]
        Matrix2 Rm90;  Rm90 << 0_r, 1_r, -1_r, 0_r;

        // Closure: R90 * R90 == R180 == -I
        Matrix2 R180 = R90 * R90;
        EXPECT_EQ(R180(0, 0), -1_r);  EXPECT_EQ(R180(0, 1), 0_r);
        EXPECT_EQ(R180(1, 0), 0_r);  EXPECT_EQ(R180(1, 1), -1_r);

        // Inverse
        EXPECT_EQ(R90.inverse(), Rm90);
        EXPECT_EQ(Rm90.inverse(), R90);
        EXPECT_EQ(R90 * Rm90, I);
        EXPECT_EQ(Rm90 * R90, I);

        // Associativity: (R90*R90)*R90 == R90*(R90*R90)
        Matrix2 left = (R90 * R90) * R90;
        Matrix2 right = R90 * (R90 * R90);
        EXPECT_EQ(left, right);

        // Identity
        EXPECT_EQ(I * R90, R90);
        EXPECT_EQ(R90 * I, R90);
    }

    // =========================================================================
    // U(1) = SO(2) – exponential and logarithm with Rational
    // =========================================================================
    TEST_F(GaugeGroupsTest, U1RationalExponentialLogarithm) {
        using Scalar = Rational;
        using Algebra = U1<Scalar>::algebra_type;   // 2x2 skew‑symmetric
        using Matrix2 = Eigen::Matrix<Scalar, 2, 2>;

        // Angle θ = 1/2 rad
        Scalar theta = Scalar(1, 2);                // 0.5

        // Build the algebra element A = [[0, -θ], [θ, 0]]
        Algebra A;
        A << 0_r, -theta,
            theta, 0_r;

        // Exponential: exp(A) = rotation by θ
        Matrix2 R = U1<Scalar>::exp(A);

        // Expected rotation matrix using delta::cos and delta::sin
        Scalar c = delta::cos(theta, delta::default_eps());
        Scalar s = delta::sin(theta, delta::default_eps());
        EXPECT_RATIONAL_NEAR(R(0, 0), c, Scalar(1, 1000000000000));
        EXPECT_RATIONAL_NEAR(R(0, 1), -s, Scalar(1, 1000000000000));
        EXPECT_RATIONAL_NEAR(R(1, 0), s, Scalar(1, 1000000000000));
        EXPECT_RATIONAL_NEAR(R(1, 1), c, Scalar(1, 1000000000000));

        // Logarithm: log(R) should recover the algebra element A
        Algebra B = U1<Scalar>::log(R);
        EXPECT_RATIONAL_NEAR(B(0, 1), -theta, Scalar(1, 1000000000000));
        EXPECT_RATIONAL_NEAR(B(1, 0), theta, Scalar(1, 1000000000000));
    }

    // =========================================================================
    // SU(2) – exact rational group axioms (no transcendental operations)
    // =========================================================================
    TEST_F(GaugeGroupsTest, SU2RationalGroupAxioms) {
        using Scalar = Rational;
        using Complex = GaussQi;
        using Matrix2c = SU2<Scalar>::matrix_type;   // Eigen::Matrix<GaussQi, 2, 2>

        // Build an SU(2) matrix with rational complex entries:
        // Choose a Pythagorean triple (3,4,5): a = 3/5, b = 4/5.
        // U = [[a + i b,   0    ],
        //      [   0    , a - i b]]
        // This matrix is unitary and has determinant 1.
        Scalar a(3, 5);
        Scalar b(4, 5);
        Complex alpha(a, b);
        Complex beta(a, -b);

        Matrix2c U = Matrix2c::Zero();
        U(0, 0) = alpha;
        U(1, 1) = beta;

        // Unitarity: U * U^† = I
        Matrix2c I = Matrix2c::Identity();
        Matrix2c product = U * U.adjoint();
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_EQ(product(i, j), I(i, j));

        // Determinant = 1
        Complex det = U.determinant();
        Complex one(1_r, 0_r);
        EXPECT_EQ(det, one);

        // Inverse
        Matrix2c inv = U.inverse();
        Matrix2c prod = U * inv;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_EQ(prod(i, j), I(i, j));
    }

    // =========================================================================
    // SU(2) – exponential and logarithm (Rational)
    // =========================================================================
    TEST_F(GaugeGroupsTest, SU2RationalExponentialLogarithm) {
        using Scalar = Rational;
        using Complex = GaussQi;
        using Algebra = SU2<Scalar>::algebra_type;   // 2x2
        using Matrix2c = SU2<Scalar>::matrix_type;

        // Build a simple algebra element: i * H, with H = [[a, b], [c, -a]]
        // where a = 1/5, b = 0, c = 0 (diagonal)
        Scalar a(1, 5);
        Complex zero(0_r, 0_r);
        Complex entry(a, 0_r);
        Algebra A;
        A << Complex(0, a), zero,   // i * a
            zero, Complex(0, -a); // -i * a

        // Exponentiate
        Matrix2c U = SU2<Scalar>::exp(A);

        // Check unitarity
        Matrix2c I = Matrix2c::Identity();
        Matrix2c UU = U * U.adjoint();
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                EXPECT_RATIONAL_NEAR(UU(i, j).real(), I(i, j).real(), Scalar(1, 1000000000000));
                EXPECT_RATIONAL_NEAR(UU(i, j).imag(), I(i, j).imag(), Scalar(1, 1000000000000));
            }

        // Check determinant ≈ 1
        Complex det = U.determinant();
        Complex one(1_r, 0_r);
        EXPECT_RATIONAL_NEAR(det.real(), one.real(), Scalar(1, 1000000000000));
        EXPECT_RATIONAL_NEAR(det.imag(), one.imag(), Scalar(1, 1000000000000));

        // Logarithm
        Algebra A2 = SU2<Scalar>::log(U);

        // A2 should be close to A (within tolerance)
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                EXPECT_RATIONAL_NEAR(A2(i, j).real(), A(i, j).real(), Scalar(1, 1000000000000));
                EXPECT_RATIONAL_NEAR(A2(i, j).imag(), A(i, j).imag(), Scalar(1, 1000000000000));
            }
    }

    // =========================================================================
    // SU(3) – exact rational group axioms
    // =========================================================================
    TEST_F(GaugeGroupsTest, SU3RationalGroupAxioms) {
        using Scalar = Rational;
        using Complex = GaussQi;
        using Matrix3c = SU3<Scalar>::matrix_type;   // Eigen::Matrix<GaussQi, 3, 3>

        // Build an SU(3) matrix with rational complex entries:
        // Diagonal: (a + i b, a - i b, 1) with a=3/5, b=4/5.
        Scalar a(3, 5);
        Scalar b(4, 5);
        Complex alpha(a, b);
        Complex beta(a, -b);
        Complex one(1_r, 0_r);

        Matrix3c U = Matrix3c::Zero();
        U(0, 0) = alpha;
        U(1, 1) = beta;
        U(2, 2) = one;

        // Unitarity: U * U^† = I
        Matrix3c I3 = Matrix3c::Identity();
        Matrix3c product = U * U.adjoint();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                EXPECT_EQ(product(i, j), I3(i, j));

        // Determinant = 1
        Complex det = U.determinant();
        Complex one_c(1_r, 0_r);
        EXPECT_EQ(det, one_c);

        // Inverse
        Matrix3c inv = U.inverse();
        Matrix3c prod = U * inv;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                EXPECT_EQ(prod(i, j), I3(i, j));
    }

    // =========================================================================
    // SU(3) – exponential and logarithm (Rational)
    // =========================================================================
    TEST_F(GaugeGroupsTest, SU3RationalExponentialLogarithm) {
        using Scalar = Rational;
        using Complex = GaussQi;
        using Algebra = SU3<Scalar>::algebra_type;   // 3x3
        using Matrix3c = SU3<Scalar>::matrix_type;

        // Build diagonal algebra: i * diag(a, b, -a-b)
        Scalar a(1, 5);
        Scalar b(2, 5);
        Scalar c = -a - b;   // -3/5
        Algebra A = Algebra::Zero();
        A(0, 0) = Complex(0, a);
        A(1, 1) = Complex(0, b);
        A(2, 2) = Complex(0, c);

        // Exponentiate
        Matrix3c U = SU3<Scalar>::exp(A);

        // Check unitarity
        Matrix3c I3 = Matrix3c::Identity();
        Matrix3c UU = U * U.adjoint();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                EXPECT_RATIONAL_NEAR(UU(i, j).real(), I3(i, j).real(), Scalar(1, 1000000000000));
                EXPECT_RATIONAL_NEAR(UU(i, j).imag(), I3(i, j).imag(), Scalar(1, 1000000000000));
            }

        // Check determinant ≈ 1
        Complex det = U.determinant();
        Complex one(1_r, 0_r);
        EXPECT_RATIONAL_NEAR(det.real(), one.real(), Scalar(1, 1000000000000));
        EXPECT_RATIONAL_NEAR(det.imag(), one.imag(), Scalar(1, 1000000000000));

        // Logarithm
        Algebra A2 = SU3<Scalar>::log(U);

        // A2 should be close to A (within tolerance)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                EXPECT_RATIONAL_NEAR(A2(i, j).real(), A(i, j).real(), Scalar(1, 1000000000000));
                EXPECT_RATIONAL_NEAR(A2(i, j).imag(), A(i, j).imag(), Scalar(1, 1000000000000));
            }
    }
    // =========================================================================
  // U(1) – from_angle_pi and to_angle_pi (exact rational multiples of π)
  // =========================================================================
    TEST_F(GaugeGroupsTest, U1FromAnglePi) {
        using U1 = U1<Rational>;
        using Matrix = U1::matrix_type;
        const Rational tol = delta::default_eps() * 10;

        // θ = 1/2  → π/2
        Matrix R = U1::from_angle_pi(Rational(1, 2));
        EXPECT_EQ(R(0, 0), 0_r);
        EXPECT_EQ(R(0, 1), -1_r);
        EXPECT_EQ(R(1, 0), 1_r);
        EXPECT_EQ(R(1, 1), 0_r);

        // θ = 1  → π
        R = U1::from_angle_pi(1_r);
        EXPECT_EQ(R(0, 0), -1_r);
        EXPECT_EQ(R(0, 1), 0_r);
        EXPECT_EQ(R(1, 0), 0_r);
        EXPECT_EQ(R(1, 1), -1_r);

        // θ = 0  → identity
        R = U1::from_angle_pi(0_r);
        EXPECT_EQ(R, U1::identity());

        // Round‑trip: from_angle_pi → to_angle_pi
        R = U1::from_angle_pi(Rational(1, 3));  // 60°
        Rational theta_pi = U1::to_angle_pi(R);
        EXPECT_RATIONAL_NEAR(theta_pi, Rational(1, 3), tol);
    }
} // namespace delta::testing