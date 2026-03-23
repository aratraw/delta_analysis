// tests/rational/eigen_integration_test.cpp
#include <gtest/gtest.h>
#include <Eigen/Core>
#include "delta/core/rational.h"
#include "test_utils.h"

namespace delta::testing {

    // -------------------------------------------------------------------------
    // 1. Matrix construction
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, MatrixConstruction) {
        Eigen::Matrix<Rational, 2, 2> m;
        m << "1/2"_r, "1/3"_r,
            "1/4"_r, "1/5"_r;

        EXPECT_EQ(m(0, 0).evaluate(), "1/2"_r);
        EXPECT_EQ(m(0, 1).evaluate(), "1/3"_r);
        EXPECT_EQ(m(1, 0).evaluate(), "1/4"_r);
        EXPECT_EQ(m(1, 1).evaluate(), "1/5"_r);
    }

    // -------------------------------------------------------------------------
    // 2. Matrix arithmetic
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, MatrixArithmetic) {
        Eigen::Matrix<Rational, 2, 2> a, b;
        a << "1/2"_r, "1/3"_r,
            "1/4"_r, "1/5"_r;
        b << "1/6"_r, "1/7"_r,
            "1/8"_r, "1/9"_r;

        // Addition
        Eigen::Matrix<Rational, 2, 2> sum = a + b;
        EXPECT_EQ(sum(0, 0).evaluate(), "1/2"_r + "1/6"_r);
        EXPECT_EQ(sum(0, 1).evaluate(), "1/3"_r + "1/7"_r);
        EXPECT_EQ(sum(1, 0).evaluate(), "1/4"_r + "1/8"_r);
        EXPECT_EQ(sum(1, 1).evaluate(), "1/5"_r + "1/9"_r);

        // Multiplication
        Eigen::Matrix<Rational, 2, 2> prod = a * b;
        // Compute expected manually
        Rational expected00 = "1/2"_r * "1/6"_r + "1/3"_r * "1/8"_r;
        Rational expected01 = "1/2"_r * "1/7"_r + "1/3"_r * "1/9"_r;
        Rational expected10 = "1/4"_r * "1/6"_r + "1/5"_r * "1/8"_r;
        Rational expected11 = "1/4"_r * "1/7"_r + "1/5"_r * "1/9"_r;
        EXPECT_EQ(prod(0, 0).evaluate(), expected00);
        EXPECT_EQ(prod(0, 1).evaluate(), expected01);
        EXPECT_EQ(prod(1, 0).evaluate(), expected10);
        EXPECT_EQ(prod(1, 1).evaluate(), expected11);
    }

    // -------------------------------------------------------------------------
    // 3. Matrix sqrt (diagonal case)
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, MatrixSqrt) {
        Eigen::Matrix<Rational, 2, 2> d;
        d << 4_r, 0_r,
            0_r, 9_r;
        auto sqrt_d = d.array().sqrt().matrix();   // elementwise sqrt

        EXPECT_EQ(sqrt_d(0, 0).evaluate(), 2_r);
        EXPECT_EQ(sqrt_d(0, 1).evaluate(), 0_r);
        EXPECT_EQ(sqrt_d(1, 0).evaluate(), 0_r);
        EXPECT_EQ(sqrt_d(1, 1).evaluate(), 3_r);
    }

    // -------------------------------------------------------------------------
    // 4. NumTraits – epsilon() should equal default_eps()
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, NumTraitsEpsilon) {
        Rational eigen_eps = Eigen::NumTraits<Rational>::epsilon();
        EXPECT_EQ(eigen_eps, delta::default_eps());
    }

} // namespace delta::testing