// tests/rational/eigen_integration_gaussqi.cpp
// ============================================================================
// EIGEN INTEGRATION TESTS FOR GAUSSQI (COMPLEX MATRIX TRANSCENDENTALS)
// ============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "delta/core/rational.h"
#include "test_utils.h"

using namespace delta;
using namespace delta::literals;

namespace delta::testing {

    // Общие алиасы типов для GaussQi
    namespace gaussqi_aliases {
        using Mat2x2 = Eigen::Matrix<GaussQi, 2, 2>;
        using Array2x2 = Eigen::Array<GaussQi, 2, 2>;
    }

    using namespace gaussqi_aliases;

    class EigenGaussQiTest : public RationalTest {
    protected:
        void SetUp() override {
            RationalTest::SetUp();
            delta::reset_default_eps();
        }
        void TearDown() override {
            delta::reset_default_eps();
            RationalTest::TearDown();
        }
    };

    static const Rational EPS = "1/10000000000000000000"_r;  // 1e-19
    static const Rational LOOSE_EPS = "1/1000000000"_r;       // 1e-9

    // ============================================================================
    // Basic GaussQi integration with Eigen
    // ============================================================================

    TEST_F(EigenGaussQiTest, GaussQiMatrixConstruction) {
        Mat2x2 Z;
        Z << GaussQi(1, 2), GaussQi(3, 4),
            GaussQi(5, 6), GaussQi(7, 8);
        EXPECT_EQ(Z(0, 0).real(), 1_r); EXPECT_EQ(Z(0, 0).imag(), 2_r);
        EXPECT_EQ(Z(1, 1).real(), 7_r); EXPECT_EQ(Z(1, 1).imag(), 8_r);
    }

    TEST_F(EigenGaussQiTest, GaussQiArithmeticAndConjugate) {
        Mat2x2 A, B;
        A << GaussQi(1, 2), GaussQi(3, 4),
            GaussQi(5, 6), GaussQi(7, 8);
        B << GaussQi(1, 0), GaussQi(0, 1),
            GaussQi(0, -1), GaussQi(1, 0);

        Mat2x2 C = A + B;
        EXPECT_EQ(C(0, 0), GaussQi(2, 2));

        auto Aconj = A.conjugate();
        EXPECT_EQ(Aconj(0, 0), conj(A(0, 0)));
        EXPECT_EQ(Aconj(0, 1), conj(A(0, 1)));

        auto Aadj = A.adjoint();
        EXPECT_EQ(Aadj(0, 0), conj(A(0, 0)));
        EXPECT_EQ(Aadj(1, 0), conj(A(0, 1)));
    }

    TEST_F(EigenGaussQiTest, GaussQiElementWiseTranscendentals) {
        Mat2x2 Z;
        Z << GaussQi(0, 0), GaussQi(1, 0),
            GaussQi(0, 1), GaussQi(1, 1);

        auto expZ = Z.unaryExpr([](const GaussQi& z) { return delta::exp(z, EPS); });
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                if (Z(i, j) == GaussQi(0, 0)) continue;
                auto recovered = delta::log(expZ(i, j), EPS);
                EXPECT_RATIONAL_NEAR(recovered.real(), Z(i, j).real(), LOOSE_EPS);
            }
    }

    TEST_F(EigenGaussQiTest, GaussQiRealImagAccessors) {
        Mat2x2 A;
        A << GaussQi(1, 2), GaussQi(3, 4),
            GaussQi(5, 6), GaussQi(7, 8);

        auto realPart = A.real();
        auto imagPart = A.imag();
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                EXPECT_EQ(realPart(i, j), A(i, j).real());
                EXPECT_EQ(imagPart(i, j), A(i, j).imag());
            }
    }

    // ============================================================================
    // MATRIX TRANSCENDENTALS FOR GAUSSQI
    // ============================================================================

    TEST_F(EigenGaussQiTest, MatrixSinCosViaExpForGaussQi) {
        Mat2x2 A;
        A << GaussQi(0, 1), GaussQi(1, 0),
            GaussQi(0, -1), GaussQi(-1, 0);

        auto sinA = delta::sin(A, EPS);
        auto cosA = delta::cos(A, EPS);

        auto sin2 = sinA * sinA;
        auto cos2 = cosA * cosA;
        auto sum = sin2 + cos2;

        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                Rational expected = (i == j) ? 1_r : 0_r;
                EXPECT_RATIONAL_NEAR(sum(i, j).real(), expected, LOOSE_EPS);
                EXPECT_RATIONAL_NEAR(sum(i, j).imag(), 0_r, LOOSE_EPS);
            }
    }

    TEST_F(EigenGaussQiTest, GaussQiMatrixExpComplex) {
        Mat2x2 A;
        A << GaussQi(0, 1), GaussQi(0, 0),
            GaussQi(0, 0), GaussQi(0, -1);

        auto E = delta::exp(A, EPS);

        Rational cos1 = delta::cos(1_r, EPS);
        Rational sin1 = delta::sin(1_r, EPS);

        EXPECT_RATIONAL_NEAR(E(0, 0).real(), cos1, LOOSE_EPS);
        EXPECT_RATIONAL_NEAR(E(0, 0).imag(), sin1, LOOSE_EPS);
        EXPECT_RATIONAL_NEAR(E(1, 1).real(), cos1, LOOSE_EPS);
        EXPECT_RATIONAL_NEAR(E(1, 1).imag(), -sin1, LOOSE_EPS);
    }

} // namespace delta::testing