// tests/rational/eigen_integration_gaussqi.cpp
// ============================================================================
// EIGEN INTEGRATION TESTS FOR GAUSSQI (COMPLEX MATRIX TRANSCENDENTALS)
// ============================================================================
//
// UPDATED 2026-05-11: HONEST TOLERANCES
// All comparisons now use the requested tolerance EPS multiplied by a
// small factor (10, 100) when identities or inverse operations are tested,
// reflecting the actual error propagation in exact rational arithmetic.
// ============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "delta/core/rational.h"
#include "delta/core/eigen_integration.h"
#include "test_utils.h"

using namespace delta;
using namespace delta::literals;

namespace delta::testing {

    // Общие алиасы типов для GaussQi
    namespace gaussqi_aliases {
        using Mat2x2 = Eigen::Matrix<GaussQi, 2, 2>;
        using Mat5x5 = Eigen::Matrix<GaussQi, 5, 5>;
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
                // exp followed by log accumulates error; allow EPS * 10
                EXPECT_RATIONAL_NEAR(recovered.real(), Z(i, j).real(), EPS * 10);
                EXPECT_RATIONAL_NEAR(recovered.imag(), Z(i, j).imag(), EPS * 10);
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

        // sin^2 + cos^2 = I, two matrix multiplications + sum ⇒ EPS * 100
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                Rational expected = (i == j) ? 1_r : 0_r;
                EXPECT_RATIONAL_NEAR(sum(i, j).real(), expected, EPS * 100);
                EXPECT_RATIONAL_NEAR(sum(i, j).imag(), 0_r, EPS * 100);
            }
    }

    TEST_F(EigenGaussQiTest, GaussQiMatrixExpComplex) {
        Mat2x2 A;
        A << GaussQi(0, 1), GaussQi(0, 0),
            GaussQi(0, 0), GaussQi(0, -1);

        auto E = delta::exp(A, EPS);

        Rational cos1 = delta::cos(1_r, EPS);
        Rational sin1 = delta::sin(1_r, EPS);

        EXPECT_RATIONAL_NEAR(E(0, 0).real(), cos1, EPS);
        EXPECT_RATIONAL_NEAR(E(0, 0).imag(), sin1, EPS);
        EXPECT_RATIONAL_NEAR(E(1, 1).real(), cos1, EPS);
        EXPECT_RATIONAL_NEAR(E(1, 1).imag(), -sin1, EPS);
    }

    // ============================================================================
    // ADDITIONAL TESTS FOR GAUSSQI MATRIX TRANSCENDENTALS
    // ============================================================================

    TEST_F(EigenGaussQiTest, ExpZeroMatrix) {
        Mat2x2 Z = Mat2x2::Zero();
        auto E = delta::exp(Z, EPS);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_EQ(E(i, j), (i == j) ? GaussQi(1, 0) : GaussQi(0, 0));
    }

    TEST_F(EigenGaussQiTest, LogIdentity) {
        Mat2x2 I = Mat2x2::Identity();
        auto L = delta::log(I, EPS);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_EQ(L(i, j), (i == j) ? GaussQi(0, 0) : GaussQi(0, 0));
    }

    TEST_F(EigenGaussQiTest, SinZeroMatrix) {
        Mat2x2 Z = Mat2x2::Zero();
        auto S = delta::sin(Z, EPS);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_EQ(S(i, j), (i == j) ? GaussQi(0, 0) : GaussQi(0, 0));
    }

    TEST_F(EigenGaussQiTest, CosZeroMatrix) {
        Mat2x2 Z = Mat2x2::Zero();
        auto C = delta::cos(Z, EPS);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_EQ(C(i, j), (i == j) ? GaussQi(1, 0) : GaussQi(0, 0));
    }

    TEST_F(EigenGaussQiTest, ExpTimesExpMinus) {
        Mat2x2 A;
        A << GaussQi(1, 2), GaussQi(3, 4),
            GaussQi(5, 6), GaussQi(7, 8);
        auto expA = delta::exp(A, EPS);
        auto expMinusA = delta::exp(-A, EPS);
        auto product = expA * expMinusA;

        // exp(A)*exp(-A) = I; two matrix exponentials + multiplication ⇒ EPS * 100
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                GaussQi expected = (i == j) ? GaussQi(1, 0) : GaussQi(0, 0);
                EXPECT_LT(delta::abs(product(i, j).real() - expected.real()), EPS * 100);
                EXPECT_LT(delta::abs(product(i, j).imag() - expected.imag()), EPS * 100);
            }
    }

    TEST_F(EigenGaussQiTest, SinSqPlusCosSq) {
        Mat2x2 A;
        A << GaussQi(1, 2), GaussQi(3, 4),
            GaussQi(5, 6), GaussQi(7, 8);
        auto S = delta::sin(A, EPS);
        auto C = delta::cos(A, EPS);
        auto Identity = S * S + C * C;

        // sin^2 + cos^2 = I; two multiplications + sum ⇒ EPS * 100
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                GaussQi expected = (i == j) ? GaussQi(1, 0) : GaussQi(0, 0);
                EXPECT_LT(delta::abs(Identity(i, j).real() - expected.real()), EPS * 100);
                EXPECT_LT(delta::abs(Identity(i, j).imag() - expected.imag()), EPS * 100);
            }
    }

    TEST_F(EigenGaussQiTest, DiagonalMatrixExp) {
        Mat5x5 D = Mat5x5::Zero();
        for (int i = 0; i < 5; ++i) D(i, i) = GaussQi(i + 1, 0);
        auto result = delta::exp(D, EPS);
        for (int i = 0; i < 5; ++i) {
            GaussQi expected = delta::exp(D(i, i), EPS);
            EXPECT_LT(delta::abs(result(i, i).real() - expected.real()), EPS);
            EXPECT_LT(delta::abs(result(i, i).imag() - expected.imag()), EPS);
            for (int j = 0; j < 5; ++j)
                if (i != j) EXPECT_EQ(result(i, j), GaussQi(0, 0));
        }
    }

    TEST_F(EigenGaussQiTest, DiagonalMatrixLog) {
        Mat5x5 D = Mat5x5::Zero();
        for (int i = 0; i < 5; ++i) D(i, i) = GaussQi(i + 2, 0);
        auto result = delta::log(D, EPS);
        for (int i = 0; i < 5; ++i) {
            GaussQi expected = delta::log(D(i, i), EPS);
            EXPECT_LT(delta::abs(result(i, i).real() - expected.real()), EPS);
            EXPECT_LT(delta::abs(result(i, i).imag() - expected.imag()), EPS);
            for (int j = 0; j < 5; ++j)
                if (i != j) EXPECT_EQ(result(i, j), GaussQi(0, 0));
        }
    }

    TEST_F(EigenGaussQiTest, DiagonalMatrixSin) {
        Mat5x5 D = Mat5x5::Zero();
        for (int i = 0; i < 5; ++i) D(i, i) = GaussQi(i + 1, 0);
        auto result = delta::sin(D, EPS);
        for (int i = 0; i < 5; ++i) {
            GaussQi expected = delta::sin(D(i, i), EPS);
            EXPECT_LT(delta::abs(result(i, i).real() - expected.real()), EPS);
            EXPECT_LT(delta::abs(result(i, i).imag() - expected.imag()), EPS);
            for (int j = 0; j < 5; ++j)
                if (i != j) EXPECT_EQ(result(i, j), GaussQi(0, 0));
        }
    }

    TEST_F(EigenGaussQiTest, DiagonalMatrixCos) {
        Mat5x5 D = Mat5x5::Zero();
        for (int i = 0; i < 5; ++i) D(i, i) = GaussQi(i + 1, 0);
        auto result = delta::cos(D, EPS);
        for (int i = 0; i < 5; ++i) {
            GaussQi expected = delta::cos(D(i, i), EPS);
            EXPECT_LT(delta::abs(result(i, i).real() - expected.real()), EPS);
            EXPECT_LT(delta::abs(result(i, i).imag() - expected.imag()), EPS);
            for (int j = 0; j < 5; ++j)
                if (i != j) EXPECT_EQ(result(i, j), GaussQi(0, 0));
        }
    }

    TEST_F(EigenGaussQiTest, SqrtSquarePositiveDefinite) {
        Mat2x2 A;
        A << GaussQi(5, 0), GaussQi(2, 0),
            GaussQi(2, 0), GaussQi(5, 0);
        auto sqrtA = delta::sqrt(A, EPS);
        auto square = sqrtA * sqrtA;

        // sqrt(A)^2 = A; one sqrt (iterative) + one multiplication ⇒ EPS * 100
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                EXPECT_LT(delta::abs(square(i, j).real() - A(i, j).real()), EPS * 100);
                EXPECT_LT(delta::abs(square(i, j).imag() - A(i, j).imag()), EPS * 100);
            }
    }

    TEST_F(EigenGaussQiTest, LogSingularThrows) {
        Mat2x2 singular;
        singular << GaussQi(1, 0), GaussQi(2, 0),
            GaussQi(2, 0), GaussQi(4, 0);
        EXPECT_THROW(delta::log(singular, EPS), std::domain_error);
    }

    TEST_F(EigenGaussQiTest, MatrixExpWithExpression) {
        Mat2x2 A, B;
        A << GaussQi(1, 0), GaussQi(0, 0),
            GaussQi(0, 0), GaussQi(1, 0);
        B << GaussQi(0, 1), GaussQi(1, 0),
            GaussQi(1, 0), GaussQi(0, 0);
        auto expr = (3_r * A + 2_r * B).eval();
        auto E = delta::exp(expr, EPS);
        EXPECT_EQ(E.rows(), 2);
        EXPECT_EQ(E.cols(), 2);
        EXPECT_TRUE(E(0, 0) != GaussQi(0, 0));
    }

} // namespace delta::testing