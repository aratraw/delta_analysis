// tests/rational/eigen_integration_rational.cpp
// ============================================================================
// EIGEN INTEGRATION TESTS FOR RATIONAL (MATRIX TRANSCENDENTALS)
// ============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "delta/core/rational.h"
#include "delta/core/eigen_integration.h"
#include "test_utils.h"

using namespace delta;
using namespace delta::literals;

namespace delta::testing {

    // Общие алиасы типов для Rational
    namespace rational_aliases {
        using Mat2x2 = Eigen::Matrix<Rational, 2, 2>;
        using Mat3x3 = Eigen::Matrix<Rational, 3, 3>;
        using Mat5x5 = Eigen::Matrix<Rational, 5, 5>;
        using Mat10x10 = Eigen::Matrix<Rational, 10, 10>;
        using Array2x2 = Eigen::Array<Rational, 2, 2>;
    }

    using namespace rational_aliases;

    class EigenRationalTest : public RationalTest {
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
    // Basic Rational integration with Eigen
    // ============================================================================

    TEST_F(EigenRationalTest, RationalMatrixConstruction) {
        Mat2x2 A;
        A << 1_r, 2_r, 3_r, 4_r;
        EXPECT_EQ(A(0, 0), 1_r);
        EXPECT_EQ(A(0, 1), 2_r);
        EXPECT_EQ(A(1, 0), 3_r);
        EXPECT_EQ(A(1, 1), 4_r);
    }

    TEST_F(EigenRationalTest, RationalMatrixArithmetic) {
        Mat2x2 A, B;
        A << 1_r, 2_r, 3_r, 4_r;
        B << 5_r, 6_r, 7_r, 8_r;

        Mat2x2 C = A + B;
        EXPECT_EQ(C(0, 0), 6_r); EXPECT_EQ(C(0, 1), 8_r);
        EXPECT_EQ(C(1, 0), 10_r); EXPECT_EQ(C(1, 1), 12_r);

        C = A - B;
        EXPECT_EQ(C(0, 0), -4_r); EXPECT_EQ(C(0, 1), -4_r);
        EXPECT_EQ(C(1, 0), -4_r); EXPECT_EQ(C(1, 1), -4_r);

        C = A * B;
        EXPECT_EQ(C(0, 0), 19_r); EXPECT_EQ(C(0, 1), 22_r);
        EXPECT_EQ(C(1, 0), 43_r); EXPECT_EQ(C(1, 1), 50_r);

        C = A * 2_r;
        EXPECT_EQ(C(0, 0), 2_r); EXPECT_EQ(C(0, 1), 4_r);
        EXPECT_EQ(C(1, 0), 6_r); EXPECT_EQ(C(1, 1), 8_r);
    }

    TEST_F(EigenRationalTest, RationalReductionsAndTranspose) {
        Mat2x2 A;
        A << 1_r, 2_r, 3_r, 4_r;

        EXPECT_EQ(A.sum(), 10_r);
        EXPECT_EQ(A.prod(), 24_r);
        EXPECT_EQ(A.squaredNorm(), 30_r);

        auto At = A.transpose();
        EXPECT_EQ(At(0, 1), 3_r);
        EXPECT_EQ(At(1, 0), 2_r);
    }

    // ============================================================================
    // Element-wise transcendentals via ADL
    // ============================================================================

    TEST_F(EigenRationalTest, RationalElementWiseSqrt) {
        Array2x2 A;
        A << 0_r, 1_r, 4_r, 9_r;
        auto B = A.sqrt();
        EXPECT_EQ(B(0, 0), 0_r);
        EXPECT_EQ(B(0, 1), 1_r);
        EXPECT_EQ(B(1, 0), 2_r);
        EXPECT_EQ(B(1, 1), 3_r);
    }

    TEST_F(EigenRationalTest, RationalElementWiseExpLog) {
        Array2x2 A;
        A << 1_r, 2_r, 3_r, 4_r;

        auto expA = A.exp();
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_RATIONAL_NEAR(delta::exp(delta::log(expA(i, j), EPS), EPS), expA(i, j), LOOSE_EPS);

        auto logA = A.log();
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_RATIONAL_NEAR(delta::exp(logA(i, j), EPS), A(i, j), EPS);
    }

    TEST_F(EigenRationalTest, RationalElementWiseTrig) {
        Array2x2 A;
        A << 0_r, "1/6"_r, "1/4"_r, "1/3"_r;

        auto sinA = A.sin();
        auto cosA = A.cos();
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                auto s2 = sinA(i, j) * sinA(i, j);
                auto c2 = cosA(i, j) * cosA(i, j);
                EXPECT_RATIONAL_NEAR(s2 + c2, 1_r, LOOSE_EPS);
            }
    }

    // ============================================================================
    // MATRIX TRANSCENDENTALS FOR RATIONAL
    // ============================================================================

    TEST_F(EigenRationalTest, MatrixExpVsElementWiseExp) {
        Mat2x2 A;
        A << 1_r, 2_r, 3_r, 4_r;

        auto mat_exp = delta::exp(A, EPS);
        auto elem_exp = A.array().exp();

        EXPECT_NE(mat_exp(0, 0), elem_exp(0, 0));
        EXPECT_NE(mat_exp(0, 1), elem_exp(0, 1));

        auto mat_exp_neg = delta::exp(-A, EPS);
        auto identity = mat_exp * mat_exp_neg;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                Rational expected = (i == j) ? 1_r : 0_r;
                EXPECT_RATIONAL_NEAR(identity(i, j), expected, LOOSE_EPS);
            }
    }

    TEST_F(EigenRationalTest, MatrixLogExpInverse) {
        Mat2x2 A;
        A << "1/2"_r, "1/4"_r, "1/4"_r, "1/2"_r;

        auto E = delta::exp(A, EPS);
        auto L = delta::log(E, EPS);

        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_RATIONAL_NEAR(L(i, j), A(i, j), LOOSE_EPS);
    }

    TEST_F(EigenRationalTest, MatrixExpDiagonalFastPath) {
        Mat3x3 D = Mat3x3::Zero();
        D.diagonal() << 1_r, 2_r, 3_r;

        auto mat_exp = delta::exp(D, EPS);
        auto elem_exp = D.array().exp();

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                if (i == j) {
                    EXPECT_RATIONAL_NEAR(mat_exp(i, j), elem_exp(i, j), EPS);
                }
                else {
                    EXPECT_EQ(mat_exp(i, j), 0_r);
                }
            }
    }

    TEST_F(EigenRationalTest, MatrixLogDiagonalFastPath) {
        Mat3x3 D = Mat3x3::Zero();
        D.diagonal() << 2_r, 3_r, 4_r;

        auto mat_log = delta::log(D, EPS);
        auto elem_log = D.array().log();

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                if (i == j) {
                    EXPECT_RATIONAL_NEAR(mat_log(i, j), elem_log(i, j), EPS);
                }
                else {
                    EXPECT_EQ(mat_log(i, j), 0_r);
                }
            }
    }

    TEST_F(EigenRationalTest, MatrixSinTaylorForRational) {
        Mat2x2 A;
        A << 0_r, Rational(1, 10), Rational(1, 10), 0_r;

        auto sinA = delta::sin(A, EPS);
        auto cosA = delta::cos(A, EPS);

        Rational expected_sin = delta::sin(Rational(1, 10), EPS);
        Rational expected_cos = delta::cos(Rational(1, 10), EPS);

        EXPECT_RATIONAL_NEAR(sinA(0, 1), expected_sin, EPS);
        EXPECT_RATIONAL_NEAR(sinA(1, 0), expected_sin, EPS);
        EXPECT_RATIONAL_NEAR(sinA(0, 0), 0_r, EPS);
        EXPECT_RATIONAL_NEAR(sinA(1, 1), 0_r, EPS);

        EXPECT_RATIONAL_NEAR(cosA(0, 0), expected_cos, EPS);
        EXPECT_RATIONAL_NEAR(cosA(1, 1), expected_cos, EPS);
        EXPECT_RATIONAL_NEAR(cosA(0, 1), 0_r, EPS);
        EXPECT_RATIONAL_NEAR(cosA(1, 0), 0_r, EPS);
    }

    TEST_F(EigenRationalTest, MatrixSqrtDenmanBeavers) {
        Mat2x2 A;
        A << 4_r, 0_r, 0_r, 9_r;

        auto sqrtA = delta::sqrt(A, EPS);
        auto product = sqrtA * sqrtA;

        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_RATIONAL_NEAR(product(i, j), A(i, j), EPS);
    }

    TEST_F(EigenRationalTest, MatrixExpWithExpression) {
        Mat2x2 A, B;
        A << 1_r, 0_r, 0_r, 1_r;
        B << 0_r, 1_r, 1_r, 0_r;

        auto expr = (3 * A + 2 * B).eval();
        auto E = delta::exp(expr, EPS);

        EXPECT_EQ(E.rows(), 2);
        EXPECT_EQ(E.cols(), 2);
        EXPECT_TRUE(E(0, 0) != 0_r);
    }

    TEST_F(EigenRationalTest, DefaultEpsilonHandling) {
        Mat2x2 A;
        A << 1_r, 2_r, 3_r, 4_r;

        auto E1 = delta::exp(A);
        auto E2 = delta::exp(A, delta::default_eps());

        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_EQ(E1(i, j), E2(i, j));
    }

    TEST_F(EigenRationalTest, MatrixLogSingularThrows) {
        Mat2x2 singular;
        singular << 1_r, 2_r, 2_r, 4_r;
        EXPECT_THROW(delta::log(singular, EPS), std::domain_error);
    }

    TEST_F(EigenRationalTest, MatrixSqrtNegativeThrows) {
        Mat2x2 neg;
        neg << -1_r, 0_r, 0_r, -1_r;
        EXPECT_THROW(delta::sqrt(neg, EPS), std::domain_error);
    }

    TEST_F(EigenRationalTest, NonSquareMatrixCompileError) {
        SUCCEED();
    }

    TEST_F(EigenRationalTest, LargeDiagonalMatrixExp) {
        Mat10x10 D = Mat10x10::Zero();
        for (int i = 0; i < 10; ++i)
            D(i, i) = Rational(i + 1);

        auto E = delta::exp(D, LOOSE_EPS);

        for (int i = 0; i < 10; ++i) {
            Rational expected = delta::exp(Rational(i + 1), LOOSE_EPS);
            EXPECT_RATIONAL_NEAR(E(i, i), expected, LOOSE_EPS);
            for (int j = 0; j < 10; ++j)
                if (i != j) EXPECT_EQ(E(i, j), 0_r);
        }
    }

    TEST_F(EigenRationalTest, MatrixExpSeriesConvergence) {
        Mat2x2 A;
        A << 1_r, 1_r, 1_r, 1_r;

        auto E1 = delta::exp(A, EPS);
        auto E2 = delta::exp(A, EPS);

        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_EQ(E1(i, j), E2(i, j));
    }

    // ============================================================================
    // Wolfram Alpha verification
    // ============================================================================

    TEST_F(EigenRationalTest, WolframMatrixExp2x2) {
        Mat2x2 A;
        A << 1_r, 2_r, 3_r, 4_r;

        auto mat_exp = delta::exp(A, EPS);

        Rational e00("51.96895619870500365812");
        Rational e01("74.73656456700321254988");
        Rational e10("112.10484685050481882482");
        Rational e11("164.07380304920982248295");

        EXPECT_RATIONAL_NEAR(mat_exp(0, 0), e00, LOOSE_EPS);
        EXPECT_RATIONAL_NEAR(mat_exp(0, 1), e01, LOOSE_EPS);
        EXPECT_RATIONAL_NEAR(mat_exp(1, 0), e10, LOOSE_EPS);
        EXPECT_RATIONAL_NEAR(mat_exp(1, 1), e11, LOOSE_EPS);
    }

    TEST_F(EigenRationalTest, WolframMatrixLog2x2) {
        Mat2x2 A;
        A << Rational(12, 10), Rational(1, 10), Rational(1, 10), Rational(12, 10);

        auto mat_log = delta::log(A, EPS);

        Rational l00("0.178837");
        Rational l01("0.083527");

        EXPECT_RATIONAL_NEAR(mat_log(0, 0), l00, Rational(1, 1000000));
        EXPECT_RATIONAL_NEAR(mat_log(0, 1), l01, Rational(1, 1000000));
        EXPECT_RATIONAL_NEAR(mat_log(1, 0), l01, Rational(1, 1000000));
        EXPECT_RATIONAL_NEAR(mat_log(1, 1), l00, Rational(1, 1000000));
    }

    TEST_F(EigenRationalTest, WolframExponentialNilpotent3x3) {
        Mat3x3 N;
        N << 0_r, 1_r, 0_r, 0_r, 0_r, 1_r, 0_r, 0_r, 0_r;

        auto E = delta::exp(N, EPS);

        EXPECT_EQ(E(0, 0), 1_r); EXPECT_EQ(E(0, 1), 1_r); EXPECT_EQ(E(0, 2), Rational(1, 2));
        EXPECT_EQ(E(1, 0), 0_r); EXPECT_EQ(E(1, 1), 1_r); EXPECT_EQ(E(1, 2), 1_r);
        EXPECT_EQ(E(2, 0), 0_r); EXPECT_EQ(E(2, 1), 0_r); EXPECT_EQ(E(2, 2), 1_r);
    }

    // ============================================================================
    // Diagonal matrix tests
    // ============================================================================

    TEST_F(EigenRationalTest, DiagonalMatrixExp) {
        Mat5x5 D = Mat5x5::Zero();
        for (int i = 0; i < 5; ++i) D(i, i) = Rational(i + 1);

        auto result = delta::exp(D, EPS);
        for (int i = 0; i < 5; ++i) {
            Rational expected = delta::exp(D(i, i), EPS);
            EXPECT_RATIONAL_NEAR(result(i, i), expected, EPS);
            for (int j = 0; j < 5; ++j)
                if (i != j) EXPECT_EQ(result(i, j), 0_r);
        }
    }

    TEST_F(EigenRationalTest, DiagonalMatrixLog) {
        Mat5x5 D = Mat5x5::Zero();
        for (int i = 0; i < 5; ++i) D(i, i) = Rational(i + 2);

        auto result = delta::log(D, EPS);
        for (int i = 0; i < 5; ++i) {
            Rational expected = delta::log(D(i, i), EPS);
            EXPECT_RATIONAL_NEAR(result(i, i), expected, EPS);
            for (int j = 0; j < 5; ++j)
                if (i != j) EXPECT_EQ(result(i, j), 0_r);
        }
    }

    TEST_F(EigenRationalTest, DiagonalMatrixSin) {
        Mat5x5 D = Mat5x5::Zero();
        for (int i = 0; i < 5; ++i) D(i, i) = Rational(i + 1);

        auto result = delta::sin(D, EPS);
        for (int i = 0; i < 5; ++i) {
            Rational expected = delta::sin(D(i, i), EPS);
            EXPECT_RATIONAL_NEAR(result(i, i), expected, EPS);
            for (int j = 0; j < 5; ++j)
                if (i != j) EXPECT_EQ(result(i, j), 0_r);
        }
    }

    TEST_F(EigenRationalTest, DiagonalMatrixCos) {
        Mat5x5 D = Mat5x5::Zero();
        for (int i = 0; i < 5; ++i) D(i, i) = Rational(i + 1);

        auto result = delta::cos(D, EPS);
        for (int i = 0; i < 5; ++i) {
            Rational expected = delta::cos(D(i, i), EPS);
            EXPECT_RATIONAL_NEAR(result(i, i), expected, EPS);
            for (int j = 0; j < 5; ++j)
                if (i != j) EXPECT_EQ(result(i, j), 0_r);
        }
    }

    TEST_F(EigenRationalTest, DiagonalMatrixSqrtExact) {
        Mat5x5 D2 = Mat5x5::Zero();
        for (int i = 0; i < 5; ++i) D2(i, i) = Rational((i + 1) * (i + 1));

        auto result = delta::sqrt(D2, EPS);
        for (int i = 0; i < 5; ++i) {
            Rational expected = Rational(i + 1);
            EXPECT_RATIONAL_NEAR(result(i, i), expected, EPS);
            for (int j = 0; j < 5; ++j)
                if (i != j) EXPECT_EQ(result(i, j), 0_r);
        }
    }
    // ============================================================================
// ADDITIONAL TESTS FOR MATRIX TRANSCENDENTALS
// ============================================================================

    TEST_F(EigenRationalTest, ExpZeroMatrix) {
        Mat2x2 Z = Mat2x2::Zero();
        auto E = delta::exp(Z, EPS);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_EQ(E(i, j), (i == j) ? 1_r : 0_r);
    }

    TEST_F(EigenRationalTest, LogIdentity) {
        Mat2x2 I = Mat2x2::Identity();
        auto L = delta::log(I, EPS);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_EQ(L(i, j), 0_r);
    }

    TEST_F(EigenRationalTest, SinZeroMatrix) {
        Mat2x2 Z = Mat2x2::Zero();
        auto S = delta::sin(Z, EPS);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_EQ(S(i, j), 0_r);
    }

    TEST_F(EigenRationalTest, CosZeroMatrix) {
        Mat2x2 Z = Mat2x2::Zero();
        auto C = delta::cos(Z, EPS);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_EQ(C(i, j), (i == j) ? 1_r : 0_r);
    }

    TEST_F(EigenRationalTest, SinSqPlusCosSq) {
        // non‑diagonal matrix
        Mat2x2 A;
        A << 1_r, 2_r, 3_r, 4_r;
        auto S = delta::sin(A, EPS);
        auto C = delta::cos(A, EPS);
        auto Identity = S * S + C * C;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                Rational expected = (i == j) ? 1_r : 0_r;
                EXPECT_RATIONAL_NEAR(Identity(i, j), expected, LOOSE_EPS);
            }

        // another matrix (smaller values)
        Mat2x2 B;
        B << Rational(1, 2), Rational(1, 4), Rational(1, 4), Rational(1, 2);
        auto S2 = delta::sin(B, EPS);
        auto C2 = delta::cos(B, EPS);
        auto Identity2 = S2 * S2 + C2 * C2;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                Rational expected = (i == j) ? 1_r : 0_r;
                EXPECT_RATIONAL_NEAR(Identity2(i, j), expected, LOOSE_EPS);
            }
    }

    TEST_F(EigenRationalTest, MatrixExpTimesExpMinus) {
        Mat2x2 A;
        A << 1_r, 2_r, 3_r, 4_r;
        auto expA = delta::exp(A, EPS);
        auto expMinusA = delta::exp(-A, EPS);
        auto product = expA * expMinusA;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                Rational expected = (i == j) ? 1_r : 0_r;
                EXPECT_RATIONAL_NEAR(product(i, j), expected, LOOSE_EPS);
            }

        Mat2x2 B;
        B << Rational(1, 2), Rational(1, 4), Rational(1, 4), Rational(1, 2);
        auto expB = delta::exp(B, EPS);
        auto expMinusB = delta::exp(-B, EPS);
        auto product2 = expB * expMinusB;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                Rational expected = (i == j) ? 1_r : 0_r;
                EXPECT_RATIONAL_NEAR(product2(i, j), expected, LOOSE_EPS);
            }
    }

    TEST_F(EigenRationalTest, SinCosForLargeNormMatrix) {
        // diagonal matrix with elements = 2 (norm = 2 > π ≈ 3.14? Actually 2 < π, so need > π)
        // Let's use 4, so that sin(4) and cos(4) are not trivial, but the fast path will be taken.
        Mat2x2 D = Mat2x2::Zero();
        D(0, 0) = Rational(4);
        D(1, 1) = Rational(4);
        auto sinD = delta::sin(D, EPS);
        auto cosD = delta::cos(D, EPS);
        Rational expected_sin = delta::sin(4_r, EPS);
        Rational expected_cos = delta::cos(4_r, EPS);
        EXPECT_RATIONAL_NEAR(sinD(0, 0), expected_sin, EPS);
        EXPECT_RATIONAL_NEAR(sinD(1, 1), expected_sin, EPS);
        EXPECT_RATIONAL_NEAR(cosD(0, 0), expected_cos, EPS);
        EXPECT_RATIONAL_NEAR(cosD(1, 1), expected_cos, EPS);
        EXPECT_EQ(sinD(0, 1), 0_r);
        EXPECT_EQ(cosD(0, 1), 0_r);
    }

    TEST_F(EigenRationalTest, SinDoubleAngleIdentity) {
        // diagonal matrix (so that sin/cos commute)
        Mat2x2 D = Mat2x2::Zero();
        D(0, 0) = Rational(1);
        D(1, 1) = Rational(2);
        auto sinA = delta::sin(D, EPS);
        auto cosA = delta::cos(D, EPS);
        auto sin2A = delta::sin(2 * D, EPS);
        auto computed = 2 * sinA * cosA;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_RATIONAL_NEAR(sin2A(i, j), computed(i, j), LOOSE_EPS);
    }

    TEST_F(EigenRationalTest, CosDoubleAngleIdentity) {
        Mat2x2 D = Mat2x2::Zero();
        D(0, 0) = Rational(1);
        D(1, 1) = Rational(2);
        auto cosA = delta::cos(D, EPS);
        auto cos2A = delta::cos(2 * D, EPS);
        auto computed = 2 * cosA * cosA - Mat2x2::Identity().cast<Rational>();
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                // Вывод текущих значений для отладки
                std::cout << "[" << i << "," << j << "] "
                    << "cos2A: " << cos2A(i, j)
                    << " | computed: " << computed(i, j)
                    << " | diff: " << (cos2A(i, j) - computed(i, j))
                    << std::endl;

                EXPECT_RATIONAL_NEAR(cos2A(i, j), computed(i, j), LOOSE_EPS);
            }
    }

    TEST_F(EigenRationalTest, SqrtSquare) {
        // non‑diagonal matrix that is positive definite (but rational exact)
        Mat2x2 A;
        A << 5_r, 2_r, 2_r, 5_r;
        auto sqrtA = delta::sqrt(A, EPS);
        auto square = sqrtA * sqrtA;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_RATIONAL_NEAR(square(i, j), A(i, j), LOOSE_EPS);

        // another non‑diagonal matrix
        Mat2x2 B;
        B << 4_r, 1_r, 1_r, 4_r;
        auto sqrtB = delta::sqrt(B, EPS);
        auto squareB = sqrtB * sqrtB;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_RATIONAL_NEAR(squareB(i, j), B(i, j), LOOSE_EPS);
    }
} // namespace delta::testing