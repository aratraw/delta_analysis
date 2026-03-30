// tests/calculus/test_rational_embedding.cpp
#include <gtest/gtest.h>
#include "test_fixtures.h"

namespace delta::testing {

    /**
     * @class RationalEmbeddingTest
     * @brief Tests for the embedding of rational numbers into real numbers
     *        via fundamental sequences.
     */
    class RationalEmbeddingTest : public DeltaTest {};

    /**
     * @test Verify that two real numbers constructed from the same rational
     *       are considered equal and approximately equal within a tolerance.
     */
    TEST_F(RationalEmbeddingTest, RationalToReal) {
        RealNumber r1(3_r);
        RealNumber r2(3_r);

        EXPECT_TRUE(r1 == r2);
        EXPECT_TRUE(r1.approx_equal(r2, Rational(1, 1000)));

        RealNumber r3(4_r);
        EXPECT_FALSE(r1 == r3);
    }

    /**
     * @test Verify that two different fundamental sequences converging to the
     *       same rational number produce equivalent real numbers.
     */
    TEST_F(RationalEmbeddingTest, DifferentRepresentationsSameRational) {
        // Constant sequence for 3
        auto seq1 = std::make_shared<FundamentalSequence<ExponentialModulus>>(
            [](std::size_t) { return 3_r; }, ExponentialModulus(Rational(0), Rational(1, 2)), 0);
        // Sequence converging to 3: 3 + 1/2^n
        auto seq2 = std::make_shared<FundamentalSequence<ExponentialModulus>>(
            [](std::size_t n) { return 3_r + Rational(1) / pow2(n); },
            ExponentialModulus(Rational(1), Rational(1, 2)), 0);

        RealNumber r1(seq1);
        RealNumber r2(seq2);

        EXPECT_TRUE(r1 == r2);
        EXPECT_TRUE(r1.approx_equal(r2, Rational(1, 1000)));
    }

} // namespace delta::testing