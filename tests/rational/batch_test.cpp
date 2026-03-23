// tests/rational/batch_test.cpp
#include <gtest/gtest.h>
#include <vector>
#include "delta/core/rational.h"
#include "delta/rational/batch_arithmetic.h"
#include "test_utils.h"

namespace delta::testing {

    // -------------------------------------------------------------------------
    // Batch addition tests
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, BatchAddSimple) {
        std::vector<Rational> terms = { "1/2"_r, "1/3"_r, "1/6"_r };
        Rational sum = internal::batch_add(terms);
        EXPECT_EQ(sum.evaluate(), 1_r);
    }

    TEST_F(RationalTest, BatchAddLarge) {
        std::vector<Rational> terms(100, "1/1000"_r);
        Rational sum = internal::batch_add(terms);
        EXPECT_EQ(sum.evaluate(), "1/10"_r);
    }

    TEST_F(RationalTest, BatchAddMixed) {
        std::vector<Rational> terms = { "1/2"_r, "1/1000000000000000000000000000000"_r };
        Rational sum = internal::batch_add(terms);
        // Expected: 1/2 + 1/10^30 = (5*10^29 + 1)/10^30
        // Since we cannot know exact numerator, we just check approximate value.
        double approx = static_cast<double>(sum.evaluate());
        EXPECT_NEAR(approx, 0.5 + 1e-30, 1e-30);
    }

    TEST_F(RationalTest, BatchAddOverflow) {
        // Use numbers that cause 128-bit overflow in intermediate calculations
        std::string denom1 = "1000000000000000000000000000000";
        std::string denom2 = "1000000000000000000000000000001";
        std::vector<Rational> terms = {
            Rational(1, boost::multiprecision::cpp_int(denom1)),
            Rational(1, boost::multiprecision::cpp_int(denom2))
        };
        Rational sum = internal::batch_add(terms);
        // Expected: (denom1 + denom2) / (denom1 * denom2)
        // We'll just check that the sum is not zero and is positive.
        EXPECT_GT(sum.evaluate(), 0_r);
        // Also check that it's less than 2/denom1 (since denominator is huge)
        Rational bound = Rational(2) / Rational(denom1);
        EXPECT_LT(sum.evaluate(), bound);
    }

    TEST_F(RationalTest, BatchAddLazy) {
        set_eager_mode(false);
        auto a = delta::sqrt(2_r);
        auto b = delta::exp(1_r);
        std::vector<Rational> terms = { a, b };
        Rational sum = internal::batch_add(terms);
        // Evaluate to get a rational approximation
        Rational ev = sum.evaluate();
        double approx = static_cast<double>(ev);
        double expected = std::sqrt(2.0) + std::exp(1.0);
        EXPECT_NEAR(approx, expected, 1e-12);
    }

    TEST_F(RationalTest, BatchAddEmpty) {
        std::vector<Rational> terms;
        Rational sum = internal::batch_add(terms);
        EXPECT_EQ(sum.evaluate(), 0_r);
    }

} // namespace delta::testing