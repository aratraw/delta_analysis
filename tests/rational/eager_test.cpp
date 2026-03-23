// tests/rational/eager_test.cpp
#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "test_utils.h"

namespace delta::testing {

    // -------------------------------------------------------------------------
    // 1. Eager mode flag (default false)
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, EagerModeFlag) {
        EXPECT_FALSE(eager_mode());            // default
        set_eager_mode(true);
        EXPECT_TRUE(eager_mode());
        set_eager_mode(false);
        EXPECT_FALSE(eager_mode());
    }

    // -------------------------------------------------------------------------
    // 2. Eager evaluation – operations produce concrete numbers, not lazy nodes
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, EagerEvaluation) {
        set_eager_mode(true);
        Rational sum = "1/2"_r + "1/3"_r;
        EXPECT_FALSE(sum.is_lazy());
        EXPECT_EQ(sum.evaluate(), "5/6"_r);

        Rational diff = "1/2"_r - "1/3"_r;
        EXPECT_FALSE(diff.is_lazy());
        EXPECT_EQ(diff.evaluate(), "1/6"_r);

        Rational prod = "2/3"_r * "3/4"_r;
        EXPECT_FALSE(prod.is_lazy());
        EXPECT_EQ(prod.evaluate(), "1/2"_r);

        Rational quot = "2/3"_r / "4/5"_r;
        EXPECT_FALSE(quot.is_lazy());
        EXPECT_EQ(quot.evaluate(), "5/6"_r);
    }

    // -------------------------------------------------------------------------
    // 3. ScopedEagerEval – inside block eager, after block lazy
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, ScopedEagerEval) {
        // Initially lazy mode
        set_eager_mode(false);

        Rational sum1;
        {
            ScopedEagerEval guard;
            // Inside block, eager mode
            sum1 = "1/2"_r + "1/3"_r;
            EXPECT_FALSE(sum1.is_lazy());
            EXPECT_EQ(sum1.evaluate(), "5/6"_r);
        }
        // After block, mode restored to lazy
        Rational sum2 = "1/2"_r + "1/3"_r;
        EXPECT_TRUE(sum2.is_lazy());
        EXPECT_EQ(sum2.evaluate(), "5/6"_r);
    }

    // -------------------------------------------------------------------------
    // 4. Eager transcendentals – sqrt, exp, etc. return evaluated values
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, EagerTranscendentals) {
        set_eager_mode(true);

        // sqrt(4) = 2 exactly
        Rational s = delta::sqrt(4_r);
        EXPECT_FALSE(s.is_lazy());
        EXPECT_EQ(s.evaluate(), 2_r);

        // sqrt(2) approximate
        Rational s2 = delta::sqrt(2_r);
        EXPECT_FALSE(s2.is_lazy());
        double val = static_cast<double>(s2.evaluate());
        EXPECT_NEAR(val, std::sqrt(2.0), 1e-12);

        // exp(0) = 1
        Rational e0 = delta::exp(0_r);
        EXPECT_FALSE(e0.is_lazy());
        EXPECT_EQ(e0.evaluate(), 1_r);

        // exp(1) approximate
        Rational e1 = delta::exp(1_r);
        EXPECT_FALSE(e1.is_lazy());
        double e_val = static_cast<double>(e1.evaluate());
        EXPECT_NEAR(e_val, std::exp(1.0), 1e-12);

        // log(1) = 0
        Rational l1 = delta::log(1_r);
        EXPECT_FALSE(l1.is_lazy());
        EXPECT_EQ(l1.evaluate(), 0_r);

        // sin(0) = 0
        Rational sin0 = delta::sin(0_r);
        EXPECT_FALSE(sin0.is_lazy());
        EXPECT_EQ(sin0.evaluate(), 0_r);

        // cos(0) = 1
        Rational cos0 = delta::cos(0_r);
        EXPECT_FALSE(cos0.is_lazy());
        EXPECT_EQ(cos0.evaluate(), 1_r);

        // acos(1) = 0
        Rational acos1 = delta::acos(1_r);
        EXPECT_FALSE(acos1.is_lazy());
        EXPECT_EQ(acos1.evaluate(), 0_r);
    }

} // namespace delta::testing