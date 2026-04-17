#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "test_utils.h"

namespace delta::testing {

    class TranscendentalTest : public RationalTest {};

    // -------------------------------------------------------------------------
    // Eager трансцендентные (всегда возвращают Rational)
    // -------------------------------------------------------------------------
    TEST_F(TranscendentalTest, EagerSqrt) {
        Rational s4 = delta::sqrt(4_r);
        EXPECT_EQ(s4, 2_r);

        Rational s2 = delta::sqrt(2_r);
        Rational expected_sqrt2 = Rational("14142135623730950488/10000000000000000000");
        EXPECT_RATIONAL_NEAR(s2, expected_sqrt2, "1/1000000000000"_r);
    }

    TEST_F(TranscendentalTest, EagerExp) {
        Rational e0 = delta::exp(0_r);
        EXPECT_EQ(e0, 1_r);

        Rational e1 = delta::exp(1_r);
        Rational expected_e = Rational("27182818284590452354/10000000000000000000");
        EXPECT_RATIONAL_NEAR(e1, expected_e, "1/1000000000000"_r);
    }

    TEST_F(TranscendentalTest, EagerLog) {
        Rational l1 = delta::log(1_r);
        EXPECT_EQ(l1, 0_r);

        Rational l2 = delta::log(2_r);
        Rational expected_log2 = Rational("69314718055994530942/100000000000000000000");
        EXPECT_RATIONAL_NEAR(l2, expected_log2, "1/1000000000000"_r);
    }

    TEST_F(TranscendentalTest, EagerSinCos) {
        Rational s0 = delta::sin(0_r);
        EXPECT_EQ(s0, 0_r);

        Rational c0 = delta::cos(0_r);
        EXPECT_EQ(c0, 1_r);
    }

    TEST_F(TranscendentalTest, EagerPiE) {
        Rational p = delta::pi();
        Rational expected_pi = Rational("31415926535897932384626433832795028841971693993751/10000000000000000000000000000000000000000000000000");
        EXPECT_RATIONAL_NEAR(p, expected_pi, "1/1000000000000"_r);

        Rational e = delta::e();
        Rational expected_e = Rational("27182818284590452353602874713526624977572470936996/10000000000000000000000000000000000000000000000000");
        EXPECT_RATIONAL_NEAR(e, expected_e, "1/1000000000000"_r);
    }
    // -------------------------------------------------------------------------
    // Проверка точности с разным eps (eager версия)
    // -------------------------------------------------------------------------
    TEST_F(TranscendentalTest, PrecisionParameter) {
        Rational high = delta::sqrt(2_r, "1/1000000000000000000000000000000"_r);
        Rational low = delta::sqrt(2_r, "1/100"_r);
        Rational diff = delta::abs(high - low);
        double d = diff.to_double();
        EXPECT_GT(d, 1e-6);
    }
    // -------------------------------------------------------------------------
    // Lazy трансцендентные (явно вызываем lazy_*)
    // -------------------------------------------------------------------------
    TEST_F(TranscendentalTest, LazySqrt) {
        auto s = delta::lazy_sqrt(2_r);
        static_assert(std::is_same_v<decltype(s), LazyRational>);
        Rational expected_sqrt2 = Rational("14142135623730950488/10000000000000000000");
        EXPECT_RATIONAL_NEAR(s.eval(), expected_sqrt2, "1/1000000000000"_r);
    }

    TEST_F(TranscendentalTest, LazyExp) {
        auto e = delta::lazy_exp(1_r);
        static_assert(std::is_same_v<decltype(e), LazyRational>);
        Rational expected_e = Rational("27182818284590452354/10000000000000000000");
        EXPECT_RATIONAL_NEAR(e.eval(), expected_e, "1/1000000000000"_r);
    }

    TEST_F(TranscendentalTest, LazyPi) {
        auto p = delta::lazy_pi();
        static_assert(std::is_same_v<decltype(p), LazyRational>);
        Rational expected_pi = Rational("31415926535897932384626433832795028841971693993751/10000000000000000000000000000000000000000000000000");
        EXPECT_RATIONAL_NEAR(p.eval(), expected_pi, "1/1000000000000"_r);
    }



} // namespace delta::testing