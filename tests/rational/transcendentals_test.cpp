// tests/rational/transcendentals_test.cpp
#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "test_utils.h"

namespace delta::testing {

    class RationalTranscendentalTest : public RationalTest {};

    // Вспомогательная функция для получения double из Rational
    inline double to_double(const Rational& r) {
        return internal::to_double(r.to_value());
    }

    // -------------------------------------------------------------------------
    // 1. Sqrt
    // -------------------------------------------------------------------------
    TEST_F(RationalTranscendentalTest, Sqrt) {
        // Exact values
        Rational s4 = delta::sqrt(4_r);
        EXPECT_EQ(s4.eval(), 2_r);

        // Approximate values
        Rational s2 = delta::sqrt(2_r);
        Rational expected_sqrt2 = Rational("14142135623730950488/10000000000000000000");
        EXPECT_RATIONAL_NEAR(s2.eval(), expected_sqrt2, "1/1000000000000"_r);
    }

    // -------------------------------------------------------------------------
    // 2. Exp
    // -------------------------------------------------------------------------
    TEST_F(RationalTranscendentalTest, Exp) {
        // Exact values
        Rational e0 = delta::exp(0_r);
        EXPECT_EQ(e0.eval(), 1_r);

        // Approximate values
        Rational e1 = delta::exp(1_r);
        Rational expected_e = Rational("27182818284590452354/10000000000000000000");
        EXPECT_RATIONAL_NEAR(e1.eval(), expected_e, "1/1000000000000"_r);
    }

    // -------------------------------------------------------------------------
    // 3. Log
    // -------------------------------------------------------------------------
    TEST_F(RationalTranscendentalTest, Log) {
        // Exact values
        Rational l1 = delta::log(1_r);
        EXPECT_EQ(l1.eval(), 0_r);

        // Approximate values
        Rational l2 = delta::log(2_r);
        Rational expected_log2 = Rational("69314718055994530942/100000000000000000000");
        EXPECT_RATIONAL_NEAR(l2.eval(), expected_log2, "1/1000000000000"_r);

        // log(e) ≈ 1
        Rational le = delta::log(delta::e());
        EXPECT_RATIONAL_NEAR(le.eval(), 1_r, "1/1000000000000"_r);
    }

    // -------------------------------------------------------------------------
    // 4. Sin
    // -------------------------------------------------------------------------
    TEST_F(RationalTranscendentalTest, Sin) {
        // Exact values
        Rational s0 = delta::sin(0_r);
        EXPECT_EQ(s0.eval(), 0_r);

        // sin(π/2) ≈ 1
        Rational s_pi2 = delta::sin(delta::pi() / 2_r);
        EXPECT_RATIONAL_NEAR(s_pi2.eval(), 1_r, "1/1000000000000"_r);

        // sin(π) ≈ 0
        Rational s_pi = delta::sin(delta::pi());
        EXPECT_RATIONAL_NEAR(s_pi.eval(), 0_r, "1/1000000000000"_r);
    }

    // -------------------------------------------------------------------------
    // 5. Cos
    // -------------------------------------------------------------------------
    TEST_F(RationalTranscendentalTest, Cos) {
        // Exact values
        Rational c0 = delta::cos(0_r);
        EXPECT_EQ(c0.eval(), 1_r);

        // cos(π/2) ≈ 0
        Rational c_pi2 = delta::cos(delta::pi() / 2_r);
        EXPECT_RATIONAL_NEAR(c_pi2.eval(), 0_r, "1/1000000000000"_r);

        // cos(π) ≈ -1
        Rational c_pi = delta::cos(delta::pi());
        EXPECT_RATIONAL_NEAR(c_pi.eval(), -1_r, "1/1000000000000"_r);
    }

    // -------------------------------------------------------------------------
    // 6. Acos
    // -------------------------------------------------------------------------
    TEST_F(RationalTranscendentalTest, Acos) {
        // acos(1) = 0
        Rational a1 = delta::acos(1_r);
        EXPECT_EQ(a1.eval(), 0_r);

        // acos(0) ≈ π/2
        Rational a0 = delta::acos(0_r);
        Rational half_pi = delta::pi() / 2_r;
        EXPECT_RATIONAL_NEAR(a0.eval(), half_pi, "1/1000000000000"_r);

        // acos(-1) ≈ π
        Rational a_1 = delta::acos(-1_r);
        Rational pi = delta::pi();
        EXPECT_RATIONAL_NEAR(a_1.eval(), pi, "1/1000000000000"_r);
    }

    // -------------------------------------------------------------------------
    // 7. Pi
    // -------------------------------------------------------------------------
    TEST_F(RationalTranscendentalTest, Pi) {
        Rational p = delta::pi();
        Rational expected_pi = Rational("31415926535897932384626433832795028841971693993751/10000000000000000000000000000000000000000000000000");
        EXPECT_RATIONAL_NEAR(p.eval(), expected_pi, "1/1000000000000"_r);
    }

    // -------------------------------------------------------------------------
    // 8. E
    // -------------------------------------------------------------------------
    TEST_F(RationalTranscendentalTest, E) {
        Rational e = delta::e();
        Rational expected_e = Rational("27182818284590452353602874713526624977572470936996/10000000000000000000000000000000000000000000000000");
        EXPECT_RATIONAL_NEAR(e.eval(), expected_e, "1/1000000000000"_r);
    }

    // -------------------------------------------------------------------------
    // 9. Precision parameter (eps) affects approximation quality
    // -------------------------------------------------------------------------
    TEST_F(RationalTranscendentalTest, PrecisionParameter) {
        Rational high = delta::sqrt(2_r, "1/1000000000000000000000000000000"_r); // 1e-30
        Rational low = delta::sqrt(2_r, "1/100"_r); // 1e-2
        Rational diff = delta::abs(high.eval() - low.eval());
        // Сравниваем diff > 1e-6, используя to_double
        double d = to_double(diff);
        EXPECT_GT(d, 1e-6);
    }

    // -------------------------------------------------------------------------
    // 10. Lazy transcendentals (non‑eager mode)
    // -------------------------------------------------------------------------
    TEST_F(RationalTranscendentalTest, LazyTranscendentals) {
        set_eager_mode(false);
        Rational s = delta::sqrt(2_r);
        EXPECT_TRUE(s.is_lazy());
        // But evaluation should still work
        Rational expected_sqrt2 = Rational("14142135623730950488/10000000000000000000");
        EXPECT_RATIONAL_NEAR(s.eval(), expected_sqrt2, "1/1000000000000"_r);
    }

} // namespace delta::testing