// tests/rational/transcendentals_test.cpp
#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "test_utils.h"

namespace delta::testing {

    // -------------------------------------------------------------------------
    // 1. Sqrt
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, Sqrt) {
        // Exact values
        Rational s4 = delta::sqrt(4_r);
        EXPECT_EQ(s4.evaluate(), 2_r);

        // Approximate values
        Rational s2 = delta::sqrt(2_r);
        double val = static_cast<double>(s2.evaluate());
        EXPECT_NEAR(val, std::sqrt(2.0), 1e-12);
    }

    // -------------------------------------------------------------------------
    // 2. Exp
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, Exp) {
        // Exact values
        Rational e0 = delta::exp(0_r);
        EXPECT_EQ(e0.evaluate(), 1_r);

        // Approximate values
        Rational e1 = delta::exp(1_r);
        double val = static_cast<double>(e1.evaluate());
        EXPECT_NEAR(val, std::exp(1.0), 1e-12);
    }

    // -------------------------------------------------------------------------
    // 3. Log
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, Log) {
        // Exact values
        Rational l1 = delta::log(1_r);
        EXPECT_EQ(l1.evaluate(), 0_r);

        // Approximate values
        Rational l2 = delta::log(2_r);
        double val = static_cast<double>(l2.evaluate());
        EXPECT_NEAR(val, std::log(2.0), 1e-12);

        // log(e) ≈ 1
        Rational le = delta::log(delta::e());
        double ev = static_cast<double>(le.evaluate());
        EXPECT_NEAR(ev, 1.0, 1e-12);
    }

    // -------------------------------------------------------------------------
    // 4. Sin
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, Sin) {
        // Exact values
        Rational s0 = delta::sin(0_r);
        EXPECT_EQ(s0.evaluate(), 0_r);

        // sin(π/2) ≈ 1
        Rational s_pi2 = delta::sin(delta::pi() / 2_r);
        double val = static_cast<double>(s_pi2.evaluate());
        EXPECT_NEAR(val, 1.0, 1e-12);

        // sin(π) ≈ 0
        Rational s_pi = delta::sin(delta::pi());
        double vpi = static_cast<double>(s_pi.evaluate());
        EXPECT_NEAR(vpi, 0.0, 1e-12);
    }

    // -------------------------------------------------------------------------
    // 5. Cos
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, Cos) {
        // Exact values
        Rational c0 = delta::cos(0_r);
        EXPECT_EQ(c0.evaluate(), 1_r);

        // cos(π/2) ≈ 0
        Rational c_pi2 = delta::cos(delta::pi() / 2_r);
        double val = static_cast<double>(c_pi2.evaluate());
        EXPECT_NEAR(val, 0.0, 1e-12);

        // cos(π) ≈ -1
        Rational c_pi = delta::cos(delta::pi());
        double vpi = static_cast<double>(c_pi.evaluate());
        EXPECT_NEAR(vpi, -1.0, 1e-12);
    }

    // -------------------------------------------------------------------------
    // 6. Acos
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, Acos) {
        // acos(1) = 0
        Rational a1 = delta::acos(1_r);
        EXPECT_EQ(a1.evaluate(), 0_r);

        // acos(0) ≈ π/2
        Rational a0 = delta::acos(0_r);
        double val = static_cast<double>(a0.evaluate());
        double half_pi = std::acos(0.0);
        EXPECT_NEAR(val, half_pi, 1e-12);

        // acos(-1) ≈ π
        Rational a_1 = delta::acos(-1_r);
        double v = static_cast<double>(a_1.evaluate());
        EXPECT_NEAR(v, std::acos(-1.0), 1e-12);
    }

    // -------------------------------------------------------------------------
    // 7. Pi
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, Pi) {
        Rational p = delta::pi();
        double val = static_cast<double>(p.evaluate());
        EXPECT_NEAR(val, 3.141592653589793, 1e-12);
    }

    // -------------------------------------------------------------------------
    // 8. E
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, E) {
        Rational e = delta::e();
        double val = static_cast<double>(e.evaluate());
        EXPECT_NEAR(val, 2.718281828459045, 1e-12);
    }

    // -------------------------------------------------------------------------
    // 9. Precision parameter (eps) affects approximation quality
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, PrecisionParameter) {
        Rational high = delta::sqrt(2_r, "1e-30"_r);
        Rational low = delta::sqrt(2_r, "1e-2"_r);
        double diff = std::abs(static_cast<double>(high - low));
        EXPECT_GT(diff, 1e-6);   // significantly different
    }

    // -------------------------------------------------------------------------
    // 10. Lazy transcendentals (non‑eager mode)
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, LazyTranscendentals) {
        set_eager_mode(false);
        Rational s = delta::sqrt(2_r);
        EXPECT_TRUE(s.is_lazy());
        // But evaluation should still work
        double val = static_cast<double>(s.evaluate());
        EXPECT_NEAR(val, std::sqrt(2.0), 1e-12);
    }

} // namespace delta::testing