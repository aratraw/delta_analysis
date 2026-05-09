// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/complex/gauss_qi_test.cpp
// ============================================================================
// GAUSSIAN RATIONAL NUMBERS – COMPLETE TEST SUITE
// ============================================================================
//
// This file tests the GaussQi class and its transcendental extensions:
//   - Constructors (default, Rational, two Rationals, strings, integers).
//   - Accessors real(), imag().
//   - Arithmetic operators (+, -, *, /) and compound assignments.
//   - Unary minus, conjugation, norm.
//   - Comparisons (==, !=).
//   - String representation and to_double.
//   - Transcendentals: exp, log, sqrt, pow (integer and complex), abs, arg.
//   - Inverse relationships: exp(log(z)) ≈ z, log(exp(z)) ≈ z.
//   - Error handling: division by zero, log(0), 0^0.
//   - Integration with Eigen (if enabled) – basic matrix operations.
//
// All tests are eager – no lazy expression trees involved.
// ============================================================================

#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "test_utils.h"

namespace delta::testing {

    class GaussQiTest : public RationalTest {};

    // -------------------------------------------------------------------------
    // 1. Constructors and basic accessors
    // -------------------------------------------------------------------------

    TEST_F(GaussQiTest, Constructors) {
        // Default
        GaussQi z0;
        EXPECT_EQ(z0.real(), 0_r);
        EXPECT_EQ(z0.imag(), 0_r);

        // From Rational (real only)
        GaussQi z1(123_r);
        EXPECT_EQ(z1.real(), 123_r);
        EXPECT_EQ(z1.imag(), 0_r);

        // From two Rationals
        GaussQi z2(1_r, 2_r);
        EXPECT_EQ(z2.real(), 1_r);
        EXPECT_EQ(z2.imag(), 2_r);

        // From integers
        GaussQi z3(3, -4);
        EXPECT_EQ(z3.real(), 3_r);
        EXPECT_EQ(z3.imag(), -4_r);

        // From strings (single number)
        GaussQi z4("5/2");
        EXPECT_EQ(z4.real(), "5/2"_r);
        EXPECT_EQ(z4.imag(), 0_r);

        // From string pair
        GaussQi z5("1/3", "2/5");
        EXPECT_EQ(z5.real(), "1/3"_r);
        EXPECT_EQ(z5.imag(), "2/5"_r);
    }

    TEST_F(GaussQiTest, Accessors) {
        GaussQi z(7, -11);
        EXPECT_EQ(z.real(), 7_r);
        EXPECT_EQ(z.imag(), -11_r);

        GaussQi w;
        w.real(1_r);
        w.imag(2_r);
        EXPECT_EQ(w.real(), 1_r);
        EXPECT_EQ(w.imag(), 2_r);
    }

    // -------------------------------------------------------------------------
    // 2. Arithmetic operators
    // -------------------------------------------------------------------------

    TEST_F(GaussQiTest, Arithmetic) {
        GaussQi a(1, 2);
        GaussQi b(3, 4);

        // Addition
        GaussQi sum = a + b;
        EXPECT_EQ(sum.real(), 4_r);
        EXPECT_EQ(sum.imag(), 6_r);

        // Subtraction
        GaussQi diff = a - b;
        EXPECT_EQ(diff.real(), -2_r);
        EXPECT_EQ(diff.imag(), -2_r);

        // Multiplication: (1+2i)*(3+4i) = -5 + 10i
        GaussQi prod = a * b;
        EXPECT_EQ(prod.real(), -5_r);
        EXPECT_EQ(prod.imag(), 10_r);

        // Division: (1+2i)/(3+4i) = 11/25 + 2/25 i
        GaussQi quot = a / b;
        EXPECT_EQ(quot.real(), "11/25"_r);
        EXPECT_EQ(quot.imag(), "2/25"_r);
    }

    TEST_F(GaussQiTest, CompoundAssignments) {
        GaussQi a(1, 2);
        GaussQi b(3, 4);

        a += b;
        EXPECT_EQ(a.real(), 4_r);
        EXPECT_EQ(a.imag(), 6_r);

        a -= b;
        EXPECT_EQ(a.real(), 1_r);
        EXPECT_EQ(a.imag(), 2_r);

        a *= b;
        EXPECT_EQ(a.real(), -5_r);
        EXPECT_EQ(a.imag(), 10_r);

        a /= b;
        EXPECT_EQ(a.real(), 1_r);
        EXPECT_EQ(a.imag(), 2_r);
    }
    TEST_F(GaussQiTest, MixedScalarArithmetic) {
        GaussQi z(2, 3);
        Rational s(4);

        GaussQi sum = z + s;
        EXPECT_EQ(sum.real(), 6_r);
        EXPECT_EQ(sum.imag(), 3_r);

        GaussQi sum2 = s + z;
        EXPECT_EQ(sum2.real(), 6_r);
        EXPECT_EQ(sum2.imag(), 3_r);

        GaussQi diff = z - s;
        EXPECT_EQ(diff.real(), -2_r);
        EXPECT_EQ(diff.imag(), 3_r);

        GaussQi diff2 = s - z;
        EXPECT_EQ(diff2.real(), 2_r);
        EXPECT_EQ(diff2.imag(), -3_r);

        GaussQi prod = z * s;
        EXPECT_EQ(prod.real(), 8_r);
        EXPECT_EQ(prod.imag(), 12_r);

        GaussQi prod2 = s * z;
        EXPECT_EQ(prod2.real(), 8_r);
        EXPECT_EQ(prod2.imag(), 12_r);

        GaussQi quot = z / s;
        EXPECT_EQ(quot.real(), "1/2"_r);
        EXPECT_EQ(quot.imag(), "3/4"_r);

        GaussQi quot2 = s / z;
        EXPECT_EQ(quot2.real(), "8/13"_r);
        EXPECT_EQ(quot2.imag(), "-12/13"_r);
    }

    TEST_F(GaussQiTest, UnaryMinusAndConjugate) {
        GaussQi z(1, -2);
        GaussQi neg = -z;
        EXPECT_EQ(neg.real(), -1_r);
        EXPECT_EQ(neg.imag(), 2_r);

        GaussQi conj_z = z.conj();
        EXPECT_EQ(conj_z.real(), 1_r);
        EXPECT_EQ(conj_z.imag(), 2_r);

        EXPECT_EQ(delta::conj(z), conj_z);
    }

    TEST_F(GaussQiTest, Norm) {
        GaussQi z(3, 4);
        EXPECT_EQ(z.norm(), 25_r);
    }

    // -------------------------------------------------------------------------
    // 3. Comparisons
    // -------------------------------------------------------------------------

    TEST_F(GaussQiTest, Equality) {
        GaussQi a(1, 2);
        GaussQi b(1, 2);
        GaussQi c(1, 3);
        EXPECT_EQ(a, b);
        EXPECT_NE(a, c);
    }

    // -------------------------------------------------------------------------
    // 4. String representation and double conversion
    // -------------------------------------------------------------------------

    TEST_F(GaussQiTest, StringRepresentation) {
        GaussQi z1(1, 2);
        EXPECT_EQ(z1.to_string(), "(1,2)");

        GaussQi z2(5, 0);
        EXPECT_EQ(z2.to_string(), "5");

        GaussQi z3(0, -3);
        EXPECT_EQ(z3.to_string(), "(0,-3)");
    }

    TEST_F(GaussQiTest, ToDouble) {
        GaussQi z(1, 2);
        auto [re, im] = z.to_double();
        EXPECT_DOUBLE_EQ(re, 1.0);
        EXPECT_DOUBLE_EQ(im, 2.0);
    }

    // -------------------------------------------------------------------------
    // 5. Transcendentals – exp, log, sqrt, pow, abs, arg
    // -------------------------------------------------------------------------

    static const Rational EPS = "1/100000000000000000000"_r; // 1e-20

    TEST_F(GaussQiTest, ExpLogInverse) {
        GaussQi z(2, 3);
        GaussQi logz = delta::log(z, EPS);
        GaussQi exp_logz = delta::exp(logz, EPS);
        EXPECT_RATIONAL_NEAR(exp_logz, z, EPS * 100);

        GaussQi w("1/2"_r, 1_r);
        GaussQi expw = delta::exp(w, EPS);
        GaussQi log_expw = delta::log(expw, EPS);
        EXPECT_RATIONAL_NEAR(log_expw, w, EPS * 100);
    }

    TEST_F(GaussQiTest, ExpRealImag) {
        GaussQi e0 = delta::exp(GaussQi(0), EPS);
        EXPECT_EQ(e0.real(), 1_r);
        EXPECT_EQ(e0.imag(), 0_r);

        GaussQi i_pi(0, delta::pi(EPS));
        GaussQi exp_ipi = delta::exp(i_pi, EPS);
        EXPECT_RATIONAL_NEAR(exp_ipi.real(), -1_r, EPS);
        EXPECT_LE(delta::abs(exp_ipi.imag()), EPS);
    }

    TEST_F(GaussQiTest, Sqrt) {
        GaussQi s1 = delta::sqrt(GaussQi(1), EPS);
        EXPECT_EQ(s1.real(), 1_r);
        EXPECT_EQ(s1.imag(), 0_r);

        GaussQi s_neg1 = delta::sqrt(GaussQi(-1), EPS);
        EXPECT_RATIONAL_NEAR(s_neg1.real(), 0_r, EPS);
        EXPECT_RATIONAL_NEAR(s_neg1.imag(), 1_r, EPS);

        GaussQi z(1, 2);
        GaussQi sqrtz = delta::sqrt(z, EPS);
        GaussQi check = sqrtz * sqrtz;
        EXPECT_RATIONAL_NEAR(check, z, EPS * 100);
    }

    TEST_F(GaussQiTest, AbsAndArg) {
        GaussQi z(1, 1);
        Rational m = delta::abs(z, EPS);
        EXPECT_RATIONAL_NEAR(m * m, 2_r, EPS);

        Rational arg = delta::arg(z, EPS);
        Rational pi4 = delta::pi(EPS) / 4_r;
        EXPECT_RATIONAL_NEAR(arg, pi4, EPS);

        GaussQi neg_one(-1, 0);
        Rational arg_neg = delta::arg(neg_one, EPS);
        EXPECT_RATIONAL_NEAR(arg_neg, delta::pi(EPS), EPS);
    }

    TEST_F(GaussQiTest, PowIntegerExponent) {
        GaussQi base(2, 1);   // 2+i
        GaussQi p2 = delta::pow(base, 2);
        EXPECT_EQ(p2.real(), 3_r);
        EXPECT_EQ(p2.imag(), 4_r);

        GaussQi p0 = delta::pow(base, 0);
        EXPECT_EQ(p0.real(), 1_r);
        EXPECT_EQ(p0.imag(), 0_r);

        GaussQi p_neg = delta::pow(base, -1);
        EXPECT_EQ(p_neg.real(), "2/5"_r);
        EXPECT_EQ(p_neg.imag(), "-1/5"_r);
    }

    TEST_F(GaussQiTest, PowGeneralExponent) {
        GaussQi z(2, 1);
        GaussQi w(0, 1);   // i
        GaussQi zi = delta::pow(z, w, EPS);
        EXPECT_GT(delta::abs(zi, EPS), 0_r);
    }

    // -------------------------------------------------------------------------
    // 6. Error handling
    // -------------------------------------------------------------------------

    TEST_F(GaussQiTest, DivisionByZero) {
        GaussQi z(1, 2);
        GaussQi zero(0, 0);
        EXPECT_THROW(z / zero, std::domain_error);
        EXPECT_THROW(z / 0_r, std::domain_error);
    }

    TEST_F(GaussQiTest, LogOfZero) {
        GaussQi zero(0, 0);
        EXPECT_THROW(delta::log(zero, EPS), std::domain_error);
    }

    TEST_F(GaussQiTest, ZeroToZero) {
        GaussQi zero(0, 0);
        EXPECT_THROW(delta::pow(zero, zero, EPS), std::domain_error);
    }

    // -------------------------------------------------------------------------
    // 7. User-defined literals _qi
    // -------------------------------------------------------------------------

    TEST_F(GaussQiTest, Literals) {
        using namespace delta::literals;

        auto z1 = "1+2i"_qi;
        EXPECT_EQ(z1.real(), 1_r);
        EXPECT_EQ(z1.imag(), 2_r);

        auto z2 = "1-2i"_qi;
        EXPECT_EQ(z2.real(), 1_r);
        EXPECT_EQ(z2.imag(), -2_r);

        auto z3 = "i"_qi;
        EXPECT_EQ(z3.real(), 0_r);
        EXPECT_EQ(z3.imag(), 1_r);

        auto z4 = "-i"_qi;
        EXPECT_EQ(z4.real(), 0_r);
        EXPECT_EQ(z4.imag(), -1_r);

        auto z5 = "2i"_qi;
        EXPECT_EQ(z5.real(), 0_r);
        EXPECT_EQ(z5.imag(), 2_r);

        auto z6 = "-3i"_qi;
        EXPECT_EQ(z6.real(), 0_r);
        EXPECT_EQ(z6.imag(), -3_r);

        auto z7 = "1+i"_qi;
        EXPECT_EQ(z7.real(), 1_r);
        EXPECT_EQ(z7.imag(), 1_r);

        auto z8 = "1-i"_qi;
        EXPECT_EQ(z8.real(), 1_r);
        EXPECT_EQ(z8.imag(), -1_r);

        auto z9 = "1/2+3/4i"_qi;
        EXPECT_EQ(z9.real(), "1/2"_r);
        EXPECT_EQ(z9.imag(), "3/4"_r);

        auto z10 = "0.333i"_qi;
        EXPECT_EQ(z10.real(), 0_r);
        EXPECT_EQ(z10.imag(), "333/1000"_r);

        auto z11 = "2.5-1.2i"_qi;
        EXPECT_EQ(z11.real(), "5/2"_r);
        EXPECT_EQ(z11.imag(), "-6/5"_r);

        auto z12 = "42"_qi;
        EXPECT_EQ(z12.real(), 42_r);
        EXPECT_EQ(z12.imag(), 0_r);

        auto z13 = "-17/3"_qi;
        EXPECT_EQ(z13.real(), "-17/3"_r);
        EXPECT_EQ(z13.imag(), 0_r);

        auto z14 = "(1/2,-3/4)"_qi;
        EXPECT_EQ(z14.real(), "1/2"_r);
        EXPECT_EQ(z14.imag(), "-3/4"_r);

        auto z15 = " 1 + 2i "_qi;
        EXPECT_EQ(z15.real(), 1_r);
        EXPECT_EQ(z15.imag(), 2_r);
    }
} // namespace delta::testing