// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/rational/rational_test.cpp
#pragma once
#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "test_utils.h"

namespace delta::testing {
    class RationalBasicTest : public RationalTest {};
    // -------------------------------------------------------------------------
    // 1. Constructors
    // -------------------------------------------------------------------------
    TEST_F(RationalBasicTest, Constructors) {
        // Default constructor
        Rational a;
        EXPECT_EQ(a.to_string(), "0");

        // Integer constructors
        Rational b(123_r);
        EXPECT_EQ(b.to_string(), "123");
        Rational c(-45_r);
        EXPECT_EQ(c.to_string(), "-45");

        // String constructors
        Rational d("0.5"_r);
        EXPECT_EQ(d.to_string(), "1/2");
        Rational e("1/3"_r);
        EXPECT_EQ(e.to_string(), "1/3");

        // Explicit two‑int constructor
        Rational f(1, 5);
        EXPECT_EQ(f.to_string(), "1/5");
    }

    // -------------------------------------------------------------------------
    // 2. String parsing
    // -------------------------------------------------------------------------
    TEST_F(RationalBasicTest, StringParsing) {
        // Целые числа
        EXPECT_EQ("123"_r.to_string(), "123");
        EXPECT_EQ("-456"_r.to_string(), "-456");

        // Десятичные дроби
        EXPECT_EQ("0.75"_r.to_string(), "3/4");
        EXPECT_EQ("-0.125"_r.to_string(), "-1/8");
        EXPECT_EQ("0.0"_r.to_string(), "0");

        // Обыкновенные дроби
        EXPECT_EQ("5/8"_r.to_string(), "5/8");
        EXPECT_EQ("-7/9"_r.to_string(), "-7/9");
        EXPECT_EQ("0/1"_r.to_string(), "0");
    }

    // -------------------------------------------------------------------------
    // 3. Arithmetic
    // -------------------------------------------------------------------------
    TEST_F(RationalBasicTest, Arithmetic) {
        std::cerr << "Arithmetic test: start" << std::endl;
        // Addition
        Rational sum = "1/2"_r + "1/3"_r;
        EXPECT_EQ(sum.to_string(), "5/6");

        // Subtraction
        Rational diff = "1/2"_r - "1/3"_r;
        EXPECT_EQ(diff.to_string(), "1/6");

        // Multiplication
        Rational prod = "2/3"_r * "3/4"_r;
        EXPECT_EQ(prod.to_string(), "1/2");

        // Division
        Rational quot = "2/3"_r / "4/5"_r;
        EXPECT_EQ(quot.to_string(), "5/6");
    }

    // -------------------------------------------------------------------------
    // 4. Compound assignments
    // -------------------------------------------------------------------------
    TEST_F(RationalBasicTest, CompoundAssignments) {
        Rational a = "1/2"_r;
        a += "1/3"_r;
        EXPECT_EQ(a.to_string(), "5/6");

        Rational b = "1/2"_r;
        b -= "1/3"_r;
        EXPECT_EQ(b.to_string(), "1/6");

        Rational c = "2/3"_r;
        c *= "3/4"_r;
        EXPECT_EQ(c.to_string(), "1/2");

        Rational d = "2/3"_r;
        d /= "4/5"_r;
        EXPECT_EQ(d.to_string(), "5/6");
    }

    // -------------------------------------------------------------------------
    // 5. Negation
    // -------------------------------------------------------------------------
    TEST_F(RationalBasicTest, Negation) {
        Rational a = -"1/2"_r;
        EXPECT_EQ(a.to_string(), "-1/2");
    }

    // -------------------------------------------------------------------------
    // 6. Abs
    // -------------------------------------------------------------------------
    TEST_F(RationalBasicTest, Abs) {
        Rational a = delta::abs("-1/2"_r);
        EXPECT_EQ(a.to_string(), "1/2");
    }

    // -------------------------------------------------------------------------
    // 7. Comparison operators
    // -------------------------------------------------------------------------
    TEST_F(RationalBasicTest, Comparison) {
        EXPECT_TRUE(("1/2"_r < "3/4"_r));
        EXPECT_FALSE(("1/2"_r > "3/4"_r));
        EXPECT_TRUE(("1/2"_r == "2/4"_r));
        EXPECT_TRUE(("1/2"_r != "2/3"_r));
        EXPECT_TRUE(("1/2"_r <= "2/4"_r));
        EXPECT_TRUE(("1/2"_r >= "2/4"_r));
    }

    // -------------------------------------------------------------------------
    // 8. to_string roundtrip
    // -------------------------------------------------------------------------
    TEST_F(RationalBasicTest, ToFromString) {
        Rational r = "123/456"_r;
        std::string s = r.to_string();
        Rational r2(s);
        EXPECT_EQ(r2, r);
    }

    // -------------------------------------------------------------------------
    // 9. Canonical form (denominator positive, gcd == 1)
    // -------------------------------------------------------------------------
    TEST_F(RationalBasicTest, CanonicalForm) {

        Rational sum = "2/6"_r + "1/6"_r;
        EXPECT_TRUE(is_reduced(sum));
        EXPECT_EQ(sum.to_string(), "1/2");
    }

    // -------------------------------------------------------------------------
    // 10. Denominator does not explode on chain of additions
    // -------------------------------------------------------------------------
    TEST_F(RationalBasicTest, DenominatorDoesNotExplode) {

        Rational sum = 0_r;
        for (int i = 1; i <= 10; ++i) {
            sum += Rational(1, i);
        }
        // The exact denominator is 2520
        std::string s = sum.to_string();
        size_t slash = s.find('/');
        ASSERT_NE(slash, std::string::npos);
        std::string den_str = s.substr(slash + 1);
        EXPECT_EQ(den_str, "2520");
    }

    // -------------------------------------------------------------------------
    // 11. Cross‑cancellation (large numbers)
    // -------------------------------------------------------------------------
    TEST_F(RationalBasicTest, CrossCancellation) {
        Rational a = "99999999999999999999/1"_r;
        Rational b = "1/99999999999999999999"_r;
        Rational c = (a * b);
        EXPECT_EQ(c, 1_r);
    }

    // -------------------------------------------------------------------------
    // 12. Large powers (integer exponent)
    // -------------------------------------------------------------------------
    TEST_F(RationalBasicTest, LargePowers) {
        Rational base = "2/3"_r;
        Rational pow10 = delta::pow(base, 10);
        EXPECT_EQ(pow10.to_string(), "1024/59049");

        Rational pow_neg = delta::pow(base, -10);
        EXPECT_EQ(pow_neg.to_string(), "59049/1024");
    }

    // -------------------------------------------------------------------------
    // 13. Division by zero throws
    // -------------------------------------------------------------------------
    TEST_F(RationalBasicTest, DivisionByZero) {
        EXPECT_THROW("1/2"_r / 0_r, std::exception);
        EXPECT_THROW(Rational(1, 0), std::exception);
    }

    // -------------------------------------------------------------------------
    // 14. Zero representation
    // -------------------------------------------------------------------------
    TEST_F(RationalBasicTest, ZeroRepresentation) {
        Rational zero = 0_r;
        EXPECT_EQ(zero.to_string(), "0");
    }

} // namespace delta::testing