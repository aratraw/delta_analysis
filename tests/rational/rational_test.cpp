// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/rational/rational_test.cpp
// ============================================================================
// BASIC TESTS FOR RATIONAL (EAGER, IMMUTABLE RATIONAL NUMBERS)
// ============================================================================
//
// This file tests the core functionality of the Rational class:
//   - Constructors (default, integer, string, numerator/denominator).
//   - String parsing (integers, decimals, fractions).
//   - Arithmetic operators (+, -, *, /) and compound assignments.
//   - Negation and absolute value.
//   - Comparison operators.
//   - to_string round‑trip and canonical form (gcd reduction, positive denominator).
//   - Denominator growth behaviour (does not explode unnecessarily).
//   - Cross‑cancellation in multiplication.
//   - Large integer exponentiation (positive and negative exponents).
//   - Division by zero (exception handling).
//   - Zero representation.
//
// All tests are eager (immediate) and use rational comparisons.
// ============================================================================

#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "test_utils.h"

namespace delta::testing {
    class RationalBasicTest : public RationalTest {};

    // -------------------------------------------------------------------------
    // 1. Constructors
    // -------------------------------------------------------------------------

    /**
     * @test Constructors
     * @brief Tests default, integer, and string constructors.
     */
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

    /**
     * @test StringParsing
     * @brief Verifies that strings are correctly parsed into rationals
     *        (integers, decimal fractions, common fractions).
     */
    TEST_F(RationalBasicTest, StringParsing) {
        // Integers
        EXPECT_EQ("123"_r.to_string(), "123");
        EXPECT_EQ("-456"_r.to_string(), "-456");

        // Decimal fractions
        EXPECT_EQ("0.75"_r.to_string(), "3/4");
        EXPECT_EQ("-0.125"_r.to_string(), "-1/8");
        EXPECT_EQ("0.0"_r.to_string(), "0");

        // Common fractions
        EXPECT_EQ("5/8"_r.to_string(), "5/8");
        EXPECT_EQ("-7/9"_r.to_string(), "-7/9");
        EXPECT_EQ("0/1"_r.to_string(), "0");
    }

    // -------------------------------------------------------------------------
    // 3. Arithmetic
    // -------------------------------------------------------------------------

    /**
     * @test Arithmetic
     * @brief Checks addition, subtraction, multiplication, and division.
     */
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

    /**
     * @test CompoundAssignments
     * @brief Tests +=, -=, *=, /= operators.
     */
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

    /**
     * @test Negation
     * @brief Verifies unary minus.
     */
    TEST_F(RationalBasicTest, Negation) {
        Rational a = -"1/2"_r;
        EXPECT_EQ(a.to_string(), "-1/2");
    }

    // -------------------------------------------------------------------------
    // 6. Abs
    // -------------------------------------------------------------------------

    /**
     * @test Abs
     * @brief Checks absolute value function.
     */
    TEST_F(RationalBasicTest, Abs) {
        Rational a = delta::abs("-1/2"_r);
        EXPECT_EQ(a.to_string(), "1/2");
    }

    // -------------------------------------------------------------------------
    // 7. Comparison operators
    // -------------------------------------------------------------------------

    /**
     * @test Comparison
     * @brief Tests <, >, ==, !=, <=, >=.
     */
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

    /**
     * @test ToFromString
     * @brief Converts a rational to a string and back; should yield the same value.
     */
    TEST_F(RationalBasicTest, ToFromString) {
        Rational r = "123/456"_r;
        std::string s = r.to_string();
        Rational r2(s);
        EXPECT_EQ(r2, r);
    }

    // -------------------------------------------------------------------------
    // 9. Canonical form (denominator positive, gcd == 1)
    // -------------------------------------------------------------------------

    /**
     * @test CanonicalForm
     * @brief After arithmetic, the result is reduced (gcd = 1, denominator positive).
     */
    TEST_F(RationalBasicTest, CanonicalForm) {
        Rational sum = "2/6"_r + "1/6"_r;
        EXPECT_TRUE(is_reduced(sum));
        EXPECT_EQ(sum.to_string(), "1/2");
    }

    // -------------------------------------------------------------------------
    // 10. Denominator does not explode on chain of additions
    // -------------------------------------------------------------------------

    /**
     * @test DenominatorDoesNotExplode
     * @brief Sum of 1 + 1/2 + ... + 1/10 yields denominator 2520 (least common multiple),
     *        not an astronomically large number.
     */
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

    /**
     * @test CrossCancellation
     * @brief Multiplication of a very large numerator by its reciprocal yields 1.
     */
    TEST_F(RationalBasicTest, CrossCancellation) {
        Rational a = "99999999999999999999/1"_r;
        Rational b = "1/99999999999999999999"_r;
        Rational c = (a * b);
        EXPECT_EQ(c, 1_r);
    }

    // -------------------------------------------------------------------------
    // 12. Large powers (integer exponent)
    // -------------------------------------------------------------------------

    /**
     * @test LargePowers
     * @brief Raises (2/3) to the 10th and to the -10th power.
     */
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

    /**
     * @test DivisionByZero
     * @brief Division by zero (both runtime and construction) should throw an exception.
     */
    TEST_F(RationalBasicTest, DivisionByZero) {
        EXPECT_THROW("1/2"_r / 0_r, std::exception);
        EXPECT_THROW(Rational(1, 0), std::exception);
    }

    // -------------------------------------------------------------------------
    // 14. Zero representation
    // -------------------------------------------------------------------------

    /**
     * @test ZeroRepresentation
     * @brief Zero is represented as "0".
     */
    TEST_F(RationalBasicTest, ZeroRepresentation) {
        Rational zero = 0_r;
        EXPECT_EQ(zero.to_string(), "0");
    }

} // namespace delta::testing