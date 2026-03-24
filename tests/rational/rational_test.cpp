// tests/rational/rational_test.cpp
#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "test_utils.h"

namespace delta::testing {

    // -------------------------------------------------------------------------
    // 1. Constructors
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, Constructors) {
        // Default constructor
        Rational a;
        EXPECT_EQ(delta::to_string(a), "0");

        // Integer constructors
        Rational b(123_r);
        EXPECT_EQ(delta::to_string(b), "123");
        Rational c(-45_r);
        EXPECT_EQ(delta::to_string(c), "-45");

        // String constructors
        Rational d("0.5"_r);
        EXPECT_EQ(delta::to_string(d), "1/2");
        Rational e("1/3"_r);
        EXPECT_EQ(delta::to_string(e), "1/3");

        // Explicit two‑int constructor
        Rational f(1, 5);
        EXPECT_EQ(delta::to_string(f), "1/5");
    }

    // -------------------------------------------------------------------------
    // 2. String parsing
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, StringParsing) {
        // Целые числа
        EXPECT_EQ(delta::to_string("123"_r), "123");
        EXPECT_EQ(delta::to_string("-456"_r), "-456");

        // Десятичные дроби
        EXPECT_EQ(delta::to_string("0.75"_r), "3/4");
        EXPECT_EQ(delta::to_string("-0.125"_r), "-1/8");
        EXPECT_EQ(delta::to_string("0.0"_r), "0");

        // Обыкновенные дроби
        EXPECT_EQ(delta::to_string("5/8"_r), "5/8");
        EXPECT_EQ(delta::to_string("-7/9"_r), "-7/9");
        EXPECT_EQ(delta::to_string("0/1"_r), "0");
    }

    // -------------------------------------------------------------------------
    // 3. Arithmetic
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, Arithmetic) {
        std::cerr << "Arithmetic test: start" << std::endl;
        // Addition
        Rational sum = "1/2"_r + "1/3"_r;
        EXPECT_EQ(delta::to_string(sum), "5/6");

        // Subtraction
        Rational diff = "1/2"_r - "1/3"_r;
        EXPECT_EQ(delta::to_string(diff), "1/6");

        // Multiplication
        Rational prod = "2/3"_r * "3/4"_r;
        EXPECT_EQ(delta::to_string(prod), "1/2");

        // Division
        Rational quot = "2/3"_r / "4/5"_r;
        EXPECT_EQ(delta::to_string(quot), "5/6");
    }

    // -------------------------------------------------------------------------
    // 4. Compound assignments
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, CompoundAssignments) {
        Rational a = "1/2"_r;
        a += "1/3"_r;
        EXPECT_EQ(delta::to_string(a), "5/6");

        Rational b = "1/2"_r;
        b -= "1/3"_r;
        EXPECT_EQ(delta::to_string(b), "1/6");

        Rational c = "2/3"_r;
        c *= "3/4"_r;
        EXPECT_EQ(delta::to_string(c), "1/2");

        Rational d = "2/3"_r;
        d /= "4/5"_r;
        EXPECT_EQ(delta::to_string(d), "5/6");
    }

    // -------------------------------------------------------------------------
    // 5. Negation
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, Negation) {
        Rational a = -"1/2"_r;
        EXPECT_EQ(delta::to_string(a), "-1/2");
    }

    // -------------------------------------------------------------------------
    // 6. Abs
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, Abs) {
        Rational a = delta::abs("-1/2"_r);
        EXPECT_EQ(delta::to_string(a), "1/2");
    }

    // -------------------------------------------------------------------------
    // 7. Comparison operators
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, Comparison) {
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
    TEST_F(RationalTest, ToFromString) {
        Rational r = "123/456"_r;
        std::string s = delta::to_string(r);
        Rational r2(s);
        EXPECT_EQ(r2, r);
    }

    // -------------------------------------------------------------------------
    // 9. Canonical form (denominator positive, gcd == 1)
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, CanonicalForm) {
        ScopedEagerEval eager;
        Rational sum = "2/6"_r + "1/6"_r;
        EXPECT_TRUE(is_reduced(sum));
        EXPECT_EQ(delta::to_string(sum), "1/2");
    }

    // -------------------------------------------------------------------------
    // 10. Denominator does not explode on chain of additions
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, DenominatorDoesNotExplode) {
        ScopedEagerEval eager;
        Rational sum = 0_r;
        for (int i = 1; i <= 10; ++i) {
            sum += Rational(1, i);
        }
        // The exact denominator is 2520
        std::string s = delta::to_string(sum);
        size_t slash = s.find('/');
        ASSERT_NE(slash, std::string::npos);
        std::string den_str = s.substr(slash + 1);
        EXPECT_EQ(den_str, "2520");
    }

    // -------------------------------------------------------------------------
    // 11. Cross‑cancellation (large numbers)
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, CrossCancellation) {
        Rational a = "99999999999999999999/1"_r;
        Rational b = "1/99999999999999999999"_r;
        Rational c = (a * b).simplify();
        EXPECT_EQ(c, 1_r);
    }

    // -------------------------------------------------------------------------
    // 12. Large powers (integer exponent)
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, LargePowers) {
        Rational base = "2/3"_r;
        Rational pow10 = delta::pow(base, 10);
        EXPECT_EQ(delta::to_string(pow10), "1024/59049");

        Rational pow_neg = delta::pow(base, -10);
        EXPECT_EQ(delta::to_string(pow_neg), "59049/1024");
    }

    // -------------------------------------------------------------------------
    // 13. Division by zero throws
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, DivisionByZero) {
        EXPECT_THROW("1/2"_r / 0_r, std::exception);
        EXPECT_THROW(Rational(1, 0), std::exception);
    }

    // -------------------------------------------------------------------------
    // 14. Zero representation
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, ZeroRepresentation) {
        Rational zero = 0_r;
        EXPECT_EQ(delta::to_string(zero), "0");
    }

} // namespace delta::testing