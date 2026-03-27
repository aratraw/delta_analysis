// tests/rational/interval_test.cpp
#include <gtest/gtest.h>
#include <limits>
#include "delta/rational/interval.h"
#include "test_utils.h"

namespace delta::testing {

    using internal::Interval;

    class IntervalTest : public RationalTest {
    protected:
        const double inf = std::numeric_limits<double>::infinity();
    };

    // -------------------------------------------------------------------------
    // 1. Constructors
    // -------------------------------------------------------------------------
    TEST_F(IntervalTest, Constructors) {
        Interval a(1.0);
        EXPECT_DOUBLE_EQ(a.lower(), 1.0);
        EXPECT_DOUBLE_EQ(a.upper(), 1.0);

        Interval b(1.0, 2.0);
        EXPECT_DOUBLE_EQ(b.lower(), 1.0);
        EXPECT_DOUBLE_EQ(b.upper(), 2.0);
    }

    // -------------------------------------------------------------------------
    // 2. Addition
    // -------------------------------------------------------------------------
    TEST_F(IntervalTest, Addition) {
        Interval a(1.0, 2.0);
        Interval b(0.5, 1.0);
        Interval c = a + b;
        // Expected [1.5, 3.0] but extended by one ulp outward
        EXPECT_LE(c.lower(), 1.5);
        EXPECT_GE(c.upper(), 3.0);
        // Check that it contains the theoretical interval.
        EXPECT_GE(c.lower(), 1.5 - std::numeric_limits<double>::epsilon());
        EXPECT_LE(c.upper(), 3.0 + std::numeric_limits<double>::epsilon());
    }

    // -------------------------------------------------------------------------
    // 3. Subtraction
    // -------------------------------------------------------------------------
    TEST_F(IntervalTest, Subtraction) {
        Interval a(2.0, 3.0);
        Interval b(1.0, 2.0);
        Interval c = a - b;  // [2-2, 3-1] = [0,2]
        EXPECT_LE(c.lower(), 0.0);
        EXPECT_GE(c.upper(), 2.0);
    }

    // -------------------------------------------------------------------------
    // 4. Multiplication
    // -------------------------------------------------------------------------
    TEST_F(IntervalTest, Multiplication) {
        Interval a(1.0, 2.0);
        Interval b(3.0, 4.0);
        Interval c = a * b;  // min = 1*3=3, max = 2*4=8
        EXPECT_LE(c.lower(), 3.0);
        EXPECT_GE(c.upper(), 8.0);

        // Test negative intervals
        Interval d(-2.0, -1.0);
        Interval e = a * d;  // min = 2*(-2)= -4, max = 1*(-1)= -1
        EXPECT_LE(e.lower(), -4.0);
        EXPECT_GE(e.upper(), -1.0);
    }

    // -------------------------------------------------------------------------
    // 5. Division (without zero)
    // -------------------------------------------------------------------------
    TEST_F(IntervalTest, Division) {
        Interval a(4.0, 8.0);
        Interval b(2.0, 4.0);
        Interval c = a / b;  // min = 4/4=1, max = 8/2=4
        EXPECT_LE(c.lower(), 1.0);
        EXPECT_GE(c.upper(), 4.0);
    }

    // -------------------------------------------------------------------------
    // 6. Division by zero (interval containing zero)
    // -------------------------------------------------------------------------
    TEST_F(IntervalTest, DivisionByZero) {
        Interval a(1.0, 2.0);
        Interval b(-1.0, 1.0);
        Interval c = a / b;
        // Should return (-inf, +inf)
        EXPECT_EQ(c.lower(), -inf);
        EXPECT_EQ(c.upper(), inf);
    }

    // -------------------------------------------------------------------------
    // 7. Negation
    // -------------------------------------------------------------------------
    TEST_F(IntervalTest, Negation) {
        Interval a(1.0, 2.0);
        Interval b = -a;
        EXPECT_DOUBLE_EQ(b.lower(), -2.0);
        EXPECT_DOUBLE_EQ(b.upper(), -1.0);
    }

    // -------------------------------------------------------------------------
    // 8. Comparison (for non‑overlapping intervals)
    // -------------------------------------------------------------------------
    TEST_F(IntervalTest, Comparison) {
        Interval a(1.0, 2.0);
        Interval b(3.0, 4.0);
        EXPECT_TRUE(a < b);
        EXPECT_FALSE(a > b);
        EXPECT_TRUE(a <= b);
        EXPECT_FALSE(a >= b);
    }

    // -------------------------------------------------------------------------
    // 9. Overlaps
    // -------------------------------------------------------------------------
    TEST_F(IntervalTest, Overlaps) {
        Interval a(1.0, 3.0);
        Interval b(2.0, 4.0);
        EXPECT_TRUE(a.overlaps(b));
        EXPECT_TRUE(b.overlaps(a));

        Interval c(1.0, 2.0);
        Interval d(3.0, 4.0);
        EXPECT_FALSE(c.overlaps(d));
    }

} // namespace delta::testing