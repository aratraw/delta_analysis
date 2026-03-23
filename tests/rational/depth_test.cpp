// tests/rational/depth_test.cpp
#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "delta/rational/context.h"   // for MAX_LAZY_DEPTH
#include "test_utils.h"

namespace delta::testing {

    // Helper to create a left‑associative chain of additions: (((0 + 1) + 1) + ...)
    Rational build_addition_chain(int n) {
        Rational sum = 0_r;
        for (int i = 0; i < n; ++i) {
            sum = sum + 1_r;
        }
        return sum;
    }

    // -------------------------------------------------------------------------
    // 1. Depth tracking
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, DepthTracking) {
        set_eager_mode(false);

        // Build a tree of depth 5 (e.g., ((((0+1)+1)+1)+1))
        Rational root = build_addition_chain(5);
        EXPECT_EQ(root.depth(), 5);
    }

    // -------------------------------------------------------------------------
    // 2. Depth overflow – tree should be collapsed when depth > MAX_LAZY_DEPTH
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, DepthOverflow) {
        set_eager_mode(false);

        int over = internal::MAX_LAZY_DEPTH + 100;   // well above the limit
        Rational root = build_addition_chain(over);

        // The depth should never exceed the limit; the root may become evaluated.
        EXPECT_LE(root.depth(), internal::MAX_LAZY_DEPTH);
        // After overflow, the result should be a concrete rational (non‑lazy)
        EXPECT_FALSE(root.is_lazy());
    }

    // -------------------------------------------------------------------------
    // 3. Eager on overflow – explicit check that the node is replaced by evaluated value
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, EagerOnOverflow) {
        set_eager_mode(false);

        // Build a chain just below the limit
        int just_below = internal::MAX_LAZY_DEPTH - 1;
        Rational root = build_addition_chain(just_below);
        EXPECT_TRUE(root.is_lazy());          // still lazy

        // Add one more to exceed the limit
        root = root + 1_r;
        // Depth should be <= MAX_LAZY_DEPTH, and result should be non‑lazy
        EXPECT_LE(root.depth(), internal::MAX_LAZY_DEPTH);
        EXPECT_FALSE(root.is_lazy());
    }

    // -------------------------------------------------------------------------
    // 4. No premature collapse – chain below the limit stays lazy
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, NoPrematureCollapse) {
        set_eager_mode(false);

        int below = 500;   // 500 < MAX_LAZY_DEPTH (which is 1000)
        Rational root = build_addition_chain(below);
        EXPECT_TRUE(root.is_lazy());
        EXPECT_EQ(root.depth(), below);
    }

} // namespace delta::testing