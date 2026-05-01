// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

#include <gtest/gtest.h>
#include "../test_fixtures.h"

using namespace delta::testing;

/**
 * @class DifferentiabilityTest
 * @brief Tests for differentiability concepts (though here we only check basic
 *        path invariants; detailed differentiability checks are in calculus/).
 */
class DifferentiabilityTest : public DeltaTest {};

/**
 * @test For a quadratic function on a dyadic path, verify that after several
 *       refinements the grid remains sorted, bounds are preserved, and the
 *       level has increased. This is a sanity check; actual differentiability
 *       properties are tested elsewhere.
 */
TEST_F(DifferentiabilityTest, QuadraticAtMidpoint) {
    ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
    auto path = make_midpoint_path(grid0);

    auto func = [](const Addr& x) { return x * x; };

    for (int i = 0; i < 5; ++i) {
        path.advance(func);
    }

    // Verify that the grid remains correct
    EXPECT_TRUE(is_sorted(path.current_grid()));
    EXPECT_TRUE(bounds_match(path.current_grid(), 0_r, 1_r));
    EXPECT_GT(path.level(), 0);
}

/**
 * @test For the absolute value function (non‑differentiable at 0) on a dyadic
 *       path, check that the grid stays sorted and bounds are preserved.
 *       This only verifies that the path machinery works even for non‑smooth
 *       functions.
 */
TEST_F(DifferentiabilityTest, AbsoluteValueNotDifferentiable) {
    ListGrid<Addr, Compare> grid0({ -1_r, 0_r, 1_r });
    auto path = make_midpoint_path(grid0);

    auto func = [](const Addr& x) { return x >= 0_r ? x : -x; };

    for (int i = 0; i < 5; ++i) {
        path.advance(func);
    }

    EXPECT_TRUE(is_sorted(path.current_grid()));
    EXPECT_TRUE(bounds_match(path.current_grid(), -1_r, 1_r));
    EXPECT_GT(path.level(), 0);
}