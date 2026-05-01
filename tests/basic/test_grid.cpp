// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

//test_grid.cpp
#include <gtest/gtest.h>
#include "../test_fixtures.h"
#include "delta/core/list_grid.h"

using namespace delta::testing;

/**
 * @class GridTest
 * @brief Tests for the ListGrid class (basic grid functionality).
 *
 * Verifies construction, sorting invariants, and refinement operations
 * with different point insertion strategies.
 */
class GridTest : public DeltaTest {};

/**
 * @test Verify that a grid can be constructed from a sorted initializer list,
 *       that its size and elements are correct, and that it is sorted.
 */
TEST_F(GridTest, Construction) {
    ListGrid<Addr, Compare> grid({ 1_r, 2_r, 3_r });
    EXPECT_EQ(grid.size(), 3);
    EXPECT_EQ(grid[0], 1_r);
    EXPECT_EQ(grid[1], 2_r);
    EXPECT_EQ(grid[2], 3_r);
    EXPECT_TRUE(is_sorted(grid));
    EXPECT_TRUE(bounds_match(grid, 1_r, 3_r));
}

/**
 * @test Construction with a sorted list should not throw.
 */
TEST_F(GridTest, SortedInputPasses) {
    EXPECT_NO_THROW((ListGrid<Addr, Compare>({ 1_r, 2_r, 3_r })));
}

/**
 * @test Refinement using the midpoint operator should insert the midpoint
 *       between every pair of consecutive points, preserving order and bounds.
 */
TEST_F(GridTest, RefineMidpoint) {
    ListGrid<Addr, Compare> grid({ 0_r, 1_r });
    auto refined = grid.refine([](const Addr& x, const Addr& y) {
        return (x + y) / 2_r;
        });
    EXPECT_EQ(refined.size(), 3);
    EXPECT_EQ(refined[0], 0_r);
    EXPECT_EQ(refined[1], 1_r / 2_r);
    EXPECT_EQ(refined[2], 1_r);
    EXPECT_TRUE(is_sorted(refined));
    EXPECT_TRUE(bounds_match(refined, 0_r, 1_r));
}

/**
 * @test Refinement with a general lambda (here a fixed fraction λ=1/3) should
 *       insert points at the specified fraction, maintaining order.
 */
TEST_F(GridTest, RefineLambda) {
    ListGrid<Addr, Compare> grid({ 0_r, 1_r });
    Rational lambda = 1_r / 3_r;
    auto refined = grid.refine([lambda](const Addr& x, const Addr& y) {
        return x + lambda * (y - x);
        });
    EXPECT_EQ(refined.size(), 3);
    EXPECT_EQ(refined[0], 0_r);
    EXPECT_EQ(refined[1], 1_r / 3_r);
    EXPECT_EQ(refined[2], 1_r);
}