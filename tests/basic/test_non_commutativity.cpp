// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/basic/test_non_commutativity.cpp
#include <gtest/gtest.h>
#include <sstream>
#include "../test_fixtures.h"

using namespace delta::testing;

/**
 * @class NonCommutativityTest
 * @brief Tests for non‑commutativity of delta operators.
 *
 * Verifies that applying operators in different orders yields different grids,
 * while preserving sorting and bounds.
 */
class NonCommutativityTest : public DeltaTest {};

/**
 * @test Using two different fixed‑lambda operators (1/3 and 2/3) in opposite
 *       orders should produce different grids after two refinement steps.
 */
TEST_F(NonCommutativityTest, Lambda13vs23) {
    FixedLambdaOperator op13(1_r / 3_r);
    FixedLambdaOperator op23(2_r / 3_r);

    using OpType = FixedLambdaOperator;
    auto strategy1 = make_dynamic_strategy<OpType>({ op13, op23 });
    auto strategy2 = make_dynamic_strategy<OpType>({ op23, op13 });

    ListGrid<Addr, Compare> grid0({ 0_r, 1_r });

    auto path1 = make_path(grid0, std::move(strategy1));
    auto path2 = make_path(grid0, std::move(strategy2));

    auto dummy = [](const Addr&) { return Addr(0); };

    path1.advance(dummy);
    path1.advance(dummy);
    path2.advance(dummy);
    path2.advance(dummy);

    auto grid1 = path1.current_grid();
    auto grid2 = path2.current_grid();

    // Invariant checks
    EXPECT_TRUE(is_sorted(grid1));
    EXPECT_TRUE(is_sorted(grid2));
    EXPECT_TRUE(bounds_match(grid1, 0_r, 1_r));
    EXPECT_TRUE(bounds_match(grid2, 0_r, 1_r));

    // Check specific values (important for this test)
    EXPECT_EQ(grid1[0], 0_r);
    EXPECT_EQ(grid1[1], 2_r / 9_r);
    EXPECT_EQ(grid1[2], 1_r / 3_r);
    EXPECT_EQ(grid1[3], 7_r / 9_r);
    EXPECT_EQ(grid1[4], 1_r);

    EXPECT_EQ(grid2[0], 0_r);
    EXPECT_EQ(grid2[1], 2_r / 9_r);
    EXPECT_EQ(grid2[2], 2_r / 3_r);
    EXPECT_EQ(grid2[3], 7_r / 9_r);
    EXPECT_EQ(grid2[4], 1_r);

    // Grids must differ
    EXPECT_NE(grid1[2], grid2[2]);
}