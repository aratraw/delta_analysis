// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/calculus/test_riemann_sum.cpp
#include <gtest/gtest.h>
#include "test_fixtures.h"
#include "delta/calculus/riemann_sum.h"

namespace delta::testing {

    /**
     * @class RiemannSumTest
     * @brief Tests for the Riemann sum functions (left, right, tagged).
     *
     * Verifies that Riemann sums on dyadic grids give the expected values
     * for simple functions and that edge cases (empty grid, single point)
     * are handled correctly.
     */
    class RiemannSumTest : public DeltaTest {};

    /**
     * @test Left Riemann sum of f(x)=x on a dyadic path.
     *       At level 0 (grid [0,1]) the sum is 0.
     *       At level 1 (grid [0, 1/2, 1]) the sum is 1/4.
     *       At level 2 (grid [0, 1/4, 1/2, 3/4, 1]) the sum is 3/8.
     */
    TEST_F(RiemannSumTest, LeftSumIdentityOnDyadic) {
        ListGrid<Addr, Compare> grid({ 0_r, 1_r });
        auto path = make_midpoint_path(grid);

        auto func = [](const Addr& x) { return x; };

        // Level 0: grid [0,1], left sum = 0 * 1 = 0
        auto sum0 = calculus::left_riemann_sum(path.current_grid(), func);
        EXPECT_EQ(sum0, 0_r);

        path.advance(func);
        // Level 1: grid [0, 1/2, 1], left sum = 0*0.5 + 0.5*0.5 = 0.25
        auto sum1 = calculus::left_riemann_sum(path.current_grid(), func);
        EXPECT_EQ(sum1, 1_r / 4_r);

        path.advance(func);
        // Level 2: grid [0, 1/4, 1/2, 3/4, 1], left sum =
        // 0*0.25 + 0.25*0.25 + 0.5*0.25 + 0.75*0.25 = (0+0.25+0.5+0.75)*0.25 = 1.5*0.25 = 0.375
        auto sum2 = calculus::left_riemann_sum(path.current_grid(), func);
        EXPECT_EQ(sum2, 3_r / 8_r);
    }

    /**
     * @test Right Riemann sum of f(x)=x on a dyadic path.
     *       At level 0 the sum is 1.
     *       At level 1 the sum is 3/4.
     *       At level 2 the sum is 5/8.
     */
    TEST_F(RiemannSumTest, RightSumIdentityOnDyadic) {
        ListGrid<Addr, Compare> grid({ 0_r, 1_r });
        auto path = make_midpoint_path(grid);

        auto func = [](const Addr& x) { return x; };

        auto sum0 = calculus::right_riemann_sum(path.current_grid(), func);
        EXPECT_EQ(sum0, 1_r); // 1 * 1

        path.advance(func);
        auto sum1 = calculus::right_riemann_sum(path.current_grid(), func);
        EXPECT_EQ(sum1, 3_r / 4_r); // 0.5*0.5 + 1*0.5 = 0.75

        path.advance(func);
        auto sum2 = calculus::right_riemann_sum(path.current_grid(), func);
        // 0.25*0.25 + 0.5*0.25 + 0.75*0.25 + 1*0.25 = (0.25+0.5+0.75+1)*0.25 = 2.5*0.25 = 0.625
        EXPECT_EQ(sum2, 5_r / 8_r);
    }

    /**
     * @test Tagged Riemann sum with a left‑point tagger on grid [0,1,2].
     *       For f(x)=x², the left‑tagged sum should use the left endpoint of each subinterval.
     */
    TEST_F(RiemannSumTest, TaggedSumWithLeftTagger) {
        ListGrid<Addr, Compare> grid({ 0_r, 1_r, 2_r });
        auto func = [](const Addr& x) { return x * x; };

        auto left_tagger = [](const Addr& left, const Addr&) { return left; };
        auto sum = calculus::tagged_riemann_sum(grid, func, left_tagger);
        auto expected = func(0_r) * (1_r - 0_r) + func(1_r) * (2_r - 1_r);
        EXPECT_EQ(sum, expected);
    }

    /**
     * @test Tagged Riemann sum with a right‑point tagger on grid [0,1,2].
     *       For f(x)=x², the right‑tagged sum should use the right endpoint of each subinterval.
     */
    TEST_F(RiemannSumTest, TaggedSumWithRightTagger) {
        ListGrid<Addr, Compare> grid({ 0_r, 1_r, 2_r });
        auto func = [](const Addr& x) { return x * x; };

        auto right_tagger = [](const Addr&, const Addr& right) { return right; };
        auto sum = calculus::tagged_riemann_sum(grid, func, right_tagger);
        auto expected = func(1_r) * (1_r - 0_r) + func(2_r) * (2_r - 1_r);
        EXPECT_EQ(sum, expected);
    }

    /**
     * @test Riemann sum on an empty grid should return zero.
     */
    TEST_F(RiemannSumTest, EmptyGridReturnsZero) {
        ListGrid<Addr, Compare> empty;
        auto func = [](const Addr&) { return 1_r; };
        auto sum = calculus::left_riemann_sum(empty, func);
        EXPECT_EQ(sum, 0_r);
    }

    /**
     * @test Riemann sum on a single‑point grid should return zero.
     */
    TEST_F(RiemannSumTest, SinglePointGridReturnsZero) {
        ListGrid<Addr, Compare> grid({ 42_r });
        auto func = [](const Addr&) { return 1_r; };
        auto sum = calculus::left_riemann_sum(grid, func);
        EXPECT_EQ(sum, 0_r);
    }

} // namespace delta::testing