// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/basic/test_sqrt2.cpp
#include <gtest/gtest.h>
#include <vector>
#include "../test_fixtures.h"
#include "delta/rational/transcendentals.h"

using namespace delta::testing;

/**
 * @class Sqrt2Test
 * @brief Tests for approximating √2 using dyadic refinement.
 */
class Sqrt2Test : public DeltaTest {};

/**
 * @test Approximate √2 by tracking the interval that contains it on a dyadic path.
 *
 * The initial grid is [0,2]. After each refinement, we locate the interval that
 * contains √2 and record its left endpoint. The differences between successive
 * left endpoints should decay as 2/2^i. Finally, we check that the grid remains sorted.
 */
TEST_F(Sqrt2Test, DyadicApproximation) {
    ListGrid<Addr, Compare> grid0({ 0_r, 2_r });
    auto path = make_midpoint_path(grid0);

    // Lambda to find the interval containing sqrt(2) using exact rational comparison
    auto contains_sqrt2 = [](const ListGrid<Addr, Compare>& grid) -> Addr {
        const auto& data = grid.data();
        for (size_t i = 0; i + 1 < data.size(); ++i) {
            const Addr& left = data[i];
            const Addr& right = data[i + 1];
            // Check if left^2 <= 2 <= right^2
            if (left * left <= 2_r && right * right >= 2_r) {
                return left;
            }
        }
        return Addr(-1);
        };

    std::vector<Addr> left_endpoints;
    // Dummy function for path advancement (values not used for grid generation with midpoint operator)
    auto dummy_func = [](const Addr&) { return 0_r; };

    for (int i = 0; i < 10; ++i) {
        left_endpoints.push_back(contains_sqrt2(path.current_grid()));
        path.advance(dummy_func);
    }

    // Check that the sequence of left endpoints converges
    for (size_t i = 1; i < left_endpoints.size(); ++i) {
        Addr diff = left_endpoints[i] - left_endpoints[i - 1];
        if (diff < 0) diff = -diff;
        // Expected bound: 2 / 2^i
        Rational expected = Rational(2) / delta::pow(Rational(2), static_cast<int>(i));
        // Allow a small tolerance for rational approximations (though differences should be exact powers of two)
        Rational tolerance = Rational(1, 1000000000000);
        EXPECT_LE(diff, expected + tolerance)
            << "Difference at step " << i << " = " << diff << ", expected <= " << expected;
    }

    // Invariant: all grids are sorted
    EXPECT_TRUE(is_sorted(path.current_grid()));
    EXPECT_TRUE(bounds_match(path.current_grid(), 0_r, 2_r));
}