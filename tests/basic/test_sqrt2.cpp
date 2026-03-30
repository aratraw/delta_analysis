#include <gtest/gtest.h>
#include <vector>
#include "../test_fixtures.h"

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

    auto contains_sqrt2 = [](const ListGrid<Addr, Compare>& grid) -> Addr {
        const auto& data = grid.data();
        // Find the interval containing sqrt(2) ≈ 1.41421356
        for (size_t i = 0; i + 1 < data.size(); ++i) {
            if (data[i] <= 141421356_r / 100000000_r && data[i + 1] >= 141421356_r / 100000000_r) {
                return data[i];
            }
        }
        return Addr(-1);
        };

    std::vector<Addr> left_endpoints;
    for (int i = 0; i < 10; ++i) {
        left_endpoints.push_back(contains_sqrt2(path.current_grid()));
        path.advance([](const Addr&) { return Addr(0); });
    }

    // Check that the sequence of left endpoints converges
    for (size_t i = 1; i < left_endpoints.size(); ++i) {
        Addr diff = left_endpoints[i] - left_endpoints[i - 1];
        // The difference should decrease roughly as 2/2^i
        double expected = 2.0 / (1 << i);
        EXPECT_LE(diff.to_double(), 2.0 / (1 << i) + 1e-12);
    }

    // Invariant: all grids are sorted
    EXPECT_TRUE(is_sorted(path.current_grid()));
    EXPECT_TRUE(bounds_match(path.current_grid(), 0_r, 2_r));
}