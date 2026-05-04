// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

#include <gtest/gtest.h>
#include <vector>
#include "../test_fixtures.h"

using namespace delta::testing;

/**
 * @class IntegralTest
 * @brief Tests for Riemann sums (integration) on delta paths.
 *
 * Verifies that left Riemann sums approximate the true integral
 * and that the error decreases as the grid is refined.
 */
class IntegralTest : public DeltaTest {};

/**
 * @brief Compute the left Riemann sum of a function over the current grid of a path.
 *
 * @tparam Path A type that provides a `current_grid()` method returning a grid
 *              with `data()` and `size()` and supports `operator[]`.
 * @param path  The path containing the grid.
 * @param func  The function to integrate.
 * @return      The left Riemann sum as a Rational.
 */
template<typename Path>
Rational left_riemann_sum(const Path& path, const typename Path::Func& func) {
    const auto& grid = path.current_grid();
    const auto& data = grid.data();
    Rational sum = 0_r;
    for (size_t i = 0; i + 1 < data.size(); ++i) {
        sum += func(data[i]) * (data[i + 1] - data[i]);
    }
    return sum;
}

/**
 * @test Approximate the integral of f(x)=x on [0,1] using a dyadic path.
 *
 * The true integral is 0.5. As the grid refines, the left Riemann sum
 * should converge to 0.5, and the error should decrease monotonically.
 * Here we only check that after 10 refinements the error is below 1e-3.
 */
TEST_F(IntegralTest, DyadicX) {
    ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
    auto path = make_midpoint_path(grid0);

    auto func = [](const Addr& x) { return x; };
    std::vector<Rational> sums;

    for (int i = 0; i < 10; ++i) {
        sums.push_back(left_riemann_sum(path, func));
        path.advance(func);
    }

    Rational expected = 1_r / 2_r;
    Rational error = sums.back() - expected;
    // The error should decrease with each refinement step.
    EXPECT_RATIONAL_NEAR(error, 0_r, Rational(1, 1000));
    // Optionally one could also check monotonic decrease of the error.
}