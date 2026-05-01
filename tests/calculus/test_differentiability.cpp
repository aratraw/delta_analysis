// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/calculus/test_differentiability.cpp
#include <gtest/gtest.h>
#include "test_fixtures.h"

namespace delta::testing {

    /**
     * @class DifferentiabilityTest
     * @brief Tests for differentiability checks using modulus of convergence.
     *
     * Verifies that the function check_differentiability correctly determines
     * whether a function has a given derivative at a point, based on the
     * difference quotients on a sequence of refined grids.
     */
    class DifferentiabilityTest : public DeltaTest {};

    /**
     * @test Identity function f(x)=x at x=1/2.
     *       The derivative should be 1, and the error should be zero,
     *       so any modulus with C=0 works.
     */
    TEST_F(DifferentiabilityTest, IdentityFunction) {
        ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
        auto path = make_midpoint_path(grid0);
        auto func = [](const Addr& x) { return x; };

        using GridType = ListGrid<Addr, Compare>;
        std::vector<GridType> grids;
        grids.push_back(path.current_grid());
        for (int i = 0; i < 5; ++i) {
            path.advance(func);
            grids.push_back(path.current_grid());
        }

        Addr x = 1_r / 2_r;
        Dist D = 1_r;
        PowerModulus<Rational> modulus(0_r, 1_r);
        Rational tolerance = Rational(1, 1000000000000);

        bool diff = check_differentiability(grids, x, func, D, modulus, 1, tolerance);
        EXPECT_TRUE(diff);
    }

    /**
     * @test Quadratic function f(x)=x² at x=1/2.
     *       The derivative is 1, and the error is bounded by the grid step,
     *       so the linear modulus ω(δ)=δ should be satisfied.
     */
    TEST_F(DifferentiabilityTest, QuadraticAtHalf) {
        ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
        auto path = make_midpoint_path(grid0);
        auto func = [](const Addr& x) { return x * x; };

        using GridType = ListGrid<Addr, Compare>;
        std::vector<GridType> grids;
        grids.push_back(path.current_grid());
        for (int i = 0; i < 5; ++i) {
            path.advance(func);
            grids.push_back(path.current_grid());
        }

        Addr x = 1_r / 2_r;
        Dist D = 1_r; // 2 * 0.5
        PowerModulus<Rational> modulus(1_r, 1_r);
        Rational tolerance = Rational(1, 1000000000000);

        bool diff = check_differentiability(grids, x, func, D, modulus, 1, tolerance);
        EXPECT_TRUE(diff);
    }

    /**
     * @test Quadratic function f(x)=x² at x=1/4.
     *       The derivative is 1/2. The point first appears at some level;
     *       we check from that level onward, using the linear modulus.
     */
    TEST_F(DifferentiabilityTest, QuadraticAtQuarter) {
        ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
        auto path = make_midpoint_path(grid0);
        auto func = [](const Addr& x) { return x * x; };

        using GridType = ListGrid<Addr, Compare>;
        std::vector<GridType> grids;
        grids.push_back(path.current_grid());
        for (int i = 0; i < 6; ++i) {
            path.advance(func);
            grids.push_back(path.current_grid());
        }

        Addr x = 1_r / 4_r;
        Dist D = 1_r / 2_r; // 2 * 0.25
        PowerModulus<Rational> modulus(1_r, 1_r);
        Rational tolerance = Rational(1, 1000000000000);

        std::size_t first_level = 0;
        for (; first_level < grids.size(); ++first_level) {
            if (find_address_index(grids[first_level], x) >= 0) break;
        }
        ASSERT_LT(first_level, grids.size());

        bool diff = check_differentiability(grids, x, func, D, modulus, first_level, tolerance);
        EXPECT_TRUE(diff);
    }

    /**
     * @test Absolute value function f(x)=|x| at x=0.
     *       It is not differentiable at zero, so the test should fail
     *       even with a linear modulus.
     */
    TEST_F(DifferentiabilityTest, AbsoluteValueNotDifferentiableAtZero) {
        ListGrid<Addr, Compare> grid0({ -1_r, 0_r, 1_r });
        auto path = make_midpoint_path(grid0);
        auto func = [](const Addr& x) { return x >= 0_r ? x : -x; };

        using GridType = ListGrid<Addr, Compare>;
        std::vector<GridType> grids;
        grids.push_back(path.current_grid());
        for (int i = 0; i < 5; ++i) {
            path.advance(func);
            grids.push_back(path.current_grid());
        }

        Addr x = 0_r;
        Dist D = 0_r;
        PowerModulus<Rational> modulus(1_r, 1_r);
        Rational tolerance = Rational(1, 1000000000000);

        bool diff = check_differentiability(grids, x, func, D, modulus, 0, tolerance);
        EXPECT_FALSE(diff);
    }

} // namespace delta::testing