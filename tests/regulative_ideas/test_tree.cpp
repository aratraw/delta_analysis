// tests/regulative_ideas/test_tree.cpp
#include <gtest/gtest.h>
#include "../test_fixtures.h"
#include "delta/core/tree_grid.h"
#include "delta/core/delta_path.h"
#include "delta/calculus/riemann_sum.h"

namespace delta::testing {

    /**
     * @class TreePathTest
     * @brief Test suite for the tree‑based regulative idea (binary tree addresses).
     *
     * This fixture tests the behaviour of TreeDeltaPath and the associated tree‑adapted
     * Riemann sum (tree_riemann_sum). It verifies that integrals over the binary tree
     * of characteristic functions and constant functions yield the expected values.
     */
    class TreePathTest : public DeltaTest {};

    /**
     * @test DirichletIntegral
     * @brief Approximate the integral of a function that is 1 on right‑half leaves
     *        and 0 elsewhere, using tree refinement.
     *
     * The function returns 1 for addresses ending with '1', 0 for those ending with '0',
     * and 0 for the root. As the tree refines, the integral should approach 0.5
     * (half the measure). The test checks that after the first level the integral is
     * near 0.5, and that it stabilises (changes slowly) in subsequent levels.
     */
    TEST_F(TreePathTest, DirichletIntegral) {
        TreeDeltaPath<Rational> path;
        auto func = [](const std::string& addr) -> Rational {
            if (addr.empty()) return 0_r;
            return (addr.back() == '0') ? 0_r : 1_r;
            };

        Rational prev = 0_r;
        Rational half = Rational(1, 2);
        Rational tolerance = Rational(1, 10); // 0.1

        for (int level = 0; level <= 5; ++level) {
            auto integral = calculus::tree_riemann_sum(path, func);
            if (level > 0) {
                EXPECT_RATIONAL_NEAR(integral, half, tolerance);
                if (level > 1) {
                    // Change should be small
                    Rational change = integral - prev;
                    if (change < 0) change = -change;
                    EXPECT_LE(change, Rational(2, 10)); // 0.2
                }
            }
            prev = integral;
            path.advance();
        }
    }

    /**
     * @test LevelZeroGrid
     * @brief Verify that a newly constructed TreeDeltaPath contains only the root node.
     */
    TEST_F(TreePathTest, LevelZeroGrid) {
        TreeDeltaPath<Rational> path; // level = 0
        const auto& grid = path.current_grid();
        EXPECT_EQ(grid.size(), 1);
        EXPECT_EQ(grid[0], "");
    }

    /**
     * @test ConstantFunctionIntegral
     * @brief Check that the integral of a constant function on the tree equals that constant
     *        at every refinement level.
     */
    TEST_F(TreePathTest, ConstantFunctionIntegral) {
        TreeDeltaPath<Rational> path;
        auto func = [](const std::string&) { return Rational(5, 2); }; // 2.5

        Rational prev = 0_r;
        for (int level = 0; level <= 5; ++level) {
            Rational integral = calculus::tree_riemann_sum(path, func);
            EXPECT_EQ(integral, Rational(5, 2));
            if (level > 0) {
                EXPECT_EQ(integral, prev);
            }
            prev = integral;
            path.advance();
        }
    }

    /**
     * @test LeftHalfCharacteristic
     * @brief Integral of the characteristic function of the left half of the tree
     *        (addresses ending with '0').
     *
     * The expected value is 0.5. The test checks that after each refinement the
     * computed integral is near 0.5 and that consecutive integrals are close.
     */
    TEST_F(TreePathTest, LeftHalfCharacteristic) {
        TreeDeltaPath<Rational> path;
        auto func = [](const std::string& addr) -> Rational {
            if (addr.empty()) return 0_r;
            return (addr.back() == '0') ? 1_r : 0_r;
            };

        Rational prev = 0_r;
        Rational half = Rational(1, 2);
        Rational tolerance = Rational(1, 10); // 0.1

        for (int level = 1; level <= 5; ++level) {
            path.advance();
            Rational integral = calculus::tree_riemann_sum(path, func);
            EXPECT_RATIONAL_NEAR(integral, half, tolerance);
            if (level > 1) {
                Rational change = integral - prev;
                if (change < 0) change = -change;
                EXPECT_LE(change, Rational(2, 10)); // 0.2
            }
            prev = integral;
        }
    }

    /**
     * @test RightHalfCharacteristic
     * @brief Integral of the characteristic function of the right half of the tree
     *        (addresses ending with '1').
     *
     * Symmetric to LeftHalfCharacteristic; also should converge to 0.5.
     */
    TEST_F(TreePathTest, RightHalfCharacteristic) {
        TreeDeltaPath<Rational> path;
        auto func = [](const std::string& addr) -> Rational {
            if (addr.empty()) return 0_r;
            return (addr.back() == '1') ? 1_r : 0_r;
            };

        Rational prev = 0_r;
        Rational half = Rational(1, 2);
        Rational tolerance = Rational(1, 10); // 0.1

        for (int level = 1; level <= 5; ++level) {
            path.advance();
            Rational integral = calculus::tree_riemann_sum(path, func);
            EXPECT_RATIONAL_NEAR(integral, half, tolerance);
            if (level > 1) {
                Rational change = integral - prev;
                if (change < 0) change = -change;
                EXPECT_LE(change, Rational(2, 10)); // 0.2
            }
            prev = integral;
        }
    }

} // namespace delta::testing