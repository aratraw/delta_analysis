// tests/calculus/test_continuity.cpp
#include <gtest/gtest.h>
#include "test_fixtures.h"

namespace delta::testing {

    /**
     * @class ContinuityTest
     * @brief Tests for continuity checks using modulus of continuity.
     *
     * Verifies that the function check_continuity_level correctly determines
     * whether a given function satisfies a prescribed modulus of continuity
     * on a sequence of refined grids.
     */
    class ContinuityTest : public DeltaTest {};

    /**
     * @test Identity function f(x)=x on a dyadic path.
     *       The modulus ω(δ)=δ (C=1, α=1) should be satisfied exactly.
     */
    TEST_F(ContinuityTest, IdentityFunctionOnDyadicPath) {
        ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
        auto path = make_midpoint_path(grid0);
        auto func = [](const Addr& x) { return x; };
        EuclideanValueMetric vm;

        PowerModulus<Rational> modulus(1_r, 1_r);

        for (std::size_t n = 0; n <= 5; ++n) {
            const auto& grid = path.current_grid();
            bool ok = check_continuity_level(grid, func, vm, modulus, Rational(1, 1000000000000));
            EXPECT_TRUE(ok) << "Failed at level " << n;
            if (n < 5) path.advance(func);
        }
    }

    /**
     * @test Constant function f(x)=5. Any modulus with C=0 works,
     *       so the test should always pass.
     */
    TEST_F(ContinuityTest, ConstantFunction) {
        ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
        auto path = make_midpoint_path(grid0);
        auto func = [](const Addr&) { return 5_r; };
        EuclideanValueMetric vm;

        PowerModulus<Rational> modulus(0_r, 1_r);

        for (std::size_t n = 0; n <= 5; ++n) {
            const auto& grid = path.current_grid();
            bool ok = check_continuity_level(grid, func, vm, modulus, Rational(1, 1000000000000));
            EXPECT_TRUE(ok) << "Failed at level " << n;
            if (n < 5) path.advance(func);
        }
    }

    /**
     * @test Quadratic function f(x)=x² on a dyadic path.
     *       The modulus ω(δ)=2δ (Lipschitz constant 2) should be satisfied.
     */
    TEST_F(ContinuityTest, QuadraticFunction) {
        ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
        auto path = make_midpoint_path(grid0);
        auto func = [](const Addr& x) { return x * x; };
        EuclideanValueMetric vm;

        PowerModulus<Rational> modulus(2_r, 1_r);

        for (std::size_t n = 0; n <= 5; ++n) {
            const auto& grid = path.current_grid();
            bool ok = check_continuity_level(grid, func, vm, modulus, Rational(1, 1000000000000));
            EXPECT_TRUE(ok) << "Failed at level " << n;
            if (n < 5) path.advance(func);
        }
    }

    /**
     * @test Square root function f(x)=√x (Hölder continuous with α=0.5).
     *       The modulus ω(δ)=√δ should be satisfied (within tolerance).
     */
    TEST_F(ContinuityTest, SqrtFunction) {
        ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
        auto path = make_midpoint_path(grid0);
        // Approximate sqrt(x) as a Rational with 1e‑12 accuracy
        auto func = [](const Addr& x) -> Rational {
            double val = std::sqrt(x.to_double());
            return Rational(static_cast<int64_t>(val * 1e12), 1e12);
            };
        EuclideanValueMetric vm;
        // Modulus of continuity for sqrt: ω(δ) = √δ (C=1, α=0.5)
        PowerModulus<Rational> modulus(1_r, Rational(1, 2));

        for (std::size_t n = 0; n <= 5; ++n) {
            const auto& grid = path.current_grid();
            bool ok = check_continuity_level(grid, func, vm, modulus, Rational(1, 100000));
            EXPECT_TRUE(ok) << "Failed at level " << n;
            if (n < 5) path.advance(func);
        }
    }

} // namespace delta::testing