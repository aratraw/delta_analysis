// tests/geometry/test_discrete_action.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "delta/geometry/discrete_action.h"
#include "delta/core/uniform_grid.h"
#include "delta/core/delta_path.h"
#include "delta/core/delta_operator.h"
#include "delta/core/operational_function.h"
#include "../test_fixtures.h"

namespace delta::testing {

    // -----------------------------------------------------------------------------
    // Test fixture for discrete action tests (1D uniform grid)
    // -----------------------------------------------------------------------------
    class DiscreteActionTest : public DeltaTest {
    protected:
        using Scalar = double;
        using Addr = Scalar;
        using Grid = UniformGrid<Addr>;
        using Compare = std::less<Addr>;
        using Between = LessBetweenness;
        using AddrMetric = EuclideanMetric;
        using ValMetric = EuclideanValueMetric;
        using Strategy = StaticStrategy<MidpointOperator>;
        using Path = DeltaPath<Addr, Scalar, Scalar, Between, AddrMetric, ValMetric, Strategy, Compare>;
        using Field = OperationalFunction<Addr, Scalar, Grid>;

        Grid grid{ 0.0, 0.1, 11 }; // [0,1] with 11 points
        Strategy strategy{ MidpointOperator{} };
        Path path{ grid, strategy, Between{}, AddrMetric{}, ValMetric{} };

        // Helper to create a field from a function
        Field make_field(std::function<Scalar(Addr)> func) {
            return Field(grid, func);
        }

        // Linear trajectory x(t) = t
        Field linear_field() {
            return make_field([](Addr t) { return t; });
        }

        // Constant field
        Field constant_field(Scalar c) {
            return make_field([c](Addr) { return c; });
        }
    };

    // -----------------------------------------------------------------------------
    // FreeParticleAction tests
    // -----------------------------------------------------------------------------

    TEST_F(DiscreteActionTest, FreeParticleActionValue) {
        FreeParticleAction<Field, Path> action(1.0); // mass = 1

        // For linear trajectory x(t)=t on [0,1], action = ∫₀¹ ½ (dx/dt)² dt = 0.5
        Field x = linear_field();
        Scalar S = action.evaluate(x, path);

        // With 11 points, approximation should be close to 0.5
        EXPECT_NEAR(S, 0.5, 1e-2);

        // Refine grid and check convergence
        UniformGrid<Addr> refined_grid(0.0, 0.05, 21);
        Path refined_path(refined_grid, strategy, Between{}, AddrMetric{}, ValMetric{});
        Field x_refined = make_field([](Addr t) { return t; });
        S = action.evaluate(x_refined, refined_path);
        EXPECT_NEAR(S, 0.5, 5e-3); // should be more accurate
    }

    TEST_F(DiscreteActionTest, FreeParticleVariation) {
        FreeParticleAction<Field, Path> action(1.0);
        Field x = linear_field();

        // For a linear trajectory, the Euler-Lagrange equation is d²x/dt² = 0,
        // which is satisfied exactly by x(t)=t. The discrete variation at interior
        // points should be zero (up to discretization error).
        Addr interior = 0.5;
        Scalar var = action.variation(x, path, interior);
        EXPECT_NEAR(var, 0.0, 1e-12);

        // At boundary points, variation is defined as zero (Dirichlet conditions)
        Scalar var_left = action.variation(x, path, grid[0]);
        EXPECT_NEAR(var_left, 0.0, 1e-12);

        Scalar var_right = action.variation(x, path, grid[grid.size() - 1]);
        EXPECT_NEAR(var_right, 0.0, 1e-12);

        // For a non-solution field (e.g., constant), variation should be non-zero
        Field c = constant_field(0.5);
        var = action.variation(c, path, interior);
        EXPECT_NE(std::abs(var), 0.0);
    }

    // -----------------------------------------------------------------------------
    // ScalarFieldAction tests
    // -----------------------------------------------------------------------------

    TEST_F(DiscreteActionTest, ScalarFieldActionZeroPotential) {
        // Action: S = ∫ [½ (∂φ)² + V(φ)] dx, with V=0
        auto V = [](Scalar) { return 0.0; };
        auto dV = [](Scalar) { return 0.0; };
        ScalarFieldAction<Field, Path> action(V, dV, 1.0); // stiffness = 1

        Field phi = linear_field(); // φ(x)=x
        Scalar S = action.evaluate(phi, path);

        // Expected: ∫₀¹ ½ (1)² dx = 0.5
        EXPECT_NEAR(S, 0.5, 1e-2);

        // Variation for a linear function should be small (since φ''=0)
        Addr interior = 0.5;
        Scalar var = action.variation(phi, path, interior);
        EXPECT_NEAR(var, 0.0, 1e-10);
    }

    TEST_F(DiscreteActionTest, ScalarFieldActionQuadraticPotential) {
        // V(φ) = ½ φ², so dV = φ
        auto V = [](Scalar phi) { return 0.5 * phi * phi; };
        auto dV = [](Scalar phi) { return phi; };
        ScalarFieldAction<Field, Path> action(V, dV, 1.0);

        Field phi = linear_field(); // φ(x)=x

        // Compute action numerically
        Scalar S = action.evaluate(phi, path);

        // Analytical: ∫₀¹ (½ (1)² + ½ x²) dx = ½ ∫₀¹ (1 + x²) dx = ½ [x + x³/3]₀¹ = ½ (1 + 1/3) = ½ * 4/3 = 2/3 ≈ 0.6667
        EXPECT_NEAR(S, 2.0 / 3.0, 1e-2);

        // Variation at interior point should not be zero (since φ not a solution of -φ'' + φ = 0)
        Addr interior = 0.5;
        Scalar var = action.variation(phi, path, interior);
        EXPECT_NE(std::abs(var), 0.0);

        // For a constant field φ=0, variation should be zero (trivial solution)
        Field zero = constant_field(0.0);
        var = action.variation(zero, path, interior);
        EXPECT_NEAR(var, 0.0, 1e-12);
    }

} // namespace delta::testing