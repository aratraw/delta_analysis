// tests/numerical/test_variational_solvers.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "delta/numerical/variational_solvers.h"
#include "delta/geometry/discrete_action.h"
#include "delta/core/uniform_grid.h"
#include "delta/core/delta_path.h"
#include "delta/core/delta_operator.h"
#include "delta/core/operational_function.h"
#include "../test_fixtures.h"

namespace delta::testing {

    using namespace delta::numerical;
    using namespace delta::geometry;
    using namespace delta::core;

    // -----------------------------------------------------------------------------
    // Test fixture for variational solvers using 1D uniform grid and free particle action
    // -----------------------------------------------------------------------------
    class VariationalSolverTest : public DeltaTest {
    protected:
        using Scalar = double;
        using Point = Scalar;
        using Grid = UniformGrid<Point>;
        using Compare = std::less<Point>;
        using Between = LessBetweenness;
        using AddrMetric = EuclideanMetric;
        using ValMetric = EuclideanValueMetric;
        using Strategy = StaticStrategy<MidpointOperator>;
        using Path = DeltaPath<Point, Scalar, Scalar, Between, AddrMetric, ValMetric, Strategy, Compare>;
        using Field = OperationalFunction<Point, Scalar, Grid>;
        using Action = FreeParticleAction<Field, Path>;

        Grid grid{ 0.0, 0.1, 11 }; // [0,1] with 11 points
        Strategy strategy{ MidpointOperator{} };
        Path path{ grid, strategy, Between{}, AddrMetric{}, ValMetric{} };

        // Helper to create initial field (constant or linear)
        Field make_constant_field(Scalar c) {
            return Field(grid, [c](Point) { return c; });
        }

        Field make_linear_field(Scalar a = 1.0, Scalar b = 0.0) {
            return Field(grid, [a, b](Point x) { return a * x + b; });
        }

        // Compute gradient norm squared (sum of squares of variations)
        Scalar gradient_norm_sq(const Action& action, const Field& field, const Path& path) {
            Scalar sum = 0.0;
            for (const auto& addr : path.current_grid()) {
                Scalar g = action.variation(field, path, addr);
                sum += g * g;
            }
            return sum;
        }
    };

    // -----------------------------------------------------------------------------
    // Test gradient descent for free particle action
    // -----------------------------------------------------------------------------
    TEST_F(VariationalSolverTest, GradientDescentConvergesToLinear) {
        // Action: free particle with mass 1
        Action action(1.0);

        // Initial guess: constant field u(x) = 0.5 (not a minimizer)
        Field u = make_constant_field(0.5);

        // Exact minimizer for free particle action with fixed endpoints u(0)=0, u(1)=1
        // But our action does not fix endpoints; it's defined on all points with natural boundary conditions?
        // Actually FreeParticleAction variation formula returns zero for boundary points (Dirichlet).
        // The minimizer with fixed endpoints is linear, but our action treats boundaries as Dirichlet (variation=0).
        // So we should set boundary conditions manually? In solve_euler_lagrange_gradient_descent, we don't impose BCs.
        // The function updates all points including boundaries. That means boundaries will also change.
        // For a free particle with natural boundaries (Neumann), the minimizer is constant? Wait, Euler-Lagrange u''=0 with natural BCs u'=0 gives constant. So constant is minimizer. Our test constant initial should already be minimizer? But we need to check.

        // Let's modify: we want to test convergence to linear interpolation with fixed endpoints.
        // We can fix endpoints by not updating them in the solver, or we can use the fact that variation at boundaries returns 0,
        // so they won't change. In FreeParticleAction, variation at boundaries returns 0 (as implemented in discrete_action.h).
        // So boundaries are effectively Dirichlet with values given by initial field. If we set initial field to have u(0)=0, u(1)=1,
        // and linear elsewhere, that's already a minimizer. But if we start with constant, boundaries will stay constant (0.5, 0.5),
        // and interior will adjust to satisfy u''=0 with those boundary values? That yields linear function between 0.5 and 0.5, i.e., constant 0.5. So constant is the minimizer for that boundary condition. So we need to set initial boundaries to 0 and 1.

        // Let's create initial field with u(0)=0, u(1)=1, and constant elsewhere (which is discontinuous). That's not a solution.
        Field u_init(grid, [](Point x) {
            if (x == 0.0) return 0.0;
            if (x == 1.0) return 1.0;
            return 0.5; // arbitrary interior values
            });

        // Before optimization, gradient norm should be non-zero
        Scalar grad_norm0 = gradient_norm_sq(action, u_init, path);
        EXPECT_GT(grad_norm0, 1e-6);

        // Run gradient descent
        bool converged = solve_euler_lagrange_gradient_descent(action, u_init, path, 0.01, 500, 1e-8);

        EXPECT_TRUE(converged);

        // After convergence, gradient norm should be small
        Scalar grad_norm_final = gradient_norm_sq(action, u_init, path);
        EXPECT_LT(grad_norm_final, 1e-6);

        // Check that solution is approximately linear between 0 and 1
        for (Point x : grid) {
            Scalar exact = x; // linear interpolation from 0 to 1
            EXPECT_NEAR(u_init(x), exact, 0.02); // allow some error due to discretization
        }
    }

    // -----------------------------------------------------------------------------
    // Test that gradient descent reduces the action
    // -----------------------------------------------------------------------------
    TEST_F(VariationalSolverTest, GradientDescentDecreasesAction) {
        Action action(1.0);

        // Create a random initial field
        Field u(grid, [](Point x) {
            // Use a simple bump not satisfying Euler-Lagrange
            return x * (1.0 - x) + 0.2; // parabola with value 0.2 at boundaries
            });

        Scalar S0 = action.evaluate(u, path);

        // Run a few iterations
        bool converged = solve_euler_lagrange_gradient_descent(action, u, path, 0.01, 50, 1e-12);
        // We don't expect full convergence, but action should decrease

        Scalar S1 = action.evaluate(u, path);
        EXPECT_LT(S1, S0);
    }

    // -----------------------------------------------------------------------------
    // Test with exact linear solution: gradient should be zero
    // -----------------------------------------------------------------------------
    TEST_F(VariationalSolverTest, ExactSolutionHasZeroGradient) {
        Action action(1.0);

        // Linear solution u(x) = x (satisfies u''=0)
        Field u_exact = make_linear_field(1.0, 0.0);

        Scalar grad_norm = gradient_norm_sq(action, u_exact, path);
        // Should be zero except at boundaries where variation is defined zero anyway
        EXPECT_NEAR(grad_norm, 0.0, 1e-12);
    }

} // namespace delta::testing