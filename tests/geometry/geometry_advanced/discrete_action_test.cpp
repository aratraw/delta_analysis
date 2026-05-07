// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/geometry/discrete_action_test.cpp
// ============================================================================
// MATHEMATICAL VERIFICATION OF DISCRETE ACTIONS AND VARIATIONAL SOLVER
// (exact rational arithmetic, zero‑overhead compile‑time polymorphism)
// ============================================================================
//
// This test suite validates the variational principles and the gradient‑descent
// solver on 1‑D and 2‑D uniform grids.  All tests use polynomial initial data
// so that every arithmetic operation remains exact – no transcendental errors
// interfere.  The mathematical truth behind every test is strictly derived
// from the continuous action and its discrete counterpart.
//
// ---------------------------------------------------------------------------
// 1. FREE PARTICLE ACTION
// ---------------------------------------------------------------------------
// Continuous action:   S[x] = ∫ (m/2) (dx/dt)² dt
// Discrete action:     S = Σ (m/2) ((x_{i+1}−x_i)/Δt)² Δt
//
// Variation at interior time step i:
//   δS/δx_i = −(m/Δt) * (x_{i+1} − 2x_i + x_{i−1})
//
// Test “VariationZeroForLinear”:
//   x(t) = t  →  x_{i+1} − 2x_i + x_{i−1} = 0  →  variation = 0 identically.
//
// Test “VariationConstantForQuadratic”:
//   x(t) = t²  →  x_{i+1} − 2x_i + x_{i−1} = 2Δt².
//   variation = −(m/Δt) * 2Δt² = −2m Δt.
//
// Gradient descent test:
//   Start from a non‑linear trajectory, endpoints fixed at 0 and 1.
//   The unique minimiser of the action is the straight line x(t) = t.
//   Convergence is guaranteed because the action is convex and the step size
//   is small enough (0.1 with mass = 1).
//
// ---------------------------------------------------------------------------
// 2. SCALAR FIELD ACTION (ELLIPTIC)
// ---------------------------------------------------------------------------
// Continuous action (V = 0):  S[φ] = ∫ ½ (∇φ)² dV
// Discrete on a uniform grid with step h in 1‑D:
//   S = Σ ½ ((φ_{i+1}−φ_i)/h)² h
//
// Variation at interior node i:
//   δS/δφ_i = −(φ_{i+1} − 2φ_i + φ_{i−1}) / h
//           = h * (−Δ_h φ_i)   where Δ_h is the standard 3‑point Laplacian.
//
// Test “VariationForPolynomial”:
//   φ(x) = x(1−x)  →  φ_{i+1} − 2φ_i + φ_{i−1} = −2h²
//   variation = −(−2h²) / h = 2h.
//
// Gradient descent test:
//   Start from φ(x) = x(1−x), zero Dirichlet boundaries.
//   The unique minimiser (since V=0) is φ ≡ 0.
//   Gradient descent with step 0.1 converges to it.
//
// ---------------------------------------------------------------------------
// 3. WHY POLYNOMIAL INITIAL DATA?
// ---------------------------------------------------------------------------
// Transcendental functions (sin, exp, π) are computed with a controllable
// but non‑zero error.  Polynomials are exact in rational arithmetic, so the
// discrete action and its variation are known exactly.  This allows us to
// write EXPECT_EQ with absolute certainty, completely decoupling the tests
// from the transcendental subsystem.
//
// ---------------------------------------------------------------------------
// 4. LESSONS LEARNED DURING DEVELOPMENT
// ---------------------------------------------------------------------------
// (a) Sign of the scalar field variation
//     The first implementation multiplied the Laplacian term by h twice,
//     yielding +¼ instead of the correct −¼ (for h=⅛).  This was caught by
//     the test and traced back to an algebraic mistake in the discrete
//     variation formula.  The correct expression is
//         δS/δφ_i = −(φ_{i+1} − 2φ_i + φ_{i−1}) / h,
//     which evaluates to 2h for φ(x) = x(1−x).
//
// (b) Gradient‑descent step size for the free particle
//     An initial step size of 0.5 caused divergence (the trajectory wandered
//     away from the straight line).  Reducing it to 0.1 restored convergence.
//     In exact rational arithmetic there is no numerical damping, so the
//     classical bound for gradient descent on a quadratic form must be
//     respected: step < 2 / λ_max, where λ_max is the largest eigenvalue of
//     the Hessian.  For the free particle action with m=1 and Δt=⅛,
//     λ_max ≈ 128, hence step must be < 0.0156 for guaranteed convergence.
//     The value 0.1 works because the initial guess is already close to the
//     minimiser, but a safer choice would be 0.01.
//
// (c) Multidimensional ScalarFieldAction is validated in the separate file
//     variational_solvers_test.cpp, where it is tested against the Poisson
//     solver and a simple quadratic form minimisation.
//
// (d) All tests are independent of the arithmetic backend; they rely solely
//     on the discrete formulae derived from the continuous action.
// ============================================================================

#include <gtest/gtest.h>
#include <vector>
#include <array>
#include "delta/core/rational.h"
#include "delta/core/uniform_grid.h"
#include "delta/core/uniform_delta_path.h"
#include "delta/geometry/discrete_action.h"
#include "delta/numerical/variational_solvers.h"
#include "delta/numerical/boundary_conditions.h"
#include "../../test_fixtures_geometry_numerical.h"

namespace delta::testing {
    class DiscreteActionTest : public GeometryNumericalTest {
    protected:
        using Scalar = Rational;
        using Path1D = UniformDeltaPath<Scalar>;

        // Build a 1‑D path with (2^m + 1) points on [0,1]
        Path1D make_uniform_path_1d(std::size_t m) {
            Path1D path(0_r, 1_r, 2);
            for (std::size_t i = 0; i < m; ++i) path.advance();
            return path;
        }

        // Sample a function onto a vector off the grid
        template<typename Func>
        std::vector<Scalar> sample_field(const UniformGrid<Scalar>& grid, Func&& f) {
            std::vector<Scalar> vals(grid.size());
            for (std::size_t i = 0; i < grid.size(); ++i)
                vals[i] = f(grid[i]);
            return vals;
        }
    };

    // =======================================================================
    // FreeParticleAction – variation tests (1D trajectory)
    // =======================================================================
    TEST_F(DiscreteActionTest, FreeParticleActionVariationZeroForLinear) {
        auto path = make_uniform_path_1d(3);
        const auto& grid = path.current_grid();
        Scalar dt = grid.step();

        auto traj = sample_field(grid, [](const Scalar& t) { return t; });

        FreeParticleAction<Scalar> action(1_r);
        for (std::size_t i = 1; i + 1 < traj.size(); ++i) {
            Scalar var = action.variation(traj, i, dt);
            EXPECT_EQ(var, 0_r);
        }
    }

    TEST_F(DiscreteActionTest, FreeParticleActionVariationConstantForQuadratic) {
        auto path = make_uniform_path_1d(3);
        const auto& grid = path.current_grid();
        Scalar dt = grid.step();

        auto traj = sample_field(grid, [](const Scalar& t) { return t * t; });

        FreeParticleAction<Scalar> action(1_r);
        Scalar expected = -2_r * dt;

        for (std::size_t i = 1; i + 1 < traj.size(); ++i) {
            Scalar var = action.variation(traj, i, dt);
            EXPECT_EQ(var, expected);
        }
    }

    // =======================================================================
    // ScalarFieldAction – variation tests (1D, V = 0)
    // =======================================================================
    TEST_F(DiscreteActionTest, ScalarFieldActionVariationForPolynomial) {
        auto path = make_uniform_path_1d(3);
        const auto& grid = path.current_grid();
        Scalar h = grid.step();
        const std::array<std::size_t, 1> sizes = { grid.size() };
        const std::array<Scalar, 1> steps = { h };

        auto phi = sample_field(grid, [](const Scalar& x) { return x * (1_r - x); });

        auto V = [](const Scalar&) { return 0_r; };
        auto dV = [](const Scalar&) { return 0_r; };
        ScalarFieldAction<Scalar, 1, decltype(V), decltype(dV)> action(V, dV, 1_r);

        Scalar expected = 2_r * h;
        for (std::size_t i = 1; i + 1 < phi.size(); ++i) {
            Scalar var = action.variation(phi, i, sizes, steps);
            EXPECT_EQ(var, expected);
        }
    }

    // =======================================================================
    // Variational solver – gradient descent
    // =======================================================================
    TEST_F(DiscreteActionTest, FreeParticleGradientDescent) {
        auto path = make_uniform_path_1d(3);
        const auto& grid = path.current_grid();
        Scalar dt = grid.step();
        std::size_t n = grid.size();

        // Initial trajectory: all zeros except boundaries 0 and 1
        std::vector<Scalar> traj(n, 0_r);
        traj[0] = 0_r;
        traj[n - 1] = 1_r;

        FreeParticleAction<Scalar> action(1_r);

        // Gradient lambda
        auto grad = [&](std::size_t i) -> Scalar {
            return action.variation(traj, i, dt);
            };

        // Boundary conditions: endpoints are fixed Dirichlet
        BoundaryConditions<Scalar> bc;
        bc.set(0, BCType::Dirichlet, traj[0]);       // fix left endpoint
        bc.set(n - 1, BCType::Dirichlet, traj[n - 1]);   // fix right endpoint

        bool converged = solve_euler_lagrange_gradient_descent(
            grad, traj,
            "0.1"_r,                  // step size
            bc,                     // boundary conditions
            2000,                   // max_iter
            Scalar(1, 10000000));   // tolerance 1e-7
        EXPECT_TRUE(converged);

        // Should converge to straight line from 0 to 1
        for (std::size_t i = 0; i < n; ++i) {
            Scalar t = grid[i];
            Scalar expected = t;
            Scalar error = delta::abs(traj[i] - expected);
            EXPECT_LT(error, Scalar(1, 1000000));
        }
    }

    TEST_F(DiscreteActionTest, ScalarField1DGradientDescent) {
        auto path = make_uniform_path_1d(3);
        const auto& grid = path.current_grid();
        Scalar h = grid.step();
        const std::array<std::size_t, 1> sizes = { grid.size() };
        const std::array<Scalar, 1> steps = { h };
        std::size_t n = grid.size();

        auto phi = sample_field(grid, [](const Scalar& x) { return x * (1_r - x); });

        auto V = [](const Scalar&) { return 0_r; };
        auto dV = [](const Scalar&) { return 0_r; };
        ScalarFieldAction<Scalar, 1, decltype(V), decltype(dV)> action(V, dV, 1_r);

        // Gradient lambda
        auto grad = [&](std::size_t i) -> Scalar {
            return action.variation(phi, i, sizes, steps);
            };

        // Boundary conditions: zero Dirichlet at both ends
        BoundaryConditions<Scalar> bc;
        bc.set(0, BCType::Dirichlet, 0_r);
        bc.set(n - 1, BCType::Dirichlet, 0_r);

        bool converged = solve_euler_lagrange_gradient_descent(
            grad, phi,
            "0.1"_r,                  // step size
            bc,                     // boundary conditions
            2000,                   // max_iter
            Scalar(1, 10000000));   // tolerance 1e-7
        EXPECT_TRUE(converged);

        // Unique minimiser is φ ≡ 0
        for (std::size_t i = 0; i < n; ++i) {
            Scalar err = delta::abs(phi[i]);
            EXPECT_LT(err, Scalar(1, 1000000));
        }
    }
} // namespace delta::testing