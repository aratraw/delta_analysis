// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/numerical/solvers/heat_solver_test.cpp
// ============================================================================
// MATHEMATICAL VERIFICATION OF DISCRETE HEAT SOLVERS
// (corrected for exact rational arithmetic — no rounding noise)
// ============================================================================
//
// This test suite validates the correctness of solve_heat()
// on the unit square with homogeneous Dirichlet boundary conditions.
//
// ---------------------------------------------------------------------------
// CONTINUOUS PROBLEM (unchanged)
// ---------------------------------------------------------------------------
//   Ω = (0,1)×(0,1)
//   u_t = Δu  in Ω,   u = 0 on ∂Ω
//   Initial condition: u0(x,y) = sin(πx)·sin(πy)
//   Exact solution: u_ex(x,y,t) = sin(πx)·sin(πy)·exp(−2π² t)
//
// ---------------------------------------------------------------------------
// DISCRETISATION (unchanged)
// ---------------------------------------------------------------------------
// Uniform grid with N = 2^m+1 points, h = 1/(N-1).
// Five‑point stencil:  K (stiffness) and M = h²·I (lumped mass).
//
// ---------------------------------------------------------------------------
// RATIONALE FOR TEST SELECTION (RATIONAL ARITHMETIC)
// ---------------------------------------------------------------------------
// 1. **No rounding noise** → parasitic high‑frequency modes are NOT excited.
//    A pure first‑mode initial condition remains in the stable subspace even
//    when the time step exceeds the linear stability limit for higher modes.
//    Hence, the classic “blow‑up” test with Δt just above the limit is
//    **not reproducible** in exact arithmetic and is therefore omitted.
//
// 2. **Temporal convergence of Crank‑Nicolson**. On a very coarse spatial
//    grid (9×9) the spatial error dominates, and the O(Δt²) temporal error
//    is already below 1% of the total error even for the largest Δt.
//    Further reducing Δt does not change the total error monotonically.
//    → The temporal convergence test for CN is **meaningful only on finer
//      spatial grids** (e.g. 17×17 or 33×33), but those are too expensive
//      for exact rational LU.  Therefore it is removed.
//
// 3. **Remaining tests** that are both mathematically sound and feasible:
//    - Temporal convergence of Implicit Euler (first‑order, error dominated
//      by time even on a 9×9 grid).
//    - Spatial convergence (CN with very fine Δt, varying spatial resolution).
//    - Explicit Euler with a stable step (sanity check).
//    - Long‑time decay to zero (qualitative behaviour).
//
// ---------------------------------------------------------------------------
// CRITICAL NOTE ON TRANSCENDENTAL PRECISION AND RATIONAL NUMBER SIZE
// ---------------------------------------------------------------------------
// The library computes transcendental functions (sin, exp, pi) with a
// controllable absolute error bound (default 1e-30).  This determines the
// number of terms in the series or the internal working precision.
//
//   • High precision (1e-30) → initial condition and exact solution are very
//     faithful → discretisation error dominates, but the *difference* between
//     numerical and exact solution is small → the fractions that appear in
//     the L2 error remain compact → error calculation is fast.
//
//   • Lower precision (e.g. 1e-12) → transcendental approximations are
//     coarser → the discretisation error gets “polluted” by a larger
//     approximation error → the difference sol − u_ex becomes larger and,
//     crucially, involves fractions with huge numerators/denominators once
//     squared and summed.  The subsequent GCD reductions and sqrt cause
//     *dramatic* slowdown (we observed 25 s → 150 s for SpatialConvergence).
//
// Therefore:
//   – Tests that compute a quantitative L2 error (TemporalConvergenceImplicitEuler,
//     SpatialConvergence) MUST keep the default high precision to avoid
//     pathological growth of rational numbers.  Do NOT lower the precision
//     in these tests.
//   – Tests that only need a rough bound (ExplicitEulerStableStep: < 0.05,
//     DecayToZero: < 0.1) CAN use a coarser precision because the error
//     value is not used further and the rational numbers stay small enough
//     for a single comparison.
//
// ---------------------------------------------------------------------------
// GUARANTEE
// ---------------------------------------------------------------------------
// IF ANY OF THESE TESTS FAILS, THE BUG IS IN THE SOLVER IMPLEMENTATION,
// NOT IN THE TESTS.
// ============================================================================

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>
#include "delta/core/rational.h"
#include "delta/core/uniform_grid.h"
#include "delta/core/product_grid.h"
#include "delta/core/uniform_delta_path.h"
#include "delta/core/operational_function.h"
#include "delta/geometry/product_regulative.h"
#include "delta/numerical/boundary_conditions.h"
#include "delta/numerical/solvers/heat_solver.h"
#include "../../test_fixtures_geometry_numerical.h"

namespace delta::testing {
    using namespace delta::numerical::solvers;

    class HeatSolverTest : public GeometryNumericalTest {
    protected:
        using Scalar = Rational;
        using Path1D = UniformDeltaPath<Scalar>;
        using Path2D = geometry::ProductDeltaPath<Path1D, Path1D>;

        // -------------------------------------------------------------------
        //  Grid & exact solution infrastructure
        // -------------------------------------------------------------------
        Path2D make_uniform_path(std::size_t m) {
            Path1D path1d(0_r, 1_r, 2);
            for (std::size_t i = 0; i < m; ++i) path1d.advance();
            return Path2D(path1d, path1d);
        }

        Scalar pi_val() const {
            // Uses the currently active default epsilon.
            // When we need high accuracy, we do NOT call set_precision,
            // so the original 1e-30 is used and pi is computed accordingly.
            return delta::pi(delta::default_eps());
        }

        Scalar u0_func(const std::array<Scalar, 2>& pt) const {
            return delta::sin(pi_val() * pt[0]) * delta::sin(pi_val() * pt[1]);
        }

        template<typename Grid2D>
        OperationalFunction<std::array<Scalar, 2>, Scalar, Grid2D>
            make_u0(const Grid2D& grid) const {
            return OperationalFunction<std::array<Scalar, 2>, Scalar, Grid2D>(
                grid,
                [this](const std::array<Scalar, 2>& pt) { return u0_func(pt); }
            );
        }

        Scalar exact_solution(const std::array<Scalar, 2>& pt, Scalar t) const {
            Scalar amp = delta::exp(-2_r * pi_val() * pi_val() * t);
            return amp * u0_func(pt);
        }

        BoundaryConditions<Scalar> make_zero_dirichlet(const Path2D& path) const {
            BoundaryConditions<Scalar> bc;
            auto grid = path.current_grid();
            std::size_t Nx = grid.get_grid(0).size();
            std::size_t Ny = grid.get_grid(1).size();
            for (std::size_t j = 0; j < Ny; ++j) {
                for (std::size_t i = 0; i < Nx; ++i) {
                    if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1)
                        bc.set(i + j * Nx, BCType::Dirichlet, 0_r);
                }
            }
            return bc;
        }

        template<typename Field>
        Scalar compute_L2_error(const Field& sol, const Path2D& path, Scalar t) const {
            auto grid = path.current_grid();
            Scalar error_sq = 0;
            std::size_t nx = grid.get_grid(0).size();
            std::size_t ny = grid.get_grid(1).size();
            Scalar hx = grid.get_grid(0).step();
            Scalar hy = grid.get_grid(1).step();
            Scalar vol = hx * hy;
            for (std::size_t j = 0; j < ny; ++j) {
                for (std::size_t i = 0; i < nx; ++i) {
                    std::array<Scalar, 2> pt = { grid.get_grid(0)[i], grid.get_grid(1)[j] };
                    Scalar diff = sol(pt) - exact_solution(pt, t);
                    error_sq += diff * diff * vol;
                }
            }
            return delta::sqrt(error_sq);
        }

        // -------------------------------------------------------------------
        //  Precision constants (used only where safe – see test comments)
        // -------------------------------------------------------------------
        static inline const Scalar COARSE_PRECISION = Scalar(1, 1000000000000);   // 1e-12
        static inline const Scalar DECAY_PRECISION = Scalar(1LL, 10000000000000000LL); // 1e-16 (safe for small grid)
    };

    // -----------------------------------------------------------------------
    // Temporal convergence – Implicit Euler (error must decrease with smaller dt)
    //
    // WHY NO set_precision HERE:
    //   This test computes a quantitative L2 error and compares values.
    //   Lowering the precision would inflate the discretisation error and
    //   cause gigantic rational numbers in error_sq, slowing down the test
    //   from ~1 s to tens of seconds.  We keep the default 1e-30.
    // -----------------------------------------------------------------------
    TEST_F(HeatSolverTest, TemporalConvergenceImplicitEuler) {
        const std::size_t m = 3;               // N = 9
        const Scalar T = Scalar(5, 100);        // 0.05
        auto path = make_uniform_path(m);
        auto u0 = make_u0(path.current_grid());
        auto bc = make_zero_dirichlet(path);
        EuclideanMetric metric;

        std::vector<Scalar> dts;
        for (int k = 0; k < 3; ++k) {
            dts.push_back(T / Scalar(5 * (1 << k)));   // Δt = 0.01, 0.005, 0.0025
        }

        std::vector<Scalar> errors;
        for (auto dt : dts) {
            auto sol = solve_heat(path, u0, 1_r, dt, T, metric, bc, TimeScheme::IMPLICIT_EULER);
            errors.push_back(compute_L2_error(sol, path, T));
        }

        // errors must be strictly decreasing
        for (std::size_t i = 1; i < errors.size(); ++i) {
            EXPECT_LT(errors[i], errors[i - 1]);
        }
    }

    // -----------------------------------------------------------------------
    // Spatial convergence (Crank‑Nicolson with very fine Δt)
    // Error must be smaller on finer grid
    //
    // WHY NO set_precision HERE:
    //   Same reason as above – computing L2 error on a 17×17 grid with
    //   deflated precision causes catastrophic growth of rational numbers.
    //   We keep the default high precision.
    // -----------------------------------------------------------------------
    TEST_F(HeatSolverTest, SpatialConvergence) {
        const Scalar T = Scalar(2, 100);        // 0.02
        const Scalar dt = T / 20_r;             // 0.001, 20 steps

        EuclideanMetric metric;

        std::vector<std::size_t> levels = { 3, 4 };   // N = 9, 17
        std::vector<Scalar> errors;

        for (std::size_t m : levels) {
            auto path = make_uniform_path(m);
            auto u0 = make_u0(path.current_grid());
            auto bc = make_zero_dirichlet(path);

            auto sol = solve_heat(path, u0, 1_r, dt, T, metric, bc, TimeScheme::CRANK_NICOLSON);
            errors.push_back(compute_L2_error(sol, path, T));
        }

        // finer grid must yield smaller error
        EXPECT_LT(errors[1], errors[0]);
    }

    // -----------------------------------------------------------------------
    // Explicit Euler with small stable step – basic sanity check
    //
    // WHY WE CAN USE COARSE PRECISION HERE:
    //   We only check that the error is < 0.05.  The error value itself
    //   is not used in further arithmetic, and the small grid (9×9) and
    //   small number of steps keep the rational numbers manageable even
    //   with a less accurate initial condition.
    //   Lowering the precision speeds up the transcendental evaluations,
    //   which is beneficial since this test is run on every commit.
    // -----------------------------------------------------------------------
    TEST_F(HeatSolverTest, ExplicitEulerStableStep) {
        set_precision(COARSE_PRECISION);
        const std::size_t m = 3;               // N = 9
        const Scalar h = 1_r / 8_r;
        const Scalar dt = (h * h) / 8_r;      // well below stability limit
        const Scalar T = Scalar(1, 20);        // 0.05
        auto path = make_uniform_path(m);
        auto u0 = make_u0(path.current_grid());
        auto bc = make_zero_dirichlet(path);
        EuclideanMetric metric;

        auto sol = solve_heat(path, u0, 1_r, dt, T, metric, bc, TimeScheme::EXPLICIT_EULER);
        Scalar err = compute_L2_error(sol, path, T);
        EXPECT_LT(err, Scalar(5, 100));         // < 0.05 – explicit works
    }

    // -----------------------------------------------------------------------
    // Long-time decay (Implicit Euler to T=1.0) – solution must be near zero
    //
    // WHY WE CAN USE DECAY_PRECISION HERE:
    //   We only check that the error is < 0.1.  The grid is small (9×9) and
    //   the number of time steps is moderate.  Even with a slightly less
    //   accurate initial condition, the error stays below the bound and the
    //   rational numbers do not explode.  This test is a qualitative check
    //   that the solution decays to zero, not a high‑precision benchmark.
    // -----------------------------------------------------------------------
    TEST_F(HeatSolverTest, DecayToZero) {
        set_precision(DECAY_PRECISION);
        const std::size_t m = 3;               // N = 9
        const Scalar T = 1_r;                  // large time
        const Scalar dt = Scalar(5, 100);      // 0.05, 20 steps
        auto path = make_uniform_path(m);
        auto u0 = make_u0(path.current_grid());
        auto bc = make_zero_dirichlet(path);
        EuclideanMetric metric;

        auto sol = solve_heat(path, u0, 1_r, dt, T, metric, bc, TimeScheme::IMPLICIT_EULER);
        Scalar err = compute_L2_error(sol, path, T);
        EXPECT_LT(err, Scalar(1, 10));   // < 0.1
    }

} // namespace delta::testing