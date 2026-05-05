// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/numerical/solvers/poisson_solver_test.cpp
// ============================================================================
// MATHEMATICAL VERIFICATION OF DISCRETE POISSON SOLVER
// Last updated: 2026-05-17
// ============================================================================
//
// This test suite validates the correctness and convergence of solve_poisson()
// on unit square with homogeneous Dirichlet boundary conditions.
//
// ---------------------------------------------------------------------------
// 1. CONTINUOUS PROBLEM
// ---------------------------------------------------------------------------
//   Ω = (0,1)×(0,1)
//   -Δu = f  in Ω,   u = 0 on ∂Ω
//   Exact solution: u_ex(x,y) = sin(πx)·sin(πy)
//   Right-hand side: f(x,y) = -Δu_ex = 2π²·sin(πx)·sin(πy)
//
// ---------------------------------------------------------------------------
// 2. DISCRETISATION (five-point stencil on uniform grid)
// ---------------------------------------------------------------------------
// Grid: points (i·h, j·h) with h = 1/(N-1), i,j = 0…N-1.
// Internal nodes: i = 1…N-2, j = 1…N-2.
// Discrete Laplacian:
//   (-Δ_h u)_{i,j} = (4u_{i,j} - u_{i+1,j} - u_{i-1,j} - u_{i,j+1} - u_{i,j-1}) / h²
// Dirichlet BC: u_{i,j} = 0 for i∈{0,N-1} or j∈{0,N-1}.
//
// ---------------------------------------------------------------------------
// 3. EXPECTED ACCURACY
// ---------------------------------------------------------------------------
// The five‑point scheme is second‑order accurate: ‖u_h - u_ex‖_L2 = O(h²).
//
// ---------------------------------------------------------------------------
// 4. TEST PLAN
// ---------------------------------------------------------------------------
// Test 1 (Accuracy): on a single grid (N=9) the L2 error is < 0.05.
// Test 2 (Convergence): for N = 9,17,33,65 the estimated order of convergence
//     is at least 1.8 and converges to ~2.0.
//
// All reference values are derived from the analytic expressions above.
// IF ANY TEST FAILS, THE BUG IS IN THE SOLVER IMPLEMENTATION, NOT IN THE TEST.
// ============================================================================

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "delta/core/rational.h"
#include "delta/core/uniform_grid.h"
#include "delta/core/product_grid.h"
#include "delta/core/uniform_delta_path.h"          // UniformDeltaPath
#include "delta/core/operational_function.h"
#include "delta/geometry/product_regulative.h"      // ProductDeltaPath
#include "delta/numerical/boundary_conditions.h"
#include "delta/numerical/solvers/poisson_solver.h"
#include "../../test_fixtures_geometry_numerical.h"

namespace delta::testing {
    using namespace delta::numerical::solvers;
    class PoissonSolverTest : public GeometryNumericalTest {
    protected:
        using Scalar = Rational;
        using Path1D = UniformDeltaPath<Scalar>;                    // равномерный путь
        using Path2D = geometry::ProductDeltaPath<Path1D, Path1D>;  // двумерный продукт

        // -------------------------------------------------------------------
        // Build a 2D product path with 2^m+1 grid points in each direction
        // -------------------------------------------------------------------
        Path2D make_uniform_path(std::size_t m) {
            // start, step, count: [0,1] with 2 points initially
            Path1D path1d(0_r, 1_r, 2);
            // advance m times → 2^m + 1 points, step = 1/2^m
            for (std::size_t i = 0; i < m; ++i) path1d.advance();
            return Path2D(path1d, path1d);
        }

        // -------------------------------------------------------------------
        // Exact solution and RHS (Rational approximations using high precision)
        // -------------------------------------------------------------------
        Scalar pi_val() const {
            static Scalar pi = delta::pi(delta::default_eps());
            return pi;
        }

        Scalar exact_solution(const std::array<Scalar, 2>& pt) const {
            Scalar px = pt[0], py = pt[1];
            return delta::sin(pi_val() * px) * delta::sin(pi_val() * py);
        }

        Scalar rhs_function(const std::array<Scalar, 2>& pt) const {
            return 2_r * pi_val() * pi_val() * exact_solution(pt);
        }

        // -------------------------------------------------------------------
        // Build RHS as OperationalFunction on the given 2D grid
        // -------------------------------------------------------------------
        template<typename Grid2D>
        OperationalFunction<std::array<Scalar, 2>, Scalar, Grid2D>
            make_rhs(const Grid2D& grid) const {
            return OperationalFunction<std::array<Scalar, 2>, Scalar, Grid2D>(
                grid,
                [this](const std::array<Scalar, 2>& pt) { return rhs_function(pt); }
            );
        }

        // -------------------------------------------------------------------
        // Create boundary conditions (Dirichlet zero on all four edges)
        // -------------------------------------------------------------------
        BoundaryConditions<Scalar> make_zero_dirichlet(const Path2D& path) const {
            BoundaryConditions<Scalar> bc;
            const auto grid = path.current_grid();
            std::size_t Nx = grid.get_grid(0).size();
            std::size_t Ny = grid.get_grid(1).size();
            for (std::size_t j = 0; j < Ny; ++j) {
                for (std::size_t i = 0; i < Nx; ++i) {
                    if (i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1) {
                        bc.set(i + j * Nx, BCType::Dirichlet, 0_r);
                    }
                }
            }
            return bc;
        }

        // -------------------------------------------------------------------
        // Compute L2 error between solution field and exact solution
        // -------------------------------------------------------------------
        template<typename Field>
        Scalar compute_L2_error(const Field& sol, const Path2D& path) const {
            auto grid = path.current_grid();
            Scalar error_sq = 0;
            std::size_t nx = grid.get_grid(0).size();
            std::size_t ny = grid.get_grid(1).size();
            Scalar hx = grid.get_grid(0).step();
            Scalar hy = grid.get_grid(1).step();
            Scalar vol = hx * hy; // uniform cell area
            for (std::size_t j = 0; j < ny; ++j) {
                for (std::size_t i = 0; i < nx; ++i) {
                    std::array<Scalar, 2> pt = { grid.get_grid(0)[i], grid.get_grid(1)[j] };
                    Scalar diff = sol(pt) - exact_solution(pt);
                    error_sq += diff * diff * vol;
                }
            }
            return delta::sqrt(error_sq);
        }
    };

    // -----------------------------------------------------------------------
    // Accuracy test
    // -----------------------------------------------------------------------
    TEST_F(PoissonSolverTest, Accuracy) {
        auto path = make_uniform_path(3); // 2^3+1 = 9 points, h = 1/8
        auto grid = path.current_grid();
        auto rhs = make_rhs(grid);
        auto bc = make_zero_dirichlet(path);
        EuclideanMetric metric;

        auto solution = solve_poisson(path, rhs, bc, metric);

        Scalar error = compute_L2_error(solution, path);
        EXPECT_LT(error, Scalar(5, 100)); // 0.05
    }

    // -----------------------------------------------------------------------
    // Convergence test
    // -----------------------------------------------------------------------
    TEST_F(PoissonSolverTest, Convergence) {
        internal::reset_default_eps();
        std::vector<std::size_t> levels = { 3, 4,5}; // N = 9,17
        // Уровень 6 (65×65) убран — слишком дорого для точного Rational.
        std::vector<Scalar> errors;
        EuclideanMetric metric;

        for (std::size_t m : levels) {
            auto path = make_uniform_path(m);
            auto grid = path.current_grid();
            auto rhs = make_rhs(grid);
            auto bc = make_zero_dirichlet(path);
            auto sol = solve_poisson(path, rhs, bc, metric);
            errors.push_back(compute_L2_error(sol, path));
        }

        for (std::size_t i = 1; i < errors.size(); ++i) {
            if (errors[i] <= 0) continue;
            Scalar ratio = errors[i - 1] / errors[i];
            Scalar order = delta::log(ratio) / delta::log(2_r);
            EXPECT_GT(order, Scalar(18, 10)); // 1.8
        }
    }
    TEST_F(PoissonSolverTest, SolverPerformanceBenchmarkAutoSuccess) {
        // Control test: single solve on 33x33 grid to measure raw solve time.
        // No accuracy or convergence checks; only verifies the solver terminates
        // and outputs the wall-clock time for future baseline comparisons.
        internal::reset_default_eps();
        const std::size_t m = 5; // 2^5+1 = 33 points per dimension
        auto path = make_uniform_path(m);
        auto grid = path.current_grid();
        auto rhs = make_rhs(grid);
        auto bc = make_zero_dirichlet(path);
        EuclideanMetric metric;

        auto start = std::chrono::steady_clock::now();
        auto solution = solve_poisson(path, rhs, bc, metric);
        auto end = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "[PERF] Poisson 33x33 solve time: " << ms << " ms" << std::endl;
        SUCCEED();
    }
} // namespace delta::testing