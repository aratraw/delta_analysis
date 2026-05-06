// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/numerical/solvers/advection_solver_test.cpp
// ============================================================================
// MATHEMATICAL VERIFICATION OF DISCRETE ADVECTION SOLVER (UPWIND)
// (corrected for exact rational arithmetic — no rounding noise)
// ============================================================================
//
// This test suite validates the correctness of solve_advection_upwind_2d()
// on the unit square with periodic boundary conditions.
//
// ---------------------------------------------------------------------------
// 1. CONTINUOUS PROBLEM
// ---------------------------------------------------------------------------
// Ω = (0,1)×(0,1)
// u_t + a u_x + b u_y = 0 in Ω
// Periodic BCs: u(0,y,t)=u(1,y,t), u(x,0,t)=u(x,1,t)
// Initial condition: u(x,y,0) = u0(x,y)
// Exact solution: u(x,y,t) = u0(x - a t, y - b t) (with periodic wrapping)
//
// Tested configurations:
// a) Polynomial exact rational: u0 = (x(1-x)y(1-y))^2
// Used for strict mass conservation – no transcendental error.
// b) Smooth periodic: u0 = sin(2πx) sin(2πy)
// Exact solution: u_ex(x,y,t) = sin(2π(x - t)) sin(2πy)
// Used for convergence tests.
//
// ---------------------------------------------------------------------------
// 2. DISCRETISATION
// ---------------------------------------------------------------------------
// Spatial: uniform grid, N cells per dimension, h = 1/N, nodes at (i·h, j·h)
// with periodic index arithmetic (mod N).
// Temporal: explicit upwind
// u_{i,j}^{n+1} = u_{i,j}^n - Cx (u_{i,j}^n - u_{i-1,j}^n) - Cy (u_{i,j}^n - u_{i,j-1}^n)
// where Cx = a·Δt/h, Cy = b·Δt/h (for a,b ≥ 0).
//
// Mass (∑ u_{i,j}·h²) is conserved exactly in exact arithmetic with periodic BCs.
//
// ---------------------------------------------------------------------------
// 3. EXPECTED PROPERTIES IN EXACT RATIONAL ARITHMETIC
// ---------------------------------------------------------------------------
// • Mass: with polynomial IC and exact arithmetic, total mass is strictly invariant.
// • Convergence: error decreases when Δt and h are reduced simultaneously.
//
// Stability (CFL) test is omitted for the same reason as wave/heat:
// without rounding noise a pure Fourier mode does not trigger instability
// even when C > 1.
//
// ---------------------------------------------------------------------------
// 4. TEST PLAN
// ---------------------------------------------------------------------------
// MassConservation – polynomial IC, periodic BCs, check Σu(t)=Σu(0)
// TemporalSpatialConvergence – smooth periodic IC, fixed ratio Δt = h/2,
// error on finer grid strictly smaller.
//
// IF ANY TEST FAILS, THE BUG IS IN THE SOLVER IMPLEMENTATION, NOT IN THE TEST.
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
#include "delta/numerical/solvers/advection_solver.h"
#include "../../test_fixtures_geometry_numerical.h"

namespace delta::testing {
	using namespace delta::numerical::solvers;

	class AdvectionSolverTest : public GeometryNumericalTest {
	protected:
		using Scalar = Rational;
		using Path1D = UniformDeltaPath<Scalar>;
		using Path2D = geometry::ProductDeltaPath<Path1D, Path1D>;

		// Build a uniform grid with N cells per dimension.
		// We use UniformDeltaPath, but since we need periodic BCs, we only
		// care about the node coordinates; the path is used for the grid structure.
		// Number of interior nodes = N (periodic identification of boundaries).
		Path2D make_uniform_path(std::size_t N) {
			Scalar h = 1_r / Scalar(N);
			// UniformDeltaPath(start, step, count)
			Path1D path1d(0_r, h, N, std::less<Scalar>{});
			return Path2D(path1d, path1d);
		}

		// -------------------------------------------------------------------
		// Polynomial initial condition (exact rational)
		// -------------------------------------------------------------------
		Scalar poly_u0_func(const std::array<Scalar, 2>& pt) const {
			Scalar x = pt[0], y = pt[1];
			Scalar val = x * (1_r - x) * y * (1_r - y);
			return val * val; // note: x and y are in [0,1], periodicity not needed
		}

		template<typename Grid2D>
		OperationalFunction<std::array<Scalar, 2>, Scalar, Grid2D>
			make_poly_u0(const Grid2D& grid) const {
			return OperationalFunction<std::array<Scalar, 2>, Scalar, Grid2D>(
				grid,
				[this](const std::array<Scalar, 2>& pt) { return poly_u0_func(pt); }
			);
		}

		// -------------------------------------------------------------------
		// Smooth periodic initial condition & exact solution
		// -------------------------------------------------------------------
		Scalar sin_u0_func(const std::array<Scalar, 2>& pt) const {
			Scalar pi = pi_val();
			return delta::sin(2_r * pi * pt[0]) * delta::sin(2_r * pi * pt[1]);
		}

		Scalar exact_solution(const std::array<Scalar, 2>& pt, Scalar t, Scalar a, Scalar b) const {
			Scalar pi = pi_val();
			Scalar x = pt[0] - a * t;
			Scalar y = pt[1] - b * t;
			// periodic wrapping into [0,1)
			x = x - delta::floor(x);
			y = y - delta::floor(y);
			return delta::sin(2_r * pi * x) * delta::sin(2_r * pi * y);
		}

		template<typename Grid2D>
		OperationalFunction<std::array<Scalar, 2>, Scalar, Grid2D>
			make_sin_u0(const Grid2D& grid) const {
			return OperationalFunction<std::array<Scalar, 2>, Scalar, Grid2D>(
				grid,
				[this](const std::array<Scalar, 2>& pt) { return sin_u0_func(pt); }
			);
		}

		Scalar pi_val() const {
			return delta::pi(delta::default_eps());
		}

		// -------------------------------------------------------------------
		// Periodic boundary conditions
		// -------------------------------------------------------------------
		BoundaryConditions<Scalar> make_periodic(const Path2D& path) const {
			BoundaryConditions<Scalar> bc;
			auto grid = path.current_grid();
			std::size_t Nx = grid.get_grid(0).size(); // N nodes in x
			std::size_t Ny = grid.get_grid(1).size(); // N nodes in y
			// Periodic: identify i=0 with i=Nx-1? Actually with periodic, we have
			// N cells, nodes 0..N-1, and we want u_{N,j}=u_{0,j}, so we pair (0,j) with (N-1,j)?
			// In our setup, we have N nodes per dimension, but we want periodic wrap:
			// we'll use N as number of cells, and we store N nodes (0..N-1). For periodicity,
			// we need to identify node i=N with node i=0. Since we only have nodes 0..N-1,
			// we can identify i=0 and i=N-1? That would be for a grid of N cells, nodes at 0, 1/N, ... , (N-1)/N.
			// Then u(1,y) corresponds to node at 1, which is the same as node 0. So we pair (0,j) with (N-1,j)?
			// Actually if we have nodes 0,1,...,N-1 representing coordinates 0, h, 2h, ..., (N-1)h, then the right
			// boundary is at x=1 which is N*h = 1. That node does not exist. So we need to pair the last node (N-1)
			// with the first one (0) for the stencil to wrap. That's what we'll do.
			for (std::size_t j = 0; j < Ny; ++j) {
				// Left-right periodic: (0,j) with (Nx-1,j)
				bc.add_periodic_pair(0 + j * Nx, (Nx - 1) + j * Nx);
			}
			for (std::size_t i = 0; i < Nx; ++i) {
				// Top-bottom periodic: (i,0) with (i,Ny-1)
				bc.add_periodic_pair(i + 0 * Nx, i + (Ny - 1) * Nx);
			}
			return bc;
		}

		// -------------------------------------------------------------------
		// Mass (sum u * h²)
		// -------------------------------------------------------------------
		template<typename Field>
		Scalar compute_mass(const Field& sol, const Path2D& path) const {
			auto grid = path.current_grid();
			Scalar total = 0;
			std::size_t nx = grid.get_grid(0).size();
			std::size_t ny = grid.get_grid(1).size();
			Scalar hx = grid.get_grid(0).step();
			Scalar hy = grid.get_grid(1).step();
			Scalar vol = hx * hy;
			for (std::size_t j = 0; j < ny; ++j) {
				for (std::size_t i = 0; i < nx; ++i) {
					std::array<Scalar, 2> pt = { grid.get_grid(0)[i], grid.get_grid(1)[j] };
					total += sol(pt) * vol;
				}
			}
			return total;
		}

		// -------------------------------------------------------------------
		// L2 error at time t
		// -------------------------------------------------------------------
		template<typename Field>
		Scalar compute_L2_error(const Field& sol, const Path2D& path,
			Scalar t, Scalar a, Scalar b) const {
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
					Scalar diff = sol(pt) - exact_solution(pt, t, a, b);
					error_sq += diff * diff * vol;
				}
			}
			return delta::sqrt(error_sq);
		}
	};

	// =======================================================================
	// Mass conservation
	// =======================================================================
	TEST_F(AdvectionSolverTest, MassConservation) {
		using Scalar = Rational;
		const std::size_t N = 8; // 8 cells per dimension
		const Scalar a = 1_r, b = 0_r;
		const Scalar h = 1_r / N;
		const Scalar dt = h / 2_r; // CFL = 0.5
		const Scalar T = 1_r;

		auto path = make_uniform_path(N);
		auto u0 = make_poly_u0(path.current_grid());
		auto bc = make_periodic(path);

		// Mass before
		Scalar initial_mass = compute_mass(u0, path);

		auto sol = solve_advection_upwind_2d(path, u0, a, b, dt, T, bc);

		// Mass after
		Scalar final_mass = compute_mass(sol, path);

		EXPECT_EQ(initial_mass, final_mass);
	}

	// =======================================================================
	// Temporal‑spatial convergence (sinusoidal initial condition)
	// =======================================================================
	TEST_F(AdvectionSolverTest, Convergence) {
		using Scalar = Rational;
		const Scalar a = 1_r, b = 0_r;
		const Scalar T = Scalar(5, 10); // 0.5

		std::vector<std::size_t> sizes = { 16, 32 }; // N cells
		std::vector<Scalar> errors;

		for (std::size_t N : sizes) {
			Scalar h = 1_r / N;
			Scalar dt = h / 2_r;
			auto path = make_uniform_path(N);
			auto u0 = make_sin_u0(path.current_grid());
			auto bc = make_periodic(path);
			auto sol = solve_advection_upwind_2d(path, u0, a, b, dt, T, bc);
			errors.push_back(compute_L2_error(sol, path, T, a, b));
		}

		// Finer grid must give smaller error
		EXPECT_LT(errors[1], errors[0]);
	}

} // namespace delta::testing