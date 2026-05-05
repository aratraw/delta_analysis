// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/numerical/solvers/poisson_solver.h
// ============================================================================
// DISCRETE POISSON SOLVER (five‑point stencil, uniform product grid)
// ============================================================================
// Returns: OperationalFunction over the grid of the given path.
// ============================================================================

// =============================================================================
// FUTURE OPTIMISATIONS – BEYOND EXACT SPARSE LU
// =============================================================================
//
// The current implementation uses Eigen::SparseLU with the exact Rational
// scalar type.  While mathematically pristine, it suffers from exponential
// growth of numerators/denominators and fill‑in during factorisation, making
// it impractical for grids larger than about 17×17.  The Δ‑analysis framework
// provides a natural remedy: the path already generates a hierarchy of nested
// grids, which is the perfect substrate for **multilevel algorithms**.
//
// ---------------------------------------------------------------------------
// 1. GEOMETRIC MULTIGRID (GMG) – THE MAIN CANDIDATE
// ---------------------------------------------------------------------------
// Every Δ‑path (UniformDeltaPath or DeltaPath) naturally stores the sequence
// S₀ ⊂ S₁ ⊂ … of grids.  A V‑cycle multigrid solver would:
//   a) Smooth the error on the finest grid with a cheap iterative method
//      (damped Jacobi, Gauss‑Seidel) that does NOT create new denominators
//      (only linear combinations of existing values).
//   b) Restrict the residual to the next coarser grid.
//   c) Repeat down to the coarsest grid (e.g. 3×3 or 5×5).
//   d) Solve exactly on the coarsest grid – this LU solve is negligible.
//   e) Prolongate the correction back up, smoothing again.
//
// The key advantage for Rational: the exact factorisation is performed only
// on a tiny matrix (≲ 25 unknowns), where it is instantaneous.  All other
// operations are sparse matrix‑vector products with small integer coefficients
// (the stencil entries never change), so the rational numbers remain compact.
// The overall complexity becomes O(N) per cycle, with convergence factors
// typically < 0.1 per V‑cycle.
//
// ---------------------------------------------------------------------------
// 2. ALGEBRAIC MULTIGRID (AMG) / HIERARCHICAL MATRICES
// ---------------------------------------------------------------------------
// If we later support unstructured grids (e.g. simplicial complexes), GMG
// must be replaced by AMG, which constructs coarse spaces algebraically.
// The same principle applies: keep exact factorisation on the coarsest level
// only, and use cheap smoothers everywhere else.
//
// ---------------------------------------------------------------------------
// 3. DOMAIN DECOMPOSITION
// ---------------------------------------------------------------------------
// Divide the domain into subdomains, solve exactly on each small subdomain,
// and “stitch” them together via boundary multipliers (FETI, Neumann‑Neumann).
// Subdomain solves are again tiny LU factorisations, keeping Rational
// numbers small.  This approach fits naturally with ProductGrid, where
// subdomains can be rectangular blocks.
//
// ---------------------------------------------------------------------------
// 4. ITERATIVE SOLVERS WITH RATIONAL PRECONDITIONERS
// ---------------------------------------------------------------------------
// Conjugate Gradient (CG) with a cheap preconditioner (diagonal, ILU(0),
// multigrid) avoids any global factorisation.  CG iterations only require
// matrix‑vector products, which for the five‑point stencil are O(N) and
// involve only the constants 4 and –1.  Preconditioners must be chosen to
// not introduce large denominators.
//
// ---------------------------------------------------------------------------
// 5. MODULAR (CHINESE REMAINDER) SOLVER
// ---------------------------------------------------------------------------
// Solve the system modulo several small primes (using fast integer
// arithmetic), then reconstruct the exact rational solution via the Chinese
// Remainder Theorem and rational reconstruction.  This completely avoids
// rational arithmetic in the inner loop and can handle significantly larger
// systems while retaining exactness.  It is, however, more complex to
// implement and requires a reliable bound on the solution size.
//
// ---------------------------------------------------------------------------
// 6. CONCLUSION – BACKLOG FOR THE NEXT DEVELOPMENT CYCLE
// ---------------------------------------------------------------------------
// The current solver is feature‑complete and passes validation.  When
// performance on larger grids becomes necessary, the recommended path is:
//   1. Implement Geometric Multigrid using the existing path hierarchy.
//   2. Keep exact factorisation only on the coarsest level.
//   3. Investigate modular methods if exactness must be preserved at scale.
//
// Until then, we limit tests to grid sizes where direct LU is still feasible
// (N ≤ 17 is safe; N = 33 is acceptable for a single baseline measurement).
// =============================================================================
#pragma once

#include <Eigen/Sparse>
#include <array>
#include <vector>
#include <stdexcept>
#include "delta/core/uniform_grid.h"
#include "delta/core/product_grid.h"
#include "delta/core/operational_function.h"
#include "delta/geometry/product_regulative.h"
#include "delta/numerical/boundary_conditions.h"

namespace delta::numerical::solvers {

    /**
     * @brief Solve -Δu = f on a 2D uniform product grid.
     *
     * @tparam Path2D  ProductDeltaPath whose current_grid() yields
     *                 ProductGrid<UniformGrid<Scalar>, 2>.
     * @tparam BC      BoundaryConditions<Scalar> type.
     * @tparam Metric  Metric type (unused for uniform grid, reserved for future).
     *
     * @param path   Path providing the current grid.
     * @param rhs    OperationalFunction containing the right‑hand side f.
     * @param bc     Boundary conditions.
     * @param metric Metric object (unused).
     * @return       OperationalFunction representing the solution u.
     */
    template<typename Path2D, typename Scalar, typename BC, typename Metric>
    auto solve_poisson(
        const Path2D& path,
        const OperationalFunction<std::array<Scalar, 2>, Scalar,
        ProductGrid<UniformGrid<Scalar>, 2>>&rhs,
        BC& bc,
        const Metric& /*metric*/)
    {
        using Grid2D = ProductGrid<UniformGrid<Scalar>, 2>;
        using OF = OperationalFunction<std::array<Scalar, 2>, Scalar, Grid2D>;

        auto grid2d = path.current_grid();
        const auto& grid_x = grid2d.get_grid(0);
        const auto& grid_y = grid2d.get_grid(1);
        std::size_t nx = grid_x.size();
        std::size_t ny = grid_y.size();
        std::size_t n = nx * ny;
        Scalar hx = grid_x.step();
        Scalar hy = grid_y.step();

        // Assemble matrix (unscaled – RHS will be scaled by hx*hy)
        Eigen::SparseMatrix<Scalar> A(n, n);
        std::vector<Eigen::Triplet<Scalar>> triplets;
        triplets.reserve(n * 5);
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) continue;
                std::size_t row = i + j * nx;
                triplets.emplace_back(row, row, 4_r);
                triplets.emplace_back(row, row - 1, -1_r);
                triplets.emplace_back(row, row + 1, -1_r);
                triplets.emplace_back(row, row - nx, -1_r);
                triplets.emplace_back(row, row + nx, -1_r);
            }
        }
        A.setFromTriplets(triplets.begin(), triplets.end());

        // Build RHS vector b
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b(n);
        b.setZero();
        const Scalar cell_area = hx * hy;
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                std::array<Scalar, 2> pt = { grid_x[i], grid_y[j] };
                std::size_t idx = i + j * nx;
                // rhs is OperationalFunction, supports operator()
                b(idx) = rhs(pt) * cell_area;
            }
        }

        // Lumped mass vector (placeholder – not used for Dirichlet)
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> M(n);
        M.setConstant(cell_area); // actual dual areas would be better, but works for Dirichlet only

        // Apply boundary conditions
        apply_boundary_conditions(A, b, M, n, bc);

        // Solve
        Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
        solver.compute(A);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Poisson solver: decomposition failed");
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u = solver.solve(b);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Poisson solver: solve failed");

        // Wrap solution into OperationalFunction
        // Use a lambda that captures u and grid dimensions
        return OF(grid2d, [&](const std::array<Scalar, 2>& pt) -> Scalar {
            std::size_t i = delta::detail::uniform_index(pt[0], grid_x);
            std::size_t j = delta::detail::uniform_index(pt[1], grid_y);
            return u(i + j * nx);
            });
    }

} // namespace delta::numerical::solvers