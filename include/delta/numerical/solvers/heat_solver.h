// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/numerical/solvers/heat_solver.h
// ============================================================================
// DISCRETE HEAT SOLVER (five‑point stencil, uniform product grid)
// ============================================================================
// Supported time schemes: Explicit Euler, Implicit Euler, Crank‑Nicolson.
// Returns: OperationalFunction over the grid of the given path at time T.
// ============================================================================

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
#include "delta/numerical/solvers/common.h"       // TimeScheme, assembly helpers

namespace delta::numerical::solvers {

    template<typename Path2D, typename Scalar, typename BC, typename Metric>
    auto solve_heat(
        const Path2D& path,
        const OperationalFunction<std::array<Scalar, 2>, Scalar,
        ProductGrid<UniformGrid<Scalar>, 2>>&u0,
        Scalar alpha,
        Scalar dt,
        Scalar T,
        const Metric& /*metric*/,
        BC& bc,
        TimeScheme scheme)
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
        const Scalar mass_lump = hx * hy;

        // -----------------------------------------------------------------
        // 1. Assemble stiffness matrix K (unscaled) and lumped mass vector
        // -----------------------------------------------------------------
        Eigen::SparseMatrix<Scalar> K = assemble_laplacian_5pt<Scalar>(nx, ny);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> M_vec = lumped_mass_vector<Scalar>(nx, ny, hx, hy);

        // -----------------------------------------------------------------
        // 2. Initial condition vector
        // -----------------------------------------------------------------
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u_vec(n);
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                std::array<Scalar, 2> pt = { grid_x[i], grid_y[j] };
                u_vec(i + j * nx) = u0(pt);
            }
        }

        // -----------------------------------------------------------------
        // 3. Number of time steps
        // -----------------------------------------------------------------
        std::size_t steps = static_cast<std::size_t>((T / dt).convert_to<double>());
        if (steps == 0) steps = 1;

        // -----------------------------------------------------------------
        // 4. Precompute implicit system matrix and factorise (if needed)
        // -----------------------------------------------------------------
        Eigen::SparseMatrix<Scalar> A;
        Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> solver;
        if (scheme != TimeScheme::EXPLICIT_EULER) {
            Scalar theta = (scheme == TimeScheme::CRANK_NICOLSON) ? Scalar(1, 2) : Scalar(1);
            A = K;
            A *= (theta * dt * alpha);
            for (std::size_t k = 0; k < n; ++k) {
                A.coeffRef(k, k) += mass_lump;
            }
            // Apply boundary conditions once to the system matrix
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dummy_rhs(n);
            dummy_rhs.setZero();
            apply_boundary_conditions(A, dummy_rhs, M_vec, n, bc);
            solver.compute(A);
            if (solver.info() != Eigen::Success)
                throw std::runtime_error("Heat solver: decomposition failed");
        }

        // -----------------------------------------------------------------
        // 5. Time stepping loop
        // -----------------------------------------------------------------
        for (std::size_t step = 0; step < steps; ++step) {
            if (scheme == TimeScheme::EXPLICIT_EULER) {
                Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Ku = K * u_vec;
                Scalar coeff = (dt * alpha) / mass_lump;
                u_vec -= coeff * Ku;
                // Enforce Dirichlet zero on boundaries
                for (std::size_t j = 0; j < ny; ++j) {
                    for (std::size_t i = 0; i < nx; ++i) {
                        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
                            u_vec(i + j * nx) = 0;
                        }
                    }
                }
            }
            else {
                Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b(n);
                if (scheme == TimeScheme::IMPLICIT_EULER) {
                    b = mass_lump * u_vec;
                }
                else { // Crank‑Nicolson
                    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Ku = K * u_vec;
                    b = mass_lump * u_vec - (dt * alpha / 2_r) * Ku;
                }
                // Apply Dirichlet values to RHS (the matrix A already has BC applied)
                for (std::size_t i = 0; i < n; ++i) {
                    if (bc.has(i) && bc.type(i) == BCType::Dirichlet) {
                        b(i) = bc.value(i);
                    }
                }
                u_vec = solver.solve(b);
                if (solver.info() != Eigen::Success)
                    throw std::runtime_error("Heat solver: solve failed at step");
            }
        }

        // -----------------------------------------------------------------
        // 6. Wrap result into OperationalFunction
        //
        // *** CRITICAL FIX ***
        // The solution vector `u_vec` must be moved into the returned lambda
        // to ensure its lifetime matches that of the OperationalFunction.
        // Capturing by reference (`[&]`) is unsafe because `u_vec` is destroyed
        // when `solve_heat` returns.  The same fix must be applied to
        // `poisson_solver.h` (and any other solver returning an
        // OperationalFunction backed by a local Eigen vector).
        // -----------------------------------------------------------------
        return OF(grid2d, [u = std::move(u_vec), &grid_x, &grid_y, nx](const std::array<Scalar, 2>& pt) -> Scalar {
            std::size_t i = delta::detail::uniform_index(pt[0], grid_x);
            std::size_t j = delta::detail::uniform_index(pt[1], grid_y);
            return u(i + j * nx);
            });
    }

} // namespace delta::numerical::solvers