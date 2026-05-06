// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/numerical/solvers/wave_solver.h
// ============================================================================
// DISCRETE WAVE SOLVER (leapfrog, uniform product grid)
// ============================================================================
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
#include "delta/numerical/solvers/common.h"       // assembly helpers

namespace delta::numerical::solvers {

    template<typename Path2D, typename Scalar, typename BC, typename Metric>
    auto solve_wave_leapfrog(
        const Path2D& path,
        const OperationalFunction<std::array<Scalar, 2>, Scalar,
        ProductGrid<UniformGrid<Scalar>, 2>>&u0,
        const OperationalFunction<std::array<Scalar, 2>, Scalar,
        ProductGrid<UniformGrid<Scalar>, 2>>&v0,
        Scalar c,
        Scalar dt,
        Scalar T,
        const Metric& /*metric*/,
        BC& bc)
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
        // 1. Assemble stiffness matrix K and lumped mass vector
        // -----------------------------------------------------------------
        Eigen::SparseMatrix<Scalar> K = assemble_laplacian_5pt<Scalar>(nx, ny);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> M_vec = lumped_mass_vector<Scalar>(nx, ny, hx, hy);

        // -----------------------------------------------------------------
        // 2. Initial condition vectors u^0 and v^0
        // -----------------------------------------------------------------
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u_prev(n);  // will be u^{n-1}
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u_cur(n);   // will be u^n
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> v_vec(n);   // v^0 (only used for first step)

        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                std::array<Scalar, 2> pt = { grid_x[i], grid_y[j] };
                std::size_t idx = i + j * nx;
                u_prev(idx) = u0(pt);
                v_vec(idx) = v0(pt);
            }
        }

        // Enforce Dirichlet BCs on the initial state (u^0)
        for (std::size_t j = 0; j < ny; ++j)
            for (std::size_t i = 0; i < nx; ++i)
                if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
                    u_prev(i + j * nx) = 0_r;

        // -----------------------------------------------------------------
        // 3. First time step (second‑order)
        //    u¹ = u⁰ + Δt·v⁰ – (c²Δt²/2) M⁻¹ K u⁰
        // -----------------------------------------------------------------
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Ku0 = K * u_prev;
        Scalar coeff = (c * c * dt * dt) / (2_r * mass_lump);
        u_cur = u_prev + dt * v_vec - coeff * Ku0;

        // Enforce Dirichlet BCs on u¹ as well
        for (std::size_t j = 0; j < ny; ++j)
            for (std::size_t i = 0; i < nx; ++i)
                if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
                    u_cur(i + j * nx) = 0_r;

        // -----------------------------------------------------------------
        // 4. Number of time steps
        // -----------------------------------------------------------------
        std::size_t steps = static_cast<std::size_t>((T / dt).convert_to<double>());
        if (steps == 0) steps = 1;

        // -----------------------------------------------------------------
        // 5. Main leapfrog loop – only interior nodes are updated
        // -----------------------------------------------------------------
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u_next(n);
        for (std::size_t step = 1; step < steps; ++step) {
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Ku = K * u_cur;
            for (std::size_t j = 1; j < ny - 1; ++j) {
                for (std::size_t i = 1; i < nx - 1; ++i) {
                    std::size_t idx = i + j * nx;
                    u_next(idx) = 2_r * u_cur(idx) - u_prev(idx) - (c * c * dt * dt / mass_lump) * Ku(idx);
                }
            }
            // Boundary nodes stay zero
            for (std::size_t j = 0; j < ny; ++j)
                for (std::size_t i = 0; i < nx; ++i)
                    if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1)
                        u_next(i + j * nx) = 0_r;

            // shift for next step
            u_prev = u_cur;
            u_cur = u_next;
        }

        // -----------------------------------------------------------------
        // 6. Wrap result into OperationalFunction
        // -----------------------------------------------------------------
        return OF(grid2d, [u = std::move(u_cur), &grid_x, &grid_y, nx](const std::array<Scalar, 2>& pt) -> Scalar {
            std::size_t i = delta::detail::uniform_index(pt[0], grid_x);
            std::size_t j = delta::detail::uniform_index(pt[1], grid_y);
            return u(i + j * nx);
            });
    }

} // namespace delta::numerical::solvers