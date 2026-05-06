// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/numerical/solvers/advection_solver.h
// ============================================================================
// DISCRETE ADVECTION SOLVER (upwind, uniform product grid, periodic BCs)
// ============================================================================
// Solves u_t + a u_x + b u_y = 0 with constant velocities a, b ≥ 0.
// Returns: OperationalFunction over the grid of the given path at time T.
// ============================================================================

#pragma once

#include <Eigen/Sparse>         // only for Eigen::Matrix (we don't need Sparse here)
#include <array>
#include <vector>
#include <stdexcept>
#include <functional>
#include "delta/core/uniform_grid.h"
#include "delta/core/product_grid.h"
#include "delta/core/operational_function.h"
#include "delta/geometry/product_regulative.h"
#include "delta/numerical/boundary_conditions.h"
#include "delta/numerical/solvers/common.h"         // not strictly needed, but for consistency

namespace delta::numerical::solvers {

    template<typename Path2D, typename Scalar, typename BC>
    auto solve_advection_upwind_2d(
        const Path2D& path,
        const OperationalFunction<std::array<Scalar, 2>, Scalar,
        ProductGrid<UniformGrid<Scalar>, 2>>&u0,
        Scalar a,
        Scalar b,
        Scalar dt,
        Scalar T,
        BC& bc)
    {
        using Grid2D = ProductGrid<UniformGrid<Scalar>, 2>;
        using OF = OperationalFunction<std::array<Scalar, 2>, Scalar, Grid2D>;

        auto grid2d = path.current_grid();
        const auto& grid_x = grid2d.get_grid(0);
        const auto& grid_y = grid2d.get_grid(1);
        std::size_t nx = grid_x.size();   // number of nodes in x (for periodic, usually N cells → N nodes)
        std::size_t ny = grid_y.size();
        std::size_t n = nx * ny;
        Scalar hx = grid_x.step();
        Scalar hy = grid_y.step();

        // Courant numbers (non‑negative velocities assumed)
        Scalar Cx = a * dt / hx;
        Scalar Cy = b * dt / hy;

        // Periodic index helper
        auto wrap = [](std::ptrdiff_t idx, std::size_t mod) -> std::size_t {
            if (idx < 0) return static_cast<std::size_t>(idx + static_cast<std::ptrdiff_t>(mod));
            if (static_cast<std::size_t>(idx) >= mod) return static_cast<std::size_t>(idx - static_cast<std::ptrdiff_t>(mod));
            return static_cast<std::size_t>(idx);
            };

        // Initial condition vector
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u_cur(n);
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                std::array<Scalar, 2> pt = { grid_x[i], grid_y[j] };
                u_cur(i + j * nx) = u0(pt);
            }
        }

        // Number of time steps
        std::size_t steps = static_cast<std::size_t>((T / dt).convert_to<int>());
        if (steps == 0) steps = 1;

        // Time iteration
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> u_next(n);
        for (std::size_t step = 0; step < steps; ++step) {
            for (std::size_t j = 0; j < ny; ++j) {
                for (std::size_t i = 0; i < nx; ++i) {
                    std::size_t idx = i + j * nx;
                    std::size_t im1 = wrap(static_cast<std::ptrdiff_t>(i) - 1, nx);
                    std::size_t jm1 = wrap(static_cast<std::ptrdiff_t>(j) - 1, ny);

                    Scalar u_im1 = u_cur(im1 + j * nx);
                    Scalar u_jm1 = u_cur(i + jm1 * nx);
                    Scalar u_ij = u_cur(idx);

                    u_next(idx) = u_ij - Cx * (u_ij - u_im1) - Cy * (u_ij - u_jm1);
                }
            }
            u_cur.swap(u_next);
        }

        // Wrap result
        return OF(grid2d, [u = std::move(u_cur), &grid_x, &grid_y, nx](const std::array<Scalar, 2>& pt) -> Scalar {
            std::size_t i = delta::detail::uniform_index(pt[0], grid_x);
            std::size_t j = delta::detail::uniform_index(pt[1], grid_y);
            return u(i + j * nx);
            });
    }

} // namespace delta::numerical::solvers