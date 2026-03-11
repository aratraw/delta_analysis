// include/delta/numerical/solvers/advection_solver.h
#pragma once

#include "delta/core/grid_concept.h"
#include <vector>

namespace delta::numerical {

    /**
     * @brief Advection equation (upwind) on 1D Cartesian grid.
     *
     * Solves ∂u/∂t + a ∂u/∂x = 0 with upwind scheme.
     *
     * @tparam Grid    OrderedGrid (1D).
     * @tparam Metric  Metric.
     * @param grid      The grid.
     * @param u0        Initial condition.
     * @param a         Advection speed.
     * @param dt        Time step.
     * @param num_steps Number of steps.
     * @param metric    Metric for grid spacing.
     * @return std::vector<Value> Solution at final time.
     */
    template<typename Grid, typename Metric>
    std::vector<typename Grid::value_type>
        solve_advection_upwind_1d(const Grid& grid,
            const std::vector<typename Grid::value_type>& u0,
            double a, double dt, std::size_t num_steps,
            const Metric& metric) {
        using Value = typename Grid::value_type;
        std::size_t n = grid.size();

        std::vector<Value> u = u0;
        std::vector<Value> u_new(n);

        for (std::size_t step = 0; step < num_steps; ++step) {
            for (std::size_t i = 0; i < n; ++i) {
                if (a > 0) {
                    if (i == 0) {
                        // left boundary – use constant (Dirichlet)
                        u_new[i] = u[i];
                    }
                    else {
                        Value h = metric(grid[i], grid[i - 1]); // step to the left
                        u_new[i] = u[i] - a * dt / h * (u[i] - u[i - 1]);
                    }
                }
                else {
                    if (i == n - 1) {
                        u_new[i] = u[i];
                    }
                    else {
                        Value h = metric(grid[i + 1], grid[i]);
                        u_new[i] = u[i] - a * dt / h * (u[i + 1] - u[i]);
                    }
                }
            }
            u = u_new;
        }
        return u;
    }

} // namespace delta::numerical