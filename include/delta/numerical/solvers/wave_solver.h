// include/delta/numerical/solvers/wave_solver.h
#pragma once

#include "delta/core/grid_concept.h"
#include "delta/numerical/grid_differences.h"
#include <Eigen/Sparse>
#include <vector>
#include <functional>

namespace delta::numerical {

    /**
     * @brief Wave equation (explicit leapfrog) on Cartesian grids.
     *
     * @tparam Grid    OrderedGrid (points must have scalar type).
     * @tparam Metric  Metric.
     * @param grid      The grid.
     * @param u0        Initial displacement.
     * @param v0        Initial velocity.
     * @param f         Source term f(t, index).
     * @param c         Wave speed.
     * @param dt        Time step.
     * @param num_steps Number of steps.
     * @param metric    Metric for distances.
     * @return std::vector<std::vector<Value>> Solution at each time step.
     */
    template<typename Grid, typename Metric>
    std::vector<std::vector<typename Grid::value_type>>
        solve_wave_explicit_cartesian(const Grid& grid,
            const std::vector<typename Grid::value_type>& u0,
            const std::vector<typename Grid::value_type>& v0,
            const std::function<typename Grid::value_type(double, std::size_t)>& f,
            double c, double dt, std::size_t num_steps,
            const Metric& metric) {
        using Value = typename Grid::value_type;
        std::size_t n = grid.size();

        auto A = build_laplacian_matrix(grid, metric);

        std::vector<Value> u_prev = u0;
        std::vector<Value> u_curr(n);
        // Initial step: u_curr = u0 + dt * v0
        for (std::size_t i = 0; i < n; ++i) {
            u_curr[i] = u0[i] + dt * v0[i];
        }

        std::vector<std::vector<Value>> result;
        result.push_back(u0);
        result.push_back(u_curr);

        for (std::size_t step = 1; step < num_steps; ++step) {
            double t = step * dt;
            std::vector<Value> u_next(n, 0);
            // Compute A * u_curr
            Eigen::Matrix<Value, Eigen::Dynamic, 1> u_curr_eig(n);
            for (std::size_t i = 0; i < n; ++i) u_curr_eig(i) = u_curr[i];
            Eigen::Matrix<Value, Eigen::Dynamic, 1> Au = A * u_curr_eig;

            for (std::size_t i = 0; i < n; ++i) {
                Value fi = f(t, i);
                u_next[i] = 2 * u_curr[i] - u_prev[i] + dt * dt * (c * c * Au(i) + fi);
            }
            u_prev = u_curr;
            u_curr = u_next;
            result.push_back(u_curr);
        }
        return result;
    }

} // namespace delta::numerical