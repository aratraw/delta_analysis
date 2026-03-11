// include/delta/numerical/convergence_tests.h
#pragma once

#include <vector>
#include <cmath>
#include <functional>
#include <iostream>

namespace delta::numerical {

    /**
     * @brief Compute L2 error between numerical and exact solutions on a grid.
     *
     * @tparam Grid Type satisfying GridConcept.
     * @tparam Value Scalar type.
     * @param grid The grid.
     * @param numerical Numerical solution values at grid points.
     * @param exact Exact solution function.
     * @return Value L2 error.
     */
    template<typename Grid, typename Value, typename ExactFunc>
    Value l2_error(const Grid& grid,
        const std::vector<Value>& numerical,
        ExactFunc&& exact) {
        Value sum{ 0 };
        Value total_measure{ 0 };
        for (std::size_t i = 0; i < grid.size(); ++i) {
            Value diff = numerical[i] - exact(grid[i]);
            // Need measure for each point. For 1D, use average of adjacent intervals.
            // For general grid, we approximate by the volume of the Voronoi cell.
            // This is complex. For simplicity, just sum squared differences.
            sum += diff * diff;
            total_measure += Value{ 1 };
        }
        return std::sqrt(sum / total_measure);
    }

    /**
     * @brief Print convergence table for a sequence of grids.
     *
     * @tparam Grids Container of grids.
     * @tparam Solutions Container of solution vectors.
     * @param grids Vector of grids (each level).
     * @param solutions Vector of solution vectors.
     * @param exact Exact solution function.
     * @param title Title for the table.
     */
    template<typename Grids, typename Solutions, typename ExactFunc>
    void convergence_table(const Grids& grids,
        const Solutions& solutions,
        ExactFunc&& exact,
        const std::string& title = "Convergence") {
        std::cout << "\n" << title << "\n";
        std::cout << "Level\tSize\tError\t\tOrder\n";
        double prev_error = 0.0;
        std::size_t prev_size = 0;
        for (std::size_t lvl = 0; lvl < grids.size(); ++lvl) {
            const auto& grid = grids[lvl];
            const auto& sol = solutions[lvl];
            double error = l2_error(grid, sol, exact);
            std::cout << lvl << "\t" << grid.size() << "\t" << error;
            if (lvl > 0) {
                double order = std::log(prev_error / error) / std::log(static_cast<double>(prev_size) / grid.size());
                std::cout << "\t" << order;
            }
            std::cout << "\n";
            prev_error = error;
            prev_size = grid.size();
        }
    }

} // namespace delta::numerical