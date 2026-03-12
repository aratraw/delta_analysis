// include/delta/numerical/variational_solvers.h
#pragma once

#include "delta/geometry/discrete_action.h"
#include "delta/core/path_concept.h"
#include "delta/geometry/tensor_field.h"
#include <vector>
#include <functional>
#include <cmath>

namespace delta::numerical {

    /**
     * @brief Solve Euler-Lagrange equations using gradient descent.
     *
     * @tparam Action Type satisfying Action concept (must provide variation method).
     * @tparam Field  Type of the field (must support at(addr), set(addr, value), and value_type).
     * @tparam Path   Path type (must provide current_grid() returning a range of addresses).
     * @param action  The discrete action.
     * @param field   Initial guess (will be modified in-place).
     * @param path    The path providing the current grid.
     * @param step_size Learning rate.
     * @param max_iter Maximum iterations.
     * @param tol     Tolerance for gradient norm (squared L2 norm of variations).
     * @return bool   True if converged (gradient norm < tol), false otherwise.
     */
    template<typename Action, typename Field, typename Path>
    bool solve_euler_lagrange_gradient_descent(
        const Action& action,
        Field& field,
        const Path& path,
        typename Field::value_type step_size,
        std::size_t max_iter = 1000,
        typename Field::value_type tol = 1e-8)
    {
        using Value = typename Field::value_type;

        for (std::size_t iter = 0; iter < max_iter; ++iter) {
            Value grad_norm_sq = Value{ 0 };

            // Store variations to avoid in-place interference
            std::vector<Value> variations;
            const auto& grid = path.current_grid();
            variations.reserve(grid.size());

            // Compute variations and accumulate squared norm
            for (const auto& addr : grid) {
                Value g = action.variation(field, path, addr);
                variations.push_back(g);
                grad_norm_sq += g * g;   // assumes multiplication yields scalar square
            }

            // Check convergence
            if (grad_norm_sq < tol * tol) {
                return true;
            }

            // Update field using stored variations
            auto it_var = variations.begin();
            for (const auto& addr : grid) {
                Value current = field.at(addr);
                field.set(addr, current - step_size * (*it_var));
                ++it_var;
            }
        }

        return false; // did not converge within max_iter
    }

} // namespace delta::numerical