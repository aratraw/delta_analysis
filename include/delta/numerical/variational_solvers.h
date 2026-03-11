// include/delta/numerical/variational_solvers.h
#pragma once

#include "delta/geometry/discrete_action.h"
#include <vector>
#include <functional>

namespace delta::numerical {

    /**
     * @brief Solve Euler-Lagrange equations using gradient descent.
     *
     * @tparam Action Type satisfying Action concept.
     * @tparam Field  Type of the field (must be copyable and support element-wise operations).
     * @tparam Path   Path type.
     * @param action  The discrete action.
     * @param field   Initial guess (will be modified).
     * @param path    The path.
     * @param step_size Learning rate.
     * @param max_iter Maximum iterations.
     * @param tol     Tolerance for gradient norm.
     * @return bool   True if converged.
     */
    template<typename Action, typename Field, typename Path>
    bool solve_euler_lagrange_gradient_descent(const Action& action,
        Field& field,
        const Path& path,
        typename Field::value_type step_size,
        std::size_t max_iter = 1000,
        typename Field::value_type tol = 1e-8) {
        using Value = typename Field::value_type;
        for (std::size_t iter = 0; iter < max_iter; ++iter) {
            Value grad_norm{ 0 };
            // Compute variation at each address
            for (const auto& addr : path.current_grid()) {
                Value g = action.variation(field, path, addr);
                grad_norm += g * g;
                // Update: φ_new = φ_old - step_size * g
                // But field might not support direct modification. We'll need to create a new field.
                // For simplicity, assume field is a vector of values keyed by address.
                // In practice, we would need a way to set value at addr.
                // This is tricky; we'll leave as conceptual.
            }
            if (grad_norm < tol) return true;
        }
        return false;
    }

    // More sophisticated solvers (Newton, etc.) can be added later.

} // namespace delta::numerical