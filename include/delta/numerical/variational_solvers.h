// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/numerical/variational_solvers.h
#pragma once

#include <vector>
#include <cstddef>
#include "delta/core/rational.h"
#include "delta/numerical/boundary_conditions.h" // now needed for BC handling

namespace delta::numerical {

    template<typename GradFunc, typename Scalar>
    bool solve_euler_lagrange_gradient_descent(
        GradFunc&& grad,
        std::vector<Scalar>& field,
        Scalar step_size,
        const BoundaryConditions<Scalar>& bc, // added BC parameter
        std::size_t max_iter = 5000,
        Scalar tol = Scalar(1, 1000000))
    {
        std::size_t n = field.size();
        if (n == 0) return true;

        for (std::size_t iter = 0; iter < max_iter; ++iter) {
            Scalar max_change = 0;
            for (std::size_t i = 0; i < n; ++i) {
                // If this node has a Dirichlet condition, its value is fixed; skip it.
                if (bc.has(i) && bc.type(i) == BCType::Dirichlet)
                    continue;

                Scalar g = grad(i);
                Scalar new_val = field[i] - step_size * g;
                Scalar change = new_val - field[i];
                if (change < Scalar(0)) change = -change;
                if (change > max_change) max_change = change;
                field[i] = new_val;
            }
            if (max_change <= tol)
                return true;
        }
        return false;
    }

} // namespace delta::numerical