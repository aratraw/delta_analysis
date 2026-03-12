// include/delta/numerical/grid_laplacian.h
#pragma once

#include <vector>
#include <Eigen/Sparse>
#include "delta/core/grid_concept.h"
#include "delta/core/regulative_idea.h"
#include "grid_ops_1d.h"  // для neighbor_indices

namespace delta::numerical {

    template<typename Grid, typename Metric>
    Eigen::SparseMatrix<typename Grid::value_type>
        build_laplacian_matrix(const Grid& grid, const Metric& metric) {
        using Value = typename Grid::value_type;
        std::size_t n = grid.size();
        std::vector<Eigen::Triplet<Value>> triplets;

        for (std::size_t i = 0; i < n; ++i) {
            const auto& point = grid[i];
            auto [left, right] = neighbor_indices(grid, static_cast<std::ptrdiff_t>(i));

            if (left < 0 || right < 0) {
                triplets.emplace_back(static_cast<int>(i), static_cast<int>(i), Value{ 1 });
                continue;
            }

            const auto& point_minus = grid[static_cast<std::size_t>(left)];
            const auto& point_plus = grid[static_cast<std::size_t>(right)];
            auto h_plus = metric(point_plus, point);
            auto h_minus = metric(point, point_minus);

            Value coeff_plus = Value{ 2 } / ((h_plus + h_minus) * h_plus);
            Value coeff_minus = Value{ 2 } / ((h_plus + h_minus) * h_minus);
            Value coeff_center = -(coeff_plus + coeff_minus);

            triplets.emplace_back(static_cast<int>(i), static_cast<int>(i), coeff_center);
            triplets.emplace_back(static_cast<int>(i), static_cast<int>(left), coeff_minus);
            triplets.emplace_back(static_cast<int>(i), static_cast<int>(right), coeff_plus);
        }

        Eigen::SparseMatrix<Value> A(static_cast<int>(n), static_cast<int>(n));
        A.setFromTriplets(triplets.begin(), triplets.end());
        return A;
    }

} // namespace delta::numerical