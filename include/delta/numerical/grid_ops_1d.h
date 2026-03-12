// include/delta/numerical/grid_ops_1d.h
#pragma once

#include <cstddef>
#include <utility>
#include "delta/core/grid_concept.h"
#include "delta/core/regulative_idea.h"

namespace delta::numerical {

    // -------------------------------------------------------------------------
    // Схемы разностей
    // -------------------------------------------------------------------------
    enum class DifferenceScheme {
        FORWARD,
        BACKWARD,
        CENTRAL
    };

    // -------------------------------------------------------------------------
    // Вспомогательные функции для одномерных упорядоченных сеток
    // -------------------------------------------------------------------------

    template<typename Grid>
    std::ptrdiff_t find_index(const Grid& grid, const typename Grid::value_type& point) {
        for (std::size_t i = 0; i < grid.size(); ++i) {
            if (grid[i] == point) return static_cast<std::ptrdiff_t>(i);
        }
        return -1;
    }

    template<typename Grid>
    std::pair<std::ptrdiff_t, std::ptrdiff_t>
        neighbor_indices(const Grid& grid, std::ptrdiff_t idx) {
        std::ptrdiff_t left = (idx > 0) ? idx - 1 : -1;
        std::ptrdiff_t right = (idx + 1 < static_cast<std::ptrdiff_t>(grid.size())) ? idx + 1 : -1;
        return { left, right };
    }

    // -------------------------------------------------------------------------
    // Одномерные разностные операторы
    // -------------------------------------------------------------------------

    template<typename Grid, typename Field, typename Metric>
    auto forward_difference(const Grid& grid,
        const Field& field,
        const Metric& metric,
        const typename Grid::value_type& point) {
        using Value = std::decay_t<decltype(field(point))>;
        std::ptrdiff_t idx = find_index(grid, point);
        if (idx < 0) return Value{ 0 };

        auto [left, right] = neighbor_indices(grid, idx);
        if (right < 0) return Value{ 0 };

        const auto& point_plus = grid[right];
        Value f_plus = field(point_plus);
        Value f_center = field(point);
        auto h_plus = metric(point_plus, point);
        return (f_plus - f_center) / h_plus;
    }

    template<typename Grid, typename Field, typename Metric>
    auto backward_difference(const Grid& grid,
        const Field& field,
        const Metric& metric,
        const typename Grid::value_type& point) {
        using Value = std::decay_t<decltype(field(point))>;
        std::ptrdiff_t idx = find_index(grid, point);
        if (idx < 0) return Value{ 0 };

        auto [left, right] = neighbor_indices(grid, idx);
        if (left < 0) return Value{ 0 };

        const auto& point_minus = grid[left];
        Value f_center = field(point);
        Value f_minus = field(point_minus);
        auto h_minus = metric(point, point_minus);
        return (f_center - f_minus) / h_minus;
    }

    template<typename Grid, typename Field, typename Metric>
    auto central_difference(const Grid& grid,
        const Field& field,
        const Metric& metric,
        const typename Grid::value_type& point) {
        using Value = std::decay_t<decltype(field(point))>;
        std::ptrdiff_t idx = find_index(grid, point);
        if (idx < 0) return Value{ 0 };

        auto [left, right] = neighbor_indices(grid, idx);
        if (left < 0 || right < 0) return Value{ 0 };

        const auto& point_minus = grid[left];
        const auto& point_plus = grid[right];
        Value f_plus = field(point_plus);
        Value f_minus = field(point_minus);
        auto h_plus = metric(point_plus, point);
        auto h_minus = metric(point, point_minus);
        return (f_plus - f_minus) / (h_plus + h_minus);
    }

    template<typename Grid, typename Field, typename Metric>
    auto laplacian_general(const Grid& grid,
        const Field& field,
        const Metric& metric,
        const typename Grid::value_type& point) {
        using Value = std::decay_t<decltype(field(point))>;
        std::ptrdiff_t idx = find_index(grid, point);
        if (idx < 0) return Value{ 0 };

        auto [left, right] = neighbor_indices(grid, idx);
        if (left < 0 || right < 0) return Value{ 0 };

        const auto& point_minus = grid[left];
        const auto& point_plus = grid[right];
        Value f_center = field(point);
        Value f_plus = field(point_plus);
        Value f_minus = field(point_minus);
        auto h_plus = metric(point_plus, point);
        auto h_minus = metric(point, point_minus);

        Value term = (f_plus - f_center) / h_plus - (f_center - f_minus) / h_minus;
        return (Value{ 2 } / (h_plus + h_minus)) * term;
    }

} // namespace delta::numerical