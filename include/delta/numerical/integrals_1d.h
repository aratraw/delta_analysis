// include/delta/numerical/integrals_1d.h
#pragma once

#include <vector>
#include <cmath>
#include "delta/core/grid_concept.h"
#include "delta/core/regulative_idea.h"
#include "delta/numerical/grid_ops_1d.h"  // для neighbor_indices, laplacian_general

namespace delta::numerical {

    // -------------------------------------------------------------------------
    // Квадратурные формулы для равномерных сеток
    // -------------------------------------------------------------------------
    template<typename T, typename Func>
    T integral_1d_uniform(const std::vector<T>& x, const Func& f) {
        if (x.size() < 2) return T(0);
        T h = x[1] - x[0];
        T sum = T(0);
        for (const auto& xi : x) sum += f(xi);
        return sum * h;
    }

    template<typename T, typename Func>
    T integral_1d_trapezoidal(const std::vector<T>& x, const Func& f) {
        if (x.size() < 2) return T(0);
        T sum = T(0);
        for (std::size_t i = 0; i < x.size() - 1; ++i) {
            T h = x[i + 1] - x[i];
            sum += (f(x[i]) + f(x[i + 1])) * h / T(2);
        }
        return sum;
    }

    // -------------------------------------------------------------------------
    // Универсальные одномерные функции (для произвольной упорядоченной сетки)
    // -------------------------------------------------------------------------
    /**
     * @brief Вычислить объём ячейки, ассоциированной с точкой в координатной сетке (1D).
     */
    template<typename Grid, typename Metric>
    auto cell_volume(const Grid& grid, std::size_t idx, const Metric& metric) {
        using Value = typename Grid::value_type;
        std::ptrdiff_t i = static_cast<std::ptrdiff_t>(idx);
        auto [left, right] = neighbor_indices(grid, i);

        if (left < 0 && right < 0) return Value{ 0 };
        if (left < 0) {
            const auto& point = grid[i];
            const auto& point_right = grid[static_cast<std::size_t>(right)];
            return metric(point_right, point);
        }
        if (right < 0) {
            const auto& point = grid[i];
            const auto& point_left = grid[static_cast<std::size_t>(left)];
            return metric(point, point_left);
        }
        const auto& point = grid[i];
        const auto& point_left = grid[static_cast<std::size_t>(left)];
        const auto& point_right = grid[static_cast<std::size_t>(right)];
        auto h_left = metric(point, point_left);
        auto h_right = metric(point_right, point);
        return (h_left + h_right) / Value{ 2 };
    }

    /**
     * @brief Вычислить дискретный интеграл функции по области, покрытой сеткой (1D).
     */
    template<typename Grid, typename Field, typename Metric>
    auto integral(const Grid& grid, const Field& field, const Metric& metric) {
        using Value = decltype(field(grid[0]));
        Value sum{ 0 };
        for (std::size_t i = 0; i < grid.size(); ++i) {
            auto vol = cell_volume(grid, i, metric);
            sum += field(grid[i]) * vol;
        }
        return sum;
    }

    /**
     * @brief Проверить тождество суммирования по частям для двух функций f и g (1D).
     */
    template<typename Grid, typename Field, typename Metric>
    bool check_summation_by_parts(const Grid& grid,
        const Field& f,
        const Field& g,
        const Metric& metric,
        const typename Field::value_type& g_boundary_right,
        double tol = 1e-12) {
        using Value = typename Field::value_type;
        Value lhs{ 0 }, rhs{ 0 };

        for (std::size_t i = 0; i < grid.size() - 1; ++i) {
            lhs += f(grid[i]) * (g(grid[i + 1]) - g(grid[i]));
        }

        for (std::size_t i = 1; i < grid.size(); ++i) {
            rhs -= (f(grid[i]) - f(grid[i - 1])) * g(grid[i]);
        }
        rhs += f(grid[grid.size() - 1]) * g_boundary_right - f(grid[0]) * g(grid[0]);

        return std::abs(lhs - rhs) < tol;
    }

    /**
     * @brief Проверить первое тождество Грина: ∫ f Δg dV = -∫ f' g' dV + f g'|_boundary (1D).
     */
    template<typename Grid, typename Field, typename Metric>
    bool check_green_first(const Grid& grid,
        const Field& f,
        const Field& g,
        const Metric& metric,
        double tol = 1e-12) {
        using Value = typename Field::value_type;

        Value lhs{ 0 };
        for (std::size_t i = 0; i < grid.size(); ++i) {
            Value lap_g = laplacian_general(grid, g, metric, grid[i]);
            Value vol = cell_volume(grid, i, metric);
            lhs += f(grid[i]) * lap_g * vol;
        }

        // Правая часть (не реализована) – заглушка
        (void)lhs;
        return false;
    }

} // namespace delta::numerical