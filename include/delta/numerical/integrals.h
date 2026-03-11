// include/delta/numerical/integrals.h
#pragma once

#include <functional>
#include <vector>
#include <cmath>
#include "delta/core/grid_concept.h"
#include "delta/core/operational_function.h"
#include "delta/core/regulative_idea.h"
#include "grid_differences.h"  // для neighbor_indices, laplacian_general

namespace delta::numerical {

    // 1D uniform grid (left Riemann sum)
    template<typename T, typename Func>
    T integral_1d_uniform(const std::vector<T>& x, const Func& f) {
        if (x.size() < 2) return T(0);
        T h = x[1] - x[0];
        T sum = T(0);
        for (const auto& xi : x) {
            sum += f(xi);
        }
        return sum * h;
    }

    // 1D trapezoidal rule
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

    // 2D uniform grid (product grid, left Riemann sum)
    template<typename T, typename Func>
    T integral_2d_uniform(const std::vector<T>& x, const std::vector<T>& y, const Func& f) {
        if (x.size() < 2 || y.size() < 2) return T(0);
        T hx = x[1] - x[0];
        T hy = y[1] - y[0];
        T sum = T(0);
        for (const auto& xi : x) {
            for (const auto& yj : y) {
                sum += f(xi, yj);
            }
        }
        return sum * hx * hy;
    }

    // 2D non-uniform rectangular grid (left Riemann sum on each cell)
    template<typename T, typename Func>
    T integral_2d_left(const std::vector<T>& x, const std::vector<T>& y, const Func& f) {
        if (x.size() < 2 || y.size() < 2) return T(0);
        T sum = T(0);
        for (std::size_t i = 0; i < x.size() - 1; ++i) {
            T hx = x[i + 1] - x[i];
            for (std::size_t j = 0; j < y.size() - 1; ++j) {
                T hy = y[j + 1] - y[j];
                sum += f(x[i], y[j]) * hx * hy;
            }
        }
        return sum;
    }

    // -------------------------------------------------------------------------
    // Дискретный интеграл с весовыми объёмами (для произвольной упорядоченной сетки)
    // -------------------------------------------------------------------------

    /**
     * @brief Вычислить объём ячейки, ассоциированной с точкой в координатной сетке.
     *
     * Для одномерной сетки: объём = среднее арифметическое шагов вперёд и назад,
     * если точка не граничная. Для граничной – шаг в одну сторону.
     *
     * @param grid   Упорядоченная сетка.
     * @param idx    Индекс точки.
     * @param metric Метрика.
     * @return Объём (скаляр).
     */
    template<typename Grid, typename Metric>
    auto cell_volume(const Grid& grid, std::size_t idx, const Metric& metric) {
        using Value = typename Grid::value_type;
        std::ptrdiff_t i = static_cast<std::ptrdiff_t>(idx);
        auto [left, right] = neighbor_indices(grid, i);

        if (left < 0 && right < 0) return Value{ 0 }; // одна точка
        if (left < 0) {
            // левая граница: объём = шаг вправо
            const auto& point = grid[i];
            const auto& point_right = grid[static_cast<std::size_t>(right)];
            return metric(point_right, point);
        }
        if (right < 0) {
            // правая граница
            const auto& point = grid[i];
            const auto& point_left = grid[static_cast<std::size_t>(left)];
            return metric(point, point_left);
        }
        // внутренняя точка
        const auto& point = grid[i];
        const auto& point_left = grid[static_cast<std::size_t>(left)];
        const auto& point_right = grid[static_cast<std::size_t>(right)];
        auto h_left = metric(point, point_left);
        auto h_right = metric(point_right, point);
        return (h_left + h_right) / Value{ 2 };
    }

    /**
     * @brief Вычислить дискретный интеграл функции по области, покрытой сеткой.
     *
     * ∫ f dV ≈ Σ_i f(x_i) * V_i, где V_i – объём ячейки вокруг x_i.
     *
     * @param grid   Сетка.
     * @param field  Функция (операциональная).
     * @param metric Метрика.
     * @return Значение интеграла.
     */
    template<typename Grid, typename Field, typename Metric>
    auto integral(const Grid& grid,
        const Field& field,
        const Metric& metric) {
        using Value = decltype(field(grid[0]));
        Value sum{ 0 };
        for (std::size_t i = 0; i < grid.size(); ++i) {
            auto vol = cell_volume(grid, i, metric);
            sum += field(grid[i]) * vol;
        }
        return sum;
    }

    // -------------------------------------------------------------------------
    // Суммирование по частям (1D)
    // -------------------------------------------------------------------------

    /**
     * @brief Проверить тождество суммирования по частям для двух функций f и g.
     *
     * Тождество: Σ_{k=0}^{K-1} f_k (g_{k+1} - g_k) = - Σ_{k=1}^{K} (f_k - f_{k-1}) g_k + f_K g_{K+1} - f_0 g_0.
     * Здесь индексы соответствуют точкам сетки, и g_{K+1} – значение за правой границей (может быть задано).
     *
     * @param grid        Сетка.
     * @param f           Функция f.
     * @param g           Функция g.
     * @param metric      Метрика (не используется, но оставлена для единообразия).
     * @param g_boundary_right Значение g справа от последней точки (если нужно).
     * @param tol         Допуск.
     * @return true, если левая и правая части совпадают.
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

        // Левая часть: Σ f_i (g_{i+1} - g_i)
        for (std::size_t i = 0; i < grid.size() - 1; ++i) {
            lhs += f(grid[i]) * (g(grid[i + 1]) - g(grid[i]));
        }

        // Правая часть: - Σ (f_i - f_{i-1}) g_i + f_{K} g_{K+1} - f_0 g_0
        for (std::size_t i = 1; i < grid.size(); ++i) {
            rhs -= (f(grid[i]) - f(grid[i - 1])) * g(grid[i]);
        }
        rhs += f(grid[grid.size() - 1]) * g_boundary_right - f(grid[0]) * g(grid[0]);

        return std::abs(lhs - rhs) < tol;
    }

    // -------------------------------------------------------------------------
    // Дискретные тождества Грина (1D)
    // -------------------------------------------------------------------------

    /**
     * @brief Проверить первое тождество Грина: ∫ f Δg dV = -∫ f' g' dV + f g'|_boundary.
     *
     * @param grid   Сетка.
     * @param f      Функция f.
     * @param g      Функция g.
     * @param metric Метрика.
     * @param tol    Допуск.
     * @return true, если тождество выполняется.
     */
    template<typename Grid, typename Field, typename Metric>
    bool check_green_first(const Grid& grid,
        const Field& f,
        const Field& g,
        const Metric& metric,
        double tol = 1e-12) {
        using Value = typename Field::value_type;

        // Вычисляем левую часть: ∫ f Δg
        Value lhs{ 0 };
        for (std::size_t i = 0; i < grid.size(); ++i) {
            Value lap_g = laplacian_general(grid, g, metric, grid[i]);
            Value vol = cell_volume(grid, i, metric);
            lhs += f(grid[i]) * lap_g * vol;
        }

        // Вычисляем правую часть: -∫ f' g' dV + граничные члены
        // Для этого нужны градиенты f и g.
        // Пока не реализовано – возвращаем false.
        (void)lhs; // заглушка
        return false;
    }

} // namespace delta::numerical