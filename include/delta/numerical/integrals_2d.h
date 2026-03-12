// include/delta/numerical/integrals_2d.h
#pragma once

#include <cmath>
#include "delta/numerical/cartesian_grid.h"
#include "delta/numerical/concepts.h"

namespace delta::numerical {

    // -------------------------------------------------------------------------
    // Квадратурные формулы для равномерных 2D сеток
    // -------------------------------------------------------------------------
    template<typename T, typename Func>
    T integral_2d_uniform(const std::vector<T>& x, const std::vector<T>& y, const Func& f) {
        if (x.size() < 2 || y.size() < 2) return T(0);
        T hx = x[1] - x[0];
        T hy = y[1] - y[0];
        T sum = T(0);
        for (const auto& xi : x)
            for (const auto& yj : y)
                sum += f(xi, yj);
        return sum * hx * hy;
    }

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
    // Геометрические величины и операторы для 2D декартовых сеток
    // -------------------------------------------------------------------------
    template<typename Grid2D>
    typename Grid2D::value_type cell_volume_2d(const Grid2D& grid,
        std::size_t i, std::size_t j,
        const typename Grid2D::scalar_type& hx,
        const typename Grid2D::scalar_type& hy) {
        using Value = typename Grid2D::value_type;
        Value vol = hx * hy;
        if (i == 0 || i == grid.x_grid().size() - 1) vol /= 2;
        if (j == 0 || j == grid.y_grid().size() - 1) vol /= 2;
        return vol;
    }

    template<typename Grid2D, typename Field, typename Metric>
    Eigen::Matrix<typename Field::value_type, 2, 1>
        gradient_2d(const Grid2D& grid, const Field& f,
            std::size_t i, std::size_t j,
            const typename Grid2D::scalar_type& hx,
            const typename Grid2D::scalar_type& hy) {
        using Scalar = typename Field::value_type;
        Scalar dfdx = 0, dfdy = 0;
        auto p = grid.point_at(i, j);

        if (i > 0 && i + 1 < grid.x_grid().size())
            dfdx = (f(grid.point_at(i + 1, j)) - f(grid.point_at(i - 1, j))) / (2 * hx);
        else if (i == 0)
            dfdx = (f(grid.point_at(i + 1, j)) - f(p)) / hx;
        else if (i == grid.x_grid().size() - 1)
            dfdx = (f(p) - f(grid.point_at(i - 1, j))) / hx;

        if (j > 0 && j + 1 < grid.y_grid().size())
            dfdy = (f(grid.point_at(i, j + 1)) - f(grid.point_at(i, j - 1))) / (2 * hy);
        else if (j == 0)
            dfdy = (f(grid.point_at(i, j + 1)) - f(p)) / hy;
        else if (j == grid.y_grid().size() - 1)
            dfdy = (f(p) - f(grid.point_at(i, j - 1))) / hy;

        return Eigen::Matrix<Scalar, 2, 1>(dfdx, dfdy);
    }

    // -------------------------------------------------------------------------
    // Проверка тождества суммирования по частям в 2D (только x-направление)
    // -------------------------------------------------------------------------
    template<typename Grid2D, typename Field, typename Metric>
    bool check_summation_by_parts_2d_x(const Grid2D& grid,
        const Field& f,
        const Field& g,
        const Metric& metric,
        double tolerance = 1e-12) {
        using Scalar = typename Field::value_type;
        const auto& xg = grid.x_grid();
        const auto& yg = grid.y_grid();
        Scalar hx = xg.step(), hy = yg.step();
        std::size_t nx = xg.size(), ny = yg.size();

        Scalar lhs = 0;          // ∫ (∂f/∂x) g dV
        Scalar rhs = 0;          // -∫ f (∂g/∂x) dV
        Scalar boundary = 0;     // ∮ f g n_x dS

        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                Scalar vol = cell_volume_2d(grid, i, j, hx, hy);
                auto grad_f = gradient_2d(grid, f, i, j, hx, hy);
                auto grad_g = gradient_2d(grid, g, i, j, hx, hy);

                lhs += grad_f.x() * g(grid.point_at(i, j)) * vol;
                rhs += f(grid.point_at(i, j)) * grad_g.x() * vol;
            }
        }

        Scalar edge_len = hy;
        for (std::size_t j = 0; j < ny; ++j) {
            auto p_left = grid.point_at(0, j);
            auto p_right = grid.point_at(nx - 1, j);
            boundary += f(p_left) * g(p_left) * (-edge_len);
            boundary += f(p_right) * g(p_right) * edge_len;
        }

        Scalar error = std::abs(lhs + rhs - boundary);
        return error <= tolerance;
    }

} // namespace delta::numerical