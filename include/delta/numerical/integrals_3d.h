// include/delta/numerical/integrals_3d.h
#pragma once

#include <cmath>
#include "delta/numerical/cartesian_grid.h"
#include "delta/numerical/concepts.h"

namespace delta::numerical {

    // -------------------------------------------------------------------------
    // Геометрические величины и операторы для 3D декартовых сеток
    // -------------------------------------------------------------------------
    template<typename Grid3D>
    typename Grid3D::value_type cell_volume_3d(const Grid3D& grid,
        std::size_t i, std::size_t j, std::size_t k,
        const typename Grid3D::scalar_type& hx,
        const typename Grid3D::scalar_type& hy,
        const typename Grid3D::scalar_type& hz) {
        using Value = typename Grid3D::value_type;
        Value vol = hx * hy * hz;
        if (i == 0 || i == grid.x_grid().size() - 1) vol /= 2;
        if (j == 0 || j == grid.y_grid().size() - 1) vol /= 2;
        if (k == 0 || k == grid.z_grid().size() - 1) vol /= 2;
        return vol;
    }

    template<typename Grid3D, typename Field, typename Metric>
    Eigen::Matrix<typename Field::value_type, 3, 1>
        gradient_3d(const Grid3D& grid, const Field& f,
            std::size_t i, std::size_t j, std::size_t k,
            const typename Grid3D::scalar_type& hx,
            const typename Grid3D::scalar_type& hy,
            const typename Grid3D::scalar_type& hz) {
        using Scalar = typename Field::value_type;
        Scalar dfdx = 0, dfdy = 0, dfdz = 0;
        auto p = grid.point_at(i, j, k);

        if (i > 0 && i + 1 < grid.x_grid().size())
            dfdx = (f(grid.point_at(i + 1, j, k)) - f(grid.point_at(i - 1, j, k))) / (2 * hx);
        else if (i == 0)
            dfdx = (f(grid.point_at(i + 1, j, k)) - f(p)) / hx;
        else if (i == grid.x_grid().size() - 1)
            dfdx = (f(p) - f(grid.point_at(i - 1, j, k))) / hx;

        if (j > 0 && j + 1 < grid.y_grid().size())
            dfdy = (f(grid.point_at(i, j + 1, k)) - f(grid.point_at(i, j - 1, k))) / (2 * hy);
        else if (j == 0)
            dfdy = (f(grid.point_at(i, j + 1, k)) - f(p)) / hy;
        else if (j == grid.y_grid().size() - 1)
            dfdy = (f(p) - f(grid.point_at(i, j - 1, k))) / hy;

        if (k > 0 && k + 1 < grid.z_grid().size())
            dfdz = (f(grid.point_at(i, j, k + 1)) - f(grid.point_at(i, j, k - 1))) / (2 * hz);
        else if (k == 0)
            dfdz = (f(grid.point_at(i, j, k + 1)) - f(p)) / hz;
        else if (k == grid.z_grid().size() - 1)
            dfdz = (f(p) - f(grid.point_at(i, j, k - 1))) / hz;

        return Eigen::Matrix<Scalar, 3, 1>(dfdx, dfdy, dfdz);
    }

    template<typename Grid3D, typename VecField, typename Metric>
    typename VecField::scalar_type
        divergence_3d(const Grid3D& grid, const VecField& v,
            std::size_t i, std::size_t j, std::size_t k,
            const typename Grid3D::scalar_type& hx,
            const typename Grid3D::scalar_type& hy,
            const typename Grid3D::scalar_type& hz) {
        using Scalar = typename VecField::scalar_type;
        Scalar dvx_dx = 0, dvy_dy = 0, dvz_dz = 0;
        auto p = grid.point_at(i, j, k);
        const auto& vec = v.at(p);

        if (i > 0 && i + 1 < grid.x_grid().size())
            dvx_dx = (v.at(grid.point_at(i + 1, j, k))(0) - v.at(grid.point_at(i - 1, j, k))(0)) / (2 * hx);
        else if (i == 0)
            dvx_dx = (v.at(grid.point_at(i + 1, j, k))(0) - vec(0)) / hx;
        else if (i == grid.x_grid().size() - 1)
            dvx_dx = (vec(0) - v.at(grid.point_at(i - 1, j, k))(0)) / hx;

        if (j > 0 && j + 1 < grid.y_grid().size())
            dvy_dy = (v.at(grid.point_at(i, j + 1, k))(1) - v.at(grid.point_at(i, j - 1, k))(1)) / (2 * hy);
        else if (j == 0)
            dvy_dy = (v.at(grid.point_at(i, j + 1, k))(1) - vec(1)) / hy;
        else if (j == grid.y_grid().size() - 1)
            dvy_dy = (vec(1) - v.at(grid.point_at(i, j - 1, k))(1)) / hy;

        if (k > 0 && k + 1 < grid.z_grid().size())
            dvz_dz = (v.at(grid.point_at(i, j, k + 1))(2) - v.at(grid.point_at(i, j, k - 1))(2)) / (2 * hz);
        else if (k == 0)
            dvz_dz = (v.at(grid.point_at(i, j, k + 1))(2) - vec(2)) / hz;
        else if (k == grid.z_grid().size() - 1)
            dvz_dz = (vec(2) - v.at(grid.point_at(i, j, k - 1))(2)) / hz;

        return dvx_dx + dvy_dy + dvz_dz;
    }

    // -------------------------------------------------------------------------
    // Проверка тождества суммирования по частям в 3D (x-направление)
    // -------------------------------------------------------------------------
    template<typename Grid3D, typename Field, typename Metric>
    bool check_summation_by_parts_3d_x(const Grid3D& grid,
        const Field& f,
        const Field& g,
        const Metric& metric,
        double tolerance = 1e-12) {
        using Scalar = typename Field::value_type;
        const auto& xg = grid.x_grid();
        const auto& yg = grid.y_grid();
        const auto& zg = grid.z_grid();
        Scalar hx = xg.step(), hy = yg.step(), hz = zg.step();
        std::size_t nx = xg.size(), ny = yg.size(), nz = zg.size();

        Scalar lhs = 0, rhs = 0, boundary = 0;

        for (std::size_t k = 0; k < nz; ++k) {
            for (std::size_t j = 0; j < ny; ++j) {
                for (std::size_t i = 0; i < nx; ++i) {
                    Scalar vol = cell_volume_3d(grid, i, j, k, hx, hy, hz);
                    auto grad_f = gradient_3d(grid, f, i, j, k, hx, hy, hz);
                    auto grad_g = gradient_3d(grid, g, i, j, k, hx, hy, hz);

                    lhs += grad_f.x() * g(grid.point_at(i, j, k)) * vol;
                    rhs += f(grid.point_at(i, j, k)) * grad_g.x() * vol;
                }
            }
        }

        Scalar face_area = hy * hz;
        for (std::size_t k = 0; k < nz; ++k) {
            for (std::size_t j = 0; j < ny; ++j) {
                auto p_left = grid.point_at(0, j, k);
                auto p_right = grid.point_at(nx - 1, j, k);
                boundary += f(p_left) * g(p_left) * (-face_area);
                boundary += f(p_right) * g(p_right) * face_area;
            }
        }

        Scalar error = std::abs(lhs + rhs - boundary);
        return error <= tolerance;
    }

} // namespace delta::numerical