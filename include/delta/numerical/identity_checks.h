// include/delta/numerical/identity_checks.h
#pragma once

#include "delta/geometry/tensor_field.h"
#include "delta/numerical/cartesian_grid.h"
#include "delta/numerical/grid_curl.h"
#include "delta/numerical/integrals_3d.h"
#include "delta/numerical/concepts.h"
#include <cmath>

namespace delta::numerical {

    // -----------------------------------------------------------------------------
    // Проверка curl grad f = 0 для 3D декартовой сетки
    // -----------------------------------------------------------------------------
    template<typename Grid, typename ScalarField, typename Metric>
        requires CartesianGrid3D<Grid>&&
    IsMetric<Metric, typename Grid::value_type, typename Grid::scalar_type>
        bool check_curl_grad_zero(const Grid& grid,
            const ScalarField& f,
            const Metric& metric,
            double tolerance = 1e-12) {
        using Scalar = typename ScalarField::value_type;
        using Point = typename Grid::value_type;
        geometry::TensorField<Point, Scalar, 1, 3> grad;

        const auto& xg = grid.x_grid();
        const auto& yg = grid.y_grid();
        const auto& zg = grid.z_grid();
        Scalar hx = xg.step(), hy = yg.step(), hz = zg.step();
        std::size_t nx = xg.size(), ny = yg.size(), nz = zg.size();

        for (std::size_t i = 0; i < nx; ++i) {
            for (std::size_t j = 0; j < ny; ++j) {
                for (std::size_t k = 0; k < nz; ++k) {
                    auto p = grid.point_at(i, j, k);
                    Scalar dfdx = 0, dfdy = 0, dfdz = 0;
                    if (i > 0 && i + 1 < nx)
                        dfdx = (f(grid.point_at(i + 1, j, k)) - f(grid.point_at(i - 1, j, k))) / (2 * hx);
                    else if (i == 0 && i + 1 < nx)
                        dfdx = (f(grid.point_at(i + 1, j, k)) - f(p)) / hx;
                    else if (i == nx - 1 && i > 0)
                        dfdx = (f(p) - f(grid.point_at(i - 1, j, k))) / hx;

                    if (j > 0 && j + 1 < ny)
                        dfdy = (f(grid.point_at(i, j + 1, k)) - f(grid.point_at(i, j - 1, k))) / (2 * hy);
                    else if (j == 0 && j + 1 < ny)
                        dfdy = (f(grid.point_at(i, j + 1, k)) - f(p)) / hy;
                    else if (j == ny - 1 && j > 0)
                        dfdy = (f(p) - f(grid.point_at(i, j - 1, k))) / hy;

                    if (k > 0 && k + 1 < nz)
                        dfdz = (f(grid.point_at(i, j, k + 1)) - f(grid.point_at(i, j, k - 1))) / (2 * hz);
                    else if (k == 0 && k + 1 < nz)
                        dfdz = (f(grid.point_at(i, j, k + 1)) - f(p)) / hz;
                    else if (k == nz - 1 && k > 0)
                        dfdz = (f(p) - f(grid.point_at(i, j, k - 1))) / hz;

                    grad.set(p, Eigen::Matrix<Scalar, 3, 1>(dfdx, dfdy, dfdz));
                }
            }
        }

        auto curl_grad = discrete_curl(grid, grad, metric);

        double max_error = 0.0;
        for (const auto& [p, vec] : curl_grad) {
            double err = std::sqrt(vec.squaredNorm());
            if (err > max_error) max_error = err;
        }
        return max_error <= tolerance;
    }

    // -----------------------------------------------------------------------------
    // Проверка div curl v = 0 для 3D декартовой сетки
    // -----------------------------------------------------------------------------
    template<typename Grid, typename VecField, typename Metric>
        requires CartesianGrid3D<Grid>&&
    IsMetric<Metric, typename Grid::value_type, typename Grid::scalar_type>
        bool check_div_curl_zero(const Grid& grid,
            const VecField& v,
            const Metric& metric,
            double tolerance = 1e-12) {
        using Scalar = typename VecField::scalar_type;
        auto curl_v = discrete_curl(grid, v, metric);

        const auto& xg = grid.x_grid();
        const auto& yg = grid.y_grid();
        const auto& zg = grid.z_grid();
        Scalar hx = xg.step(), hy = yg.step(), hz = zg.step();
        std::size_t nx = xg.size(), ny = yg.size(), nz = zg.size();

        double max_error = 0.0;
        for (std::size_t i = 0; i < nx; ++i) {
            for (std::size_t j = 0; j < ny; ++j) {
                for (std::size_t k = 0; k < nz; ++k) {
                    auto p = grid.point_at(i, j, k);
                    const auto& w = curl_v.at(p);

                    Scalar dwx_dx = 0, dwy_dy = 0, dwz_dz = 0;
                    if (i > 0 && i + 1 < nx)
                        dwx_dx = (curl_v.at(grid.point_at(i + 1, j, k))(0) - curl_v.at(grid.point_at(i - 1, j, k))(0)) / (2 * hx);
                    else if (i == 0 && i + 1 < nx)
                        dwx_dx = (curl_v.at(grid.point_at(i + 1, j, k))(0) - w(0)) / hx;
                    else if (i == nx - 1 && i > 0)
                        dwx_dx = (w(0) - curl_v.at(grid.point_at(i - 1, j, k))(0)) / hx;

                    if (j > 0 && j + 1 < ny)
                        dwy_dy = (curl_v.at(grid.point_at(i, j + 1, k))(1) - curl_v.at(grid.point_at(i, j - 1, k))(1)) / (2 * hy);
                    else if (j == 0 && j + 1 < ny)
                        dwy_dy = (curl_v.at(grid.point_at(i, j + 1, k))(1) - w(1)) / hy;
                    else if (j == ny - 1 && j > 0)
                        dwy_dy = (w(1) - curl_v.at(grid.point_at(i, j - 1, k))(1)) / hy;

                    if (k > 0 && k + 1 < nz)
                        dwz_dz = (curl_v.at(grid.point_at(i, j, k + 1))(2) - curl_v.at(grid.point_at(i, j, k - 1))(2)) / (2 * hz);
                    else if (k == 0 && k + 1 < nz)
                        dwz_dz = (curl_v.at(grid.point_at(i, j, k + 1))(2) - w(2)) / hz;
                    else if (k == nz - 1 && k > 0)
                        dwz_dz = (w(2) - curl_v.at(grid.point_at(i, j, k - 1))(2)) / hz;

                    Scalar div = dwx_dx + dwy_dy + dwz_dz;
                    double err = std::abs(div);
                    if (err > max_error) max_error = err;
                }
            }
        }
        return max_error <= tolerance;
    }

} // namespace delta::numerical