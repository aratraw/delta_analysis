// include/delta/numerical/grid_curl.h
#pragma once

#include "delta/geometry/tensor_field.h"
#include "delta/numerical/cartesian_grid.h"
#include "delta/numerical/concepts.h"

namespace delta::numerical {

    template<typename Grid, typename VecField, typename Metric>
        requires CartesianGrid3D<Grid>
    auto discrete_curl(const Grid& grid,
        const VecField& vec_field,
        const Metric& metric) {
        using Scalar = typename VecField::scalar_type;
        using Point = typename Grid::value_type;
        geometry::TensorField<Point, Scalar, 1, 3> result;

        const auto& xg = grid.x_grid();
        const auto& yg = grid.y_grid();
        const auto& zg = grid.z_grid();

        Scalar hx = xg.step();
        Scalar hy = yg.step();
        Scalar hz = zg.step();

        std::size_t nx = xg.size();
        std::size_t ny = yg.size();
        std::size_t nz = zg.size();

        for (std::size_t i = 0; i < nx; ++i) {
            for (std::size_t j = 0; j < ny; ++j) {
                for (std::size_t k = 0; k < nz; ++k) {
                    Point p = grid.point_at(i, j, k);
                    const auto& v = vec_field.at(p);

                    Scalar curl_x = 0, curl_y = 0, curl_z = 0;

                    if (j > 0 && j + 1 < ny && k > 0 && k + 1 < nz) {
                        Scalar dvz_dy = (vec_field.at(grid.point_at(i, j + 1, k))(2) -
                            vec_field.at(grid.point_at(i, j - 1, k))(2)) / (2 * hy);
                        Scalar dvy_dz = (vec_field.at(grid.point_at(i, j, k + 1))(1) -
                            vec_field.at(grid.point_at(i, j, k - 1))(1)) / (2 * hz);
                        curl_x = dvz_dy - dvy_dz;
                    }

                    if (i > 0 && i + 1 < nx && k > 0 && k + 1 < nz) {
                        Scalar dvx_dz = (vec_field.at(grid.point_at(i, j, k + 1))(0) -
                            vec_field.at(grid.point_at(i, j, k - 1))(0)) / (2 * hz);
                        Scalar dvz_dx = (vec_field.at(grid.point_at(i + 1, j, k))(2) -
                            vec_field.at(grid.point_at(i - 1, j, k))(2)) / (2 * hx);
                        curl_y = dvx_dz - dvz_dx;
                    }

                    if (i > 0 && i + 1 < nx && j > 0 && j + 1 < ny) {
                        Scalar dvy_dx = (vec_field.at(grid.point_at(i + 1, j, k))(1) -
                            vec_field.at(grid.point_at(i - 1, j, k))(1)) / (2 * hx);
                        Scalar dvx_dy = (vec_field.at(grid.point_at(i, j + 1, k))(0) -
                            vec_field.at(grid.point_at(i, j - 1, k))(0)) / (2 * hy);
                        curl_z = dvy_dx - dvx_dy;
                    }

                    Point curl_vec(curl_x, curl_y, curl_z);
                    result.set(p, curl_vec);
                }
            }
        }
        return result;
    }

} // namespace delta::numerical