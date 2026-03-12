// include/delta/numerical/integrals_green.h
#pragma once

#include <cmath>
#include "delta/numerical/cartesian_grid.h"
#include "delta/numerical/concepts.h"
#include "integrals_2d.h"   // для gradient_2d, cell_volume_2d
#include "integrals_3d.h"   // для gradient_3d, cell_volume_3d, divergence_3d

namespace delta::numerical {

    // -------------------------------------------------------------------------
    // Первое тождество Грина в 2D
    // -------------------------------------------------------------------------
    template<typename Grid2D, typename Field, typename Metric>
    bool check_green_first_2d(const Grid2D& grid,
        const Field& f,
        const Field& g,
        const Metric& metric,
        double tolerance = 1e-12) {
        using Scalar = typename Field::value_type;
        const auto& xg = grid.x_grid();
        const auto& yg = grid.y_grid();
        Scalar hx = xg.step(), hy = yg.step();
        std::size_t nx = xg.size(), ny = yg.size();

        Scalar lhs = 0;          // ∫ f Δg dV
        Scalar rhs_integral = 0; // -∫ ∇f·∇g dV
        Scalar boundary = 0;     // ∮ f (∇g·n) dS

        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                Scalar vol = cell_volume_2d(grid, i, j, hx, hy);
                auto p = grid.point_at(i, j);

                Scalar lapl_g = 0;
                if (i > 0 && i + 1 < nx && j > 0 && j + 1 < ny) {
                    lapl_g = (g(grid.point_at(i + 1, j)) + g(grid.point_at(i - 1, j)) - 2 * g(p)) / (hx * hx)
                        + (g(grid.point_at(i, j + 1)) + g(grid.point_at(i, j - 1)) - 2 * g(p)) / (hy * hy);
                }
                lhs += f(p) * lapl_g * vol;

                auto grad_f = gradient_2d(grid, f, i, j, hx, hy);
                auto grad_g = gradient_2d(grid, g, i, j, hx, hy);
                rhs_integral += grad_f.dot(grad_g) * vol;
            }
        }

        // Граничные вклады
        // Нижняя и верхняя границы (y = const)
        for (std::size_t i = 0; i < nx; ++i) {
            auto p_down = grid.point_at(i, 0);
            auto grad_down = gradient_2d(grid, g, i, 0, hx, hy);
            boundary += f(p_down) * (-grad_down.y()) * hx;

            auto p_up = grid.point_at(i, ny - 1);
            auto grad_up = gradient_2d(grid, g, i, ny - 1, hx, hy);
            boundary += f(p_up) * (grad_up.y()) * hx;
        }
        // Левая и правая границы (x = const)
        for (std::size_t j = 0; j < ny; ++j) {
            auto p_left = grid.point_at(0, j);
            auto grad_left = gradient_2d(grid, g, 0, j, hx, hy);
            boundary += f(p_left) * (-grad_left.x()) * hy;

            auto p_right = grid.point_at(nx - 1, j);
            auto grad_right = gradient_2d(grid, g, nx - 1, j, hx, hy);
            boundary += f(p_right) * (grad_right.x()) * hy;
        }

        Scalar error = std::abs(lhs + rhs_integral - boundary);
        return error <= tolerance;
    }

    // -------------------------------------------------------------------------
    // Второе тождество Грина в 2D: ∫ (u Δv - v Δu) dV = ∮ (u ∇v - v ∇u)·n dS
    // -------------------------------------------------------------------------
    template<typename Grid2D, typename Field, typename Metric>
    bool check_green_second_2d(const Grid2D& grid,
        const Field& u,
        const Field& v,
        const Metric& metric,
        double tolerance = 1e-12) {
        using Scalar = typename Field::value_type;
        const auto& xg = grid.x_grid();
        const auto& yg = grid.y_grid();
        Scalar hx = xg.step(), hy = yg.step();
        std::size_t nx = xg.size(), ny = yg.size();

        Scalar lhs = 0;          // ∫ (u Δv - v Δu) dV
        Scalar boundary = 0;     // ∮ (u ∇v - v ∇u)·n dS

        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                Scalar vol = cell_volume_2d(grid, i, j, hx, hy);
                auto p = grid.point_at(i, j);

                Scalar lapl_u = 0, lapl_v = 0;
                if (i > 0 && i + 1 < nx && j > 0 && j + 1 < ny) {
                    lapl_u = (u(grid.point_at(i + 1, j)) + u(grid.point_at(i - 1, j)) - 2 * u(p)) / (hx * hx)
                        + (u(grid.point_at(i, j + 1)) + u(grid.point_at(i, j - 1)) - 2 * u(p)) / (hy * hy);
                    lapl_v = (v(grid.point_at(i + 1, j)) + v(grid.point_at(i - 1, j)) - 2 * v(p)) / (hx * hx)
                        + (v(grid.point_at(i, j + 1)) + v(grid.point_at(i, j - 1)) - 2 * v(p)) / (hy * hy);
                }
                lhs += (u(p) * lapl_v - v(p) * lapl_u) * vol;
            }
        }

        // Граничные вклады: сумма по всем граням (u ∇v - v ∇u)·n * длина грани
        // Нижняя грань (y = 0)
        for (std::size_t i = 0; i < nx; ++i) {
            auto p = grid.point_at(i, 0);
            auto grad_u = gradient_2d(grid, u, i, 0, hx, hy);
            auto grad_v = gradient_2d(grid, v, i, 0, hx, hy);
            // нормаль направлена вниз: (0, -1)
            boundary += (u(p) * (-grad_v.y()) - v(p) * (-grad_u.y())) * hx;
        }
        // Верхняя грань (y = ny-1)
        for (std::size_t i = 0; i < nx; ++i) {
            auto p = grid.point_at(i, ny - 1);
            auto grad_u = gradient_2d(grid, u, i, ny - 1, hx, hy);
            auto grad_v = gradient_2d(grid, v, i, ny - 1, hx, hy);
            // нормаль вверх: (0, 1)
            boundary += (u(p) * (grad_v.y()) - v(p) * (grad_u.y())) * hx;
        }
        // Левая грань (x = 0)
        for (std::size_t j = 0; j < ny; ++j) {
            auto p = grid.point_at(0, j);
            auto grad_u = gradient_2d(grid, u, 0, j, hx, hy);
            auto grad_v = gradient_2d(grid, v, 0, j, hx, hy);
            // нормаль влево: (-1, 0)
            boundary += (u(p) * (-grad_v.x()) - v(p) * (-grad_u.x())) * hy;
        }
        // Правая грань (x = nx-1)
        for (std::size_t j = 0; j < ny; ++j) {
            auto p = grid.point_at(nx - 1, j);
            auto grad_u = gradient_2d(grid, u, nx - 1, j, hx, hy);
            auto grad_v = gradient_2d(grid, v, nx - 1, j, hx, hy);
            // нормаль вправо: (1, 0)
            boundary += (u(p) * (grad_v.x()) - v(p) * (grad_u.x())) * hy;
        }

        Scalar error = std::abs(lhs - boundary);
        return error <= tolerance;
    }

    // -------------------------------------------------------------------------
    // Первое тождество Грина в 3D
    // -------------------------------------------------------------------------
    template<typename Grid3D, typename Field, typename Metric>
    bool check_green_first_3d(const Grid3D& grid,
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

        Scalar lhs = 0, rhs_integral = 0, boundary = 0;

        for (std::size_t k = 0; k < nz; ++k) {
            for (std::size_t j = 0; j < ny; ++j) {
                for (std::size_t i = 0; i < nx; ++i) {
                    Scalar vol = cell_volume_3d(grid, i, j, k, hx, hy, hz);
                    auto p = grid.point_at(i, j, k);

                    Scalar lapl_g = 0;
                    if (i > 0 && i + 1 < nx && j > 0 && j + 1 < ny && k > 0 && k + 1 < nz) {
                        lapl_g = (g(grid.point_at(i + 1, j, k)) + g(grid.point_at(i - 1, j, k)) - 2 * g(p)) / (hx * hx)
                            + (g(grid.point_at(i, j + 1, k)) + g(grid.point_at(i, j - 1, k)) - 2 * g(p)) / (hy * hy)
                            + (g(grid.point_at(i, j, k + 1)) + g(grid.point_at(i, j, k - 1)) - 2 * g(p)) / (hz * hz);
                    }
                    lhs += f(p) * lapl_g * vol;

                    auto grad_f = gradient_3d(grid, f, i, j, k, hx, hy, hz);
                    auto grad_g = gradient_3d(grid, g, i, j, k, hx, hy, hz);
                    rhs_integral += grad_f.dot(grad_g) * vol;
                }
            }
        }

        // Граничные вклады по шести граням
        // x = 0
        for (std::size_t k = 0; k < nz; ++k)
            for (std::size_t j = 0; j < ny; ++j) {
                auto p = grid.point_at(0, j, k);
                auto grad = gradient_3d(grid, g, 0, j, k, hx, hy, hz);
                boundary += f(p) * (-grad.x()) * hy * hz;
            }
        // x = nx-1
        for (std::size_t k = 0; k < nz; ++k)
            for (std::size_t j = 0; j < ny; ++j) {
                auto p = grid.point_at(nx - 1, j, k);
                auto grad = gradient_3d(grid, g, nx - 1, j, k, hx, hy, hz);
                boundary += f(p) * (grad.x()) * hy * hz;
            }
        // y = 0
        for (std::size_t k = 0; k < nz; ++k)
            for (std::size_t i = 0; i < nx; ++i) {
                auto p = grid.point_at(i, 0, k);
                auto grad = gradient_3d(grid, g, i, 0, k, hx, hy, hz);
                boundary += f(p) * (-grad.y()) * hx * hz;
            }
        // y = ny-1
        for (std::size_t k = 0; k < nz; ++k)
            for (std::size_t i = 0; i < nx; ++i) {
                auto p = grid.point_at(i, ny - 1, k);
                auto grad = gradient_3d(grid, g, i, ny - 1, k, hx, hy, hz);
                boundary += f(p) * (grad.y()) * hx * hz;
            }
        // z = 0
        for (std::size_t j = 0; j < ny; ++j)
            for (std::size_t i = 0; i < nx; ++i) {
                auto p = grid.point_at(i, j, 0);
                auto grad = gradient_3d(grid, g, i, j, 0, hx, hy, hz);
                boundary += f(p) * (-grad.z()) * hx * hy;
            }
        // z = nz-1
        for (std::size_t j = 0; j < ny; ++j)
            for (std::size_t i = 0; i < nx; ++i) {
                auto p = grid.point_at(i, j, nz - 1);
                auto grad = gradient_3d(grid, g, i, j, nz - 1, hx, hy, hz);
                boundary += f(p) * (grad.z()) * hx * hy;
            }

        Scalar error = std::abs(lhs + rhs_integral - boundary);
        return error <= tolerance;
    }

    // -------------------------------------------------------------------------
    // Второе тождество Грина в 3D
    // -------------------------------------------------------------------------
    template<typename Grid3D, typename Field, typename Metric>
    bool check_green_second_3d(const Grid3D& grid,
        const Field& u,
        const Field& v,
        const Metric& metric,
        double tolerance = 1e-12) {
        using Scalar = typename Field::value_type;
        const auto& xg = grid.x_grid();
        const auto& yg = grid.y_grid();
        const auto& zg = grid.z_grid();
        Scalar hx = xg.step(), hy = yg.step(), hz = zg.step();
        std::size_t nx = xg.size(), ny = yg.size(), nz = zg.size();

        Scalar lhs = 0;
        Scalar boundary = 0;

        for (std::size_t k = 0; k < nz; ++k) {
            for (std::size_t j = 0; j < ny; ++j) {
                for (std::size_t i = 0; i < nx; ++i) {
                    Scalar vol = cell_volume_3d(grid, i, j, k, hx, hy, hz);
                    auto p = grid.point_at(i, j, k);

                    Scalar lapl_u = 0, lapl_v = 0;
                    if (i > 0 && i + 1 < nx && j > 0 && j + 1 < ny && k > 0 && k + 1 < nz) {
                        lapl_u = (u(grid.point_at(i + 1, j, k)) + u(grid.point_at(i - 1, j, k)) - 2 * u(p)) / (hx * hx)
                            + (u(grid.point_at(i, j + 1, k)) + u(grid.point_at(i, j - 1, k)) - 2 * u(p)) / (hy * hy)
                            + (u(grid.point_at(i, j, k + 1)) + u(grid.point_at(i, j, k - 1)) - 2 * u(p)) / (hz * hz);
                        lapl_v = (v(grid.point_at(i + 1, j, k)) + v(grid.point_at(i - 1, j, k)) - 2 * v(p)) / (hx * hx)
                            + (v(grid.point_at(i, j + 1, k)) + v(grid.point_at(i, j - 1, k)) - 2 * v(p)) / (hy * hy)
                            + (v(grid.point_at(i, j, k + 1)) + v(grid.point_at(i, j, k - 1)) - 2 * v(p)) / (hz * hz);
                    }
                    lhs += (u(p) * lapl_v - v(p) * lapl_u) * vol;
                }
            }
        }

        // Граничные вклады: интеграл по поверхности (u ∇v - v ∇u)·n dS
        // x = 0
        for (std::size_t k = 0; k < nz; ++k)
            for (std::size_t j = 0; j < ny; ++j) {
                auto p = grid.point_at(0, j, k);
                auto grad_u = gradient_3d(grid, u, 0, j, k, hx, hy, hz);
                auto grad_v = gradient_3d(grid, v, 0, j, k, hx, hy, hz);
                boundary += (u(p) * (-grad_v.x()) - v(p) * (-grad_u.x())) * hy * hz;
            }
        // x = nx-1
        for (std::size_t k = 0; k < nz; ++k)
            for (std::size_t j = 0; j < ny; ++j) {
                auto p = grid.point_at(nx - 1, j, k);
                auto grad_u = gradient_3d(grid, u, nx - 1, j, k, hx, hy, hz);
                auto grad_v = gradient_3d(grid, v, nx - 1, j, k, hx, hy, hz);
                boundary += (u(p) * (grad_v.x()) - v(p) * (grad_u.x())) * hy * hz;
            }
        // y = 0
        for (std::size_t k = 0; k < nz; ++k)
            for (std::size_t i = 0; i < nx; ++i) {
                auto p = grid.point_at(i, 0, k);
                auto grad_u = gradient_3d(grid, u, i, 0, k, hx, hy, hz);
                auto grad_v = gradient_3d(grid, v, i, 0, k, hx, hy, hz);
                boundary += (u(p) * (-grad_v.y()) - v(p) * (-grad_u.y())) * hx * hz;
            }
        // y = ny-1
        for (std::size_t k = 0; k < nz; ++k)
            for (std::size_t i = 0; i < nx; ++i) {
                auto p = grid.point_at(i, ny - 1, k);
                auto grad_u = gradient_3d(grid, u, i, ny - 1, k, hx, hy, hz);
                auto grad_v = gradient_3d(grid, v, i, ny - 1, k, hx, hy, hz);
                boundary += (u(p) * (grad_v.y()) - v(p) * (grad_u.y())) * hx * hz;
            }
        // z = 0
        for (std::size_t j = 0; j < ny; ++j)
            for (std::size_t i = 0; i < nx; ++i) {
                auto p = grid.point_at(i, j, 0);
                auto grad_u = gradient_3d(grid, u, i, j, 0, hx, hy, hz);
                auto grad_v = gradient_3d(grid, v, i, j, 0, hx, hy, hz);
                boundary += (u(p) * (-grad_v.z()) - v(p) * (-grad_u.z())) * hx * hy;
            }
        // z = nz-1
        for (std::size_t j = 0; j < ny; ++j)
            for (std::size_t i = 0; i < nx; ++i) {
                auto p = grid.point_at(i, j, nz - 1);
                auto grad_u = gradient_3d(grid, u, i, j, nz - 1, hx, hy, hz);
                auto grad_v = gradient_3d(grid, v, i, j, nz - 1, hx, hy, hz);
                boundary += (u(p) * (grad_v.z()) - v(p) * (grad_u.z())) * hx * hy;
            }

        Scalar error = std::abs(lhs - boundary);
        return error <= tolerance;
    }

} // namespace delta::numerical