// include/delta/numerical/integrals.h
#pragma once

#include <functional>
#include <vector>
#include <cmath>
#include "delta/core/grid_concept.h"
#include "delta/core/operational_function.h"
#include "delta/core/regulative_idea.h"
#include "grid_differences.h"      // для neighbor_indices, laplacian_general
#include "cartesian_grid.h"        // для CartesianGrid3D (используется в многомерных проверках)

namespace delta::numerical {

    // =============================================================================
    // Базовые квадратурные формулы (для равномерных сеток)
    // =============================================================================

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

    // =============================================================================
    // Универсальные одномерные функции (для произвольной упорядоченной сетки)
    // =============================================================================

    /**
     * @brief Вычислить объём ячейки, ассоциированной с точкой в координатной сетке (1D).
     *
     * Для одномерной сетки: объём = среднее арифметическое шагов вперёд и назад,
     * если точка не граничная. Для граничной – шаг в одну сторону.
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
     * @brief Вычислить дискретный интеграл функции по области, покрытой сеткой (1D).
     *
     * ∫ f dV ≈ Σ_i f(x_i) * V_i, где V_i – объём ячейки вокруг x_i.
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
     *
     * Тождество: Σ_{k=0}^{K-1} f_k (g_{k+1} - g_k) = - Σ_{k=1}^{K} (f_k - f_{k-1}) g_k + f_K g_{K+1} - f_0 g_0.
     * Здесь индексы соответствуют точкам сетки, и g_{K+1} – значение за правой границей (может быть задано).
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

    /**
     * @brief Проверить первое тождество Грина: ∫ f Δg dV = -∫ f' g' dV + f g'|_boundary (1D).
     *
     * @note В текущей реализации правая часть не вычисляется – возвращается false.
     *       Для полноценной проверки требуется реализация градиентов.
     */
    template<typename Grid, typename Field, typename Metric>
    bool check_green_first(const Grid& grid,
        const Field& f,
        const Field& g,
        const Metric& metric,
        double tol = 1e-12) {
        using Value = typename Field::value_type;

        // Левая часть: ∫ f Δg
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

    // =============================================================================
    // Вспомогательные функции для декартовых сеток (2D и 3D)
    // =============================================================================

    // Объём ячейки для 2D равномерной сетки (с учётом границ)
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

    // Объём ячейки для 3D равномерной сетки
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

    // Градиент в 2D (возвращает вектор из двух компонент)
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

    // Градиент в 3D
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

        // x
        if (i > 0 && i + 1 < grid.x_grid().size())
            dfdx = (f(grid.point_at(i + 1, j, k)) - f(grid.point_at(i - 1, j, k))) / (2 * hx);
        else if (i == 0)
            dfdx = (f(grid.point_at(i + 1, j, k)) - f(p)) / hx;
        else if (i == grid.x_grid().size() - 1)
            dfdx = (f(p) - f(grid.point_at(i - 1, j, k))) / hx;

        // y
        if (j > 0 && j + 1 < grid.y_grid().size())
            dfdy = (f(grid.point_at(i, j + 1, k)) - f(grid.point_at(i, j - 1, k))) / (2 * hy);
        else if (j == 0)
            dfdy = (f(grid.point_at(i, j + 1, k)) - f(p)) / hy;
        else if (j == grid.y_grid().size() - 1)
            dfdy = (f(p) - f(grid.point_at(i, j - 1, k))) / hy;

        // z
        if (k > 0 && k + 1 < grid.z_grid().size())
            dfdz = (f(grid.point_at(i, j, k + 1)) - f(grid.point_at(i, j, k - 1))) / (2 * hz);
        else if (k == 0)
            dfdz = (f(grid.point_at(i, j, k + 1)) - f(p)) / hz;
        else if (k == grid.z_grid().size() - 1)
            dfdz = (f(p) - f(grid.point_at(i, j, k - 1))) / hz;

        return Eigen::Matrix<Scalar, 3, 1>(dfdx, dfdy, dfdz);
    }

    // Дивергенция в 3D
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

    // =============================================================================
    // Многомерные проверки (2D и 3D)
    // =============================================================================

    // Проверка суммирования по частям в 2D (только x-направление)
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

        // Граничные вклады: левая и правая границы (x = const)
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

    // Проверка суммирования по частям в 3D (x-направление)
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

    // Первое тождество Грина в 2D
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

                // Лапласиан Δg (5-точечный шаблон, только внутренние точки)
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

    // Первое тождество Грина в 3D
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

                    // Лапласиан Δg (7-точечный шаблон)
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

} // namespace delta::numerical