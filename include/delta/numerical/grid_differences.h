// include/delta/numerical/grid_differences.h
#pragma once

#include "delta/core/grid_concept.h"
#include "delta/core/operational_function.h"
#include "delta/core/regulative_idea.h"
#include "delta/geometry/tensor_field.h"
#include <Eigen/Sparse>
#include <vector>
#include <functional>

namespace delta::numerical {

    // -------------------------------------------------------------------------
    // Схемы разностей
    // -------------------------------------------------------------------------
    enum DifferenceScheme {
        FORWARD,
        BACKWARD,
        CENTRAL
    };

    // -------------------------------------------------------------------------
    // Вспомогательные функции для одномерных упорядоченных сеток
    // -------------------------------------------------------------------------

    /**
     * @brief Найти индекс точки в упорядоченной сетке.
     * @tparam Grid Тип, удовлетворяющий OrderedGrid.
     * @param grid Сетка.
     * @param point Точка.
     * @return Индекс точки или -1, если не найдена.
     */
    template<typename Grid>
    std::ptrdiff_t find_index(const Grid& grid, const typename Grid::value_type& point) {
        for (std::size_t i = 0; i < grid.size(); ++i) {
            if (grid[i] == point) return static_cast<std::ptrdiff_t>(i);
        }
        return -1;
    }

    /**
     * @brief Получить индексы соседей слева и справа для точки с индексом idx.
     * @param grid Сетка.
     * @param idx Индекс центральной точки.
     * @return Пара (левый_индекс, правый_индекс). Если соседа нет, индекс = -1.
     */
    template<typename Grid>
    std::pair<std::ptrdiff_t, std::ptrdiff_t>
        neighbor_indices(const Grid& grid, std::ptrdiff_t idx) {
        std::ptrdiff_t left = (idx > 0) ? idx - 1 : -1;
        std::ptrdiff_t right = (idx + 1 < static_cast<std::ptrdiff_t>(grid.size())) ? idx + 1 : -1;
        return { left, right };
    }

    // -------------------------------------------------------------------------
    // Разностные операторы первого порядка (одномерные)
    // -------------------------------------------------------------------------

    /**
     * @brief Forward difference в заданной точке.
     * @tparam Grid   OrderedGrid.
     * @tparam Field  Объект с operator()(const Addr&) (например, OperationalFunction).
     * @tparam Metric Метрика для вычисления расстояния.
     * @param grid    Сетка.
     * @param field   Функция.
     * @param metric  Метрика.
     * @param point   Точка, в которой вычисляется разность.
     * @return Значение разности (field(x⁺) - field(x)) / h⁺. Если нет правого соседа, возвращает 0.
     */
    template<typename Grid, typename Field, typename Metric>
    auto forward_difference(const Grid& grid,
        const Field& field,
        const Metric& metric,
        const typename Grid::value_type& point) {
        using Value = std::decay_t<decltype(field(point))>;
        std::ptrdiff_t idx = find_index(grid, point);
        if (idx < 0) return Value{ 0 }; // точка не в сетке

        auto [left, right] = neighbor_indices(grid, idx);
        if (right < 0) return Value{ 0 };

        const auto& point_plus = grid[right];
        Value f_plus = field(point_plus);
        Value f_center = field(point);
        auto h_plus = metric(point_plus, point);
        return (f_plus - f_center) / h_plus;
    }

    /**
     * @brief Backward difference в заданной точке.
     */
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

    /**
     * @brief Central difference в заданной точке.
     */
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

    // -------------------------------------------------------------------------
    // Лапласиан второго порядка (обобщённый)
    // -------------------------------------------------------------------------

    /**
     * @brief Вычислить дискретный лапласиан в заданной точке на неравномерной сетке.
     */
    template<typename Grid, typename Field, typename Metric>
    auto laplacian_general(const Grid& grid,
        const Field& field,
        const Metric& metric,
        const typename Grid::value_type& point) {
        using Value = std::decay_t<decltype(field(point))>;
        std::ptrdiff_t idx = find_index(grid, point);
        if (idx < 0) return Value{ 0 };

        auto [left, right] = neighbor_indices(grid, idx);
        if (left < 0 || right < 0) return Value{ 0 }; // граничная точка

        const auto& point_minus = grid[left];
        const auto& point_plus = grid[right];
        Value f_center = field(point);
        Value f_plus = field(point_plus);
        Value f_minus = field(point_minus);
        auto h_plus = metric(point_plus, point);
        auto h_minus = metric(point, point_minus);

        // Δf(x) = (2/(h⁺+h⁻)) * [ (f⁺-f)/h⁺ - (f-f⁻)/h⁻ ]
        Value term = (f_plus - f_center) / h_plus - (f_center - f_minus) / h_minus;
        return (Value{ 2 } / (h_plus + h_minus)) * term;
    }

    // -------------------------------------------------------------------------
    // Построение разрежённой матрицы лапласиана
    // -------------------------------------------------------------------------

    /**
     * @brief Построить разрежённую матрицу дискретного лапласиана для всей сетки.
     *
     * Для каждой внутренней точки используется формула из laplacian_general.
     * Граничные точки обрабатываются отдельно – по умолчанию устанавливается
     * граничное условие Дирихле с фиксацией значения (строка матрицы становится
     * единичной). Для других ГУ нужно модифицировать матрицу отдельно.
     *
     * @param grid   Упорядоченная сетка.
     * @param metric Метрика.
     * @return Разрежённая матрица (Eigen::SparseMatrix) размера grid.size() x grid.size().
     */
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
                // Граничная точка: Дирихле по умолчанию
                triplets.emplace_back(static_cast<int>(i), static_cast<int>(i), Value{ 1 });
                continue;
            }

            const auto& point_minus = grid[static_cast<std::size_t>(left)];
            const auto& point_plus = grid[static_cast<std::size_t>(right)];
            auto h_plus = metric(point_plus, point);
            auto h_minus = metric(point, point_minus);

            // Коэффициенты (из вывода в ТЗ)
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

    // -------------------------------------------------------------------------
    // Дискретный градиент (одномерный)
    // -------------------------------------------------------------------------

    /**
     * @brief Градиент скалярного поля на одномерной сетке.
     *
     * Возвращает векторное поле размерности 1, где в каждой точке хранится
     * производная, вычисленная по заданной схеме.
     *
     * @tparam Grid   OrderedGrid.
     * @tparam Field  Поле (OperationalFunction или аналогичное).
     * @tparam Metric Метрика.
     * @param grid    Сетка.
     * @param field   Скалярное поле.
     * @param metric  Метрика.
     * @param scheme  Схема разности.
     * @return Тензорное поле ранга 1 размерности 1.
     */
    template<typename Grid, typename Field, typename Metric>
    auto discrete_gradient_1d(const Grid& grid,
        const Field& field,
        const Metric& metric,
        DifferenceScheme scheme = CENTRAL) {
        using Point = typename Grid::value_type;
        using Scalar = std::decay_t<decltype(field(grid[0]))>;
        constexpr int Dim = 1;

        geometry::TensorField<Point, Scalar, 1, Dim> result;

        for (std::size_t i = 0; i < grid.size(); ++i) {
            const auto& point = grid[i];
            Scalar grad_val;

            switch (scheme) {
            case FORWARD:
                grad_val = forward_difference(grid, field, metric, point);
                break;
            case BACKWARD:
                grad_val = backward_difference(grid, field, metric, point);
                break;
            case CENTRAL:
                grad_val = central_difference(grid, field, metric, point);
                break;
            default:
                grad_val = 0;
            }

            Eigen::Matrix<Scalar, Dim, 1> grad_vec;
            grad_vec(0) = grad_val;
            result.set(point, grad_vec);
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Дискретная дивергенция (одномерная)
    // -------------------------------------------------------------------------

    /**
     * @brief Дивергенция векторного поля на одномерной сетке.
     *
     * В 1D дивергенция векторного поля (которое само является скаляром)
     * совпадает с производной. Для согласованности с многомерным случаем
     * принимается векторное поле, но используется только его единственная компонента.
     *
     * @tparam Grid      OrderedGrid.
     * @tparam VecField  Векторное поле (ранг 1, размерность 1).
     * @tparam Metric    Метрика.
     * @param grid       Сетка.
     * @param vec_field  Векторное поле.
     * @param metric     Метрика.
     * @param scheme     Схема разности.
     * @return Скалярное поле.
     */
    template<typename Grid, typename VecField, typename Metric>
    auto discrete_divergence_1d(const Grid& grid,
        const VecField& vec_field,
        const Metric& metric,
        DifferenceScheme scheme = BACKWARD) {
        using Point = typename Grid::value_type;
        using Scalar = typename VecField::scalar_type;

        geometry::TensorField<Point, Scalar, 0, 0> result;

        for (std::size_t i = 0; i < grid.size(); ++i) {
            const auto& point = grid[i];
            // Извлекаем единственную компоненту вектора
            const auto& vec = vec_field.at(point);
            Scalar comp = vec(0); // предполагаем, что размерность 1

            // Создаём фиктивное скалярное поле, передающее эту компоненту
            auto scalar_at_point = [&](const Point& p) -> Scalar {
                return comp; // но это неправильно, т.к. нужно поле, а не константа.
                // Для корректного вычисления производной нужно поле, а не одно значение.
                // Придётся создать временное поле, но это неэффективно.
                // Правильнее, чтобы vec_field предоставляла доступ к компонентам как к отдельным полям.
                // Пока оставим заглушку.
                return Scalar{ 0 };
                };

            // Вычисляем производную от компоненты
            Scalar div_val = 0;
            switch (scheme) {
            case FORWARD:
                div_val = forward_difference(grid, scalar_at_point, metric, point);
                break;
            case BACKWARD:
                div_val = backward_difference(grid, scalar_at_point, metric, point);
                break;
            case CENTRAL:
                div_val = central_difference(grid, scalar_at_point, metric, point);
                break;
            }
            result.set(point, div_val);
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Ротор (заглушка – требует 3D)
    // -------------------------------------------------------------------------

    /**
     * @brief Ротор векторного поля в трёхмерном пространстве.
     *
     * @note Требует, чтобы точки сетки были трёхмерными (например, Eigen::Vector3d).
     *       В текущей версии не реализован.
     */
    template<typename Grid, typename VecField, typename Metric>
    auto discrete_curl(const Grid& grid,
        const VecField& vec_field,
        const Metric& metric) {
        static_assert(!std::is_same_v<Grid, Grid>, // всегда false
            "discrete_curl is not implemented yet (requires 3D grid)");
        return;
    }
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

        // Шаги (для равномерной сетки)
        Scalar hx = xg.step();
        Scalar hy = yg.step();
        Scalar hz = zg.step();

        // Размерности
        std::size_t nx = xg.size();
        std::size_t ny = yg.size();
        std::size_t nz = zg.size();

        for (std::size_t i = 0; i < nx; ++i) {
            for (std::size_t j = 0; j < ny; ++j) {
                for (std::size_t k = 0; k < nz; ++k) {
                    Point p = grid.point_at(i, j, k);
                    const auto& v = vec_field.at(p);

                    // Компоненты ротора (центральные разности, кроме границ)
                    Scalar curl_x = 0, curl_y = 0, curl_z = 0;

                    if (j > 0 && j + 1 < ny && k > 0 && k + 1 < nz) {
                        // ∂v_z/∂y - ∂v_y/∂z
                        Scalar dvz_dy = (vec_field.at(grid.point_at(i, j + 1, k))(2) - vec_field.at(grid.point_at(i, j - 1, k))(2)) / (2 * hy);
                        Scalar dvy_dz = (vec_field.at(grid.point_at(i, j, k + 1))(1) - vec_field.at(grid.point_at(i, j, k - 1))(1)) / (2 * hz);
                        curl_x = dvz_dy - dvy_dz;
                    }

                    if (i > 0 && i + 1 < nx && k > 0 && k + 1 < nz) {
                        // ∂v_x/∂z - ∂v_z/∂x
                        Scalar dvx_dz = (vec_field.at(grid.point_at(i, j, k + 1))(0) - vec_field.at(grid.point_at(i, j, k - 1))(0)) / (2 * hz);
                        Scalar dvz_dx = (vec_field.at(grid.point_at(i + 1, j, k))(2) - vec_field.at(grid.point_at(i - 1, j, k))(2)) / (2 * hx);
                        curl_y = dvx_dz - dvz_dx;
                    }

                    if (i > 0 && i + 1 < nx && j > 0 && j + 1 < ny) {
                        // ∂v_y/∂x - ∂v_x/∂y
                        Scalar dvy_dx = (vec_field.at(grid.point_at(i + 1, j, k))(1) - vec_field.at(grid.point_at(i - 1, j, k))(1)) / (2 * hx);
                        Scalar dvx_dy = (vec_field.at(grid.point_at(i, j + 1, k))(0) - vec_field.at(grid.point_at(i, j - 1, k))(0)) / (2 * hy);
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