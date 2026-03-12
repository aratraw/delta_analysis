// include/delta/numerical/grid_ops_tensor_1d.h
#pragma once

#include "delta/geometry/tensor_field.h"
#include "delta/numerical/grid_ops_1d.h"
#include "delta/numerical/concepts.h"

namespace delta::numerical {

    // -------------------------------------------------------------------------
    // Дискретный градиент (одномерный) – возвращает тензорное поле ранга 1
    // -------------------------------------------------------------------------
    template<typename Grid, typename Field, typename Metric>
    auto discrete_gradient_1d(const Grid& grid,
        const Field& field,
        const Metric& metric,
        DifferenceScheme scheme = DifferenceScheme::CENTRAL) {
        using Point = typename Grid::value_type;
        using Scalar = std::decay_t<decltype(field(grid[0]))>;
        constexpr int Dim = 1;

        geometry::TensorField<Point, Scalar, 1, Dim> result;

        for (std::size_t i = 0; i < grid.size(); ++i) {
            const auto& point = grid[i];
            Scalar grad_val;

            switch (scheme) {
            case DifferenceScheme::FORWARD:
                grad_val = forward_difference(grid, field, metric, point);
                break;
            case DifferenceScheme::BACKWARD:
                grad_val = backward_difference(grid, field, metric, point);
                break;
            case DifferenceScheme::CENTRAL:
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
    // Дискретная дивергенция (одномерная) – возвращает скалярное поле
    // ВНИМАНИЕ: текущая реализация использует заглушку, так как требует
    // доступа к компонентам векторного поля как к отдельным скалярным полям.
    // -------------------------------------------------------------------------
    template<typename Grid, typename VecField, typename Metric>
    auto discrete_divergence_1d(const Grid& grid,
        const VecField& vec_field,
        const Metric& metric,
        DifferenceScheme scheme = DifferenceScheme::BACKWARD) {
        using Point = typename Grid::value_type;
        using Scalar = typename VecField::scalar_type;

        geometry::TensorField<Point, Scalar, 0, 0> result;

        for (std::size_t i = 0; i < grid.size(); ++i) {
            const auto& point = grid[i];
            const auto& vec = vec_field.at(point);
            Scalar comp = vec(0); // предполагаем размерность 1

            // Заглушка: для корректной работы нужно передавать поле компоненты,
            // а не одно значение. Здесь просто возвращаем 0.
            result.set(point, Scalar{ 0 });
        }
        return result;
    }

} // namespace delta::numerical