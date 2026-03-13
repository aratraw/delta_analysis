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
    // Вычисляет производную единственной компоненты векторного поля.
    // -------------------------------------------------------------------------
    template<typename Grid, typename VecField, typename Metric>
        requires IsMetric<Metric, typename Grid::value_type, typename VecField::scalar_type> &&
    (VecField::rank == 1) && (VecField::dim == 1)
        auto discrete_divergence_1d(const Grid& grid,
            const VecField& vec_field,
            const Metric& metric,
            DifferenceScheme scheme = DifferenceScheme::BACKWARD) {
        using Point = typename Grid::value_type;
        using Scalar = typename VecField::scalar_type;

        // Результирующее скалярное поле (ранг 0)
        geometry::TensorField<Point, Scalar, 0, 0> result;

        for (std::size_t i = 0; i < grid.size(); ++i) {
            const auto& point = grid[i];

            // Создаём функтор, возвращающий компоненту векторного поля в заданной точке
            auto component_field = [&](const Point& x) -> Scalar {
                return vec_field.at(x)(0);
                };

            Scalar div_val;
            switch (scheme) {
            case DifferenceScheme::FORWARD:
                div_val = forward_difference(grid, component_field, metric, point);
                break;
            case DifferenceScheme::BACKWARD:
                div_val = backward_difference(grid, component_field, metric, point);
                break;
            case DifferenceScheme::CENTRAL:
                div_val = central_difference(grid, component_field, metric, point);
                break;
            default:
                div_val = 0;
            }

            result.set(point, div_val);
        }
        return result;
    }

} // namespace delta::numerical