#pragma once

#include "simplicial_complex.h"
#include <Eigen/Dense>

namespace delta::geometry {

    /**
     * @class HatBasis
     * @brief Линейные базисные функции (hat functions) на симплициальном комплексе.
     *
     * Для каждого узла (вершины) определена функция φ_v, равная 1 в этой вершине,
     * 0 в остальных вершинах, и линейная на каждом симплексе.
     *
     * @tparam Complex Тип, удовлетворяющий SimplicialComplex (например, SimplicialComplex2D).
     */
    template<typename Complex>
    class HatBasis {
    public:
        using point_type = typename Complex::point_type;
        using scalar_type = typename point_type::Scalar;

        /**
         * @param mesh Ссылка на симплициальный комплекс.
         */
        explicit HatBasis(const Complex& mesh) : mesh_(mesh) {}

        /**
         * @brief Значение базисной функции для вершины v в точке p.
         *
         * @note В текущей реализации работает только если p совпадает с какой-либо вершиной.
         *       Для произвольной точки требуется поиск симплекса и барицентрические координаты.
         * @param v Индекс вершины.
         * @param p Точка.
         * @return φ_v(p)
         */
        scalar_type evaluate(std::size_t v, const point_type& p) const {
            // Проверяем, не совпадает ли p с какой-то вершиной
            for (std::size_t i = 0; i < mesh_.size(); ++i) {
                if (p == mesh_.vertex(i)) {
                    return (i == v) ? scalar_type{ 1 } : scalar_type{ 0 };
                }
            }
            // Иначе – вне вершин, пока не поддерживается
            return scalar_type{ 0 };
        }

        /**
         * @brief Градиент базисной функции для вершины v в точке p.
         *
         * Возвращает градиент (вектор) в точке p. На линейном симплексе градиент постоянен.
         * @note Реализовано только для 2D треугольников и только в точках, совпадающих с вершинами.
         * @param v Индекс вершины.
         * @param p Точка.
         * @return ∇φ_v(p)
         */
        point_type gradient(std::size_t v, const point_type& p) const {
            // Для треугольника градиент можно вычислить, зная координаты вершин.
            // Но для этого нужно знать, в каком треугольнике находится p.
            // Пока возвращаем нулевой вектор.
            return point_type::Zero();
        }

    private:
        const Complex& mesh_;
    };

} // namespace delta::geometry