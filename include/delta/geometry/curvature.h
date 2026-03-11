// include/delta/geometry/curvature.h
#pragma once
#include <numbers>
#include <cmath>
#include "connection.h"
#include "simplicial_complex.h"  // для работы с гранями

namespace delta::geometry {

    /**
     * @brief Вычисление голономии вокруг грани симплициального комплекса.
     *
     * @tparam Complex Тип, удовлетворяющий SimplicialComplex.
     * @tparam Connection Тип связности.
     * @param mesh Сетка (комплекс).
     * @param face_index Индекс грани (треугольника).
     * @param conn Связность на рёбрах.
     * @return Матрица голономии (произведение транспортов вдоль рёбер грани).
     */
    template<typename Complex, typename Connection>
    auto holonomy_around_face(const Complex& mesh,
        std::size_t face_index,
        const Connection& conn) {
        using matrix_type = typename Connection::matrix_type;
        auto tri = mesh.triangle(face_index);
        std::vector<typename Connection::edge_type> edges = {
            {tri[0], tri[1]},
            {tri[1], tri[2]},
            {tri[2], tri[0]}
        };
        return conn.holonomy(edges);
    }

    /**
     * @brief Приближение тензора кривизны из голономии.
     *
     * Для малой площади A приближённо: holonomy ≈ I + R * A, где R - тензор кривизны.
     * Возвращает R = (holonomy - I) / A.
     *
     * @param holonomy Матрица голономии.
     * @param area Площадь грани.
     * @return Матрица кривизны.
     */
    template<typename Matrix>
    Matrix curvature_from_holonomy(const Matrix& holonomy,
        const typename Matrix::Scalar& area) {
        if (area == typename Matrix::Scalar{ 0 }) {
            return Matrix::Zero();
        }
        return (holonomy - Matrix::Identity()) / area;
    }

    /**
     * @brief Вычисление скалярной кривизны в вершине для 2D симплициального комплекса
     *        с использованием метрики (дефицит угла).
     *
     * @tparam Complex Тип, удовлетворяющий SimplicialComplex.
     * @tparam Metric Тип метрики (должен вычислять длину отрезка).
     * @param mesh Сетка.
     * @param vertex_index Индекс вершины.
     * @param metric Метрика для вычисления длин рёбер.
     * @return Скалярная кривизна (дефицит угла) в вершине.
     */
    template<typename Complex, typename Metric>
    typename Complex::value_type
        vertex_curvature_deficit(const Complex& mesh,
            std::size_t vertex_index,
            const Metric& metric) {
        using Coord = typename Complex::value_type;
        // Суммируем углы прилежащих треугольников
        Coord sum_angles = Coord{ 0 };
        // Для каждого треугольника, содержащего вершину, вычислим угол при этой вершине
        // Для этого нужно найти все треугольники, инцидентные вершине.
        // В текущей реализации SimplicialComplex2D нет прямого доступа к инцидентным треугольникам,
        // поэтому придётся перебирать все треугольники (O(n_triangles)).
        // Для оптимизации можно построить структуру позже.
        // Пока реализуем простой перебор.
        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto tri = mesh.triangle(t);
            // Найдём позицию vertex_index в треугольнике
            int pos = -1;
            for (int i = 0; i < 3; ++i) {
                if (tri[i] == vertex_index) {
                    pos = i;
                    break;
                }
            }
            if (pos == -1) continue;

            // Получим координаты вершин
            auto p0 = mesh.vertex(tri[0]);
            auto p1 = mesh.vertex(tri[1]);
            auto p2 = mesh.vertex(tri[2]);

            // Вычислим угол при вершине pos
            // Для pos=0: угол между векторами (p1-p0) и (p2-p0)
            // Для pos=1: угол между (p0-p1) и (p2-p1)
            // Для pos=2: угол между (p0-p2) и (p1-p2)
            auto v1 = (pos == 0) ? (p1 - p0) : (pos == 1) ? (p0 - p1) : (p0 - p2);
            auto v2 = (pos == 0) ? (p2 - p0) : (pos == 1) ? (p2 - p1) : (p1 - p2);
            // Длины (используем метрику, которая возвращает длину)
            // В метрике обычно передаются точки, а не векторы. Нужно адаптировать.
            // Для простоты будем считать, что metric может принимать точки.
            // У нас есть точки p0,p1,p2. Вычислим длины сторон.
            // Лучше вычислять угол через теорему косинусов:
            // a = length(p1-p0), b = length(p2-p0), c = length(p1-p2) для pos=0.
            // cos(angle) = (a^2 + b^2 - c^2)/(2ab)
            // Но это требует длин сторон. Проще использовать metric для получения длин.
            Coord a, b, c;
            if (pos == 0) {
                a = metric(p1, p0);
                b = metric(p2, p0);
                c = metric(p2, p1);
            }
            else if (pos == 1) {
                a = metric(p0, p1);
                b = metric(p2, p1);
                c = metric(p2, p0);
            }
            else { // pos == 2
                a = metric(p0, p2);
                b = metric(p1, p2);
                c = metric(p1, p0);
            }
            if (a == Coord{ 0 } || b == Coord{ 0 }) continue; // вырожденный треугольник
            Coord cos_angle = (a * a + b * b - c * c) / (Coord{ 2 } * a * b);
            // Ограничим cos в [-1,1] из-за погрешностей
            if (cos_angle > Coord{ 1 }) cos_angle = Coord{ 1 };
            if (cos_angle < Coord{ -1 }) cos_angle = Coord{ -1 };
            Coord angle(std::acos(cos_angle.convert_to<double>())); // используем double для арккосинуса
            sum_angles += angle;
        }
        // Дефицит угла = 2π - сумма углов
        Coord deficit = Coord{ 2 } * Coord(std::numbers::pi) - sum_angles;
        return deficit;
    }

} // namespace delta::geometry