// include/delta/numerical/cotangent_laplacian.h
#pragma once

#include <Eigen/Sparse>
#include <vector>
#include <cmath>
#include "delta/geometry/simplicial_complex.h"
#include "delta/core/rational.h"
#include "delta/numerical/concepts.h"   // для IsMetric

namespace delta::numerical {

    /**
     * @brief Построить котангенсную матрицу Лапласа для 2D симплициального комплекса.
     *
     * @tparam Coord Скалярный тип координат (например, Rational или double).
     * @tparam Metric Тип метрики, удовлетворяющий IsMetric.
     * @param mesh 2D симплициальный комплекс (триангуляция).
     * @param metric Метрика для вычисления длин и площадей.
     * @return Eigen::SparseMatrix<Coord> разреженная матрица Лапласа размером num_vertices x num_vertices.
     */
    template<typename Coord, typename Metric>
        requires IsMetric<Metric, typename geometry::SimplicialComplex<2, Coord>::point_type, Coord>
    Eigen::SparseMatrix<Coord> build_cotangent_laplacian(
        const geometry::SimplicialComplex<2, Coord>& mesh,
        const Metric& metric)
    {
        using namespace geometry;
        using Triplet = Eigen::Triplet<Coord>;
        using vertex_index = typename SimplicialComplex<2, Coord>::vertex_index;

        std::vector<Triplet> triplets;

        std::size_t n_vertices = mesh.num_vertices();
        std::size_t n_triangles = mesh.num_triangles();
        std::size_t n_edges = mesh.num_edges();

        // Диагональные элементы (будут накоплены)
        std::vector<Coord> diag(n_vertices, Coord{ 0 });
        // Для каждого ребра накопим сумму котангенсов из прилегающих треугольников
        std::vector<Coord> edge_cot_sum(n_edges, Coord{ 0 });

        // Вычисляем площади треугольников и котангенсы
        for (std::size_t t = 0; t < n_triangles; ++t) {
            auto tri = mesh.triangle_at(t);
            vertex_index v0 = tri[0], v1 = tri[1], v2 = tri[2];

            // Длины сторон через метрику
            Coord len_01 = mesh.edge_length(static_cast<std::size_t>(mesh.find_simplex(1, { v0, v1 })), metric);
            Coord len_12 = mesh.edge_length(static_cast<std::size_t>(mesh.find_simplex(1, { v1, v2 })), metric);
            Coord len_20 = mesh.edge_length(static_cast<std::size_t>(mesh.find_simplex(1, { v2, v0 })), metric);

            // Площадь треугольника через метрику (уже есть в SimplicialComplex, но можем вычислить сами по формуле Герона)
            Coord area = mesh.cell_volume(t, metric);
            if (area <= Coord{ 0 }) continue; // защита от вырожденных треугольников

            // Котангенсы углов при каждой вершине по формуле:
            // cot(angle at v0) = (len_12^2 + len_20^2 - len_01^2) / (4 * area)
            // (угол при v0 образован сторонами v0-v1 и v0-v2, противоположная сторона v1-v2)
            Coord cot0 = (len_12 * len_12 + len_20 * len_20 - len_01 * len_01) / (Coord{ 4 } * area);
            Coord cot1 = (len_20 * len_20 + len_01 * len_01 - len_12 * len_12) / (Coord{ 4 } * area);
            Coord cot2 = (len_01 * len_01 + len_12 * len_12 - len_20 * len_20) / (Coord{ 4 } * area);

            // Индексы рёбер (ориентация не важна, нам нужен просто индекс)
            std::size_t e01 = static_cast<std::size_t>(mesh.find_simplex(1, { v0, v1 }));
            std::size_t e12 = static_cast<std::size_t>(mesh.find_simplex(1, { v1, v2 }));
            std::size_t e20 = static_cast<std::size_t>(mesh.find_simplex(1, { v2, v0 }));

            // Добавляем вклад в сумму котангенсов для каждого ребра
            edge_cot_sum[e01] += cot2;  // cot2 — угол при v2, противолежащий ребру v0-v1
            edge_cot_sum[e12] += cot0;  // cot0 — угол при v0, противолежащий ребру v1-v2
            edge_cot_sum[e20] += cot1;  // cot1 — угол при v1, противолежащий ребру v2-v0
        }

        // Формируем недиагональные элементы и накапливаем диагональ
        for (std::size_t e = 0; e < n_edges; ++e) {
            auto edge = mesh.edge_at(e);
            vertex_index i = edge[0];
            vertex_index j = edge[1];
            Coord w = edge_cot_sum[e] / Coord{ 2 }; // коэффициент Лапласа: 0.5 * (cot α + cot β)

            // Запись в матрицу (обе ориентации)
            triplets.emplace_back(static_cast<int>(i), static_cast<int>(j), -w);
            triplets.emplace_back(static_cast<int>(j), static_cast<int>(i), -w);

            // Накопление диагонали
            diag[i] += w;
            diag[j] += w;
        }

        // Добавляем диагональные элементы
        for (std::size_t i = 0; i < n_vertices; ++i) {
            triplets.emplace_back(static_cast<int>(i), static_cast<int>(i), diag[i]);
        }

        // Сборка разреженной матрицы
        Eigen::SparseMatrix<Coord> L(static_cast<int>(n_vertices), static_cast<int>(n_vertices));
        L.setFromTriplets(triplets.begin(), triplets.end());
        return L;
    }

} // namespace delta::numerical