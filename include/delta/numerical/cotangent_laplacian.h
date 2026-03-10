// include/delta/numerical/cotangent_laplacian.h
#pragma once

#include <Eigen/Sparse>
#include <vector>
#include <cmath>
#include "delta/geometry/simplicial_complex.h"

namespace delta::numerical {

    /**
     * @brief Builds the cotangent Laplace matrix for a 2D simplicial complex (triangle mesh).
     *
     * The resulting matrix L is of size N×N (N = number of vertices) and satisfies:
     * - For i ≠ j, L_ij = -0.5 * (cot α_ij + cot β_ij) if edge (i,j) exists, otherwise 0.
     * - L_ii = -∑_{j≠i} L_ij.
     *
     * Here α_ij and β_ij are the angles opposite the edge (i,j) in the two incident triangles.
     * For boundary edges, only one cotangent contributes.
     *
     * @tparam Coord Floating-point type (e.g., float, double).
     * @param mesh The input triangle mesh (SimplicialComplex2D<Coord>).
     * @return Eigen::SparseMatrix<Coord> The cotangent Laplace matrix.
     */
    template<typename Coord = double>
    Eigen::SparseMatrix<Coord> build_cotangent_laplacian(const geometry::SimplicialComplex2D<Coord>& mesh) {
        using namespace geometry;
        using Triplet = Eigen::Triplet<Coord>;
        using vertex_index = typename SimplicialComplex2D<Coord>::vertex_index;
        using edge = typename SimplicialComplex2D<Coord>::edge;
        using triangle = typename SimplicialComplex2D<Coord>::triangle;

        std::vector<Triplet> triplets;

        std::size_t n_vertices = mesh.num_vertices();
        std::size_t n_triangles = mesh.num_triangles();
        std::size_t n_edges = mesh.num_edges();

        // Предварительно выделим память для диагональных элементов
        std::vector<Coord> diag(n_vertices, 0.0);

        // Вспомогательная структура для накопления котангенсов на рёбрах
        std::vector<Coord> edge_cot_sum(n_edges, 0.0);

        // Функция для получения вектора стороны треугольника по двум вершинам
        auto get_edge_vector = [&](vertex_index i, vertex_index j) -> Eigen::Vector<Coord, 2> {
            return mesh.points()[j] - mesh.points()[i];
            };

        // Вычисление площади треугольника через векторное произведение
        auto triangle_area = [&](const triangle& tri) -> Coord {
            auto ab = get_edge_vector(tri[0], tri[1]);
            auto ac = get_edge_vector(tri[0], tri[2]);
            return Coord(0.5) * std::abs(ab.x() * ac.y() - ab.y() * ac.x());
            };

        // Вычисление котангенса угла при вершине k в треугольнике (i,j,k)
        auto cotangent_at_vertex = [&](vertex_index i, vertex_index j, vertex_index k) -> Coord {
            auto ki = get_edge_vector(k, i);
            auto kj = get_edge_vector(k, j);
            auto ij = get_edge_vector(i, j);
            double ki_len2 = ki.squaredNorm();
            double kj_len2 = kj.squaredNorm();
            double ij_len2 = ij.squaredNorm();
            // Площадь треугольника через ki,kj
            double cross = std::abs(ki.x() * kj.y() - ki.y() * kj.x());
            double area = 0.5 * cross;
            if (area <= 0) return 0.0; // защита
            double cot = (ki_len2 + kj_len2 - ij_len2) / (4.0 * area);
            return static_cast<Coord>(cot);
            };

        // Проходим по всем треугольникам
        for (std::size_t t = 0; t < n_triangles; ++t) {
            const auto& tri = mesh.triangles()[t];
            vertex_index v0 = tri[0], v1 = tri[1], v2 = tri[2];

            // Вычислим площадь треугольника (можно один раз)
            Coord area = triangle_area(tri);
            if (area <= 0) continue; // защита от вырожденных треугольников

            // Для каждого ребра треугольника вычислим котангенс противолежащего угла
            Coord cot0 = cotangent_at_vertex(v1, v2, v0); // угол при v0
            Coord cot1 = cotangent_at_vertex(v2, v0, v1); // угол при v1
            Coord cot2 = cotangent_at_vertex(v0, v1, v2); // угол при v2

            // Найдём индексы рёбер и добавим котангенсы
            auto e01 = mesh.find_edge(v0, v1);
            auto e12 = mesh.find_edge(v1, v2);
            auto e20 = mesh.find_edge(v2, v0);

            if (e01 >= 0) edge_cot_sum[e01] += cot2;
            if (e12 >= 0) edge_cot_sum[e12] += cot0;
            if (e20 >= 0) edge_cot_sum[e20] += cot1;
        }

        // Теперь для каждого ребра строим вклад в матрицу
        for (std::size_t e = 0; e < n_edges; ++e) {
            const auto& edge = mesh.edges()[e];
            vertex_index i = edge[0];
            vertex_index j = edge[1];
            Coord w = Coord(-0.5) * edge_cot_sum[e]; // L_ij = -0.5 * (cot sum)

            // Добавляем внедиагональный элемент
            triplets.emplace_back(i, j, w);
            triplets.emplace_back(j, i, w); // матрица симметрична

            // Накопление для диагонали: L_ii -= w, L_jj -= w
            diag[i] -= w;
            diag[j] -= w;
        }

        // Добавляем диагональные элементы
        for (std::size_t i = 0; i < n_vertices; ++i) {
            triplets.emplace_back(i, i, diag[i]);
        }

        // Строим разреженную матрицу
        Eigen::SparseMatrix<Coord> L(n_vertices, n_vertices);
        L.setFromTriplets(triplets.begin(), triplets.end());
        return L;
    }

} // namespace delta::numerical