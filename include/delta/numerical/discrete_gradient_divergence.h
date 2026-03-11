// include/delta/numerical/discrete_gradient_divergence.h
#pragma once

#include <Eigen/Core>
#include <vector>
#include "delta/geometry/simplicial_complex.h"
#include "delta/core/rational.h"

namespace delta::numerical {

    // Существующие шаблоны для std::vector
    template<typename Coord>
    std::vector<Coord> discrete_gradient(
        const geometry::SimplicialComplex2D<Coord>& mesh,
        const std::vector<Coord>& vertex_values)
    {
        std::size_t n_edges = mesh.num_edges();
        std::vector<Coord> gradient(n_edges);
        for (std::size_t e = 0; e < n_edges; ++e) {
            const auto& edge = mesh.edges()[e];
            gradient[e] = vertex_values[edge[1]] - vertex_values[edge[0]];
        }
        return gradient;
    }

    // Перегрузка для Eigen::VectorXd
    template<typename Coord>
    std::vector<Coord> discrete_gradient(
        const geometry::SimplicialComplex2D<Coord>& mesh,
        const Eigen::Matrix<Coord, Eigen::Dynamic, 1>& vertex_values)
    {
        std::size_t n_edges = mesh.num_edges();
        std::vector<Coord> gradient(n_edges);
        for (std::size_t e = 0; e < n_edges; ++e) {
            const auto& edge = mesh.edges()[e];
            gradient[e] = vertex_values[edge[1]] - vertex_values[edge[0]];
        }
        return gradient;
    }

    // Существующие шаблоны для raw divergence
    template<typename Coord>
    std::vector<Coord> discrete_divergence_raw(
        const geometry::SimplicialComplex2D<Coord>& mesh,
        const std::vector<Coord>& edge_values)
    {
        std::size_t n_vertices = mesh.num_vertices();
        std::vector<Coord> div(n_vertices, Coord(0));
        for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
            const auto& edge = mesh.edges()[e];
            div[edge[0]] += edge_values[e];
            div[edge[1]] -= edge_values[e];
        }
        return div;
    }

    // Перегрузка для Eigen::VectorXd
    template<typename Coord>
    std::vector<Coord> discrete_divergence_raw(
        const geometry::SimplicialComplex2D<Coord>& mesh,
        const Eigen::Matrix<Coord, Eigen::Dynamic, 1>& edge_values)
    {
        std::size_t n_vertices = mesh.num_vertices();
        std::vector<Coord> div(n_vertices, Coord(0));
        for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
            const auto& edge = mesh.edges()[e];
            div[edge[0]] += edge_values[e];
            div[edge[1]] -= edge_values[e];
        }
        return div;
    }

    // compute_vertex_dual_areas остаётся без изменений, возвращает std::vector
    template<typename Coord>
    std::vector<Coord> compute_vertex_dual_areas(const geometry::SimplicialComplex2D<Coord>& mesh) {
        std::size_t n_vertices = mesh.num_vertices();
        std::size_t n_triangles = mesh.num_triangles();
        std::vector<Coord> areas(n_vertices, Coord(0));

        auto triangle_area = [&](const typename geometry::SimplicialComplex2D<Coord>::triangle& tri) -> Coord {
            auto ab = mesh.points()[tri[1]] - mesh.points()[tri[0]];
            auto ac = mesh.points()[tri[2]] - mesh.points()[tri[0]];
            Coord det = ab.x() * ac.y() - ab.y() * ac.x();
            using std::abs;
            return abs(det) / Coord(2);
            };

        for (std::size_t t = 0; t < n_triangles; ++t) {
            const auto& tri = mesh.triangles()[t];
            Coord area = triangle_area(tri);
            for (int i = 0; i < 3; ++i) {
                areas[tri[i]] += area / Coord(3);
            }
        }
        return areas;
    }

    // discrete_divergence (с площадями) оставляем как есть, он принимает std::vector
    template<typename Coord>
    std::vector<Coord> discrete_divergence(
        const geometry::SimplicialComplex2D<Coord>& mesh,
        const std::vector<Coord>& edge_values,
        const std::vector<Coord>& vertex_areas)
    {
        auto raw = discrete_divergence_raw(mesh, edge_values);
        std::vector<Coord> result(raw.size());
        for (std::size_t i = 0; i < raw.size(); ++i) {
            result[i] = raw[i] / vertex_areas[i];
        }
        return result;
    }

} // namespace delta::numerical