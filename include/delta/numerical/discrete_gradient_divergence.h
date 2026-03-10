// include/delta/numerical/discrete_gradient_divergence.h
#pragma once

#include <Eigen/Core>
#include "delta/geometry/simplicial_complex.h"

namespace delta::numerical {

    /**
     * @brief Computes the discrete gradient of a scalar field on vertices.
     *
     * For each oriented edge (i,j) stored in the mesh (with fixed orientation low->high),
     * the gradient value is f(j) - f(i). The result is a vector of size num_edges().
     *
     * @tparam Coord Floating-point type.
     * @param mesh The triangle mesh.
     * @param vertex_values Vector of size num_vertices() containing scalar values at vertices.
     * @return Eigen::VectorXd Gradient values on edges (same order as mesh.edges()).
     */
    template<typename Coord = double>
    Eigen::VectorXd discrete_gradient(
        const geometry::SimplicialComplex2D<Coord>& mesh,
        const Eigen::VectorXd& vertex_values)
    {
        std::size_t n_edges = mesh.num_edges();
        Eigen::VectorXd gradient(n_edges);

        for (std::size_t e = 0; e < n_edges; ++e) {
            const auto& edge = mesh.edges()[e];
            // edge = {low, high} with low < high (due to our storage)
            gradient[e] = vertex_values[edge[1]] - vertex_values[edge[0]];
        }
        return gradient;
    }

    /**
     * @brief Computes the discrete divergence of a 1‑form (edge‑based field).
     *
     * The divergence at vertex v is defined as the sum over incident edges of
     * (signed) edge value, optionally divided by the dual area (Voronoi area).
     * Here we return the raw sum without area weighting; for a proper DEC
     * divergence, the result should be scaled by 1/area_v.
     *
     * The sign convention:
     *   - If the edge is stored as (low,high), for vertex low contribution = +value,
     *     for vertex high contribution = -value.
     *
     * @tparam Coord Floating-point type.
     * @param mesh The triangle mesh.
     * @param edge_values Vector of size num_edges() containing values on edges.
     * @return Eigen::VectorXd Divergence values at vertices (raw sum).
     */
    template<typename Coord = double>
    Eigen::VectorXd discrete_divergence_raw(
        const geometry::SimplicialComplex2D<Coord>& mesh,
        const Eigen::VectorXd& edge_values)
    {
        std::size_t n_vertices = mesh.num_vertices();
        Eigen::VectorXd div = Eigen::VectorXd::Zero(n_vertices);

        for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
            const auto& edge = mesh.edges()[e];
            // edge[0] = low, edge[1] = high
            div[edge[0]] += edge_values[e];
            div[edge[1]] -= edge_values[e];
        }
        return div;
    }

    /**
     * @brief Discrete divergence scaled by vertex dual areas (Voronoi areas).
     *
     * This version computes the proper DEC divergence: (δ ω)(v) = (1/A_v) * raw_sum,
     * where A_v is the area of the Voronoi cell around vertex v.
     *
     * @tparam Coord Floating-point type.
     * @param mesh The triangle mesh.
     * @param edge_values Vector of size num_edges() containing values on edges.
     * @param vertex_areas Vector of size num_vertices() containing dual areas.
     * @return Eigen::VectorXd Divergence values at vertices.
     */
    template<typename Coord = double>
    Eigen::VectorXd discrete_divergence(
        const geometry::SimplicialComplex2D<Coord>& mesh,
        const Eigen::VectorXd& edge_values,
        const Eigen::VectorXd& vertex_areas)
    {
        Eigen::VectorXd raw = discrete_divergence_raw(mesh, edge_values);
        for (std::size_t i = 0; i < mesh.num_vertices(); ++i) {
            raw[i] /= vertex_areas[i];
        }
        return raw;
    }

    /**
     * @brief Computes vertex dual areas (Voronoi areas) for a triangle mesh.
     *
     * For each vertex, the Voronoi area is the area of the region closer to that vertex
     * than to any other. In a triangle mesh, it can be approximated as 1/3 of the sum
     * of areas of incident triangles (barycentric dual). For boundary vertices, the
     * formula is adjusted.
     *
     * @tparam Coord Floating-point type.
     * @param mesh The triangle mesh.
     * @return Eigen::VectorXd Voronoi areas for each vertex.
     */
    template<typename Coord = double>
    Eigen::VectorXd compute_vertex_dual_areas(const geometry::SimplicialComplex2D<Coord>& mesh) {
        std::size_t n_vertices = mesh.num_vertices();
        std::size_t n_triangles = mesh.num_triangles();
        Eigen::VectorXd areas = Eigen::VectorXd::Zero(n_vertices);

        // Функция для вычисления площади треугольника (как в cotangent_laplacian)
        auto triangle_area = [&](const geometry::SimplicialComplex2D<Coord>::triangle& tri) {
            auto ab = mesh.points()[tri[1]] - mesh.points()[tri[0]];
            auto ac = mesh.points()[tri[2]] - mesh.points()[tri[0]];
            return 0.5 * std::abs(ab.x() * ac.y() - ab.y() * ac.x());
            };

        for (std::size_t t = 0; t < n_triangles; ++t) {
            const auto& tri = mesh.triangles()[t];
            Coord area = triangle_area(tri);
            // Каждой вершине прибавляем 1/3 площади треугольника
            for (int i = 0; i < 3; ++i) {
                areas[tri[i]] += area / 3.0;
            }
        }
        return areas;
    }

} // namespace delta::numerical