// include/delta/numerical/cotangent_laplacian.h
#pragma once

#include <Eigen/Sparse>
#include <vector>
#include <cmath>
#include "delta/geometry/simplicial_complex.h"
#include "delta/core/rational.h"

namespace delta::numerical {

    template<typename Coord = Rational>
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

        std::vector<Coord> diag(n_vertices, Coord(0));
        std::vector<Coord> edge_cot_sum(n_edges, Coord(0));

        auto get_edge_vector = [&](vertex_index i, vertex_index j) -> Eigen::Vector<Coord, 2> {
            return mesh.points()[j] - mesh.points()[i];
            };

        auto triangle_area = [&](const triangle& tri) -> Coord {
            auto ab = get_edge_vector(tri[0], tri[1]);
            auto ac = get_edge_vector(tri[0], tri[2]);
            Coord det = ab.x() * ac.y() - ab.y() * ac.x();
            using std::abs;
            return abs(det) / Coord(2);
            };

        auto cotangent_at_vertex = [&](vertex_index i, vertex_index j, vertex_index k) -> Coord {
            auto ki = get_edge_vector(k, i);
            auto kj = get_edge_vector(k, j);
            auto ij = get_edge_vector(i, j);
            Coord ki_len2 = ki.dot(ki);
            Coord kj_len2 = kj.dot(kj);
            Coord ij_len2 = ij.dot(ij);
            Coord det = ki.x() * kj.y() - ki.y() * kj.x();
            using std::abs;
            Coord area2 = abs(det); // удвоенная площадь
            if (area2 == Coord(0)) return Coord(0);
            return (ki_len2 + kj_len2 - ij_len2) / (Coord(2) * area2);
            };

        for (std::size_t t = 0; t < n_triangles; ++t) {
            const auto& tri = mesh.triangles()[t];
            vertex_index v0 = tri[0], v1 = tri[1], v2 = tri[2];

            Coord area = triangle_area(tri);
            if (area <= Coord(0)) continue;

            Coord cot0 = cotangent_at_vertex(v1, v2, v0);
            Coord cot1 = cotangent_at_vertex(v2, v0, v1);
            Coord cot2 = cotangent_at_vertex(v0, v1, v2);

            auto e01 = mesh.find_edge(v0, v1);
            auto e12 = mesh.find_edge(v1, v2);
            auto e20 = mesh.find_edge(v2, v0);

            if (e01 >= 0) edge_cot_sum[static_cast<std::size_t>(e01)] += cot2;
            if (e12 >= 0) edge_cot_sum[static_cast<std::size_t>(e12)] += cot0;
            if (e20 >= 0) edge_cot_sum[static_cast<std::size_t>(e20)] += cot1;
        }

        for (std::size_t e = 0; e < n_edges; ++e) {
            const auto& edge = mesh.edges()[e];
            vertex_index i = edge[0];
            vertex_index j = edge[1];
            Coord w = Coord(-1) * edge_cot_sum[e] / Coord(2);

            // Явное приведение к int (StorageIndex Eigen'а)
            triplets.emplace_back(static_cast<int>(i), static_cast<int>(j), w);
            triplets.emplace_back(static_cast<int>(j), static_cast<int>(i), w);

            diag[i] -= w;
            diag[j] -= w;
        }

        for (std::size_t i = 0; i < n_vertices; ++i) {
            triplets.emplace_back(static_cast<int>(i), static_cast<int>(i), diag[i]);
        }

        Eigen::SparseMatrix<Coord> L(static_cast<int>(n_vertices), static_cast<int>(n_vertices));
        L.setFromTriplets(triplets.begin(), triplets.end());
        return L;
    }

} // namespace delta::numerical