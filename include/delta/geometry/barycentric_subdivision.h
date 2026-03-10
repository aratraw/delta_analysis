// include/delta/geometry/barycentric_subdivision.h
#pragma once

#include "simplicial_complex.h"
#include <unordered_map>

namespace delta::geometry {

    template<typename Coord>
    SimplicialComplex2D<Coord> barycentric_subdivide(const SimplicialComplex2D<Coord>& complex) {
        SimplicialComplex2D<Coord> result;

        // Copy existing vertices
        for (const auto& p : complex.points()) {
            result.add_vertex(p);
        }

        // Maps: original edge -> new vertex index (midpoint)
        std::unordered_map<std::pair<std::size_t, std::size_t>, std::size_t, PairHash> edge_midpoint;
        // Maps: original triangle -> new vertex index (centroid)
        std::unordered_map<std::size_t, std::size_t> triangle_centroid;

        // Create midpoints of edges
        for (std::size_t e = 0; e < complex.edges().size(); ++e) {
            auto [v0, v1] = complex.edges()[e];
            auto p0 = complex.points()[v0];
            auto p1 = complex.points()[v1];
            point_type mid = (p0 + p1) / 2.0;
            std::size_t mid_idx = result.add_vertex(mid);
            edge_midpoint[{std::min(v0, v1), std::max(v0, v1)}] = mid_idx;
        }

        // Create centroids of triangles
        for (std::size_t t = 0; t < complex.triangles().size(); ++t) {
            auto [v0, v1, v2] = complex.triangles()[t];
            auto p0 = complex.points()[v0];
            auto p1 = complex.points()[v1];
            auto p2 = complex.points()[v2];
            point_type centroid = (p0 + p1 + p2) / 3.0;
            std::size_t c_idx = result.add_vertex(centroid);
            triangle_centroid[t] = c_idx;
        }

        // Subdivide each triangle into 6 new triangles
        for (std::size_t t = 0; t < complex.triangles().size(); ++t) {
            auto [v0, v1, v2] = complex.triangles()[t];
            std::size_t c = triangle_centroid[t];

            auto get_mid = [&](std::size_t a, std::size_t b) -> std::size_t {
                auto key = std::make_pair(std::min(a, b), std::max(a, b));
                auto it = edge_midpoint.find(key);
                if (it == edge_midpoint.end()) throw std::runtime_error("Edge midpoint not found");
                return it->second;
                };

            std::size_t m01 = get_mid(v0, v1);
            std::size_t m12 = get_mid(v1, v2);
            std::size_t m20 = get_mid(v2, v0);

            // Add the 6 triangles (order matters for orientation)
            result.add_triangle(v0, m01, c);
            result.add_triangle(m01, v1, c);
            result.add_triangle(v1, m12, c);
            result.add_triangle(m12, v2, c);
            result.add_triangle(v2, m20, c);
            result.add_triangle(m20, v0, c);
        }

        return result;
    }

} // namespace delta::geometry