// include/delta/geometry/barycentric_subdivision.h
#pragma once

#include "delta/core/grid_concept.h"
#include "delta/core/regulative_idea.h"
#include <unordered_map>
#include <boost/container_hash/hash.hpp>
#include <stdexcept>

namespace delta::geometry {

    /**
     * @brief Barycentric subdivision of a simplicial complex.
     *
     * For each edge, a midpoint is inserted. For each triangle, a centroid is inserted.
     * Then each triangle is subdivided into 6 smaller triangles by connecting the
     * centroid to the edge midpoints and vertices.
     *
     * @tparam Complex Type satisfying SimplicialComplex concept.
     *                 The vertex type must satisfy LinearAddress (to compute midpoints).
     * @param in The input complex.
     * @return A new complex of the same type after barycentric subdivision.
     */
    template<typename Complex>
        requires SimplicialComplex<Complex>&&
    LinearAddress<typename Complex::point_type>
        Complex barycentric_subdivide(const Complex& in) {
        using Coord = typename Complex::point_type;
        using Index = std::size_t;

        Complex result;

        // Copy existing vertices
        for (Index i = 0; i < in.size(); ++i) {
            result.add_vertex(in.vertex(i));
        }

        using EdgeKey = std::pair<Index, Index>;
        std::unordered_map<EdgeKey, Index, boost::hash<EdgeKey>> edge_midpoint;
        std::unordered_map<Index, Index> triangle_centroid;

        // Create edge midpoints
        for (Index e = 0; e < in.num_edges(); ++e) {
            auto [v0, v1] = in.edge(e);
            Coord p0 = in.vertex(v0);
            Coord p1 = in.vertex(v1);
            Coord mid = (p0 + p1) / Coord{ 2 };
            Index mid_idx = result.add_vertex(mid);
            edge_midpoint[{std::min(v0, v1), std::max(v0, v1)}] = mid_idx;
        }

        // Create triangle centroids
        for (Index t = 0; t < in.num_triangles(); ++t) {
            auto [v0, v1, v2] = in.triangle(t);
            Coord p0 = in.vertex(v0);
            Coord p1 = in.vertex(v1);
            Coord p2 = in.vertex(v2);
            Coord centroid = (p0 + p1 + p2) / Coord{ 3 };
            Index c_idx = result.add_vertex(centroid);
            triangle_centroid[t] = c_idx;
        }

        // Subdivide each triangle into 6 new triangles
        for (Index t = 0; t < in.num_triangles(); ++t) {
            auto [v0, v1, v2] = in.triangle(t);
            Index c = triangle_centroid[t];

            auto get_mid = [&](Index a, Index b) -> Index {
                EdgeKey key = { std::min(a, b), std::max(a, b) };
                auto it = edge_midpoint.find(key);
                if (it == edge_midpoint.end()) {
                    throw std::runtime_error("Edge midpoint not found");
                }
                return it->second;
                };

            Index m01 = get_mid(v0, v1);
            Index m12 = get_mid(v1, v2);
            Index m20 = get_mid(v2, v0);

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