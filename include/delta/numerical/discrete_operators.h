// include/delta/numerical/discrete_operators.h
#pragma once

#include "delta/geometry/simplicial_complex.h"
#include "delta/core/regulative_idea.h"
#include <vector>
#include <cmath>

namespace delta::numerical {

    /**
     * @brief Discrete gradient of a scalar field on vertices (returns values on edges).
     *
     * For each oriented edge (i,j) (as stored in the mesh, low->high), computes
     * (f_j - f_i) / length(edge). The result vector has size num_edges().
     *
     * @tparam Complex Type satisfying SimplicialComplex concept.
     * @tparam Metric  Address metric.
     * @param mesh     The simplicial complex.
     * @param vertex_values Scalar values at vertices.
     * @param metric   Metric used to compute edge lengths.
     * @return std::vector<Value> Gradient values on edges.
     */
    template<typename Complex, typename Metric>
    auto discrete_gradient(const Complex& mesh,
        const std::vector<typename Complex::value_type>& vertex_values,
        const Metric& metric) {
        using Value = typename Complex::value_type;
        std::size_t n_edges = mesh.num_edges();
        std::vector<Value> gradient(n_edges);

        for (std::size_t e = 0; e < n_edges; ++e) {
            auto [v0, v1] = mesh.edge(e);
            Value len = metric(mesh.vertex(v0), mesh.vertex(v1));
            if (len == Value{ 0 }) {
                gradient[e] = Value{ 0 };
            }
            else {
                gradient[e] = (vertex_values[v1] - vertex_values[v0]) / len;
            }
        }
        return gradient;
    }

    /**
     * @brief Raw discrete divergence of an edge‑based field (1‑form) without area scaling.
     *
     * For each vertex, sums the signed edge values: contribution from edge (low,high) is
     * +value to low, -value to high.
     *
     * @tparam Complex SimplicialComplex.
     * @param mesh     The mesh.
     * @param edge_values Values on edges (same order as mesh.edges()).
     * @return std::vector<Value> Raw divergence at vertices.
     */
    template<typename Complex, typename Value>
    std::vector<Value> discrete_divergence_raw(const Complex& mesh,
        const std::vector<Value>& edge_values) {
        std::size_t n_vertices = mesh.size();
        std::vector<Value> div(n_vertices, Value{ 0 });

        for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
            auto [v0, v1] = mesh.edge(e);
            div[v0] += edge_values[e];
            div[v1] -= edge_values[e];
        }
        return div;
    }

    /**
     * @brief Discrete divergence scaled by vertex dual areas (Voronoi).
     *
     * Computes (δ ω)(v) = (1 / A_v) * raw_divergence(v), where A_v is the Voronoi area.
     *
     * @tparam Complex SimplicialComplex.
     * @tparam Metric  Address metric.
     * @param mesh     The mesh.
     * @param edge_values Values on edges.
     * @param metric   Metric used to compute areas.
     * @return std::vector<Value> Divergence at vertices.
     */
    template<typename Complex, typename Metric>
    auto discrete_divergence(const Complex& mesh,
        const std::vector<typename Complex::value_type>& edge_values,
        const Metric& metric) {
        using Value = typename Complex::value_type;
        std::size_t n_vertices = mesh.size();

        // Compute vertex areas (Voronoi)
        std::vector<Value> vertex_areas(n_vertices, Value{ 0 });

        // Approximate as 1/3 of sum of incident triangle areas
        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto [v0, v1, v2] = mesh.triangle(t);
            auto p0 = mesh.vertex(v0);
            auto p1 = mesh.vertex(v1);
            auto p2 = mesh.vertex(v2);
            Value area = triangle_area(p0, p1, p2, metric);
            vertex_areas[v0] += area / Value{ 3 };
            vertex_areas[v1] += area / Value{ 3 };
            vertex_areas[v2] += area / Value{ 3 };
        }

        auto raw = discrete_divergence_raw(mesh, edge_values);
        std::vector<Value> result(n_vertices);
        for (std::size_t i = 0; i < n_vertices; ++i) {
            if (vertex_areas[i] != Value{ 0 }) {
                result[i] = raw[i] / vertex_areas[i];
            }
            else {
                result[i] = Value{ 0 };
            }
        }
        return result;
    }

    /**
     * @brief Discrete Laplacian of a scalar field on vertices (Δf = div grad f).
     *
     * Uses the cotangent weights formula. This is the standard DEC Laplacian for 0-forms.
     *
     * @tparam Complex SimplicialComplex.
     * @tparam Metric  Address metric (must be Euclidean for cotangent formula).
     * @param mesh     The mesh.
     * @param vertex_values Scalar values at vertices.
     * @param metric   Metric used to compute lengths and angles.
     * @return std::vector<Value> Laplacian at vertices.
     */
    template<typename Complex, typename Metric>
    std::vector<typename Complex::value_type>
        discrete_laplacian_cotangent(const Complex& mesh,
            const std::vector<typename Complex::value_type>& vertex_values,
            const Metric& metric) {
        using Value = typename Complex::value_type;
        std::size_t n_vertices = mesh.size();
        std::vector<Value> laplacian(n_vertices, Value{ 0 });

        // Compute vertex areas (Voronoi)
        std::vector<Value> vertex_areas(n_vertices, Value{ 0 });
        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto [v0, v1, v2] = mesh.triangle(t);
            auto p0 = mesh.vertex(v0);
            auto p1 = mesh.vertex(v1);
            auto p2 = mesh.vertex(v2);
            Value area = triangle_area(p0, p1, p2, metric);
            vertex_areas[v0] += area / Value{ 3 };
            vertex_areas[v1] += area / Value{ 3 };
            vertex_areas[v2] += area / Value{ 3 };
        }

        // Edge cotangent weights
        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto [v0, v1, v2] = mesh.triangle(t);
            auto p0 = mesh.vertex(v0);
            auto p1 = mesh.vertex(v1);
            auto p2 = mesh.vertex(v2);

            // Compute cotangents using metric (Euclidean assumed)
            auto cot = [&](const auto& a, const auto& b, const auto& c) -> Value {
                // cot(angle at a) = ( (b-a)·(c-a) ) / (|(b-a)×(c-a)|)
                // For Euclidean metric, we can use dot and cross.
                // For general metric, need to compute angle properly.
                // Here we assume metric gives Euclidean dot product.
                auto ab = b - a;
                auto ac = c - a;
                Value dot = ab.dot(ac);
                Value cross = abs(ab.x() * ac.y() - ab.y() * ac.x());
                if (cross == Value{ 0 }) return Value{ 0 };
                return dot / cross;
                };

            Value cot0 = cot(p1, p2, p0); // angle at v0
            Value cot1 = cot(p2, p0, p1); // angle at v1
            Value cot2 = cot(p0, p1, p2); // angle at v2

            // Edge (v0,v1)
            auto e01 = mesh.find_edge(v0, v1);
            if (e01 >= 0) {
                Value w = cot2;
                laplacian[v0] += w * (vertex_values[v0] - vertex_values[v1]);
                laplacian[v1] += w * (vertex_values[v1] - vertex_values[v0]);
            }

            // Edge (v1,v2)
            auto e12 = mesh.find_edge(v1, v2);
            if (e12 >= 0) {
                Value w = cot0;
                laplacian[v1] += w * (vertex_values[v1] - vertex_values[v2]);
                laplacian[v2] += w * (vertex_values[v2] - vertex_values[v1]);
            }

            // Edge (v2,v0)
            auto e20 = mesh.find_edge(v2, v0);
            if (e20 >= 0) {
                Value w = cot1;
                laplacian[v2] += w * (vertex_values[v2] - vertex_values[v0]);
                laplacian[v0] += w * (vertex_values[v0] - vertex_values[v2]);
            }
        }

        // Divide by vertex areas
        for (std::size_t i = 0; i < n_vertices; ++i) {
            if (vertex_areas[i] != Value{ 0 }) {
                laplacian[i] /= (Value{ 2 } * vertex_areas[i]);
            }
        }
        return laplacian;
    }

    // -------------------------------------------------------------------------
    // Helper: triangle area using metric (Euclidean assumption for now)
    // -------------------------------------------------------------------------
    namespace detail {
        template<typename Point, typename Metric>
        auto triangle_area(const Point& a, const Point& b, const Point& c, const Metric& metric) {
            // For Euclidean metric, use cross product.
            auto ab = b - a;
            auto ac = c - a;
            return abs(ab.x() * ac.y() - ab.y() * ac.x()) / 2;
            // For general metric, we would need to compute area via lengths (Heron)
        }
    }

    template<typename Point, typename Metric>
    auto triangle_area(const Point& a, const Point& b, const Point& c, const Metric& metric) {
        return detail::triangle_area(a, b, c, metric);
    }

} // namespace delta::numerical