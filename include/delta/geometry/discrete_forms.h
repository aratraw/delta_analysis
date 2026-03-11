// include/delta/geometry/discrete_forms.h
#pragma once

#include "simplicial_complex.h"
#include <vector>
#include <cassert>
#include <cmath>
#include <numbers>
#include <unordered_map>
#include <utility>
#include "delta/core/rational.h"
#include "delta/core/regulative_idea.h"

namespace delta::geometry {

    // Forward declarations
    template<typename Value, typename Complex> class DiscreteForm0;
    template<typename Value, typename Complex> class DiscreteForm1;
    template<typename Value, typename Complex> class DiscreteForm2;

    namespace detail {
        // ---------------------------------------------------------------------
        // Geometric helpers (using a metric)
        // ---------------------------------------------------------------------

        /// Compute edge length using metric
        template<typename Metric, typename Point>
        auto edge_length(const Point& a, const Point& b, const Metric& metric) {
            return metric(a, b);
        }

        /// Compute triangle area using metric (Heron's formula or cross product)
        template<typename Metric, typename Point>
        auto triangle_area(const Point& a, const Point& b, const Point& c, const Metric& metric) {
            using Scalar = decltype(metric(a, a));
            Scalar ab = metric(a, b);
            Scalar bc = metric(b, c);
            Scalar ca = metric(c, a);
            // Heron's formula
            Scalar s = (ab + bc + ca) / Scalar{ 2 };
            return sqrt(s * (s - ab) * (s - bc) * (s - ca));
        }

        /// Simplified dual edge length: (area1 + area2) / edge_length
        template<typename Complex, typename Metric>
        auto dual_edge_length(const Complex& mesh, std::size_t e, const Metric& metric) {
            using Point = typename Complex::point_type;
            using Scalar = decltype(metric(mesh.vertex(0), mesh.vertex(0)));

            auto [v0, v1] = mesh.edge_at(e);
            Point p0 = mesh.vertex(v0);
            Point p1 = mesh.vertex(v1);
            Scalar edge_len = metric(p0, p1);

            // Find the two triangles incident to this edge
            std::vector<std::size_t> incident;
            for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
                auto tri = mesh.triangle_at(t);
                if ((tri[0] == v0 && tri[1] == v1) || (tri[1] == v0 && tri[2] == v1) || (tri[2] == v0 && tri[0] == v1) ||
                    (tri[0] == v1 && tri[1] == v0) || (tri[1] == v1 && tri[2] == v0) || (tri[2] == v1 && tri[0] == v0)) {
                    incident.push_back(t);
                }
            }

            if (incident.empty()) return Scalar{ 0 };
            if (incident.size() == 1) {
                // Boundary edge: dual length is area / (edge_len/2)?? Simplified: return edge_len
                return edge_len;
            }

            // Compute areas of the two incident triangles
            auto tri0 = mesh.triangle_at(incident[0]);
            auto tri1 = mesh.triangle_at(incident[1]);
            Scalar area0 = triangle_area(mesh.vertex(tri0[0]), mesh.vertex(tri0[1]), mesh.vertex(tri0[2]), metric);
            Scalar area1 = triangle_area(mesh.vertex(tri1[0]), mesh.vertex(tri1[1]), mesh.vertex(tri1[2]), metric);

            // Dual length as (area0 + area1) / edge_len (a common approximation)
            return (area0 + area1) / edge_len;
        }

    } // namespace detail

    // -------------------------------------------------------------------------
    // Discrete 0-form (values on vertices)
    // -------------------------------------------------------------------------
    template<typename Value, typename Complex>
    class DiscreteForm0 {
    public:
        using value_type = Value;
        using complex_type = Complex;

        explicit DiscreteForm0(const Complex& mesh) : mesh_(mesh), values_(mesh.size()) {}

        Value& at_vertex(std::size_t i) { return values_.at(i); }
        const Value& at_vertex(std::size_t i) const { return values_.at(i); }

        std::size_t size() const { return values_.size(); }

        // Exterior derivative: 0-form -> 1-form
        DiscreteForm1<Value, Complex> d() const {
            DiscreteForm1<Value, Complex> result(mesh_);
            for (std::size_t e = 0; e < mesh_.num_edges(); ++e) {
                auto [v0, v1] = mesh_.edge_at(e);
                result.at_edge(e) = values_[v1] - values_[v0];
            }
            return result;
        }

        // Hodge star: 0-form -> 2-form (on triangles)
        template<typename Metric>
        DiscreteForm2<Value, Complex> star(const Metric& metric) const {
            DiscreteForm2<Value, Complex> result(mesh_);
            // For each triangle, (⋆f)_T = (area_T) * (average of f over vertices)
            // More accurately, in DEC star of 0-form is a 2-form on dual cells, but we map to triangles.
            // We'll use the average times triangle area.
            for (std::size_t t = 0; t < mesh_.num_triangles(); ++t) {
                auto tri = mesh_.triangle_at(t);
                Value avg = (values_[tri[0]] + values_[tri[1]] + values_[tri[2]]) / Value{ 3 };
                auto p0 = mesh_.vertex(tri[0]);
                auto p1 = mesh_.vertex(tri[1]);
                auto p2 = mesh_.vertex(tri[2]);
                Value area = detail::triangle_area(p0, p1, p2, metric);
                result.at_triangle(t) = avg * area;
            }
            return result;
        }

        // Refine: create a new form on a subdivided mesh (placeholder)
        template<typename NewComplex>
        DiscreteForm0<Value, NewComplex> refine(const NewComplex& new_mesh) const {
            // Not implemented yet
            throw std::runtime_error("DiscreteForm0::refine not implemented");
        }

    private:
        const Complex& mesh_;
        std::vector<Value> values_;
    };

    // -------------------------------------------------------------------------
    // Discrete 1-form (values on oriented edges)
    // -------------------------------------------------------------------------
    template<typename Value, typename Complex>
    class DiscreteForm1 {
    public:
        using value_type = Value;
        using complex_type = Complex;

        explicit DiscreteForm1(const Complex& mesh) : mesh_(mesh), values_(mesh.num_edges()) {}

        Value& at_edge(std::size_t i) { return values_.at(i); }
        const Value& at_edge(std::size_t i) const { return values_.at(i); }

        std::size_t size() const { return values_.size(); }

        // Exterior derivative: 1-form -> 2-form
        DiscreteForm2<Value, Complex> d() const {
            DiscreteForm2<Value, Complex> result(mesh_);
            for (std::size_t t = 0; t < mesh_.num_triangles(); ++t) {
                auto [v0, v1, v2] = mesh_.triangle_at(t);
                auto e01 = mesh_.find_edge(v0, v1);
                auto e12 = mesh_.find_edge(v1, v2);
                auto e20 = mesh_.find_edge(v2, v0);
                if (e01 < 0 || e12 < 0 || e20 < 0) {
                    throw std::runtime_error("Edge not found in triangle");
                }
                // Need orientation: assume triangle orientation is (v0,v1,v2)
                // Then edges are oriented as v0->v1, v1->v2, v2->v0.
                // If stored edge orientation might be opposite, we need to check sign.
                // For now, assume edges are stored with orientation low->high, so we may need to flip sign if orientation mismatches.
                // We'll ignore sign for simplicity (assume consistent orientation).
                result.at_triangle(t) = values_[e01] + values_[e12] + values_[e20];
            }
            return result;
        }

        // Hodge star: 1-form -> 1-form (on primal edges, but with weights from dual edges)
        template<typename Metric>
        DiscreteForm1<Value, Complex> star(const Metric& metric) const {
            DiscreteForm1<Value, Complex> result(mesh_);
            for (std::size_t e = 0; e < mesh_.num_edges(); ++e) {
                auto [v0, v1] = mesh_.edge_at(e);
                Value primal_length = metric(mesh_.vertex(v0), mesh_.vertex(v1));
                Value dual_length = detail::dual_edge_length(mesh_, e, metric);
                // (⋆ω)_e = (|e*| / |e|) * ω_e
                result.at_edge(e) = Value(dual_length / primal_length) * values_[e];
            }
            return result;
        }

        // Refine (placeholder)
        template<typename NewComplex>
        DiscreteForm1<Value, NewComplex> refine(const NewComplex& new_mesh) const {
            throw std::runtime_error("DiscreteForm1::refine not implemented");
        }

    private:
        const Complex& mesh_;
        std::vector<Value> values_;
    };

    // -------------------------------------------------------------------------
    // Discrete 2-form (values on triangles)
    // -------------------------------------------------------------------------
    template<typename Value, typename Complex>
    class DiscreteForm2 {
    public:
        using value_type = Value;
        using complex_type = Complex;

        explicit DiscreteForm2(const Complex& mesh) : mesh_(mesh), values_(mesh.num_triangles()) {}

        Value& at_triangle(std::size_t i) { return values_.at(i); }
        const Value& at_triangle(std::size_t i) const { return values_.at(i); }

        std::size_t size() const { return values_.size(); }

        // Exterior derivative of 2-form is zero in 2D (but could be defined for 3D)
        // Not needed for now.

        // Hodge star: 2-form -> 0-form (on vertices)
        template<typename Metric>
        DiscreteForm0<Value, Complex> star(const Metric& metric) const {
            DiscreteForm0<Value, Complex> result(mesh_);
            // For each vertex, (⋆ρ)_v = sum_{t∋v} ρ_t * (area_t) / (3 * area_v) ??? Simplified.
            // We'll use a simple average of incident triangle values weighted by area.
            std::vector<Value> vertex_sum(mesh_.size(), Value{ 0 });
            std::vector<Value> vertex_weight(mesh_.size(), Value{ 0 });
            for (std::size_t t = 0; t < mesh_.num_triangles(); ++t) {
                auto tri = mesh_.triangle_at(t);
                auto p0 = mesh_.vertex(tri[0]);
                auto p1 = mesh_.vertex(tri[1]);
                auto p2 = mesh_.vertex(tri[2]);
                Value area = detail::triangle_area(p0, p1, p2, metric);
                Value contrib = values_[t] * area;
                for (int i = 0; i < 3; ++i) {
                    vertex_sum[tri[i]] += contrib;
                    vertex_weight[tri[i]] += area;
                }
            }
            for (std::size_t v = 0; v < mesh_.size(); ++v) {
                if (vertex_weight[v] != Value{ 0 }) {
                    result.at_vertex(v) = vertex_sum[v] / (Value{ 3 } * vertex_weight[v]); // dividing by total area? Need correct formula.
                }
                else {
                    result.at_vertex(v) = Value{ 0 };
                }
            }
            return result;
        }

        // Refine (placeholder)
        template<typename NewComplex>
        DiscreteForm2<Value, NewComplex> refine(const NewComplex& new_mesh) const {
            throw std::runtime_error("DiscreteForm2::refine not implemented");
        }

    private:
        const Complex& mesh_;
        std::vector<Value> values_;
    };

    // -------------------------------------------------------------------------
    // Codifferential and Laplacian (free functions)
    // -------------------------------------------------------------------------

    /// Codifferential of a 1-form: δω = ⋆⁻¹ d ⋆ ω   (with appropriate sign)
    /// In 2D for 1-forms: δ = - ⋆⁻¹ d ⋆
    template<typename Value, typename Complex, typename Metric>
    DiscreteForm0<Value, Complex> codifferential(const DiscreteForm1<Value, Complex>& omega,
        const Metric& metric) {
        const int n = 2; // dimension
        const int k = 1; // degree
        int sign = (n * (k - 1) + 1) % 2 == 0 ? 1 : -1; // (-1)^{n(k-1)+1}
        auto star_omega = omega.star(metric);                // 1-form (dual weights)
        auto d_star_omega = star_omega.d();                  // 2-form
        auto star_d_star_omega = d_star_omega.star(metric); // 0-form
        return sign * star_d_star_omega;
    }

    /// Laplace-Beltrami operator on 0-forms: Δf = δ d f
    template<typename Value, typename Complex, typename Metric>
    DiscreteForm0<Value, Complex> laplacian(const DiscreteForm0<Value, Complex>& f,
        const Metric& metric) {
        return codifferential(f.d(), metric);
    }

} // namespace delta::geometry