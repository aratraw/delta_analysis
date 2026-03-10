// include/delta/geometry/discrete_forms.h
#pragma once

#include "simplicial_complex.h"
#include <vector>
#include <cassert>

namespace delta::geometry {

    template<int Dim, typename ValueType = double>
    class DiscreteForm {
        static_assert(Dim == 2, "Only 2D implemented");
    public:
        explicit DiscreteForm(const SimplicialComplex2D<double>& complex) : complex_(complex) {}

        void initialize() {
            vertex_values_.assign(complex_.num_vertices(), ValueType{});
            edge_values_.assign(complex_.num_edges(), ValueType{});
            triangle_values_.assign(complex_.num_triangles(), ValueType{});
        }

        // Access to values (mutable)
        ValueType& at_vertex(std::size_t idx) { return vertex_values_.at(idx); }
        const ValueType& at_vertex(std::size_t idx) const { return vertex_values_.at(idx); }

        ValueType& at_edge(std::size_t idx) { return edge_values_.at(idx); }
        const ValueType& at_edge(std::size_t idx) const { return edge_values_.at(idx); }

        ValueType& at_triangle(std::size_t idx) { return triangle_values_.at(idx); }
        const ValueType& at_triangle(std::size_t idx) const { return triangle_values_.at(idx); }

        // Exterior derivative for 0-forms (returns 1-form)
        DiscreteForm<Dim, ValueType> exterior_derivative_0() const {
            DiscreteForm<Dim, ValueType> result(complex_);
            result.initialize();

            // df(e) = f(v1) - f(v0) for edge e = (v0,v1) with orientation from v0 to v1
            for (std::size_t e = 0; e < complex_.edges().size(); ++e) {
                auto [v0, v1] = complex_.edges()[e];
                // Note: edges are stored with v0 < v1 (sorted). So orientation is from v0 to v1.
                result.at_edge(e) = at_vertex(v1) - at_vertex(v0);
            }
            return result;
        }

        // Exterior derivative for 1-forms (returns 2-form) - not yet implemented
        // DiscreteForm<Dim, ValueType> exterior_derivative_1() const;

    private:
        const SimplicialComplex2D<double>& complex_;
        std::vector<ValueType> vertex_values_;
        std::vector<ValueType> edge_values_;
        std::vector<ValueType> triangle_values_;
    };

} // namespace delta::geometry