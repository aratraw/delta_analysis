// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/numerical/gauge_theory.h
// ============================================================================
// DISCRETE GAUGE THEORY ON SIMPLICIAL COMPLEXES
// ============================================================================
//
// GaugeField<Group, Complex> stores an element of the gauge group on every
// oriented edge of a simplicial complex.  It provides Wilson action,
// gauge transformation, variation (for U(1)), and conversion to/from the
// geometric Connection class.
//
// ============================================================================

#pragma once

#include <unordered_map>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include "delta/geometry/simplicial_complex.h"
#include "delta/geometry/connection.h"
#include "delta/numerical/gauge_groups.h"
#include "delta/core/eigen_integration.h"   

namespace delta::numerical {

    template<typename Group, typename Complex>
    class GaugeField {
    public:
        using scalar_type = typename Complex::scalar_type;
        using vertex_index = typename Complex::vertex_index;
        using element_type = typename Group::matrix_type;
        using algebra_type = typename Group::algebra_type;

        /**
         * @brief Construct a trivial gauge field (all links = identity).
         */
        explicit GaugeField(const Complex& mesh)
            : mesh_(mesh)
        {
            for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
                auto [v0, v1] = mesh.edge_at(e);
                set_link(v0, v1, Group::identity());
            }
        }

        /**
         * @brief Set the link variable on the oriented edge (v0 → v1).
         */
        void set_link(vertex_index v0, vertex_index v1, const element_type& U) {
            links_[encode(v0, v1)] = U;
            links_[encode(v1, v0)] = U.inverse();
        }

        /**
         * @brief Get the link variable on the oriented edge (v0 → v1).
         */
        element_type get_link(vertex_index v0, vertex_index v1) const {
            auto it = links_.find(encode(v0, v1));
            if (it == links_.end())
                throw std::out_of_range("GaugeField::get_link: edge not found");
            return it->second;
        }

        /**
         * @brief Wilson action.
         * @param beta  Coupling constant.
         */
        scalar_type wilson_action(scalar_type beta = 1) const {
            scalar_type S = 0;
            for (std::size_t t = 0; t < mesh_.num_triangles(); ++t) {
                auto tri = mesh_.triangle_at(t);
                element_type U_tri = get_link(tri[0], tri[1])
                    * get_link(tri[1], tri[2])
                    * get_link(tri[2], tri[0]);
                // For U(1) (SO(2)) trace = 2 cos θ
                // For SU(2) trace is complex
                S += beta * (1 - (1.0 / Group::N) * Group::real_trace(U_tri));
            }
            return S;
        }

        /**
         * @brief Apply a gauge transformation.
         * @param gauge_factors  Vector of group elements for each vertex.
         */
        void gauge_transform(const std::vector<element_type>& gauge_factors) {
            if (gauge_factors.size() != mesh_.num_vertices())
                throw std::invalid_argument("gauge_transform: wrong number of factors");

            for (std::size_t e = 0; e < mesh_.num_edges(); ++e) {
                auto [v0, v1] = mesh_.edge_at(e);
                element_type U = get_link(v0, v1);
                element_type g0 = gauge_factors[v0];
                element_type g1 = gauge_factors[v1];
                element_type U_new = g1 * U * g0.inverse();
                set_link(v0, v1, U_new);
            }
        }

        /**
         * @brief Variation of the Wilson action w.r.t. a link (U(1) only).
         *
         * Returns the derivative dS/dθ for the edge with given index.
         */
        algebra_type variation(std::size_t edge_idx, scalar_type beta = 1) const
            requires (Group::N == 1)   // only for U(1)
        {
            auto [v0, v1] = mesh_.edge_at(edge_idx);
            scalar_type deriv = 0;

            // Find all triangles incident to this edge
            for (std::size_t t = 0; t < mesh_.num_triangles(); ++t) {
                auto tri = mesh_.triangle_at(t);
                // Determine orientation of the edge in this triangle
                for (int i = 0; i < 3; ++i) {
                    vertex_index a = tri[i];
                    vertex_index b = tri[(i + 1) % 3];
                    if ((a == v0 && b == v1) || (a == v1 && b == v0)) {
                        element_type U_tri = get_link(tri[0], tri[1])
                            * get_link(tri[1], tri[2])
                            * get_link(tri[2], tri[0]);
                        // For SO(2), trace = 2 cos Θ;  sin Θ is computed from the rotation matrix
                        // U_tri(1,0) = sin Θ
                        scalar_type sin_theta = U_tri(1, 0);
                        // The sign of the contribution depends on orientation
                        if (a == v0 && b == v1)
                            deriv += beta * sin_theta;
                        else
                            deriv -= beta * sin_theta;
                        break;
                    }
                }
            }

            algebra_type var;
            var << 0, -deriv,
                deriv, 0;
            return var;
        }

    private:
        const Complex& mesh_;
        std::unordered_map<std::uint64_t, element_type> links_;

        std::uint64_t encode(vertex_index a, vertex_index b) const {
            return (static_cast<std::uint64_t>(a) << 32) | static_cast<std::uint64_t>(b);
        }
    };

    // -------------------------------------------------------------------------
    // Conversion functions
    // -------------------------------------------------------------------------

    /**
     * @brief Convert a GaugeField to a Connection.
     */
    template<typename Group, typename Complex>
    delta::geometry::Connection<
        typename Complex::vertex_index,
        typename Complex::scalar_type,
        Complex::Dimension,
        typename Group::matrix_type>
        gauge_field_to_connection(const GaugeField<Group, Complex>& gf,
            const Complex& mesh) {
        using Conn = delta::geometry::Connection<
            typename Complex::vertex_index,
            typename Complex::scalar_type,
            Complex::Dimension,
            typename Group::matrix_type>;
        Conn conn;
        for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
            auto [v0, v1] = mesh.edge_at(e);
            conn.set_transport(v0, v1, gf.get_link(v0, v1));
        }
        return conn;
    }

    /**
     * @brief Convert a Connection to a GaugeField.
     */
    template<typename Group, typename Complex>
    GaugeField<Group, Complex>
        connection_to_gauge_field(
            const delta::geometry::Connection<
            typename Complex::vertex_index,
            typename Complex::scalar_type,
            Complex::Dimension,
            typename Group::matrix_type>& conn,
            const Complex& mesh) {
        GaugeField<Group, Complex> gf(mesh);
        for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
            auto [v0, v1] = mesh.edge_at(e);
            gf.set_link(v0, v1, conn.get_transport(v0, v1));
        }
        return gf;
    }

} // namespace delta::numerical