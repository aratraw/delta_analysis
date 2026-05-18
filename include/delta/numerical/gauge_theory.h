// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/numerical/gauge_theory.h
// ============================================================================
// DISCRETE GAUGE THEORY ON SIMPLICIAL COMPLEXES
// ============================================================================
//
// GaugeField<Group, Complex> stores an element of the gauge group on every
// oriented edge of a simplicial complex.  It provides:
//   - Wilson action
//   - Gauge transformation
//   - Variation (derivative) of the action w.r.t. a link (analytic for U(1),
//     formula -β/(2N)*(M-M†) for SU(2) and SU(3))
//   - Parallel transport of vectors and covectors
//   - Conversion to/from the geometric Connection class
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
        using metric_scalar_type = typename Complex::scalar_type;      // Rational
        using group_scalar_type = typename Group::ScalarType;          // Rational (U1) or GaussQi (SU2/SU3)
        using vertex_index = typename Complex::vertex_index;
        using element_type = typename Group::matrix_type;
        using algebra_type = typename Group::algebra_type;

        explicit GaugeField(const Complex& mesh) : mesh_(mesh) {
            for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
                auto [v0, v1] = mesh.edge_at(e);
                set_link(v0, v1, Group::identity());
            }
        }

        void set_link(vertex_index v0, vertex_index v1, const element_type& U) {
            links_[encode(v0, v1)] = U;
            links_[encode(v1, v0)] = U.inverse();
        }

        element_type get_link(vertex_index v0, vertex_index v1) const {
            auto it = links_.find(encode(v0, v1));
            if (it == links_.end())
                throw std::out_of_range("GaugeField::get_link: edge not found");
            return it->second;
        }

        // Wilson action: returns a metric scalar (Rational)
        metric_scalar_type wilson_action(metric_scalar_type beta = 1) const {
            metric_scalar_type S = 0;
            for (std::size_t t = 0; t < mesh_.num_triangles(); ++t) {
                auto tri = mesh_.triangle_at(t);
                element_type U_tri = get_link(tri[0], tri[1])
                    * get_link(tri[1], tri[2])
                    * get_link(tri[2], tri[0]);
                S += beta * (1 - (metric_scalar_type(1) / Group::N) * Group::real_trace(U_tri));
            }
            return S;
        }

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

        // Variation of the Wilson action with respect to the link at edge `edge_idx`.
        // Returns an algebra element (matrix over group_scalar_type).
        algebra_type variation(std::size_t edge_idx, metric_scalar_type beta = 1) const {
            auto [v0, v1] = mesh_.edge_at(edge_idx);
            const auto N = Group::N;
            algebra_type grad = algebra_type::Zero();

            for (std::size_t t = 0; t < mesh_.num_triangles(); ++t) {
                auto tri = mesh_.triangle_at(t);
                int pos0 = -1, pos1 = -1;
                for (int i = 0; i < 3; ++i) {
                    if (tri[i] == v0) pos0 = i;
                    if (tri[i] == v1) pos1 = i;
                }
                if (pos0 == -1 || pos1 == -1) continue;

                // orientation sign: +1 if (v0→v1) matches cyclic order, -1 otherwise
                int sign = 0;
                if ((pos0 == 0 && pos1 == 1) || (pos0 == 1 && pos1 == 2) || (pos0 == 2 && pos1 == 0))
                    sign = 1;
                else if ((pos0 == 1 && pos1 == 0) || (pos0 == 2 && pos1 == 1) || (pos0 == 0 && pos1 == 2))
                    sign = -1;
                else continue;

                // third vertex
                vertex_index a = v0, b = v1, c = -1;
                for (int i = 0; i < 3; ++i) {
                    if (tri[i] != a && tri[i] != b) { c = tri[i]; break; }
                }
                // M = product of the two other edges in the orientation that yields
                // U_plaq = U_e * M when sign == +1, or U_plaq = M * U_e when sign == -1
                element_type M = get_link(b, c) * get_link(c, a);

                if constexpr (Group::N == 1) {
                    // U(1): dS/dθ = β * sin(Θ) * sign
                    element_type U_plaq = get_link(tri[0], tri[1])
                        * get_link(tri[1], tri[2])
                        * get_link(tri[2], tri[0]);
                    group_scalar_type sin_theta = U_plaq(1, 0);   // sin Θ for SO(2)
                    group_scalar_type contrib = beta * sin_theta * sign;
                    grad(0, 1) -= contrib;
                    grad(1, 0) += contrib;
                }
                else {
                    // SU(2) / SU(3): contribution = -sign * β/(2N) * (M - M†)
                    algebra_type term = M - M.adjoint();
                    group_scalar_type factor = -sign * group_scalar_type(beta) / (2 * N);
                    grad += factor * term;
                }
            }
            return grad;
        }

        // Parallel transport of a vector (column) – components are group_scalar_type
        template<typename Derived>
        Eigen::Matrix<group_scalar_type, Derived::RowsAtCompileTime, 1>
            parallel_transport_vector(vertex_index from, vertex_index to,
                const Eigen::MatrixBase<Derived>& v) const {
            return get_link(from, to) * v.template cast<group_scalar_type>();
        }

        // Parallel transport of a covector (row) – components are group_scalar_type
        template<typename Derived>
        Eigen::Matrix<group_scalar_type, 1, Derived::ColsAtCompileTime>
            parallel_transport_covector(vertex_index from, vertex_index to,
                const Eigen::MatrixBase<Derived>& w) const {
            return w.template cast<group_scalar_type>() * get_link(from, to).inverse();
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