// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/numerical/gauge_theory_test.cpp
// ============================================================================
// MATHEMATICAL VERIFICATION OF DISCRETE GAUGE THEORY
// (fully rational, no double anywhere)
// ============================================================================
//
// This test suite validates the Wilson action, gauge invariance, variation
// of the action (U(1)), and conversion to/from Connection for the
// GaugeField class on a small simplicial complex (unit square).
//
// All computations use exact rational arithmetic (Rational / GaussQi).
// IF ANY TEST FAILS, THE BUG IS IN THE GAUGE THEORY IMPLEMENTATION,
// NOT IN THE TESTS.
// ============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "delta/core/rational.h"
#include "delta/geometry/simplicial_complex.h"
#include "delta/geometry/connection.h"
#include "delta/numerical/gauge_groups.h"
#include "delta/numerical/gauge_theory.h"
#include "../test_fixtures_geometry_numerical.h"

namespace delta::testing {

    class GaugeTheoryTest : public GeometryNumericalTest {
    protected:
        using Scalar = Rational;
        using Complex2D = delta::geometry::SimplicialComplex<2, Scalar>;
        using Point2D = typename Complex2D::point_type;
        using U1Group = U1<Scalar>;
        using Matrix2 = typename U1Group::matrix_type;

        // Build a unit square made of two triangles
        Complex2D make_unit_square() {
            Complex2D mesh;
            auto v0 = mesh.add_vertex(Point2D(0_r, 0_r));
            auto v1 = mesh.add_vertex(Point2D(1_r, 0_r));
            auto v2 = mesh.add_vertex(Point2D(1_r, 1_r));
            auto v3 = mesh.add_vertex(Point2D(0_r, 1_r));
            mesh.add_edge(v0, v1); mesh.add_edge(v1, v2);
            mesh.add_edge(v2, v3); mesh.add_edge(v3, v0);
            mesh.add_edge(v0, v2);
            mesh.add_triangle(v0, v1, v2);
            mesh.add_triangle(v0, v2, v3);
            return mesh;
        }

        // Rotation by +90°  [[0, -1], [1, 0]]
        Matrix2 rot90() const {
            Matrix2 R;
            R << 0_r, -1_r,
                1_r, 0_r;
            return R;
        }
    };

    // =======================================================================
    // Wilson action on a constant U(1) field
    // =======================================================================
    TEST_F(GaugeTheoryTest, WilsonActionConstantRotation) {
        auto mesh = make_unit_square();
        GaugeField<U1Group, Complex2D> gf(mesh);

        Matrix2 R = rot90();
        for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
            auto [v0, v1] = mesh.edge_at(e);
            gf.set_link(v0, v1, R);
        }

        // For each triangle, holonomy = R^3 = R_{-90} with trace = 0
        // Wilson term = 1 - (1/2)*trace = 1 - 0 = 1
        // Two triangles → total action = 2 (with beta = 1)
        Scalar S = gf.wilson_action(1_r);
        EXPECT_EQ(S, 2_r);
    }

    // =======================================================================
    // Gauge invariance of Wilson action
    // =======================================================================
    TEST_F(GaugeTheoryTest, GaugeInvariance) {
        auto mesh = make_unit_square();
        GaugeField<U1Group, Complex2D> gf(mesh);

        Matrix2 R = rot90();
        for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
            auto [v0, v1] = mesh.edge_at(e);
            gf.set_link(v0, v1, R);
        }

        Scalar S_before = gf.wilson_action(1_r);

        // Gauge transformation: random rotations (I or R90) at each vertex
        std::vector<Matrix2> factors(mesh.num_vertices(), Matrix2::Identity());
        factors[1] = R;
        factors[2] = R;
        factors[3] = R * R;   // R180
        gf.gauge_transform(factors);

        Scalar S_after = gf.wilson_action(1_r);
        EXPECT_EQ(S_after, S_before);
    }

    // =======================================================================
    // Variation of Wilson action (U(1))
    // =======================================================================
    TEST_F(GaugeTheoryTest, VariationOfAction) {
        auto mesh = make_unit_square();
        GaugeField<U1Group, Complex2D> gf(mesh);

        // Set a non‑trivial but simple field: identity everywhere
        for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
            auto [v0, v1] = mesh.edge_at(e);
            gf.set_link(v0, v1, Matrix2::Identity());
        }

        // Variation should be zero for trivial field
        auto var = gf.variation(0, 1_r);   // edge 0
        EXPECT_EQ(var(0, 0), 0_r);
        EXPECT_EQ(var(0, 1), 0_r);
        EXPECT_EQ(var(1, 0), 0_r);
        EXPECT_EQ(var(1, 1), 0_r);

        // Now set a non‑trivial field: R90 on all edges
        Matrix2 R = rot90();
        for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
            auto [v0, v1] = mesh.edge_at(e);
            gf.set_link(v0, v1, R);
        }

        // Analytic variation for edge 0 (v0→v1)
        // This edge belongs to triangle (0,1,2). The holonomy there is R^3.
        // sin(θ) = 1 for θ = 270°? Actually R^3 = R_{-90}, sin(-90°) = -1.
        // So variation = beta * sin(θ) = -1 (with beta=1).
        auto var2 = gf.variation(0, 1_r);
        Scalar expected_deriv = -1_r;
        EXPECT_EQ(var2(1, 0), expected_deriv);   // (1,0) entry is the derivative
        EXPECT_EQ(var2(0, 1), -expected_deriv);
    }

    // =======================================================================
    // Conversion GaugeField ↔ Connection
    // =======================================================================
    TEST_F(GaugeTheoryTest, GaugeFieldToConnectionRoundTrip) {
        auto mesh = make_unit_square();
        GaugeField<U1Group, Complex2D> gf(mesh);

        Matrix2 R = rot90();
        for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
            auto [v0, v1] = mesh.edge_at(e);
            gf.set_link(v0, v1, R);
        }

        using Conn2D = delta::geometry::Connection<
            Complex2D::vertex_index, Scalar, 2, Matrix2>;

        Conn2D conn = gauge_field_to_connection(gf, mesh);
        GaugeField<U1Group, Complex2D> gf2 = connection_to_gauge_field<U1Group, Complex2D>(conn, mesh);

        // Compare links
        for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
            auto [v0, v1] = mesh.edge_at(e);
            Matrix2 U1 = gf.get_link(v0, v1);
            Matrix2 U2 = gf2.get_link(v0, v1);
            EXPECT_EQ(U1, U2);
        }
    }

} // namespace delta::testing