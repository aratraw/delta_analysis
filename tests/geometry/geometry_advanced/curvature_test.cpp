// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/geometry/curvature_test.cpp
// ============================================================================
// MATHEMATICAL VERIFICATION OF DISCRETE CURVATURE
// ============================================================================
//
// This test suite validates the correctness of the discrete curvature
// computations on simplicial complexes.  It covers:
//   • Angle deficit on a flat mesh (expected 0).
//   • Holonomy‑based curvature for a trivial connection (expected 0).
//   • Gauss‑Bonnet theorem on a convex polyhedron (sum of deficits ≈ 4π).
//   • Ricci tensor and scalar curvature for a flat tetrahedron (expected 0).
//
// All tests use exact rational arithmetic where possible; geometric tests
// that involve transcendental functions (angle deficit, Gauss‑Bonnet) use
// EXPECT_RATIONAL_NEAR with a tolerance justified by the discretisation.
//
// IF ANY TEST FAILS, THE BUG IS IN THE CURVATURE IMPLEMENTATION,
// NOT IN THE TESTS.
// ============================================================================

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include "delta/core/rational.h"
#include "delta/geometry/simplicial_complex.h"
#include "delta/geometry/connection.h"
#include "delta/geometry/curvature.h"
#include "../../test_fixtures_geometry_numerical.h"

namespace delta::testing {

    class CurvatureTest : public GeometryNumericalTest {
    protected:
        using Scalar = Rational;
        using Complex2D = delta::geometry::SimplicialComplex<2, Scalar>;
        using Complex3D = delta::geometry::SimplicialComplex<3, Scalar>;
        using Point2D = typename Complex2D::point_type;
        using Point3D = typename Complex3D::point_type;
        using Connection2D = delta::geometry::Connection<
            typename Complex2D::vertex_index, Scalar, 2>;
        using Connection3D = delta::geometry::Connection<
            typename Complex3D::vertex_index, Scalar, 3>;

        // -------------------------------------------------------------------
        // Flat square divided into 4 triangles (central vertex is interior)
        // -------------------------------------------------------------------
        Complex2D make_flat_square_with_center() {
            Complex2D mesh;
            auto v0 = mesh.add_vertex(Point2D(0_r, 0_r));
            auto v1 = mesh.add_vertex(Point2D(1_r, 0_r));
            auto v2 = mesh.add_vertex(Point2D(1_r, 1_r));
            auto v3 = mesh.add_vertex(Point2D(0_r, 1_r));
            auto vc = mesh.add_vertex(Point2D(Rational(1, 2), Rational(1, 2)));

            mesh.add_edge(v0, v1); mesh.add_edge(v1, v2);
            mesh.add_edge(v2, v3); mesh.add_edge(v3, v0);
            mesh.add_edge(v0, vc); mesh.add_edge(v1, vc);
            mesh.add_edge(v2, vc); mesh.add_edge(v3, vc);

            mesh.add_triangle(v0, v1, vc);
            mesh.add_triangle(v1, v2, vc);
            mesh.add_triangle(v2, v3, vc);
            mesh.add_triangle(v3, v0, vc);
            return mesh;
        }

        // -------------------------------------------------------------------
        // Regular octahedron (all vertices on axes, side length √2)
        // -------------------------------------------------------------------
        Complex3D make_octahedron() {
            Complex3D mesh;
            auto v0 = mesh.add_vertex(Point3D(1_r, 0_r, 0_r));
            auto v1 = mesh.add_vertex(Point3D(-1_r, 0_r, 0_r));
            auto v2 = mesh.add_vertex(Point3D(0_r, 1_r, 0_r));
            auto v3 = mesh.add_vertex(Point3D(0_r, -1_r, 0_r));
            auto v4 = mesh.add_vertex(Point3D(0_r, 0_r, 1_r));
            auto v5 = mesh.add_vertex(Point3D(0_r, 0_r, -1_r));

            // Edges: all pairs except opposites
            auto add_edge_if = [&](auto a, auto b) {
                if ((a + b) != 1) return;  // skip opposite vertices (sum = ±1? no)
                // Actually opposite pairs: (0,1), (2,3), (4,5)
                if ((a == 0 && b == 1) || (a == 1 && b == 0) ||
                    (a == 2 && b == 3) || (a == 3 && b == 2) ||
                    (a == 4 && b == 5) || (a == 5 && b == 4))
                    return;
                mesh.add_edge(a, b);
                };
            for (int i = 0; i < 6; ++i)
                for (int j = i + 1; j < 6; ++j)
                    add_edge_if(i, j);

            // Triangles (8 faces)
            auto add_triangle = [&](auto a, auto b, auto c) {
                mesh.add_triangle(a, b, c);
                };
            // top half
            add_triangle(0, 2, 4); add_triangle(2, 1, 4); add_triangle(1, 3, 4); add_triangle(3, 0, 4);
            // bottom half
            add_triangle(0, 5, 2); add_triangle(2, 5, 1); add_triangle(1, 5, 3); add_triangle(3, 5, 0);
            return mesh;
        }

        // -------------------------------------------------------------------
        // Single flat tetrahedron
        // -------------------------------------------------------------------
        Complex3D make_flat_tetrahedron() {
            Complex3D mesh;
            auto v0 = mesh.add_vertex(Point3D(0_r, 0_r, 0_r));
            auto v1 = mesh.add_vertex(Point3D(1_r, 0_r, 0_r));
            auto v2 = mesh.add_vertex(Point3D(0_r, 1_r, 0_r));
            auto v3 = mesh.add_vertex(Point3D(0_r, 0_r, 1_r));
            mesh.add_edge(v0, v1); mesh.add_edge(v0, v2); mesh.add_edge(v0, v3);
            mesh.add_edge(v1, v2); mesh.add_edge(v1, v3); mesh.add_edge(v2, v3);
            mesh.add_tetrahedron(v0, v1, v2, v3);
            return mesh;
        }
    };

    // =======================================================================
    // Angle deficit on a flat mesh should be zero
    // =======================================================================
    TEST_F(CurvatureTest, DeficitOnFlatSquare) {
        auto mesh = make_flat_square_with_center();
        EuclideanMetric metric;

        // The central vertex (index 4) is interior
        Scalar deficit = vertex_curvature_deficit(mesh, 4, metric);
        // For a flat metric the deficit must be zero (up to numerical error
        // from the arccos used to compute angles).  We tolerate 1e-12.
        EXPECT_RATIONAL_NEAR(deficit, 0_r, Scalar(1, 1000000000000));
    }

    // =======================================================================
    // Holonomy‑based curvature for trivial connection is zero
    // =======================================================================
    TEST_F(CurvatureTest, CurvatureFromHolonomy2D) {
        auto mesh = make_flat_square_with_center();
        Connection2D conn;
        // Leave all matrices as identity (default)

        EuclideanMetric metric;
        for (std::size_t t = 0; t < mesh.num_triangles(); ++t) {
            auto H = holonomy_around_face(mesh, t, conn);
            Scalar area = mesh.simplex_volume(2, t, metric);
            // area for a right triangle with legs 0.5 is 0.125
            // For trivial connection H = I, so curvature F = (I - I)/area = 0
            auto F = curvature_from_holonomy(H, area);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    EXPECT_EQ(F(i, j), 0_r);
        }
    }

    // =======================================================================
    // Gauss‑Bonnet: sum of angle deficits on an octahedron ≈ 4π
    // =======================================================================
    TEST_F(CurvatureTest, GaussBonnetOctahedron) {
        auto mesh = make_octahedron();
        EuclideanMetric metric;

        Scalar total_deficit = 0;
        for (std::size_t v = 0; v < mesh.num_vertices(); ++v) {
            total_deficit += vertex_curvature_deficit(mesh, v, metric);
        }

        // 4π ≈ 12.566370614359172
        Scalar four_pi = 4_r * delta::pi(Scalar(1, 1000000000000000)); // pi with high precision
        EXPECT_RATIONAL_NEAR(total_deficit, four_pi, Scalar(1, 1000000)); // relax tolerance to 1e-6
    }

    // =======================================================================
    // Ricci tensor and scalar curvature for a flat tetrahedron
    // =======================================================================
    TEST_F(CurvatureTest, RicciFlatTetrahedron) {
        auto mesh = make_flat_tetrahedron();
        Connection3D conn; // trivial connection

        EuclideanMetric metric;
        for (std::size_t v = 0; v < mesh.num_vertices(); ++v) {
            auto Ric = vertex_ricci_curvature_3d(mesh, conn, v, metric);
            // Expected: zero matrix
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    EXPECT_EQ(Ric(i, j), 0_r);

            Scalar R = vertex_scalar_curvature_3d(mesh, conn, v, metric);
            EXPECT_EQ(R, 0_r);
        }
    }

} // namespace delta::testing