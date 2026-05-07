// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/geometry/hat_basis_test.cpp
// ============================================================================
// TESTS FOR HAT (LAGRANGE) BASIS FUNCTIONS ON SIMPLICIAL MESHES
// ============================================================================
//
// This file tests the HatBasis class on 2D triangle meshes. Verified properties:
//   - Interpolation of a linear function (exact up to rational arithmetic).
//   - Partition of unity and evaluation at vertices (φ_v(v)=1, φ_v(other)=0).
//   - Gradient of hat functions is constant on each triangle (analytic value).
//   - locate_point() correctly identifies the containing simplex.
//   - Barycentric coordinates match the values returned by evaluate().
//
// All tests use Euclidean geometry; the mesh is either a single triangle or
// a unit square split by a diagonal.
// ============================================================================

#include <gtest/gtest.h>
#include "delta/geometry/hat_basis.h"
#include "../../test_fixtures_geometry_numerical.h"

namespace delta::testing {

    class HatBasisTest : public GeometryNumericalTest {
    protected:
        using Scalar = Rational;
        using Point2D = Point<2>;

        // Helper: create a single right triangle (0,0)-(1,0)-(0,1)
        Complex<2> make_triangle_mesh() {
            Complex<2> mesh;
            auto v0 = add_vertex(mesh, Point2D(0_r, 0_r));
            auto v1 = add_vertex(mesh, Point2D(1_r, 0_r));
            auto v2 = add_vertex(mesh, Point2D(0_r, 1_r));
            add_edge(mesh, v0, v1);
            add_edge(mesh, v1, v2);
            add_edge(mesh, v2, v0);
            add_triangle(mesh, v0, v1, v2);
            return mesh;
        }

        // Helper: create a unit square split into two triangles along diagonal (0,2)
        Complex<2> make_unit_square_mesh() {
            Complex<2> mesh;
            auto v0 = add_vertex(mesh, Point2D(0_r, 0_r));
            auto v1 = add_vertex(mesh, Point2D(1_r, 0_r));
            auto v2 = add_vertex(mesh, Point2D(1_r, 1_r));
            auto v3 = add_vertex(mesh, Point2D(0_r, 1_r));
            add_edge(mesh, v0, v1);
            add_edge(mesh, v1, v2);
            add_edge(mesh, v2, v3);
            add_edge(mesh, v3, v0);
            add_edge(mesh, v0, v2); // diagonal
            add_triangle(mesh, v0, v1, v2);
            add_triangle(mesh, v0, v2, v3);
            return mesh;
        }
    };

    // -------------------------------------------------------------------------
    // Test: InterpolateLinearFunction
    // -------------------------------------------------------------------------
    /**
     * @test InterpolateLinearFunction
     * @brief Checks that linear functions are reproduced exactly (up to rational).
     *
     * For f(x,y) = x + y, interpolation at any point inside the triangle should
     * give the exact value. This verifies that the hat basis forms a partition of
     * unity and that barycentric coordinates are correct.
     */
    TEST_F(HatBasisTest, InterpolateLinearFunction) {
        auto mesh = make_triangle_mesh();
        HatBasis<Complex<2>> basis(mesh);

        // f(x,y) = x + y
        std::vector<Scalar> vertex_values(mesh.num_vertices());
        for (std::size_t i = 0; i < mesh.num_vertices(); ++i) {
            const auto& p = mesh.vertex(i);
            vertex_values[i] = p.x() + p.y();
        }

        // Test points inside and on edge
        Point2D p1("0.25"_r, "0.25"_r);
        Point2D p2("0.5"_r, "0.25"_r);
        Point2D p3("0.25"_r, "0.5"_r);
        Point2D p4("0.75"_r, 0_r);   // on the edge (v0,v1)

        Scalar val1 = basis.interpolate(p1, vertex_values);
        Scalar val2 = basis.interpolate(p2, vertex_values);
        Scalar val3 = basis.interpolate(p3, vertex_values);
        Scalar val4 = basis.interpolate(p4, vertex_values);

        Scalar eps = Rational(1, 1000000);
        EXPECT_RATIONAL_NEAR(val1, p1.x() + p1.y(), eps);
        EXPECT_RATIONAL_NEAR(val2, p2.x() + p2.y(), eps);
        EXPECT_RATIONAL_NEAR(val3, p3.x() + p3.y(), eps);
        EXPECT_RATIONAL_NEAR(val4, p4.x() + p4.y(), eps);
    }

    // -------------------------------------------------------------------------
    // Test: EvaluateHatFunctions
    // -------------------------------------------------------------------------
    /**
     * @test EvaluateHatFunctions
     * @brief Verifies φ_v(v)=1, φ_v(other vertices)=0, and matching barycentric coordinates.
     */
    TEST_F(HatBasisTest, EvaluateHatFunctions) {
        auto mesh = make_triangle_mesh();
        HatBasis<Complex<2>> basis(mesh);

        const std::size_t v0 = 0, v1 = 1, v2 = 2;

        // At vertices
        EXPECT_RATIONAL_NEAR(basis.evaluate(v0, mesh.vertex(v0)), 1_r, 0_r);
        EXPECT_RATIONAL_NEAR(basis.evaluate(v1, mesh.vertex(v0)), 0_r, 0_r);
        EXPECT_RATIONAL_NEAR(basis.evaluate(v2, mesh.vertex(v0)), 0_r, 0_r);

        EXPECT_RATIONAL_NEAR(basis.evaluate(v0, mesh.vertex(v1)), 0_r, 0_r);
        EXPECT_RATIONAL_NEAR(basis.evaluate(v1, mesh.vertex(v1)), 1_r, 0_r);
        EXPECT_RATIONAL_NEAR(basis.evaluate(v2, mesh.vertex(v1)), 0_r, 0_r);

        EXPECT_RATIONAL_NEAR(basis.evaluate(v0, mesh.vertex(v2)), 0_r, 0_r);
        EXPECT_RATIONAL_NEAR(basis.evaluate(v1, mesh.vertex(v2)), 0_r, 0_r);
        EXPECT_RATIONAL_NEAR(basis.evaluate(v2, mesh.vertex(v2)), 1_r, 0_r);

        // Inside point – compare evaluate() with barycentric coordinates
        Point2D p("0.2"_r, "0.3"_r);
        auto loc = basis.locate_point(p);
        ASSERT_TRUE(loc.has_value());
        const std::vector<Scalar>& bary = loc->second;

        Scalar eps = Rational(1, 1000000);
        EXPECT_RATIONAL_NEAR(basis.evaluate(v0, p), bary[0], eps);
        EXPECT_RATIONAL_NEAR(basis.evaluate(v1, p), bary[1], eps);
        EXPECT_RATIONAL_NEAR(basis.evaluate(v2, p), bary[2], eps);
    }

    // -------------------------------------------------------------------------
    // Test: GradientOfHatFunctionsConstantOnTriangle
    // -------------------------------------------------------------------------
    /**
     * @test GradientOfHatFunctionsConstantOnTriangle
     * @brief Checks that gradients are constant on each triangle and match the
     *        analytically known values for the reference triangle (0,0)-(1,0)-(0,1).
     *
     * The hat functions are:
     *   φ0 = 1 - x - y  → ∇φ0 = (-1, -1)
     *   φ1 = x          → ∇φ1 = (1, 0)
     *   φ2 = y          → ∇φ2 = (0, 1)
     *
     * The test evaluates gradients at an interior point and on an edge.
     */
    TEST_F(HatBasisTest, GradientOfHatFunctionsConstantOnTriangle) {
        auto mesh = make_triangle_mesh();
        HatBasis<Complex<2>> basis(mesh);

        // Triangle (0,0)-(1,0)-(0,1)
        Point2D p_inside("0.2"_r, "0.3"_r);
        Point2D p_edge("0.5"_r, 0_r);

        auto grad0_inside = basis.gradient(0, p_inside);
        auto grad1_inside = basis.gradient(1, p_inside);
        auto grad2_inside = basis.gradient(2, p_inside);

        auto grad0_edge = basis.gradient(0, p_edge);
        auto grad1_edge = basis.gradient(1, p_edge);
        auto grad2_edge = basis.gradient(2, p_edge);

        Scalar eps = Rational(1, 1000000);

        // Inside
        EXPECT_RATIONAL_NEAR(grad0_inside.x(), -1_r, eps);
        EXPECT_RATIONAL_NEAR(grad0_inside.y(), -1_r, eps);
        EXPECT_RATIONAL_NEAR(grad1_inside.x(), 1_r, eps);
        EXPECT_RATIONAL_NEAR(grad1_inside.y(), 0_r, eps);
        EXPECT_RATIONAL_NEAR(grad2_inside.x(), 0_r, eps);
        EXPECT_RATIONAL_NEAR(grad2_inside.y(), 1_r, eps);

        // On edge (should belong to the same triangle, so gradients unchanged)
        EXPECT_RATIONAL_NEAR(grad0_edge.x(), -1_r, eps);
        EXPECT_RATIONAL_NEAR(grad0_edge.y(), -1_r, eps);
        EXPECT_RATIONAL_NEAR(grad1_edge.x(), 1_r, eps);
        EXPECT_RATIONAL_NEAR(grad1_edge.y(), 0_r, eps);
        EXPECT_RATIONAL_NEAR(grad2_edge.x(), 0_r, eps);
        EXPECT_RATIONAL_NEAR(grad2_edge.y(), 1_r, eps);
    }

    // -------------------------------------------------------------------------
    // Test: LocatePoint
    // -------------------------------------------------------------------------
    /**
     * @test LocatePoint
     * @brief Verifies that locate_point correctly identifies the triangle containing a point.
     */
    TEST_F(HatBasisTest, LocatePoint) {
        auto mesh = make_unit_square_mesh();
        HatBasis<Complex<2>> basis(mesh);

        // Point in the lower‑left triangle (diagonal from (0,0) to (1,1))
        Point2D p_lower("0.3"_r, "0.3"_r);
        auto loc_lower = basis.locate_point(p_lower);
        ASSERT_TRUE(loc_lower.has_value());
        EXPECT_EQ(loc_lower->first.first, 2);   // dimension 2 (triangle)
        // Lower‑left triangle is (v0,v1,v2) : index 0 in construction order
        EXPECT_EQ(loc_lower->first.second, 0);

        // Point in the upper‑right triangle
        Point2D p_upper("0.3"_r, "0.7"_r);
        auto loc_upper = basis.locate_point(p_upper);
        ASSERT_TRUE(loc_upper.has_value());
        // Upper‑right triangle is (v0,v2,v3) : index 1
        EXPECT_EQ(loc_upper->first.second, 1);

        // Point exactly on the diagonal – both triangles possible, so only check dimension
        Point2D p_diag("0.5"_r, "0.5"_r);
        auto loc_diag = basis.locate_point(p_diag);
        ASSERT_TRUE(loc_diag.has_value());
        EXPECT_EQ(loc_diag->first.first, 2);

        // Point outside the square should not be located
        Point2D p_out("2.0"_r, "2.0"_r);
        auto loc_out = basis.locate_point(p_out);
        EXPECT_FALSE(loc_out.has_value());
    }

} // namespace delta::testing