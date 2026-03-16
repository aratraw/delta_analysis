#include <gtest/gtest.h>
#include <optional>
#include "../test_fixtures_geometry_numerical.h"
#include "delta/geometry/geometry_ops.h"

namespace delta::geometry::testing {

    using delta::testing::GeometryNumericalTest;
    using delta::operator""_r;

    /**
     * @class GeometryOpsTest
     * @brief Tests for geometry operations with Euclidean metric.
     */
    class GeometryOpsTest : public GeometryNumericalTest {
    protected:
        using Complex2D = Complex<2>;
        using Complex3D = Complex<3>;
        using Point2 = Point<2>;
        using Point3 = Point<3>;
        using VIdx2 = VertexIndex<2>;
        using VIdx3 = VertexIndex<3>;
    };

    // -----------------------------------------------------------------------------
    // edge_length tests
    // -----------------------------------------------------------------------------
    TEST_F(GeometryOpsTest, EdgeLength) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(3_r, 4_r));
        add_edge(mesh, v0, v1);

        EuclideanMetric metric;
        auto len = edge_length(mesh, 0, metric);
        EXPECT_RATIONAL_NEAR(len, 5_r, Rational(1, 1000000));
    }

    TEST_F(GeometryOpsTest, EdgeLengthZero) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(1_r, 2_r));
        VIdx2 v1 = add_vertex(mesh, Point2(1_r, 2_r));
        add_edge(mesh, v0, v1);

        EuclideanMetric metric;
        auto len = edge_length(mesh, 0, metric);
        EXPECT_EQ(len, 0_r);
    }

    // -----------------------------------------------------------------------------
    // edge_center tests
    // -----------------------------------------------------------------------------
    TEST_F(GeometryOpsTest, EdgeCenter) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(2_r, 4_r));
        add_edge(mesh, v0, v1);

        auto center = edge_center(mesh, 0);
        Point2 expected(1_r, 2_r);
        EXPECT_TRUE(vector_near<2>(center, expected, Rational(1, 1000000)));
    }

    // -----------------------------------------------------------------------------
    // triangle_center tests
    // -----------------------------------------------------------------------------
    TEST_F(GeometryOpsTest, TriangleCenter) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(3_r, 0_r));
        VIdx2 v2 = add_vertex(mesh, Point2(0_r, 3_r));
        add_edge(mesh, v0, v1);
        add_edge(mesh, v1, v2);
        add_edge(mesh, v2, v0);
        add_triangle(mesh, v0, v1, v2);

        auto center = triangle_center(mesh, 0);
        Point2 expected(1_r, 1_r);
        EXPECT_TRUE(vector_near<2>(center, expected, Rational(1, 1000000)));
    }

    // -----------------------------------------------------------------------------
    // triangle_area tests
    // -----------------------------------------------------------------------------
    TEST_F(GeometryOpsTest, TriangleArea) {
        set_precision(1_r/1000000_r);
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(3_r, 0_r));
        VIdx2 v2 = add_vertex(mesh, Point2(0_r, 4_r));
        add_edge(mesh, v0, v1);
        add_edge(mesh, v1, v2);
        add_edge(mesh, v2, v0);
        add_triangle(mesh, v0, v1, v2);

        EuclideanMetric metric;
        auto area = triangle_area(mesh, 0, metric);
        EXPECT_RATIONAL_NEAR(area, 6_r, Rational(1, 1000000));
    }

    TEST_F(GeometryOpsTest, TriangleAreaDegenerate) {
        set_precision(1_r / 1000000_r);
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(1_r, 0_r));
        VIdx2 v2 = add_vertex(mesh, Point2(2_r, 0_r));
        add_edge(mesh, v0, v1);
        add_edge(mesh, v1, v2);
        add_edge(mesh, v2, v0);
        add_triangle(mesh, v0, v1, v2);

        EuclideanMetric metric;
        auto area = triangle_area(mesh, 0, metric);
        EXPECT_RATIONAL_NEAR(area, 0_r, Rational(1, 1000000));
    }

    // -----------------------------------------------------------------------------
    // tetrahedron_volume tests
    // -----------------------------------------------------------------------------
    TEST_F(GeometryOpsTest, TetrahedronVolume) {
        Complex3D mesh;
        VIdx3 v0 = add_vertex(mesh, Point3(0_r, 0_r, 0_r));
        VIdx3 v1 = add_vertex(mesh, Point3(1_r, 0_r, 0_r));
        VIdx3 v2 = add_vertex(mesh, Point3(0_r, 1_r, 0_r));
        VIdx3 v3 = add_vertex(mesh, Point3(0_r, 0_r, 1_r));

        add_edge(mesh, v0, v1);
        add_edge(mesh, v0, v2);
        add_edge(mesh, v0, v3);
        add_edge(mesh, v1, v2);
        add_edge(mesh, v1, v3);
        add_edge(mesh, v2, v3);
        add_triangle(mesh, v0, v1, v2);
        add_triangle(mesh, v0, v1, v3);
        add_triangle(mesh, v0, v2, v3);
        add_triangle(mesh, v1, v2, v3);
        add_tetrahedron(mesh, v0, v1, v2, v3);

        EuclideanMetric metric;
        auto vol = tetrahedron_volume(mesh, 0, metric);
        EXPECT_RATIONAL_NEAR(vol, Rational(1, 6), Rational(1, 1000000));
    }

    // -----------------------------------------------------------------------------
    // cell_volume dispatcher tests
    // -----------------------------------------------------------------------------
    TEST_F(GeometryOpsTest, CellVolume2D) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(2_r, 0_r));
        VIdx2 v2 = add_vertex(mesh, Point2(0_r, 3_r));
        add_edge(mesh, v0, v1);
        add_edge(mesh, v1, v2);
        add_edge(mesh, v2, v0);
        add_triangle(mesh, v0, v1, v2);

        EuclideanMetric metric;
        auto area = cell_volume<2>(mesh, 0, metric);
        EXPECT_RATIONAL_NEAR(area, 3_r, Rational(1, 1000000));
    }

    TEST_F(GeometryOpsTest, CellVolume3D) {
        Complex3D mesh;
        VIdx3 v0 = add_vertex(mesh, Point3(0_r, 0_r, 0_r));
        VIdx3 v1 = add_vertex(mesh, Point3(1_r, 0_r, 0_r));
        VIdx3 v2 = add_vertex(mesh, Point3(0_r, 1_r, 0_r));
        VIdx3 v3 = add_vertex(mesh, Point3(0_r, 0_r, 1_r));

        add_edge(mesh, v0, v1);
        add_edge(mesh, v0, v2);
        add_edge(mesh, v0, v3);
        add_edge(mesh, v1, v2);
        add_edge(mesh, v1, v3);
        add_edge(mesh, v2, v3);
        add_triangle(mesh, v0, v1, v2);
        add_triangle(mesh, v0, v1, v3);
        add_triangle(mesh, v0, v2, v3);
        add_triangle(mesh, v1, v2, v3);
        add_tetrahedron(mesh, v0, v1, v2, v3);

        EuclideanMetric metric;
        auto vol = cell_volume<3>(mesh, 0, metric);
        EXPECT_RATIONAL_NEAR(vol, Rational(1, 6), Rational(1, 1000000));
    }

    // -----------------------------------------------------------------------------
    // edge_normal_2d tests
    // -----------------------------------------------------------------------------
    TEST_F(GeometryOpsTest, EdgeNormal2D) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(3_r, 0_r));
        add_edge(mesh, v0, v1);

        EuclideanMetric metric;
        auto normal = edge_normal_2d(mesh, 0, metric);
        Point2 expected(0_r, -3_r);
        EXPECT_TRUE(vector_near<2>(normal, expected, Rational(1, 1000000)));

        Point2 e = vertex(mesh, v1) - vertex(mesh, v0);
        EXPECT_EQ(e.dot(normal), 0_r);
    }

    TEST_F(GeometryOpsTest, EdgeNormal2DVertical) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(0_r, 4_r));
        add_edge(mesh, v0, v1);

        EuclideanMetric metric;
        auto normal = edge_normal_2d(mesh, 0, metric);
        Point2 expected(4_r, 0_r);
        EXPECT_TRUE(vector_near<2>(normal, expected, Rational(1, 1000000)));
    }

    // -----------------------------------------------------------------------------
    // edge_neighbors_2d tests
    // -----------------------------------------------------------------------------
    class EdgeNeighbors2DTest : public GeometryNumericalTest {
    protected:
        using Complex2D = Complex<2>;
        using VIdx2 = VertexIndex<2>;

        void SetUp() override {
            make_unit_square_triangulation(mesh);
            // Vertex indices: 0:(0,0), 1:(1,0), 2:(1,1), 3:(0,1)
        }

        Complex2D mesh;
    };

    TEST_F(EdgeNeighbors2DTest, InternalDiagonal) {
        auto e_idx = find_simplex(mesh, DIM_EDGE, { 0, 2 });
        ASSERT_NE(e_idx, -1);

        auto [left, right] = edge_neighbors_2d(mesh, static_cast<std::size_t>(e_idx));
        EXPECT_EQ(left, 0);
        EXPECT_TRUE(right.has_value());
        EXPECT_EQ(*right, 1);
    }

    TEST_F(EdgeNeighbors2DTest, BoundaryEdge) {
        auto e_idx = find_simplex(mesh, DIM_EDGE, { 0, 1 });
        ASSERT_NE(e_idx, -1);

        auto [left, right] = edge_neighbors_2d(mesh, static_cast<std::size_t>(e_idx));
        EXPECT_EQ(left, 0);
        EXPECT_FALSE(right.has_value());
    }

    TEST_F(EdgeNeighbors2DTest, OtherBoundary) {
        auto e_idx = find_simplex(mesh, DIM_EDGE, { 3, 0 });
        ASSERT_NE(e_idx, -1);

        auto [left, right] = edge_neighbors_2d(mesh, static_cast<std::size_t>(e_idx));
        EXPECT_EQ(left, 1);
        EXPECT_FALSE(right.has_value());
    }

} // namespace delta::geometry::testing