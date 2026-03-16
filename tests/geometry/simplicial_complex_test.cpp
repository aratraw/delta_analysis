#include <gtest/gtest.h>
#include <set>
#include "delta/geometry/simplicial_complex.h"
#include "../test_fixtures_geometry_numerical.h"

namespace delta::geometry::testing {

    using delta::testing::GeometryNumericalTest;
    using delta::operator""_r;

    /**
     * @class SimplicialComplexTest
     * @brief Tests for SimplicialComplex using proxy methods from fixture.
     */
    class SimplicialComplexTest : public GeometryNumericalTest {
    protected:
        // Type aliases for 2D and 3D complexes (using fixture's Complex)
        using Complex2D = Complex<2>;
        using Complex3D = Complex<3>;
        using Point2 = Point<2>;
        using Point3 = Point<3>;
        using VIdx2 = VertexIndex<2>;
        using VIdx3 = VertexIndex<3>;
    };

    TEST_F(SimplicialComplexTest, InitiallyEmpty) {
        Complex2D mesh;
        EXPECT_EQ(num_vertices(mesh), 0);
        EXPECT_EQ(num_edges(mesh), 0);
        EXPECT_EQ(num_triangles(mesh), 0);
        EXPECT_EQ(num_tetrahedra(mesh), 0);
        EXPECT_EQ(mesh.size(), 0); // size() не обёрнут, но это часть интерфейса OrderedGrid
        EXPECT_TRUE(begin(mesh) == end(mesh));
    }

    TEST_F(SimplicialComplexTest, AddVertices) {
        Complex2D mesh;
        Point2 p0(0_r, 0_r);
        Point2 p1(1_r, 0_r);
        Point2 p2(0_r, 1_r);

        VIdx2 idx0 = add_vertex(mesh, p0);
        VIdx2 idx1 = add_vertex(mesh, p1);
        VIdx2 idx2 = add_vertex(mesh, p2);

        EXPECT_EQ(idx0, 0);
        EXPECT_EQ(idx1, 1);
        EXPECT_EQ(idx2, 2);
        EXPECT_EQ(num_vertices(mesh), 3);
        EXPECT_EQ(mesh.size(), 3);

        EXPECT_TRUE(vertex(mesh, 0) == p0);
        EXPECT_TRUE(vertex(mesh, 1) == p1);
        EXPECT_TRUE(vertex(mesh, 2) == p2);

        // operator[] не обёрнут, но он эквивалентен vertex()
        EXPECT_TRUE(mesh[0] == p0);
        EXPECT_TRUE(mesh[1] == p1);
        EXPECT_TRUE(mesh[2] == p2);

        std::vector<Point2> points;
        for (const auto& pt : mesh) points.push_back(pt);
        ASSERT_EQ(points.size(), 3);
        EXPECT_TRUE(points[0] == p0);
        EXPECT_TRUE(points[1] == p1);
        EXPECT_TRUE(points[2] == p2);
    }

    TEST_F(SimplicialComplexTest, AddEdges) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(1_r, 0_r));
        VIdx2 v2 = add_vertex(mesh, Point2(0_r, 1_r));

        EXPECT_TRUE(add_edge(mesh, v0, v1));
        EXPECT_TRUE(add_edge(mesh, v1, v2));
        EXPECT_TRUE(add_edge(mesh, v2, v0));

        EXPECT_EQ(num_edges(mesh), 3);
        EXPECT_EQ(num_triangles(mesh), 0);

        EXPECT_FALSE(add_edge(mesh, v0, v1));
        EXPECT_FALSE(add_edge(mesh, v1, v0));

        auto e0 = edge_at(mesh, 0);
        auto e1 = edge_at(mesh, 1);
        auto e2 = edge_at(mesh, 2);

        std::set<std::pair<VIdx2, VIdx2>> expected_edges = {
            {v0, v1}, {v1, v2}, {v0, v2}
        };
        std::set<std::pair<VIdx2, VIdx2>> actual_edges = {
            {e0[0], e0[1]}, {e1[0], e1[1]}, {e2[0], e2[1]}
        };
        EXPECT_EQ(actual_edges, expected_edges);
    }

    TEST_F(SimplicialComplexTest, AddTriangle) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(1_r, 0_r));
        VIdx2 v2 = add_vertex(mesh, Point2(0_r, 1_r));

        add_edge(mesh, v0, v1);
        add_edge(mesh, v1, v2);
        add_edge(mesh, v2, v0);

        EXPECT_TRUE(add_triangle(mesh, v0, v1, v2));
        EXPECT_EQ(num_triangles(mesh), 1);

        auto tri = triangle_at(mesh, 0);
        std::set<VIdx2> expected = { v0, v1, v2 };
        std::set<VIdx2> actual = { tri[0], tri[1], tri[2] };
        EXPECT_EQ(actual, expected);
    }

    TEST_F(SimplicialComplexTest, AddTetrahedron) {
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

        EXPECT_TRUE(add_tetrahedron(mesh, v0, v1, v2, v3));
        EXPECT_EQ(num_tetrahedra(mesh), 1);

        auto tet = tetrahedron_at(mesh, 0);
        std::set<VIdx3> expected = { v0, v1, v2, v3 };
        std::set<VIdx3> actual = { tet[0], tet[1], tet[2], tet[3] };
        EXPECT_EQ(actual, expected);
    }

    TEST_F(SimplicialComplexTest, InvalidVertexIndex) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(1_r, 0_r));

        EXPECT_FALSE(add_edge(mesh, v0, 100));
        EXPECT_FALSE(add_edge(mesh, 100, v1));
        EXPECT_FALSE(add_edge(mesh, 100, 101));
        EXPECT_FALSE(add_triangle(mesh, v0, v1, 100));
    }

    TEST_F(SimplicialComplexTest, OutOfRangeAccess) {
        Complex2D mesh;
        EXPECT_THROW(vertex(mesh, 0), std::out_of_range);
        EXPECT_THROW(edge_at(mesh, 0), std::out_of_range);
        EXPECT_THROW(triangle_at(mesh, 0), std::out_of_range);
        EXPECT_THROW(tetrahedron_at(mesh, 0), std::out_of_range);
        // get_simplex не обёрнут, но можно добавить при необходимости
    }

    TEST_F(SimplicialComplexTest, ComparatorAndOrderedGrid) {
        static_assert(delta::OrderedGrid<Complex2D>,
            "SimplicialComplex must satisfy OrderedGrid concept");

        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(1_r, 0_r));
        VIdx2 v2 = add_vertex(mesh, Point2(0_r, 1_r));

        auto comp = comparator(mesh);
        EXPECT_TRUE(comp(vertex(mesh, v0), vertex(mesh, v2)));
        EXPECT_TRUE(comp(vertex(mesh, v2), vertex(mesh, v1)));
        EXPECT_FALSE(comp(vertex(mesh, v1), vertex(mesh, v0)));
    }

    TEST_F(SimplicialComplexTest, FindSimplex3D) {
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

        auto tet_idx = find_simplex(mesh, DIM_TETRAHEDRON, { v0, v1, v2, v3 });
        EXPECT_EQ(tet_idx, 0);

        EXPECT_EQ(find_simplex(mesh, DIM_TETRAHEDRON, { v3, v2, v1, v0 }), 0);

        EXPECT_EQ(find_simplex(mesh, DIM_TETRAHEDRON, { v0, v1, v2, 42 }), -1);
        EXPECT_EQ(find_simplex(mesh, 4, { v0, v1, v2, v3 }), -1); // dim 4 not supported
    }

    TEST_F(SimplicialComplexTest, UnitSquareTriangulation) {
        Complex2D mesh;
        make_unit_square_triangulation(mesh);

        EXPECT_EQ(num_vertices(mesh), 4);
        EXPECT_EQ(num_edges(mesh), 5);  // 4 sides + 1 diagonal
        EXPECT_EQ(num_triangles(mesh), 2);

        std::size_t v0 = 0, v1 = 1, v2 = 2, v3 = 3;

        EXPECT_GE(find_simplex(mesh, DIM_EDGE, { v0, v1 }), 0);
        EXPECT_GE(find_simplex(mesh, DIM_EDGE, { v1, v2 }), 0);
        EXPECT_GE(find_simplex(mesh, DIM_EDGE, { v2, v3 }), 0);
        EXPECT_GE(find_simplex(mesh, DIM_EDGE, { v3, v0 }), 0);
        EXPECT_GE(find_simplex(mesh, DIM_EDGE, { v0, v2 }), 0);
        EXPECT_EQ(find_simplex(mesh, DIM_EDGE, { v1, v3 }), -1);
    }

    // =============================================================================
    // Tests for incident_faces
    // =============================================================================

    /**
     * @class IncidentFacesTest
     * @brief Tests for incident_faces function.
     */
    class IncidentFacesTest : public GeometryNumericalTest {
    protected:
        using Complex2D = Complex<2>;
        using Complex3D = Complex<3>;
        using VIdx2 = VertexIndex<2>;
        using VIdx3 = VertexIndex<3>;
    };

    TEST_F(IncidentFacesTest, TriangleFaces) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point<2>(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point<2>(1_r, 0_r));
        VIdx2 v2 = add_vertex(mesh, Point<2>(0_r, 1_r));

        add_edge(mesh, v0, v1);
        add_edge(mesh, v1, v2);
        add_edge(mesh, v2, v0);
        add_triangle(mesh, v0, v1, v2);

        auto faces = incident_faces(mesh, DIM_TRIANGLE, 0, DIM_EDGE);
        ASSERT_EQ(faces.size(), 3);

        auto e01 = find_simplex(mesh, DIM_EDGE, { v0, v1 });
        auto e12 = find_simplex(mesh, DIM_EDGE, { v1, v2 });
        auto e20 = find_simplex(mesh, DIM_EDGE, { v2, v0 });
        ASSERT_NE(e01, -1);
        ASSERT_NE(e12, -1);
        ASSERT_NE(e20, -1);

        EXPECT_EQ(faces[0].first, static_cast<std::size_t>(e12));
        EXPECT_EQ(faces[0].second, 1);
        EXPECT_EQ(faces[1].first, static_cast<std::size_t>(e20));
        EXPECT_EQ(faces[1].second, -1);
        EXPECT_EQ(faces[2].first, static_cast<std::size_t>(e01));
        EXPECT_EQ(faces[2].second, 1);
    }

    TEST_F(IncidentFacesTest, EdgeVertices) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point<2>(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point<2>(1_r, 0_r));
        add_edge(mesh, v0, v1);

        auto faces = incident_faces(mesh, DIM_EDGE, 0, DIM_VERTEX);
        ASSERT_EQ(faces.size(), 2);

        EXPECT_EQ(faces[0].first, v1);
        EXPECT_EQ(faces[0].second, 1);
        EXPECT_EQ(faces[1].first, v0);
        EXPECT_EQ(faces[1].second, -1);
    }

    TEST_F(IncidentFacesTest, TetrahedronFaces) {
        Complex3D mesh;
        VIdx3 v0 = add_vertex(mesh, Point<3>(0_r, 0_r, 0_r));
        VIdx3 v1 = add_vertex(mesh, Point<3>(1_r, 0_r, 0_r));
        VIdx3 v2 = add_vertex(mesh, Point<3>(0_r, 1_r, 0_r));
        VIdx3 v3 = add_vertex(mesh, Point<3>(0_r, 0_r, 1_r));

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

        auto faces = incident_faces(mesh, DIM_TETRAHEDRON, 0, DIM_TRIANGLE);
        ASSERT_EQ(faces.size(), 4);

        auto t012 = find_simplex(mesh, DIM_TRIANGLE, { v0, v1, v2 });
        auto t013 = find_simplex(mesh, DIM_TRIANGLE, { v0, v1, v3 });
        auto t023 = find_simplex(mesh, DIM_TRIANGLE, { v0, v2, v3 });
        auto t123 = find_simplex(mesh, DIM_TRIANGLE, { v1, v2, v3 });
        ASSERT_NE(t012, -1);
        ASSERT_NE(t013, -1);
        ASSERT_NE(t023, -1);
        ASSERT_NE(t123, -1);

        EXPECT_EQ(faces[0].first, static_cast<std::size_t>(t123));
        EXPECT_EQ(faces[0].second, 1);
        EXPECT_EQ(faces[1].first, static_cast<std::size_t>(t023));
        EXPECT_EQ(faces[1].second, -1);
        EXPECT_EQ(faces[2].first, static_cast<std::size_t>(t013));
        EXPECT_EQ(faces[2].second, 1);
        EXPECT_EQ(faces[3].first, static_cast<std::size_t>(t012));
        EXPECT_EQ(faces[3].second, -1);
    }

    TEST_F(IncidentFacesTest, InvalidLowDim) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point<2>(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point<2>(1_r, 0_r));
        add_edge(mesh, v0, v1);

        EXPECT_THROW(incident_faces(mesh, DIM_EDGE, 0, DIM_EDGE), std::invalid_argument);
    }

} // namespace delta::geometry::testing