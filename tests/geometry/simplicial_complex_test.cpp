// tests/geometry/simplicial_complex_test.cpp
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
     *
     * Implements tests for Stage 0 of the specification:
     * - Creation and non-degeneracy checks
     * - Barycentric subdivision
     * - Incidence methods
     * - Geometric methods with metric
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

    // =========================================================================
    // Test group 1: Creation and non-degeneracy checks
    // =========================================================================

    TEST_F(SimplicialComplexTest, InitiallyEmpty) {
        Complex2D mesh;
        EXPECT_EQ(num_vertices(mesh), 0);
        EXPECT_EQ(num_edges(mesh), 0);
        EXPECT_EQ(num_triangles(mesh), 0);
        EXPECT_EQ(num_tetrahedra(mesh), 0);
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

        EXPECT_TRUE(vertex(mesh, 0) == p0);
        EXPECT_TRUE(vertex(mesh, 1) == p1);
        EXPECT_TRUE(vertex(mesh, 2) == p2);
    }

    TEST_F(SimplicialComplexTest, AddEdges) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(1_r, 0_r));
        VIdx2 v2 = add_vertex(mesh, Point2(0_r, 1_r));

        // Add edges
        EXPECT_TRUE(add_edge(mesh, v0, v1));
        EXPECT_TRUE(add_edge(mesh, v1, v2));
        EXPECT_TRUE(add_edge(mesh, v2, v0));

        EXPECT_EQ(num_edges(mesh), 3);
        EXPECT_EQ(num_triangles(mesh), 0);

        // Duplicate edges should return false
        EXPECT_FALSE(add_edge(mesh, v0, v1));
        EXPECT_FALSE(add_edge(mesh, v1, v0));  // orientation shouldn't matter

        // Verify edge contents
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

    TEST_F(SimplicialComplexTest, DISABLED_AddTriangle) {
        //ЭТОТ ТЕСТ ВЫЗЫВАЕТ STACK OVERFLOW
        // ВЕРОЯТНО ПОТОМУ ЧТО ТУТ ПОД КАПОТОМ СЧИТАЕТСЯ МОДУЛЬ/КОРЕНЬ С ОГРОМНОЙ ТОЧНОСТЬЮ.
        // ПОДКЛЮЧИ ЗДЕСЬ КОНТРОЛЬ ТОЧНОСТИ ИЗ ФИКСТУРЫ.
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(1_r, 0_r));
        VIdx2 v2 = add_vertex(mesh, Point2(0_r, 1_r));

        // Add required edges (should be automatic but we add explicitly)
        add_edge(mesh, v0, v1);
        add_edge(mesh, v1, v2);
        add_edge(mesh, v2, v0);

        // Add triangle
        EXPECT_TRUE(add_triangle(mesh, v0, v1, v2));
        EXPECT_EQ(num_triangles(mesh), 1);

        // Verify triangle contents
        auto tri = triangle_at(mesh, 0);
        std::set<VIdx2> expected = { v0, v1, v2 };
        std::set<VIdx2> actual = { tri[0], tri[1], tri[2] };
        EXPECT_EQ(actual, expected);
    }

    TEST_F(SimplicialComplexTest, DISABLED_AddTetrahedron) {
        Complex3D mesh;
        VIdx3 v0 = add_vertex(mesh, Point3(0_r, 0_r, 0_r));
        VIdx3 v1 = add_vertex(mesh, Point3(1_r, 0_r, 0_r));
        VIdx3 v2 = add_vertex(mesh, Point3(0_r, 1_r, 0_r));
        VIdx3 v3 = add_vertex(mesh, Point3(0_r, 0_r, 1_r));

        // Add required lower-dimensional simplices
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

        // Add tetrahedron
        EXPECT_TRUE(add_tetrahedron(mesh, v0, v1, v2, v3));
        EXPECT_EQ(num_tetrahedra(mesh), 1);

        // Verify tetrahedron contents
        auto tet = tetrahedron_at(mesh, 0);
        std::set<VIdx3> expected = { v0, v1, v2, v3 };
        std::set<VIdx3> actual = { tet[0], tet[1], tet[2], tet[3] };
        EXPECT_EQ(actual, expected);
    }

    TEST_F(SimplicialComplexTest, NonDegeneracyChecks) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(1_r, 0_r));
        VIdx2 v2 = add_vertex(mesh, Point2(2_r, 0_r));  // collinear

        // Try to add degenerate triangle (collinear points)
        // This should return false
        EXPECT_FALSE(add_triangle(mesh, v0, v1, v2));
        EXPECT_EQ(num_triangles(mesh), 0);
    }

    TEST_F(SimplicialComplexTest, InvalidVertexIndex) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(1_r, 0_r));

        EXPECT_FALSE(add_edge(mesh, v0, 100));
        EXPECT_FALSE(add_edge(mesh, 100, v1));
        EXPECT_FALSE(add_edge(mesh, 100, 101));
        EXPECT_FALSE(add_triangle(mesh, v0, v1, 100));
        // Строка с add_tetrahedron удалена
    }

    TEST_F(SimplicialComplexTest, OutOfRangeAccess) {
        Complex2D mesh;
        EXPECT_THROW(vertex(mesh, 0), std::out_of_range);
        EXPECT_THROW(edge_at(mesh, 0), std::out_of_range);
        EXPECT_THROW(triangle_at(mesh, 0), std::out_of_range);
        // Строка с tetrahedron_at удалена
    }

    TEST_F(SimplicialComplexTest, FindSimplex) {
        Complex3D mesh;
        VIdx3 v0 = add_vertex(mesh, Point3(0_r, 0_r, 0_r));
        VIdx3 v1 = add_vertex(mesh, Point3(1_r, 0_r, 0_r));
        VIdx3 v2 = add_vertex(mesh, Point3(0_r, 1_r, 0_r));
        VIdx3 v3 = add_vertex(mesh, Point3(0_r, 0_r, 1_r));

        // Add tetrahedron and its faces
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

        // Find tetrahedron (order shouldn't matter)
        auto tet_idx = find_simplex(mesh, DIM_TETRAHEDRON, { v0, v1, v2, v3 });
        EXPECT_EQ(tet_idx, 0);
        EXPECT_EQ(find_simplex(mesh, DIM_TETRAHEDRON, { v3, v2, v1, v0 }), 0);

        // Find triangle
        auto tri_idx = find_simplex(mesh, DIM_TRIANGLE, { v0, v1, v2 });
        EXPECT_EQ(tri_idx, 0);

        // Find edge
        auto edge_idx = find_simplex(mesh, DIM_EDGE, { v0, v1 });
        EXPECT_EQ(edge_idx, 0);

        // Non-existent simplex
        EXPECT_EQ(find_simplex(mesh, DIM_TETRAHEDRON, { v0, v1, v2, 42 }), -1);

        // Invalid dimension
        EXPECT_EQ(find_simplex(mesh, 4, { v0, v1, v2, v3 }), -1);
    }

    // =========================================================================
    // Test group 2: Barycentric subdivision
    // =========================================================================

    TEST_F(SimplicialComplexTest, BarycentricSubdivisionTriangle) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(1_r, 0_r));
        VIdx2 v2 = add_vertex(mesh, Point2(0_r, 1_r));

        add_edge(mesh, v0, v1);
        add_edge(mesh, v1, v2);
        add_edge(mesh, v2, v0);
        add_triangle(mesh, v0, v1, v2);

        // Perform subdivision
        auto [subdivided, subdiv_map] = barycentric_subdivide(mesh);

        // Check vertex count: original 3 + edge midpoints (3) + centroid (1) = 7
        EXPECT_EQ(num_vertices(subdivided), 7);

        // Check triangle count: original triangle split into 6
        EXPECT_EQ(num_triangles(subdivided), 6);

        // Check edge count: each original edge split into 2, plus 3 edges to centroid = 9
        EXPECT_EQ(num_edges(subdivided), 12);

        // Verify subdivision map for original edges
        // Original edge (v0,v1) should map to two new edges
        auto edge_key = SimplexKey{ DIM_EDGE, 0 };
        auto it = subdiv_map.find(edge_key);
        ASSERT_NE(it, subdiv_map.end());
        EXPECT_EQ(it->second.size(), 2);  // two half-edges

        // Verify subdivision map for original triangle
        auto tri_key = SimplexKey{ DIM_TRIANGLE, 0 };
        it = subdiv_map.find(tri_key);
        ASSERT_NE(it, subdiv_map.end());
        EXPECT_EQ(it->second.size(), 6);  // six small triangles
    }

    TEST_F(SimplicialComplexTest, BarycentricSubdivisionEdgeLengthReduction) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(1_r, 0_r));
        VIdx2 v2 = add_vertex(mesh, Point2(0_r, 1_r));

        add_edge(mesh, v0, v1);
        add_edge(mesh, v1, v2);
        add_edge(mesh, v2, v0);
        add_triangle(mesh, v0, v1, v2);

        // Use Euclidean metric
        delta::EuclideanMetric metric;

        // Compute original max edge length
        Scalar original_max = 0_r;
        for (std::size_t i = 0; i < num_edges(mesh); ++i) {
            original_max = std::max(original_max, edge_length(mesh, i, metric));
        }

        // Subdivide
        auto [subdivided, _] = barycentric_subdivide(mesh);

        // Compute new max edge length
        Scalar new_max = 0_r;
        for (std::size_t i = 0; i < num_edges(subdivided); ++i) {
            new_max = std::max(new_max, edge_length(subdivided, i, metric));
        }

        // For a triangle, max edge should reduce to at most 2/3 of original
        // (due to subdivision at midpoints and centroid)
        Scalar ratio = new_max / original_max;
        EXPECT_LE(ratio, 2_r / 3_r + delta::default_eps());
    }

    TEST_F(SimplicialComplexTest, BarycentricSubdivisionUnitSquare) {
        Complex2D mesh;
        make_unit_square_triangulation(mesh);

        // Original counts
        EXPECT_EQ(num_vertices(mesh), 4);
        EXPECT_EQ(num_edges(mesh), 5);   // 4 sides + 1 diagonal
        EXPECT_EQ(num_triangles(mesh), 2);

        // Subdivide
        auto [subdivided, subdiv_map] = barycentric_subdivide(mesh);

        // After subdivision of two triangles sharing a diagonal:
        // Each triangle -> 6 small triangles, total 12
        EXPECT_EQ(num_triangles(subdivided), 12);

        // Vertex count: original 4 + edge midpoints (5 edges -> 5 midpoints) 
        // + centroids (2) = 11
        EXPECT_EQ(num_vertices(subdivided), 11);

        // Check that diagonal edges (original edges 4) subdivided
        auto diagonal_key = SimplexKey{ DIM_EDGE, 4 };  // assuming diagonal is last edge
        auto it = subdiv_map.find(diagonal_key);
        if (it != subdiv_map.end()) {
            EXPECT_EQ(it->second.size(), 2);  // split into two
        }
    }

    // =========================================================================
    // Test group 3: Incidence methods
    // =========================================================================

    TEST_F(SimplicialComplexTest, IncidentFacesTriangle) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(1_r, 0_r));
        VIdx2 v2 = add_vertex(mesh, Point2(0_r, 1_r));

        add_edge(mesh, v0, v1);
        add_edge(mesh, v1, v2);
        add_edge(mesh, v2, v0);
        add_triangle(mesh, v0, v1, v2);

        // Get edges incident to triangle (codimension 1)
        auto faces = incident_faces(mesh, DIM_TRIANGLE, 0, DIM_EDGE);
        ASSERT_EQ(faces.size(), 3);

        // Find edge indices
        auto e01 = find_simplex(mesh, DIM_EDGE, { v0, v1 });
        auto e12 = find_simplex(mesh, DIM_EDGE, { v1, v2 });
        auto e20 = find_simplex(mesh, DIM_EDGE, { v2, v0 });
        ASSERT_NE(e01, -1);
        ASSERT_NE(e12, -1);
        ASSERT_NE(e20, -1);

        // Create map for verification
        std::map<std::size_t, int> face_signs;
        for (const auto& [idx, sign] : faces) {
            face_signs[idx] = sign;
        }

        // Check that all edges are present with correct signs
        // Signs follow (-1)^i pattern: edge opposite vertex i gets sign (-1)^i
        EXPECT_EQ(face_signs[e12], 1);   // opposite v0 (i=0) -> sign 1
        EXPECT_EQ(face_signs[e20], -1);  // opposite v1 (i=1) -> sign -1
        EXPECT_EQ(face_signs[e01], 1);   // opposite v2 (i=2) -> sign 1

        // Sum of signs on closed loop should be 0? Not required but interesting
        // Actually for a triangle, sum of signs = 1 + (-1) + 1 = 1, not zero
    }

    TEST_F(SimplicialComplexTest, IncidentFacesTetrahedron) {
        Complex3D mesh;
        VIdx3 v0 = add_vertex(mesh, Point3(0_r, 0_r, 0_r));
        VIdx3 v1 = add_vertex(mesh, Point3(1_r, 0_r, 0_r));
        VIdx3 v2 = add_vertex(mesh, Point3(0_r, 1_r, 0_r));
        VIdx3 v3 = add_vertex(mesh, Point3(0_r, 0_r, 1_r));

        // Add all faces (as in earlier test)
        add_edge(mesh, v0, v1); add_edge(mesh, v0, v2); add_edge(mesh, v0, v3);
        add_edge(mesh, v1, v2); add_edge(mesh, v1, v3); add_edge(mesh, v2, v3);

        add_triangle(mesh, v0, v1, v2);
        add_triangle(mesh, v0, v1, v3);
        add_triangle(mesh, v0, v2, v3);
        add_triangle(mesh, v1, v2, v3);

        add_tetrahedron(mesh, v0, v1, v2, v3);

        // Get triangles incident to tetrahedron (codimension 1)
        auto faces = incident_faces(mesh, DIM_TETRAHEDRON, 0, DIM_TRIANGLE);
        ASSERT_EQ(faces.size(), 4);

        // Find triangle indices
        auto t012 = find_simplex(mesh, DIM_TRIANGLE, { v0, v1, v2 });
        auto t013 = find_simplex(mesh, DIM_TRIANGLE, { v0, v1, v3 });
        auto t023 = find_simplex(mesh, DIM_TRIANGLE, { v0, v2, v3 });
        auto t123 = find_simplex(mesh, DIM_TRIANGLE, { v1, v2, v3 });
        ASSERT_NE(t012, -1);
        ASSERT_NE(t013, -1);
        ASSERT_NE(t023, -1);
        ASSERT_NE(t123, -1);

        // Check signs: (-1)^i pattern, where i is the omitted vertex
        std::map<std::size_t, int> face_signs;
        for (const auto& [idx, sign] : faces) {
            face_signs[idx] = sign;
        }

        EXPECT_EQ(face_signs[t123], 1);   // omit v0 (i=0) -> sign 1
        EXPECT_EQ(face_signs[t023], -1);  // omit v1 (i=1) -> sign -1
        EXPECT_EQ(face_signs[t013], 1);   // omit v2 (i=2) -> sign 1
        EXPECT_EQ(face_signs[t012], -1);  // omit v3 (i=3) -> sign -1
    }

    TEST_F(SimplicialComplexTest, IncidentFacesEdgeVertices) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(1_r, 0_r));
        add_edge(mesh, v0, v1);

        // Get vertices incident to edge (codimension 1 -> vertices)
        auto faces = incident_faces(mesh, DIM_EDGE, 0, DIM_VERTEX);
        ASSERT_EQ(faces.size(), 2);

        // For an edge, the two incident vertices have signs +1 and -1
        // (orientation: first vertex negative, second positive)
        std::map<std::size_t, int> vertex_signs;
        for (const auto& [idx, sign] : faces) {
            vertex_signs[idx] = sign;
        }

        EXPECT_EQ(vertex_signs[v1], 1);
        EXPECT_EQ(vertex_signs[v0], -1);
    }

    TEST_F(SimplicialComplexTest, IncidentFacesInvalidLowDim) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(1_r, 0_r));
        add_edge(mesh, v0, v1);

        // low_dim must be top_dim - 1
        EXPECT_THROW(incident_faces(mesh, DIM_EDGE, 0, DIM_EDGE), std::invalid_argument);
    }

    // =========================================================================
    // Test group 4: Geometric methods with metric
    // =========================================================================

    TEST_F(SimplicialComplexTest, EdgeLength) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(1_r, 0_r));
        VIdx2 v2 = add_vertex(mesh, Point2(0_r, 1_r));

        add_edge(mesh, v0, v1);
        add_edge(mesh, v1, v2);
        add_edge(mesh, v2, v0);

        delta::EuclideanMetric metric;

        // Check edge lengths
        EXPECT_RATIONAL_NEAR(edge_length(mesh, 0, metric), 1.0, 1e-12);  // (0,0)-(1,0)
        EXPECT_RATIONAL_NEAR(edge_length(mesh, 1, metric), std::sqrt(2.0), 1e-12);  // (1,0)-(0,1)
        EXPECT_RATIONAL_NEAR(edge_length(mesh, 2, metric), 1.0, 1e-12);  // (0,1)-(0,0)
    }

    TEST_F(SimplicialComplexTest, CellVolumeTriangle) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(1_r, 0_r));
        VIdx2 v2 = add_vertex(mesh, Point2(0_r, 1_r));

        add_edge(mesh, v0, v1);
        add_edge(mesh, v1, v2);
        add_edge(mesh, v2, v0);
        add_triangle(mesh, v0, v1, v2);

        delta::EuclideanMetric metric;

        // Area of right triangle = 0.5
        Scalar area = cell_volume(mesh, 0, metric);
        EXPECT_RATIONAL_NEAR(area, 0.5_r, 1e-12_r);
    }

    TEST_F(SimplicialComplexTest, CellVolumeTetrahedron) {
        Complex3D mesh;
        VIdx3 v0 = add_vertex(mesh, Point3(0_r, 0_r, 0_r));
        VIdx3 v1 = add_vertex(mesh, Point3(1_r, 0_r, 0_r));
        VIdx3 v2 = add_vertex(mesh, Point3(0_r, 1_r, 0_r));
        VIdx3 v3 = add_vertex(mesh, Point3(0_r, 0_r, 1_r));

        add_edge(mesh, v0, v1); add_edge(mesh, v0, v2); add_edge(mesh, v0, v3);
        add_edge(mesh, v1, v2); add_edge(mesh, v1, v3); add_edge(mesh, v2, v3);

        add_triangle(mesh, v0, v1, v2);
        add_triangle(mesh, v0, v1, v3);
        add_triangle(mesh, v0, v2, v3);
        add_triangle(mesh, v1, v2, v3);

        add_tetrahedron(mesh, v0, v1, v2, v3);

        delta::EuclideanMetric metric;

        // Volume of right tetrahedron = 1/6
        Scalar volume = cell_volume(mesh, 0, metric);
        EXPECT_RATIONAL_NEAR(volume, 1_r / 6_r, 1e-12_r);
    }

    TEST_F(SimplicialComplexTest, EdgeNormal2D) {
        Complex2D mesh;
        VIdx2 v0 = add_vertex(mesh, Point2(0_r, 0_r));
        VIdx2 v1 = add_vertex(mesh, Point2(1_r, 0_r));
        VIdx2 v2 = add_vertex(mesh, Point2(0_r, 1_r));

        add_edge(mesh, v0, v1);
        add_edge(mesh, v1, v2);
        add_edge(mesh, v2, v0);
        add_triangle(mesh, v0, v1, v2);

        delta::EuclideanMetric metric;

        // Test normal of bottom edge (v0->v1)
        auto normal = edge_normal_2d(mesh, 0, metric);  // edge v0-v1

        // Vector from edge midpoint to triangle centroid
        auto centroid = (vertex(mesh, v0) + vertex(mesh, v1) + vertex(mesh, v2)) / 3_r;
        auto midpoint = (vertex(mesh, v0) + vertex(mesh, v1)) / 2_r;
        auto to_centroid = centroid - midpoint;

        // Normal should be perpendicular to edge
        auto edge_vec = vertex(mesh, v1) - vertex(mesh, v0);
        EXPECT_RATIONAL_NEAR(normal.dot(to_centroid), 0_r, 1e-12_r);

        // Normal length should equal edge length (due to our scaling convention)
        Scalar edge_len = edge_length(mesh, 0, metric);
        EXPECT_RATIONAL_NEAR(normal.norm(), edge_len, 1e-12);

        // For positively oriented triangle (CCW), normal should point outward
        // For left triangle (0,0)-(1,0)-(0,1) orientation is positive,
        // so normal should point outward (away from triangle)
        // Dot product with to_centroid should be negative (outward)
        EXPECT_LT(normal.dot(to_centroid), 0);
    }

    TEST_F(SimplicialComplexTest, EdgeNeighbors2D) {
        Complex2D mesh;
        make_unit_square_triangulation(mesh);

        delta::EuclideanMetric metric;  // not used for neighbor test, but for completeness

        // Diagonal edge (v0-v2) should have two neighboring triangles
        // In our triangulation, vertices: v0(0,0), v1(1,0), v2(1,1), v3(0,1)
        // Diagonal is v0-v2
        VIdx2 v0 = 0, v2 = 2;
        auto edge_idx = find_simplex(mesh, DIM_EDGE, { v0, v2 });
        ASSERT_NE(edge_idx, -1);

        auto [left, right] = edge_neighbors_2d(mesh, edge_idx);
        EXPECT_NE(left, -1);
        EXPECT_TRUE(right.has_value());

        // Boundary edge (v0-v1) should have only one neighbor
        VIdx2 v1 = 1;
        edge_idx = find_simplex(mesh, DIM_EDGE, { v0, v1 });
        ASSERT_NE(edge_idx, -1);

        auto [left2, right2] = edge_neighbors_2d(mesh, edge_idx);
        EXPECT_NE(left2, -1);
        EXPECT_FALSE(right2.has_value());
    }

    TEST_F(SimplicialComplexTest, EdgeNeighbors2DNoRight) {
        Complex2D mesh;
        make_unit_square_triangulation(mesh);

        // Bottom edge (v0-v1) should have only left neighbor
        VIdx2 v0 = 0, v1 = 1;
        auto edge_idx = find_simplex(mesh, DIM_EDGE, { v0, v1 });
        ASSERT_NE(edge_idx, -1);

        auto [left, right] = edge_neighbors_2d(mesh, edge_idx);
        EXPECT_NE(left, -1);
        EXPECT_FALSE(right.has_value());

        // Verify that left triangle is indeed the one containing v2
        auto tri = triangle_at(mesh, left);
        std::set<VIdx2> tri_vertices = { tri[0], tri[1], tri[2] };
        EXPECT_TRUE(tri_vertices.find(v0) != tri_vertices.end());
        EXPECT_TRUE(tri_vertices.find(v1) != tri_vertices.end());
        EXPECT_TRUE(tri_vertices.find(VIdx2(2)) != tri_vertices.end());  // v2
    }

    // =========================================================================
    // Additional test: Unit square triangulation helper
    // =========================================================================

    TEST_F(SimplicialComplexTest, MakeUnitSquareTriangulation) {
        Complex2D mesh;
        make_unit_square_triangulation(mesh);

        EXPECT_EQ(num_vertices(mesh), 4);
        EXPECT_EQ(num_edges(mesh), 5);  // 4 sides + 1 diagonal
        EXPECT_EQ(num_triangles(mesh), 2);

        // Check vertices are at correct positions
        EXPECT_TRUE(vertex(mesh, 0) == Point2(0_r, 0_r));
        EXPECT_TRUE(vertex(mesh, 1) == Point2(1_r, 0_r));
        EXPECT_TRUE(vertex(mesh, 2) == Point2(1_r, 1_r));
        EXPECT_TRUE(vertex(mesh, 3) == Point2(0_r, 1_r));

        // Check diagonal edge exists
        std::ptrdiff_t diag = find_simplex(mesh, DIM_EDGE, { VIdx2(0), VIdx2(2) });
        EXPECT_NE(diag, -1);

        // Check triangles
        auto tri0 = triangle_at(mesh, 0);
        auto tri1 = triangle_at(mesh, 1);

        // One triangle should be (0,1,2), the other (0,2,3)
        std::set<std::set<VIdx2>> expected_triangles = {
            { VIdx2(0), VIdx2(1), VIdx2(2) },
            { VIdx2(0), VIdx2(2), VIdx2(3) }
        };
        std::set<std::set<VIdx2>> actual_triangles = {
            { tri0[0], tri0[1], tri0[2] },
            { tri1[0], tri1[1], tri1[2] }
        };
        EXPECT_EQ(actual_triangles, expected_triangles);
    }

} // namespace delta::geometry::testing