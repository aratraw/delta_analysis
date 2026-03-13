// tests/geometry/test_simplicial_complex_advanced.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "delta/geometry/simplicial_complex.h"
#include "../test_fixtures.h"

namespace delta::testing {

    using namespace delta::geometry;

    // -----------------------------------------------------------------------------
    // Test fixture for advanced simplicial complex tests (2D)
    // -----------------------------------------------------------------------------
    class SimplicialComplexAdvancedTest : public SimplicialComplexFixture<2, double> {
    protected:
        EuclideanMetric metric;
    };

    // -----------------------------------------------------------------------------
    // Geometric queries with metric
    // -----------------------------------------------------------------------------

    TEST_F(SimplicialComplexAdvancedTest, EdgeLength) {
        auto len0 = triangle2D.edge_length(0, metric);
        EXPECT_NEAR(len0, 1.0, 1e-12);

        auto len1 = triangle2D.edge_length(1, metric);
        EXPECT_NEAR(len1, 1.0, 1e-12);

        auto len2 = triangle2D.edge_length(2, metric);
        EXPECT_NEAR(len2, 1.0, 1e-12);
    }

    TEST_F(SimplicialComplexAdvancedTest, EdgeCenter) {
        auto center0 = triangle2D.edge_center(0);
        EXPECT_NEAR(center0.x(), 0.5, 1e-12);
        EXPECT_NEAR(center0.y(), 0.0, 1e-12);
    }

    TEST_F(SimplicialComplexAdvancedTest, CellCenter) {
        auto center = triangle2D.cell_center(0);
        double expected_x = (0.0 + 1.0 + 0.5) / 3.0;
        double expected_y = (0.0 + 0.0 + std::sqrt(3.0) / 2.0) / 3.0;
        EXPECT_NEAR(center.x(), expected_x, 1e-12);
        EXPECT_NEAR(center.y(), expected_y, 1e-12);
    }

    TEST_F(SimplicialComplexAdvancedTest, CellVolume) {
        double expected = std::sqrt(3.0) / 4.0;
        double volume = triangle2D.cell_volume(0, metric);
        EXPECT_NEAR(volume, expected, 1e-12);
    }

    TEST_F(SimplicialComplexAdvancedTest, EdgeNormal2D) {
        auto normal = triangle2D.edge_normal(0, metric);
        EXPECT_NEAR(normal.norm(), 1.0, 1e-12);
        EXPECT_NEAR(normal.dot(Eigen::Vector2d(1, 0)), 0.0, 1e-12);
        EXPECT_NEAR(normal.y(), -1.0, 1e-12);
    }

    TEST_F(SimplicialComplexAdvancedTest, EdgeNeighbors) {
        auto [left, right] = triangle2D.edge_neighbors(0);
        EXPECT_EQ(left, 0u);
        EXPECT_FALSE(right.has_value());

        auto diag_idx = square2D.find_simplex(1, { 0u, 2u });
        ASSERT_GE(diag_idx, 0);
        auto [l, r] = square2D.edge_neighbors(static_cast<std::size_t>(diag_idx));
        EXPECT_EQ(l, 0u);
        EXPECT_TRUE(r.has_value());
        EXPECT_EQ(*r, 1u);
    }

    // -----------------------------------------------------------------------------
    // Barycentric subdivision
    // -----------------------------------------------------------------------------

    TEST_F(SimplicialComplexAdvancedTest, BarycentricSubdivisionCounts) {
        auto [subdivided, subdiv_map] = barycentric_subdivide(triangle2D);

        EXPECT_EQ(subdivided.num_triangles(), 6);
        EXPECT_EQ(subdivided.num_edges(), 9);
        EXPECT_EQ(subdivided.num_vertices(), 7);

        SimplexKey key{ 2, 0 };
        auto it = subdiv_map.find(key);
        ASSERT_NE(it, subdiv_map.end());
        EXPECT_EQ(it->second.size(), 6);
        for (const auto& sk : it->second) {
            EXPECT_EQ(sk.dim, 2);
        }
    }

    TEST_F(SimplicialComplexAdvancedTest, BarycentricSubdivisionPreservesVolume) {
        double original_vol = triangle2D.cell_volume(0, metric);
        auto [subdivided, subdiv_map] = barycentric_subdivide(triangle2D);

        double total_vol = 0.0;
        for (std::size_t i = 0; i < subdivided.num_triangles(); ++i) {
            total_vol += subdivided.cell_volume(i, metric);
        }
        EXPECT_NEAR(total_vol, original_vol, 1e-12);
    }

    // -----------------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------------

    TEST(SimplicialComplexEdgeCasesTest, EmptyComplex) {
        SimplicialComplex<2, double> complex;
        EuclideanMetric metric;

        EXPECT_EQ(complex.num_vertices(), 0);
        EXPECT_EQ(complex.num_edges(), 0);
        EXPECT_EQ(complex.num_triangles(), 0);

        EXPECT_THROW(complex.edge_length(0, metric), std::out_of_range);
    }

    TEST(SimplicialComplexEdgeCasesTest, SingleVertex) {
        SimplicialComplex<2, double> complex;
        auto v0 = complex.add_vertex({ 0.0, 0.0 });
        EXPECT_EQ(complex.num_vertices(), 1);
        EXPECT_EQ(complex.num_edges(), 0);
        EXPECT_EQ(complex.num_triangles(), 0);
    }

    TEST(SimplicialComplexEdgeCasesTest, SingleEdge) {
        SimplicialComplex<2, double> complex;
        auto v0 = complex.add_vertex({ 0.0, 0.0 });
        auto v1 = complex.add_vertex({ 1.0, 0.0 });
        bool added = complex.add_edge(v0, v1);
        EXPECT_TRUE(added);
        EXPECT_EQ(complex.num_vertices(), 2);
        EXPECT_EQ(complex.num_edges(), 1);
        EXPECT_EQ(complex.num_triangles(), 0);

        EuclideanMetric metric;
        EXPECT_NEAR(complex.edge_length(0, metric), 1.0, 1e-12);
        auto center = complex.edge_center(0);
        EXPECT_NEAR(center.x(), 0.5, 1e-12);
        EXPECT_NEAR(center.y(), 0.0, 1e-12);
    }

    TEST(SimplicialComplexEdgeCasesTest, DegenerateTriangle) {
        SimplicialComplex<2, double> complex;
        auto v0 = complex.add_vertex({ 0.0, 0.0 });
        auto v1 = complex.add_vertex({ 1.0, 0.0 });
        auto v2 = complex.add_vertex({ 2.0, 0.0 });

        bool added = complex.add_triangle(v0, v1, v2);
        EXPECT_FALSE(added);
        EXPECT_EQ(complex.num_edges(), 0);
        EXPECT_EQ(complex.num_triangles(), 0);
    }

    TEST(SimplicialComplexEdgeCasesTest, DuplicateSimplex) {
        SimplicialComplex<2, double> complex;
        auto v0 = complex.add_vertex({ 0.0, 0.0 });
        auto v1 = complex.add_vertex({ 1.0, 0.0 });
        auto v2 = complex.add_vertex({ 0.0, 1.0 });

        bool added1 = complex.add_triangle(v0, v1, v2);
        EXPECT_TRUE(added1);
        bool added2 = complex.add_triangle(v0, v1, v2);
        EXPECT_FALSE(added2);

        EXPECT_EQ(complex.num_triangles(), 1);
    }

} // namespace delta::testing