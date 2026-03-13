// tests/geometry/test_simplicial_path.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "delta/geometry/simplicial_path.h"
#include "../test_fixtures.h"

namespace delta::testing {

    // -----------------------------------------------------------------------------
    // Test fixture for SimplicialDeltaPath tests (2D)
    // -----------------------------------------------------------------------------
    class SimplicialPathTest : public SimplicialComplexFixture<2, double> {
    protected:
        using Path = geometry::SimplicialDeltaPath<2, double>;
        using Point = Eigen::Vector2d;
        EuclideanMetric metric;

        // Helper to get max edge length in a complex
        double max_edge_length(const Path::Complex& complex) {
            double max_len = 0.0;
            for (std::size_t e = 0; e < complex.num_edges(); ++e) {
                double len = complex.edge_length(e, metric);
                if (len > max_len) max_len = len;
            }
            return max_len;
        }
    };

    // -----------------------------------------------------------------------------
    // Refinement counts
    // -----------------------------------------------------------------------------

    TEST_F(SimplicialPathTest, InitialCounts) {
        geometry::SimplicialDeltaPath<2, double> path(triangle2D);

        EXPECT_EQ(path.num_vertices(), 3);
        EXPECT_EQ(path.num_edges(), 3);
        EXPECT_EQ(path.num_triangles(), 1);
        EXPECT_EQ(path.level(), 0);
    }

    TEST_F(SimplicialPathTest, FirstRefinementCounts) {
        geometry::SimplicialDeltaPath<2, double> path(triangle2D);
        path.advance();

        EXPECT_EQ(path.num_vertices(), 7);
        EXPECT_EQ(path.num_edges(), 9);
        EXPECT_EQ(path.num_triangles(), 6);
        EXPECT_EQ(path.level(), 1);
    }

    TEST_F(SimplicialPathTest, SecondRefinementCounts) {
        geometry::SimplicialDeltaPath<2, double> path(triangle2D);
        path.advance();
        path.advance();

        EXPECT_EQ(path.num_triangles(), 36);
        EXPECT_GT(path.num_vertices(), 7);
        EXPECT_GT(path.num_edges(), 9);
        EXPECT_EQ(path.level(), 2);
    }

    TEST_F(SimplicialPathTest, MultipleRefinements) {
        geometry::SimplicialDeltaPath<2, double> path(triangle2D);
        std::size_t steps = 3;
        for (std::size_t i = 0; i < steps; ++i) {
            path.advance();
        }

        EXPECT_EQ(path.level(), steps);
        std::size_t expected_triangles = 1;
        for (std::size_t i = 0; i < steps; ++i) expected_triangles *= 6;
        EXPECT_EQ(path.num_triangles(), expected_triangles);
    }

    // -----------------------------------------------------------------------------
    // max_gap decreases
    // -----------------------------------------------------------------------------

    TEST_F(SimplicialPathTest, MaxGapDecreases) {
        geometry::SimplicialDeltaPath<2, double> path(triangle2D);

        double max_gap0 = path.max_gap(metric);
        EXPECT_NEAR(max_gap0, 1.0, 1e-12);

        path.advance();
        double max_gap1 = path.max_gap(metric);
        EXPECT_NEAR(max_gap1, 1.0 / std::sqrt(3.0), 1e-12);

        path.advance();
        double max_gap2 = path.max_gap(metric);
        EXPECT_LT(max_gap2, max_gap1);
    }

    TEST_F(SimplicialPathTest, MaxGapWithMetric) {
        geometry::SimplicialDeltaPath<2, double> path(triangle2D);

        struct ScaledMetric {
            double operator()(const Point& a, const Point& b) const {
                return 2.0 * (a - b).norm();
            }
        } scaled_metric;

        double max_gap_scaled = path.max_gap(scaled_metric);
        EXPECT_NEAR(max_gap_scaled, 2.0, 1e-12);

        path.advance();
        double max_gap1_scaled = path.max_gap(scaled_metric);
        double expected = 2.0 / std::sqrt(3.0);
        EXPECT_NEAR(max_gap1_scaled, expected, 1e-12);
    }

    // -----------------------------------------------------------------------------
    // Current grid access
    // -----------------------------------------------------------------------------

    TEST_F(SimplicialPathTest, CurrentGridConsistency) {
        geometry::SimplicialDeltaPath<2, double> path(triangle2D);

        const auto& grid0 = path.current_grid();
        EXPECT_EQ(grid0.num_vertices(), 3);

        path.advance();
        const auto& grid1 = path.current_grid();
        EXPECT_EQ(grid1.num_vertices(), 7);

        for (std::size_t i = 0; i < grid0.num_vertices(); ++i) {
            Point v = grid0.vertex(i);
            bool found = false;
            for (std::size_t j = 0; j < grid1.num_vertices(); ++j) {
                if (grid1.vertex(j).isApprox(v, 1e-12)) {
                    found = true;
                    break;
                }
            }
            EXPECT_TRUE(found);
        }
    }

    // -----------------------------------------------------------------------------
    // Factory from_level
    // -----------------------------------------------------------------------------

    TEST_F(SimplicialPathTest, FromLevel) {
        std::size_t target_level = 2;
        auto path = geometry::SimplicialDeltaPath<2, double>::from_level(target_level, triangle2D);

        EXPECT_EQ(path.level(), target_level);
        EXPECT_EQ(path.num_triangles(), 36);
    }

} // namespace delta::testing