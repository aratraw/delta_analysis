// tests/geometry/test_dual_complex.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "delta/geometry/dual_complex.h"
#include "delta/geometry/simplicial_complex.h"
#include "../test_fixtures.h"

namespace delta::testing {

    // -----------------------------------------------------------------------------
    // Test fixture for DualComplex tests (2D)
    // -----------------------------------------------------------------------------
    class DualComplex2DTest : public SimplicialComplexFixture<2, double> {
    protected:
        EuclideanMetric metric;
    };

    // -----------------------------------------------------------------------------
    // 2D tests
    // -----------------------------------------------------------------------------

    TEST_F(DualComplex2DTest, VertexDualAreasSquare) {
        delta::geometry::DualComplex<Complex, EuclideanMetric> dual(square2D, metric);

        double square_area = 0.0;
        for (std::size_t t = 0; t < square2D.num_triangles(); ++t) {
            square_area += square2D.cell_volume(t, metric);
        }
        EXPECT_NEAR(square_area, 1.0, 1e-12);

        double sum_vertex_areas = 0.0;
        for (std::size_t v = 0; v < square2D.num_vertices(); ++v) {
            sum_vertex_areas += dual.dual_volume(0, v);
        }
        EXPECT_NEAR(sum_vertex_areas, square_area, 1e-12);

        // Expected values: vertices 0 and 2 are adjacent to both triangles → area = 1/3,
        // vertices 1 and 3 adjacent to one triangle → area = 1/6.
        EXPECT_NEAR(dual.dual_volume(0, 0), 1.0 / 3.0, 1e-12);
        EXPECT_NEAR(dual.dual_volume(0, 1), 1.0 / 6.0, 1e-12);
        EXPECT_NEAR(dual.dual_volume(0, 2), 1.0 / 3.0, 1e-12);
        EXPECT_NEAR(dual.dual_volume(0, 3), 1.0 / 6.0, 1e-12);
    }

    TEST_F(DualComplex2DTest, EdgeDualLengthsSquare) {
        delta::geometry::DualComplex<Complex, EuclideanMetric> dual(square2D, metric);

        auto e0 = static_cast<std::size_t>(square2D.find_simplex(1, { 0u, 1u }));
        EXPECT_NEAR(dual.dual_volume(1, e0), 0.5, 1e-12);

        auto e1 = static_cast<std::size_t>(square2D.find_simplex(1, { 1u, 2u }));
        EXPECT_NEAR(dual.dual_volume(1, e1), 0.5, 1e-12);

        auto e2 = static_cast<std::size_t>(square2D.find_simplex(1, { 2u, 3u }));
        EXPECT_NEAR(dual.dual_volume(1, e2), 0.5, 1e-12);

        auto e3 = static_cast<std::size_t>(square2D.find_simplex(1, { 0u, 3u }));
        EXPECT_NEAR(dual.dual_volume(1, e3), 0.5, 1e-12);

        auto diag = static_cast<std::size_t>(square2D.find_simplex(1, { 0u, 2u }));
        double expected = (0.5 + 0.5) / std::sqrt(2.0);
        EXPECT_NEAR(dual.dual_volume(1, diag), expected, 1e-12);
    }

    // -----------------------------------------------------------------------------
    // 3D tests (using tetrahedron)
    // -----------------------------------------------------------------------------
    class DualComplex3DTest : public SimplicialComplexFixture<3, double> {
    protected:
        EuclideanMetric metric;
    };

    TEST_F(DualComplex3DTest, VertexDualVolumesTetrahedron) {
        delta::geometry::DualComplex<Complex, EuclideanMetric> dual(tetrahedron3D, metric);

        double total_vol = tetrahedron3D.cell_volume(0, metric);
        double expected_vol = std::sqrt(2.0) / 12.0;
        EXPECT_NEAR(total_vol, expected_vol, 1e-12);

        double sum_vertex_vols = 0.0;
        for (std::size_t v = 0; v < tetrahedron3D.num_vertices(); ++v) {
            sum_vertex_vols += dual.dual_volume(0, v);
        }
        EXPECT_NEAR(sum_vertex_vols, total_vol, 1e-12);

        for (std::size_t v = 0; v < tetrahedron3D.num_vertices(); ++v) {
            EXPECT_NEAR(dual.dual_volume(0, v), total_vol / 4.0, 1e-12);
        }
    }

    TEST_F(DualComplex3DTest, EdgeDualAreasTetrahedron) {
        delta::geometry::DualComplex<Complex, EuclideanMetric> dual(tetrahedron3D, metric);

        double expected = std::sqrt(2.0) / 12.0;
        for (std::size_t e = 0; e < tetrahedron3D.num_edges(); ++e) {
            EXPECT_NEAR(dual.dual_volume(1, e), expected, 1e-12);
        }
    }

    TEST_F(DualComplex3DTest, FaceDualLengthsTetrahedron) {
        delta::geometry::DualComplex<Complex, EuclideanMetric> dual(tetrahedron3D, metric);

        double vol = std::sqrt(2.0) / 12.0;
        double area = std::sqrt(3.0) / 4.0;
        double expected = vol / area;

        for (std::size_t f = 0; f < tetrahedron3D.num_triangles(); ++f) {
            EXPECT_NEAR(dual.dual_volume(2, f), expected, 1e-12);
        }
    }

} // namespace delta::testing