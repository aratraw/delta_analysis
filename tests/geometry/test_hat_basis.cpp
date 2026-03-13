// tests/geometry/test_hat_basis.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "delta/geometry/hat_basis.h"
#include "delta/geometry/simplicial_complex.h"
#include "../test_fixtures.h"

namespace delta::testing {

    // -----------------------------------------------------------------------------
    // Test fixture for HatBasis tests (2D triangle)
    // -----------------------------------------------------------------------------
    class HatBasisTest : public SimplicialComplexFixture<2, double> {
    protected:
        using Basis = geometry::HatBasis<Complex>;
        std::unique_ptr<Basis> basis;

        void SetUp() override {
            SimplicialComplexFixture::SetUp();
            basis = std::make_unique<Basis>(triangle2D);
        }

        // Helper: evaluate linear combination of basis functions with given coefficients
        double evaluate_field(const std::vector<double>& coeffs, const Eigen::Vector2d& p) const {
            double val = 0.0;
            for (std::size_t i = 0; i < coeffs.size(); ++i) {
                val += coeffs[i] * basis->evaluate(i, p);
            }
            return val;
        }
    };

    // -----------------------------------------------------------------------------
    // Interpolation of linear function
    // -----------------------------------------------------------------------------
    TEST_F(HatBasisTest, InterpolatesLinearFunction) {
        // Linear function f(x,y) = 2x + 3y
        // Values at vertices:
        // v0 (0,0): 0
        // v1 (1,0): 2
        // v2 (0.5, sqrt(3)/2): 2*0.5 + 3*(sqrt(3)/2) = 1 + (3√3)/2 ≈ 1 + 2.598 = 3.598
        std::vector<double> coeffs(3);
        coeffs[0] = 0.0;
        coeffs[1] = 2.0;
        coeffs[2] = 2 * 0.5 + 3 * (std::sqrt(3.0) / 2.0);

        // Test point inside triangle (barycentric coordinates roughly (0.2,0.3,0.5))
        Eigen::Vector2d p(0.2, 0.3);
        double exact = 2 * 0.2 + 3 * 0.3; // = 0.4 + 0.9 = 1.3

        double interpolated = evaluate_field(coeffs, p);
        EXPECT_NEAR(interpolated, exact, 1e-12);
    }

    // -----------------------------------------------------------------------------
    // Partition of unity
    // -----------------------------------------------------------------------------
    TEST_F(HatBasisTest, PartitionOfUnity) {
        // Test several points inside the triangle
        std::vector<Eigen::Vector2d> points = {
            {0.2, 0.1},
            {0.5, 0.2},
            {0.3, 0.4}
        };

        for (const auto& p : points) {
            double sum = 0.0;
            for (std::size_t i = 0; i < triangle2D.num_vertices(); ++i) {
                sum += basis->evaluate(i, p);
            }
            EXPECT_NEAR(sum, 1.0, 1e-12);
        }

        // Test on vertices themselves
        for (std::size_t i = 0; i < triangle2D.num_vertices(); ++i) {
            double sum = 0.0;
            for (std::size_t j = 0; j < triangle2D.num_vertices(); ++j) {
                sum += basis->evaluate(j, triangle2D.vertex(i));
            }
            EXPECT_NEAR(sum, 1.0, 1e-12);
        }

        // Test point outside (basis returns 0 outside)
        Eigen::Vector2d outside(2.0, 2.0);
        double sum_out = 0.0;
        for (std::size_t i = 0; i < triangle2D.num_vertices(); ++i) {
            sum_out += basis->evaluate(i, outside);
        }
        EXPECT_NEAR(sum_out, 0.0, 1e-12);
    }

    // -----------------------------------------------------------------------------
    // Gradients are constant on each simplex
    // -----------------------------------------------------------------------------
    TEST_F(HatBasisTest, GradientsConstantOnSimplex) {
        // Pick two points inside the triangle
        Eigen::Vector2d p1(0.2, 0.1);
        Eigen::Vector2d p2(0.4, 0.3);

        // For each vertex, gradient should be the same at both points
        for (std::size_t i = 0; i < triangle2D.num_vertices(); ++i) {
            auto grad1 = basis->gradient(i, p1);
            auto grad2 = basis->gradient(i, p2);
            EXPECT_TRUE(grad1.isApprox(grad2, 1e-12));
        }
    }

    TEST_F(HatBasisTest, GradientOfLinearFunction) {
        // For linear function f(x,y) = a·x + b·y + c, the gradient is constant (a,b)
        // and should equal sum_i coeffs_i * grad(phi_i).
        double a = 2.0, b = 3.0;
        std::vector<double> coeffs(3);
        coeffs[0] = a * 0 + b * 0; // 0
        coeffs[1] = a * 1 + b * 0; // 2
        coeffs[2] = a * 0.5 + b * (std::sqrt(3) / 2); // as above

        Eigen::Vector2d p(0.2, 0.3);
        Eigen::Vector2d grad_exact(a, b);

        Eigen::Vector2d grad_computed(0, 0);
        for (std::size_t i = 0; i < triangle2D.num_vertices(); ++i) {
            grad_computed += coeffs[i] * basis->gradient(i, p);
        }

        EXPECT_TRUE(grad_computed.isApprox(grad_exact, 1e-12));
    }

    // -----------------------------------------------------------------------------
    // Point location and barycentric coordinates
    // -----------------------------------------------------------------------------
    TEST_F(HatBasisTest, PointLocationAndBarycentric) {
        // Test point inside triangle
        Eigen::Vector2d p_inside(0.2, 0.3);
        auto loc = basis->locate_point_in_simplex(p_inside, triangle2D, 2, 0);
        ASSERT_TRUE(loc.has_value());
        const auto& bary = *loc;
        EXPECT_EQ(bary.size(), 3);
        // Barycentric coordinates should sum to 1
        double sum = bary[0] + bary[1] + bary[2];
        EXPECT_NEAR(sum, 1.0, 1e-12);
        // And the point should be reproduced: p = Σ bary[i] * vertex[i]
        Eigen::Vector2d reconstructed(0, 0);
        for (std::size_t i = 0; i < 3; ++i) {
            reconstructed += bary[i] * triangle2D.vertex(i);
        }
        EXPECT_TRUE(reconstructed.isApprox(p_inside, 1e-12));

        // Test point on edge
        Eigen::Vector2d p_edge(0.5, 0.0); // on edge v0-v1
        auto loc_edge = basis->locate_point_in_simplex(p_edge, triangle2D, 2, 0);
        ASSERT_TRUE(loc_edge.has_value());
        const auto& bary_edge = *loc_edge;
        // Should have bary[2] = 0 approximately
        EXPECT_NEAR(bary_edge[2], 0.0, 1e-12);
        EXPECT_NEAR(bary_edge[0] + bary_edge[1], 1.0, 1e-12);

        // Test point at vertex
        Eigen::Vector2d p_vertex = triangle2D.vertex(1);
        auto loc_vertex = basis->locate_point_in_simplex(p_vertex, triangle2D, 2, 0);
        ASSERT_TRUE(loc_vertex.has_value());
        const auto& bary_vertex = *loc_vertex;
        EXPECT_NEAR(bary_vertex[1], 1.0, 1e-12);
        EXPECT_NEAR(bary_vertex[0], 0.0, 1e-12);
        EXPECT_NEAR(bary_vertex[2], 0.0, 1e-12);

        // Test point outside
        Eigen::Vector2d p_outside(2.0, 2.0);
        auto loc_out = basis->locate_point_in_simplex(p_outside, triangle2D, 2, 0);
        EXPECT_FALSE(loc_out.has_value());
    }

} // namespace delta::testing