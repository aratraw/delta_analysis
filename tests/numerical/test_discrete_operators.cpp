// tests/numerical/test_discrete_operators.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "delta/numerical/discrete_operators.h"
#include "delta/numerical/concepts.h"
#include "../test_fixtures.h"

namespace delta::testing {

    // -----------------------------------------------------------------------------
    // Test fixture for discrete operators on 2D simplicial complexes
    // -----------------------------------------------------------------------------
    class DiscreteOperatorsTest : public SimplicialComplexFixture<2, double> {
    protected:
        using Scalar = double;
        using Vector = Eigen::Vector2d;

        delta::geometry::EuclideanMetric metric;

        // Helper to compute edge vector from v0 to v1 (oriented from lower index to higher)
        Vector edge_vector(std::size_t v0, std::size_t v1) const {
            return square2D.vertex(v1) - square2D.vertex(v0);
        }

        // Helper to compute edge direction unit vector
        Vector edge_direction(std::size_t v0, std::size_t v1) const {
            Vector e = edge_vector(v0, v1);
            return e / e.norm();
        }

        // Helper to get edge indices
        std::size_t edge_idx(std::size_t v0, std::size_t v1) const {
            auto idx = square2D.find_simplex(1, { v0, v1 });
            EXPECT_GE(idx, 0);
            return static_cast<std::size_t>(idx);
        }
    };

    // -----------------------------------------------------------------------------
    // Gradient of linear function
    // -----------------------------------------------------------------------------
    TEST_F(DiscreteOperatorsTest, GradientOfLinearFunction) {
        // Linear function f(x,y) = 2x + 3y + 1
        std::vector<Scalar> f(square2D.num_vertices());
        for (std::size_t i = 0; i < square2D.num_vertices(); ++i) {
            Vector p = square2D.vertex(i);
            f[i] = 2.0 * p.x() + 3.0 * p.y() + 1.0;
        }

        auto grad = delta::numerical::discrete_gradient(square2D, f, metric);
        EXPECT_EQ(grad.size(), square2D.num_edges());

        Vector grad_exact(2.0, 3.0);

        for (std::size_t e = 0; e < square2D.num_edges(); ++e) {
            auto [v0, v1] = square2D.edge_at(e);
            Scalar len = square2D.edge_length(e, metric);
            Vector dir = (square2D.vertex(v1) - square2D.vertex(v0)) / len;
            Scalar expected = grad_exact.dot(dir);
            Scalar computed = grad[e];
            EXPECT_NEAR(computed, expected, 1e-12);
        }
    }

    // -----------------------------------------------------------------------------
    // Divergence of gradient of linear function
    // -----------------------------------------------------------------------------
    TEST_F(DiscreteOperatorsTest, DivGradOfLinearFunction) {
        // f(x,y) = 2x + 3y + 1
        std::vector<Scalar> f(square2D.num_vertices());
        for (std::size_t i = 0; i < square2D.num_vertices(); ++i) {
            Vector p = square2D.vertex(i);
            f[i] = 2.0 * p.x() + 3.0 * p.y() + 1.0;
        }

        auto grad = delta::numerical::discrete_gradient(square2D, f, metric);
        auto div_raw = delta::numerical::discrete_divergence_raw(square2D, grad);
        auto vertex_areas = delta::numerical::compute_vertex_dual_areas(square2D, metric);
        auto div = delta::numerical::discrete_divergence(square2D, grad, metric);

        // Total raw divergence should be zero (telescoping sum)
        Scalar total_raw = 0.0;
        for (Scalar d : div_raw) total_raw += d;
        EXPECT_NEAR(total_raw, 0.0, 1e-12);

        // For a linear function, the normalized divergence should be bounded.
        Scalar max_div = 0.0;
        for (Scalar d : div) {
            if (std::abs(d) > max_div) max_div = std::abs(d);
        }
        EXPECT_LE(max_div, 1.0);
    }

    // -----------------------------------------------------------------------------
    // Divergence of a constant vector field
    // -----------------------------------------------------------------------------
    TEST_F(DiscreteOperatorsTest, DivergenceOfConstantField) {
        // 1‑form representing constant vector field V = (1,2)
        Vector V(1.0, 2.0);
        std::vector<Scalar> omega(square2D.num_edges());
        for (std::size_t e = 0; e < square2D.num_edges(); ++e) {
            auto [v0, v1] = square2D.edge_at(e);
            Vector edge_vec = square2D.vertex(v1) - square2D.vertex(v0);
            omega[e] = V.dot(edge_vec);
        }

        auto div_raw = delta::numerical::discrete_divergence_raw(square2D, omega);
        Scalar total_raw = 0.0;
        for (Scalar d : div_raw) total_raw += d;
        EXPECT_NEAR(total_raw, 0.0, 1e-12);
    }

    // -----------------------------------------------------------------------------
    // Laplacian of quadratic function
    // -----------------------------------------------------------------------------
    TEST_F(DiscreteOperatorsTest, LaplacianOfQuadratic) {
        // f(x,y) = x^2 + y^2, Δf = 4
        std::vector<Scalar> f(square2D.num_vertices());
        for (std::size_t i = 0; i < square2D.num_vertices(); ++i) {
            Vector p = square2D.vertex(i);
            f[i] = p.x() * p.x() + p.y() * p.y();
        }

        auto lap = delta::numerical::discrete_laplacian_cotangent(square2D, f, metric);

        for (std::size_t i = 0; i < lap.size(); ++i) {
            EXPECT_NEAR(lap[i], 4.0, 1.0);
        }
    }

} // namespace delta::testing