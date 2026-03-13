// tests/geometry/test_discrete_forms.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include "delta/geometry/discrete_forms.h"
#include "delta/geometry/dual_complex.h"
#include "delta/geometry/simplicial_complex.h"   // вместо barycentric_subdivision.h
#include "../test_fixtures.h"

namespace delta::testing {

    // -----------------------------------------------------------------------------
    // Test fixture for 2D discrete forms (triangle and square)
    // -----------------------------------------------------------------------------
    class DiscreteForms2DTest : public SimplicialComplexFixture<2, double> {
    protected:
        EuclideanMetric metric;
        using Dual = DualComplex<Complex, EuclideanMetric>;
        std::unique_ptr<Dual> dual;

        void SetUp() override {
            SimplicialComplexFixture::SetUp();
            dual = std::make_unique<Dual>(square2D, metric); // use square for internal edges
        }

        // Helper to create a random 0-form on square2D
        DiscreteForm<0, double, Complex> random_0form() {
            DiscreteForm<0, double, Complex> f(square2D);
            std::mt19937 rng(42);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (std::size_t i = 0; i < square2D.num_vertices(); ++i) {
                f.at(i) = dist(rng);
            }
            return f;
        }

        // Helper to create a random 1-form on square2D
        DiscreteForm<1, double, Complex> random_1form() {
            DiscreteForm<1, double, Complex> omega(square2D);
            std::mt19937 rng(43);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (std::size_t i = 0; i < square2D.num_edges(); ++i) {
                omega.at(i) = dist(rng);
            }
            return omega;
        }

        // Helper to create a random 2-form on square2D
        DiscreteForm<2, double, Complex> random_2form() {
            DiscreteForm<2, double, Complex> eta(square2D);
            std::mt19937 rng(44);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (std::size_t i = 0; i < square2D.num_triangles(); ++i) {
                eta.at(i) = dist(rng);
            }
            return eta;
        }

        // L2 inner product for 0-forms
        double inner_product(const DiscreteForm<0, double, Complex>& a,
            const DiscreteForm<0, double, Complex>& b) {
            double sum = 0.0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                sum += a.at(i) * b.at(i);
            }
            return sum;
        }

        // L2 inner product for 1-forms (using dual volumes as metric)
        double inner_product(const DiscreteForm<1, double, Complex>& a,
            const DiscreteForm<1, double, Complex>& b) {
            double sum = 0.0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                sum += a.at(i) * b.at(i) * dual->dual_volume(1, i);
            }
            return sum;
        }

        // L2 inner product for 2-forms (using dual volumes)
        double inner_product(const DiscreteForm<2, double, Complex>& a,
            const DiscreteForm<2, double, Complex>& b) {
            double sum = 0.0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                sum += a.at(i) * b.at(i) * dual->dual_volume(2, i);
            }
            return sum;
        }
    };

    // -----------------------------------------------------------------------------
    // Test fixture for 3D discrete forms (tetrahedron)
    // -----------------------------------------------------------------------------
    class DiscreteForms3DTest : public SimplicialComplexFixture<3, double> {
    protected:
        EuclideanMetric metric;
        using Dual = DualComplex<Complex, EuclideanMetric>;
        std::unique_ptr<Dual> dual;

        void SetUp() override {
            SimplicialComplexFixture::SetUp();
            dual = std::make_unique<Dual>(tetrahedron3D, metric);
        }

        // Helper to create a random 0-form on tetrahedron
        DiscreteForm<0, double, Complex> random_0form() {
            DiscreteForm<0, double, Complex> f(tetrahedron3D);
            std::mt19937 rng(45);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (std::size_t i = 0; i < tetrahedron3D.num_vertices(); ++i) {
                f.at(i) = dist(rng);
            }
            return f;
        }

        // Helper to create a random 1-form on tetrahedron
        DiscreteForm<1, double, Complex> random_1form() {
            DiscreteForm<1, double, Complex> omega(tetrahedron3D);
            std::mt19937 rng(46);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (std::size_t i = 0; i < tetrahedron3D.num_edges(); ++i) {
                omega.at(i) = dist(rng);
            }
            return omega;
        }

        // Helper to create a random 2-form on tetrahedron
        DiscreteForm<2, double, Complex> random_2form() {
            DiscreteForm<2, double, Complex> eta(tetrahedron3D);
            std::mt19937 rng(47);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (std::size_t i = 0; i < tetrahedron3D.num_triangles(); ++i) {
                eta.at(i) = dist(rng);
            }
            return eta;
        }
    };

    // =============================================================================
    // 2D Tests
    // =============================================================================

    TEST_F(DiscreteForms2DTest, ExteriorDerivativeDDZero) {
        // For a 0-form, d∘d = 0 (0-form → 1-form → 2-form)
        auto f = random_0form();
        auto df = f.d();
        auto ddf = df.d(); // 2-form
        for (std::size_t i = 0; i < square2D.num_triangles(); ++i) {
            EXPECT_NEAR(ddf.at(i), 0.0, 1e-12);
        }
        // For a 1-form, d∘d would give a 3‑form, which does not exist in 2D.
    }

    TEST_F(DiscreteForms2DTest, ExteriorDerivativeZeroForm) {
        // df on edge (i,j) with i<j equals f(j) - f(i)
        DiscreteForm<0, double, Complex> f(square2D);
        f.at(0) = 1.0;
        f.at(1) = 2.0;
        f.at(2) = 3.0;
        f.at(3) = 4.0;

        auto df = f.d();

        auto e01 = square2D.find_simplex(1, { 0u, 1u });
        auto e12 = square2D.find_simplex(1, { 1u, 2u });
        auto e23 = square2D.find_simplex(1, { 2u, 3u });
        auto e30 = square2D.find_simplex(1, { 3u, 0u }); // stored as (0,3)
        auto e02 = square2D.find_simplex(1, { 0u, 2u }); // diagonal

        EXPECT_NEAR(df.at(e01), 2.0 - 1.0, 1e-12);
        EXPECT_NEAR(df.at(e12), 3.0 - 2.0, 1e-12);
        EXPECT_NEAR(df.at(e23), 4.0 - 3.0, 1e-12);
        EXPECT_NEAR(df.at(e30), 4.0 - 1.0, 1e-12); // (0,3) orientation 0→3
        EXPECT_NEAR(df.at(e02), 3.0 - 1.0, 1e-12);
    }

    TEST_F(DiscreteForms2DTest, ExteriorDerivativeOneForm) {
        // For a 1-form ω on a triangle, dω(triangle) = ω(e01) + ω(e12) + ω(e20)
        // with signs determined by orientation of the triangle.
        DiscreteForm<1, double, Complex> omega(square2D);
        auto e01 = square2D.find_simplex(1, { 0u, 1u });
        auto e12 = square2D.find_simplex(1, { 1u, 2u });
        auto e20 = square2D.find_simplex(1, { 2u, 0u }); // stored as (0,2)

        omega.at(e01) = 1.0;
        omega.at(e12) = 2.0;
        omega.at(e20) = 3.0;

        auto domega = omega.d(); // 2‑form on triangle 0

        // Triangle (0,1,2) has orientation (0,1,2). Edge (0,1) matches → +1,
        // (1,2) matches → +1, (2,0) matches orientation (2→0) but stored as (0,2) → reversed → -1.
        double expected = 1.0 + 2.0 - 3.0;
        EXPECT_NEAR(domega.at(0), expected, 1e-12);

        omega.at(e20) = 1.0;
        domega = omega.d();
        expected = 1.0 + 2.0 - 1.0;
        EXPECT_NEAR(domega.at(0), expected, 1e-12);
    }

    TEST_F(DiscreteForms2DTest, HodgeStarInvolutive) {
        // In 2D, ⋆⋆ω = (-1)^{k(2-k)} ω
        auto f = random_0form();
        auto star_f = f.star(*dual, metric);
        auto star_star_f = star_f.star(*dual, metric); // should be +f
        for (std::size_t i = 0; i < f.size(); ++i) {
            EXPECT_NEAR(star_star_f.at(i), f.at(i), 1e-12);
        }

        auto omega = random_1form();
        auto star_omega = omega.star(*dual, metric);
        auto star_star_omega = star_omega.star(*dual, metric); // should be -omega
        for (std::size_t i = 0; i < omega.size(); ++i) {
            EXPECT_NEAR(star_star_omega.at(i), -omega.at(i), 1e-12);
        }

        auto eta = random_2form();
        auto star_eta = eta.star(*dual, metric);
        auto star_star_eta = star_eta.star(*dual, metric); // should be +eta
        for (std::size_t i = 0; i < eta.size(); ++i) {
            EXPECT_NEAR(star_star_eta.at(i), eta.at(i), 1e-12);
        }
    }

    TEST_F(DiscreteForms2DTest, HodgeStarVolumeForm) {
        // ⋆1 gives the volume 2‑form, whose value on each triangle equals its dual volume.
        DiscreteForm<0, double, Complex> one(square2D);
        for (std::size_t i = 0; i < square2D.num_vertices(); ++i) {
            one.at(i) = 1.0;
        }
        auto vol_form = one.star(*dual, metric); // 2‑form

        for (std::size_t t = 0; t < square2D.num_triangles(); ++t) {
            EXPECT_NEAR(vol_form.at(t), dual->dual_volume(2, t), 1e-12);
        }
    }

    TEST_F(DiscreteForms2DTest, CodifferentialDefinition) {
        // δ = (-1)^{n(k-1)+1} ⋆⁻¹ d ⋆  for a 1‑form in 2D: sign = -1
        auto omega = random_1form();

        auto delta_omega = codifferential(omega, *dual, metric);

        auto star_omega = omega.star(*dual, metric);
        auto d_star_omega = star_omega.d();
        auto star_d_star_omega = inverse_star(d_star_omega, *dual, metric);

        for (std::size_t i = 0; i < delta_omega.size(); ++i) {
            EXPECT_NEAR(delta_omega.at(i), -star_d_star_omega.at(i), 1e-12);
        }
    }

    TEST_F(DiscreteForms2DTest, LaplacianSelfAdjoint) {
        auto alpha = random_1form();
        auto beta = random_1form();

        auto lap_alpha = laplacian(alpha, *dual, metric);
        auto lap_beta = laplacian(beta, *dual, metric);

        double left = inner_product(lap_alpha, beta);
        double right = inner_product(alpha, lap_beta);

        EXPECT_NEAR(left, right, 1e-12);
    }

    TEST_F(DiscreteForms2DTest, LaplacianOnZeroForm) {
        // Laplacian of a constant 0‑form is zero
        DiscreteForm<0, double, Complex> constant(square2D);
        for (std::size_t i = 0; i < square2D.num_vertices(); ++i) {
            constant.at(i) = 1.0;
        }
        auto lap_const = laplacian(constant, *dual, metric);
        for (std::size_t i = 0; i < lap_const.size(); ++i) {
            EXPECT_NEAR(lap_const.at(i), 0.0, 1e-12);
        }

        // For a linear function on a square mesh, values are bounded.
        DiscreteForm<0, double, Complex> linear(square2D);
        linear.at(0) = 0.0;
        linear.at(1) = 1.0;
        linear.at(2) = 1.0;
        linear.at(3) = 0.0;
        auto lap_linear = laplacian(linear, *dual, metric);
        for (std::size_t i = 0; i < lap_linear.size(); ++i) {
            EXPECT_LE(std::abs(lap_linear.at(i)), 1.0);
        }
    }

    TEST_F(DiscreteForms2DTest, WedgeProductZeroForms) {
        DiscreteForm<0, double, Complex> f(square2D), g(square2D);
        f.at(0) = 1; f.at(1) = 2; f.at(2) = 3; f.at(3) = 4;
        g.at(0) = 5; g.at(1) = 6; g.at(2) = 7; g.at(3) = 8;

        auto h = wedge(f, g); // 0‑form
        EXPECT_EQ(h.size(), f.size());
        for (std::size_t i = 0; i < h.size(); ++i) {
            EXPECT_NEAR(h.at(i), f.at(i) * g.at(i), 1e-12);
        }
    }

    TEST_F(DiscreteForms2DTest, WedgeProductOneZero) {
        // 1‑form ∧ 0‑form = 1‑form multiplied by the interpolated 0‑form on the edge
        DiscreteForm<1, double, Complex> omega(square2D);
        for (std::size_t e = 0; e < square2D.num_edges(); ++e) {
            omega.at(e) = 1.0;
        }
        DiscreteForm<0, double, Complex> f(square2D);
        f.at(0) = 0; f.at(1) = 1; f.at(2) = 2; f.at(3) = 3;

        auto h = wedge(omega, f); // 1‑form

        auto e01 = square2D.find_simplex(1, { 0u,1u });
        auto e12 = square2D.find_simplex(1, { 1u,2u });
        auto e23 = square2D.find_simplex(1, { 2u,3u });
        auto e30 = square2D.find_simplex(1, { 0u,3u });
        auto e02 = square2D.find_simplex(1, { 0u,2u });

        EXPECT_NEAR(h.at(e01), 0.5, 1e-12);
        EXPECT_NEAR(h.at(e12), 1.5, 1e-12);
        EXPECT_NEAR(h.at(e23), 2.5, 1e-12);
        EXPECT_NEAR(h.at(e30), 1.5, 1e-12);
        EXPECT_NEAR(h.at(e02), 1.0, 1e-12);
    }

    TEST_F(DiscreteForms2DTest, WedgeProductOneOne2D) {
        // On a triangle, wedge of two 1‑forms approximates the area form.
        SimplicialComplex<2, double> tri;
        auto v0 = tri.add_vertex({ 0,0 });
        auto v1 = tri.add_vertex({ 1,0 });
        auto v2 = tri.add_vertex({ 0,1 });
        tri.add_triangle(v0, v1, v2);
        EuclideanMetric metric;
        Dual<decltype(tri), EuclideanMetric> dual_tri(tri, metric);

        DiscreteForm<1, double, decltype(tri)> dx(tri), dy(tri);
        auto e01 = tri.find_simplex(1, { v0, v1 });
        auto e12 = tri.find_simplex(1, { v1, v2 });
        auto e20 = tri.find_simplex(1, { v2, v0 });
        dx.at(e01) = 1.0; dx.at(e12) = -1.0; dx.at(e20) = 0.0;
        dy.at(e01) = 0.0; dy.at(e12) = 1.0; dy.at(e20) = -1.0;

        auto area_form = wedge(dx, dy); // 2‑form
        // Expected area = 0.5
        EXPECT_NEAR(area_form.at(0), 0.5, 1e-12);
    }

    TEST_F(DiscreteForms2DTest, RefineZeroForm) {
        auto f = random_0form();
        auto [subdivided, subdiv_map] = barycentric_subdivide(square2D);
        auto f_refined = f.refine(subdivided, subdiv_map, metric);

        // Original vertex values are preserved
        for (std::size_t i = 0; i < square2D.num_vertices(); ++i) {
            auto p = square2D.vertex(i);
            for (std::size_t j = 0; j < subdivided.num_vertices(); ++j) {
                if (subdivided.vertex(j).isApprox(p, 1e-12)) {
                    EXPECT_NEAR(f_refined.at(j), f.at(i), 1e-12);
                    break;
                }
            }
        }

        // Check interpolation of a linear function at an edge midpoint
        DiscreteForm<0, double, Complex> linear(square2D);
        linear.at(0) = 0.0; // (0,0)
        linear.at(1) = 2.0; // (1,0)
        linear.at(2) = 5.0; // (1,1)
        linear.at(3) = 3.0; // (0,1)
        auto linear_refined = linear.refine(subdivided, subdiv_map, metric);

        for (std::size_t j = 0; j < subdivided.num_vertices(); ++j) {
            auto p = subdivided.vertex(j);
            if (std::abs(p.x() - 0.5) < 1e-12 && std::abs(p.y() - 0.0) < 1e-12) {
                EXPECT_NEAR(linear_refined.at(j), 1.0, 1e-12);
                break;
            }
        }
    }

    TEST_F(DiscreteForms2DTest, RefineOneForm) {
        DiscreteForm<1, double, Complex> omega(square2D);
        for (std::size_t e = 0; e < square2D.num_edges(); ++e) {
            omega.at(e) = 1.0;
        }

        auto [subdivided, subdiv_map] = barycentric_subdivide(square2D);
        auto omega_refined = omega.refine(subdivided, subdiv_map, metric);

        for (const auto& [old_key, new_keys] : subdiv_map) {
            if (old_key.dim != 1) continue;
            std::size_t old_e = old_key.idx;
            auto edge = square2D.edge_at(old_e);
            double original_len = square2D.edge_length(old_e, metric);
            double original_integral = omega.at(old_e) * original_len;

            double sum_integral = 0.0;
            for (const auto& new_key : new_keys) {
                if (new_key.dim != 1) continue;
                std::size_t new_e = new_key.idx;
                double len = subdivided.edge_length(new_e, metric);
                sum_integral += omega_refined.at(new_e) * len;
            }
            EXPECT_NEAR(sum_integral, original_integral, 1e-12);
        }
    }

    TEST_F(DiscreteForms2DTest, RefineTwoForm) {
        DiscreteForm<2, double, Complex> eta(square2D);
        eta.at(0) = 2.0;
        eta.at(1) = 3.0;

        auto [subdivided, subdiv_map] = barycentric_subdivide(square2D);
        auto eta_refined = eta.refine(subdivided, subdiv_map, metric);

        for (const auto& [old_key, new_keys] : subdiv_map) {
            if (old_key.dim != 2) continue;
            std::size_t old_t = old_key.idx;
            double original_area = square2D.cell_volume(old_t, metric);
            double original_integral = eta.at(old_t) * original_area;

            double sum_integral = 0.0;
            for (const auto& new_key : new_keys) {
                if (new_key.dim != 2) continue;
                std::size_t new_t = new_key.idx;
                double area = subdivided.cell_volume(new_t, metric);
                sum_integral += eta_refined.at(new_t) * area;
            }
            EXPECT_NEAR(sum_integral, original_integral, 1e-12);
        }
    }

    // =============================================================================
    // 3D Tests
    // =============================================================================

    TEST_F(DiscreteForms3DTest, CurlGradZero) {
        auto f = random_0form();
        auto df = f.d();               // 1‑form (gradient)
        auto curl_df = curl(df, *dual, metric); // 1‑form
        for (std::size_t e = 0; e < tetrahedron3D.num_edges(); ++e) {
            EXPECT_NEAR(curl_df.at(e), 0.0, 1e-12);
        }
    }

    TEST_F(DiscreteForms3DTest, CurlCurlIdentity) {
        auto omega = random_1form();

        auto curl_omega = curl(omega, *dual, metric);          // 1‑form
        auto curl_curl_omega = curl(curl_omega, *dual, metric); // 1‑form

        auto d_omega = omega.d();                               // 2‑form
        auto delta_omega = codifferential(omega, *dual, metric); // 0‑form
        auto d_delta_omega = delta_omega.d();                   // 1‑form
        auto delta_d_omega = codifferential(d_omega, *dual, metric); // 1‑form

        for (std::size_t e = 0; e < tetrahedron3D.num_edges(); ++e) {
            double lhs = curl_curl_omega.at(e);
            double rhs = d_delta_omega.at(e) + delta_d_omega.at(e);
            EXPECT_NEAR(lhs, rhs, 1e-12);
        }
    }

} // namespace delta::testing