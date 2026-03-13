// tests/numerical/test_gauge_theory.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include "delta/numerical/gauge_groups.h"
#include "delta/numerical/gauge_theory.h"
#include "delta/geometry/simplicial_complex.h"
#include "../test_fixtures.h"

namespace delta::testing {


    // -----------------------------------------------------------------------------
    // Test fixture for gauge theory tests using a 2D simplicial complex (square)
    // -----------------------------------------------------------------------------
    class GaugeTheoryTest : public SimplicialComplexFixture<2, double> {
    protected:
        using Complex = SimplicialComplex<2, double>;
        EuclideanMetric metric; // not directly used but required for some methods

        void SetUp() override {
            SimplicialComplexFixture::SetUp();
            // Use square2D as the mesh
        }

        // Helper to create a GaugeField for U(1) with all links = identity
        GaugeField<U1, Complex> identity_u1_field() {
            GaugeField<U1, Complex> gf(square2D);
            // All links already initialized to identity by default
            return gf;
        }

        // Helper to create a GaugeField for SU(2) with all links = identity
        GaugeField<SU2, Complex> identity_su2_field() {
            GaugeField<SU2, Complex> gf(square2D);
            return gf;
        }

        // Helper to create a GaugeField for SU(3) with all links = identity
        GaugeField<SU3, Complex> identity_su3_field() {
            GaugeField<SU3, Complex> gf(square2D);
            return gf;
        }
    };

    // =============================================================================
    // U(1) tests
    // =============================================================================

    TEST_F(GaugeTheoryTest, U1_GroupOperations) {
        U1 a(0.5); // phase exp(i*0.5)
        U1 b(1.2); // phase exp(i*1.2)

        U1 prod = a * b;
        double expected_phase = 0.5 + 1.2;
        EXPECT_NEAR(prod.log(), expected_phase, 1e-12);

        U1 inv = a.inverse();
        EXPECT_NEAR((a * inv).log(), 0.0, 1e-12);

        U1 id = U1::identity();
        EXPECT_NEAR(id.log(), 0.0, 1e-12);

        // Test exp and log consistency
        double theta = 0.7;
        U1 g = U1::exp(theta);
        EXPECT_NEAR(g.log(), theta, 1e-12);
    }

    TEST_F(GaugeTheoryTest, U1_WilsonActionZero) {
        auto gf = identity_u1_field();
        double S = gf.wilson_action(1.0);
        EXPECT_NEAR(S, 0.0, 1e-12);
    }

    TEST_F(GaugeTheoryTest, U1_GaugeInvariance) {
        auto gf = identity_u1_field();
        // Randomize slightly
        gf.randomize(0.1);

        double S_before = gf.wilson_action(1.0);

        // Create random gauge transformation at vertices
        std::vector<U1> gauge_factors(square2D.num_vertices());
        for (auto& g : gauge_factors) {
            g = U1::exp(0.5); // fixed angle for reproducibility
        }

        gf.gauge_transform(gauge_factors);
        double S_after = gf.wilson_action(1.0);

        EXPECT_NEAR(S_before, S_after, 1e-12);
    }

    TEST_F(GaugeTheoryTest, U1_FieldStrength) {
        auto gf = identity_u1_field();
        // Set a non-trivial configuration: all links = exp(i*0.1)
        for (std::size_t e = 0; e < gf.size(); ++e) {
            gf.link(e) = U1::exp(0.1);
        }

        // Compute field strength on a face (should be product around triangle)
        for (std::size_t t = 0; t < square2D.num_triangles(); ++t) {
            auto F = gf.field_strength_algebra(t);
            // For U(1), field strength is a complex number (imaginary). Should be small.
            // Actually for constant links, the plaquette angle = 3*0.1 = 0.3, so (U - U†)/2 = i sin(0.3) ≈ i*0.2955.
            EXPECT_NEAR(std::abs(F), std::sin(0.3), 1e-12);
        }
    }

    TEST_F(GaugeTheoryTest, U1_VariationNumerical) {
        auto gf = identity_u1_field();
        gf.randomize(0.1);

        // Pick an edge index (e.g., first edge)
        std::size_t e_idx = 0;

        // Compute analytical variation
        double dS_analytical = gf.variation(e_idx, 1.0);

        // Numerical variation: perturb the link by a small epsilon, compute action difference
        double eps = 1e-6;
        U1 original = gf.link(e_idx);

        // Positive perturbation
        gf.link(e_idx) = U1::exp(original.log() + eps);
        double S_plus = gf.wilson_action(1.0);

        // Negative perturbation
        gf.link(e_idx) = U1::exp(original.log() - eps);
        double S_minus = gf.wilson_action(1.0);

        // Restore
        gf.link(e_idx) = original;

        double dS_numerical = (S_plus - S_minus) / (2.0 * eps);

        EXPECT_NEAR(dS_analytical, dS_numerical, 1e-6);
    }

    // =============================================================================
    // SU(2) tests
    // =============================================================================

    TEST_F(GaugeTheoryTest, SU2_GroupOperations) {
        // Test multiplication and inverse
        SU2 a = SU2::exp(0.3, Eigen::Vector3d(1, 0, 0));
        SU2 b = SU2::exp(0.4, Eigen::Vector3d(0, 1, 0));
        SU2 prod = a * b;
        SU2 inv = a.inverse();
        SU2 prod_inv = a * inv;
        EXPECT_TRUE(prod_inv.matrix().isApprox(SU2::identity().matrix(), 1e-12));

        // Test trace and determinant
        EXPECT_NEAR(a.matrix().determinant(), 1.0, 1e-12);
        EXPECT_NEAR(a.trace(), 2.0 * std::cos(0.15), 1e-12); // for rotation about x by 0.3, trace = 2 cos(0.15)? Wait exp(i theta sigma_x/2) has trace 2 cos(theta/2). So for theta=0.3, trace = 2 cos(0.15) ≈ 1.9775.
        double expected_trace = 2.0 * std::cos(0.15);
        EXPECT_NEAR(a.trace(), expected_trace, 1e-12);

        // Test log and exp consistency
        Eigen::Vector3d log_a = a.log();
        SU2 a_from_log = SU2::exp(log_a.norm(), log_a.normalized());
        EXPECT_TRUE(a_from_log.matrix().isApprox(a.matrix(), 1e-10));
    }

    TEST_F(GaugeTheoryTest, SU2_WilsonActionZero) {
        auto gf = identity_su2_field();
        double S = gf.wilson_action(1.0);
        EXPECT_NEAR(S, 0.0, 1e-12);
    }

    TEST_F(GaugeTheoryTest, SU2_GaugeInvariance) {
        auto gf = identity_su2_field();
        gf.randomize(0.1);

        double S_before = gf.wilson_action(1.0);

        // Random gauge transformation (fixed for reproducibility)
        std::vector<SU2> gauge_factors(square2D.num_vertices());
        for (auto& g : gauge_factors) {
            g = SU2::exp(0.2, Eigen::Vector3d(1, 1, 1).normalized());
        }

        gf.gauge_transform(gauge_factors);
        double S_after = gf.wilson_action(1.0);

        EXPECT_NEAR(S_before, S_after, 1e-12);
    }

    TEST_F(GaugeTheoryTest, SU2_VariationLieAlgebra) {
        auto gf = identity_su2_field();
        gf.randomize(0.1);

        std::size_t e_idx = 0;
        auto variation = gf.variation(e_idx, 1.0);

        // Check that variation is anti-Hermitian and traceless (element of Lie algebra su(2))
        Eigen::Matrix2cd var_mat = variation;
        EXPECT_TRUE((var_mat + var_mat.adjoint()).isZero(1e-12)); // anti-Hermitian
        EXPECT_NEAR(var_mat.trace().real(), 0.0, 1e-12);
        EXPECT_NEAR(var_mat.trace().imag(), 0.0, 1e-12);
    }

    // =============================================================================
    // SU(3) tests
    // =============================================================================

    TEST_F(GaugeTheoryTest, SU3_GroupOperations) {
        // Create a simple SU(3) matrix via exponential of an algebra element
        Eigen::Matrix3cd alg = Eigen::Matrix3cd::Zero();
        alg(0, 1) = std::complex<double>(0.1, 0.2);
        alg(1, 0) = -std::conj(alg(0, 1));
        alg(1, 2) = std::complex<double>(0.3, 0.4);
        alg(2, 1) = -std::conj(alg(1, 2));
        alg(2, 0) = std::complex<double>(0.5, 0.6);
        alg(0, 2) = -std::conj(alg(2, 0));
        // Make traceless
        alg(0, 0) = std::complex<double>(0.0, 0.1);
        alg(1, 1) = std::complex<double>(0.0, 0.2);
        alg(2, 2) = -alg(0, 0) - alg(1, 1);

        SU3 a = SU3::exp(alg);
        SU3 b = SU3::exp(alg * 0.5);
        SU3 prod = a * b;
        SU3 inv = a.inverse();
        SU3 prod_inv = a * inv;

        EXPECT_TRUE(prod_inv.matrix().isApprox(SU3::identity().matrix(), 1e-10));
        EXPECT_NEAR(a.matrix().determinant(), 1.0, 1e-10);

        // Test log and exp consistency (approximate)
        Eigen::Matrix3cd log_a = a.log();
        SU3 a_from_log = SU3::exp(log_a);
        EXPECT_TRUE(a_from_log.matrix().isApprox(a.matrix(), 1e-8));
    }

    TEST_F(GaugeTheoryTest, SU3_WilsonActionZero) {
        auto gf = identity_su3_field();
        double S = gf.wilson_action(1.0);
        EXPECT_NEAR(S, 0.0, 1e-12);
    }

    TEST_F(GaugeTheoryTest, SU3_GaugeInvariance) {
        auto gf = identity_su3_field();
        gf.randomize(0.1);

        double S_before = gf.wilson_action(1.0);

        // Random gauge transformation (simplified: use small random algebra elements)
        std::vector<SU3> gauge_factors(square2D.num_vertices());
        for (auto& g : gauge_factors) {
            Eigen::Matrix3cd alg = Eigen::Matrix3cd::Zero();
            alg(0, 1) = std::complex<double>(0.05, 0.0);
            alg(1, 0) = -std::conj(alg(0, 1));
            alg(1, 2) = std::complex<double>(0.0, 0.05);
            alg(2, 1) = -std::conj(alg(1, 2));
            alg(2, 0) = std::complex<double>(0.05, 0.05);
            alg(0, 2) = -std::conj(alg(2, 0));
            alg(0, 0) = std::complex<double>(0.0, 0.05);
            alg(1, 1) = std::complex<double>(0.0, 0.05);
            alg(2, 2) = -alg(0, 0) - alg(1, 1);
            g = SU3::exp(alg);
        }

        gf.gauge_transform(gauge_factors);
        double S_after = gf.wilson_action(1.0);

        EXPECT_NEAR(S_before, S_after, 1e-12);
    }

    TEST_F(GaugeTheoryTest, SU3_VariationLieAlgebra) {
        auto gf = identity_su3_field();
        gf.randomize(0.1);

        std::size_t e_idx = 0;
        auto variation = gf.variation(e_idx, 1.0);

        // Check that variation is anti-Hermitian and traceless (element of Lie algebra su(3))
        Eigen::Matrix3cd var_mat = variation;
        EXPECT_TRUE((var_mat + var_mat.adjoint()).isZero(1e-12)); // anti-Hermitian
        EXPECT_NEAR(var_mat.trace().real(), 0.0, 1e-12);
        EXPECT_NEAR(var_mat.trace().imag(), 0.0, 1e-12);
    }

} // namespace delta::testing