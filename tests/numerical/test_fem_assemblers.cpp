// tests/numerical/test_fem_assemblers.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "delta/numerical/fem_assemblers.h"
#include "delta/numerical/concepts.h"
#include "../test_fixtures.h"

namespace delta::testing {

    // -----------------------------------------------------------------------------
    // Test fixture for FEM assemblers on 2D simplicial complexes
    // -----------------------------------------------------------------------------
    class FEMAssemblersTest : public SimplicialComplexFixture<2, double> {
    protected:
        using Scalar = double;
        delta::geometry::EuclideanMetric metric;

        // Helper to compute total area of the mesh
        Scalar total_area() const {
            Scalar area = 0.0;
            for (std::size_t t = 0; t < square2D.num_triangles(); ++t) {
                area += square2D.cell_volume(t, metric);
            }
            return area;
        }

        // Helper to check if a matrix is symmetric (within tolerance)
        template<typename Matrix>
        bool is_symmetric(const Matrix& M, double tol = 1e-12) const {
            return (M - M.transpose()).norm() < tol;
        }
    };

    // -----------------------------------------------------------------------------
    // Mass matrix tests
    // -----------------------------------------------------------------------------

    TEST_F(FEMAssemblersTest, MassMatrixSize) {
        auto M = delta::numerical::assemble_mass_matrix(square2D, metric);
        EXPECT_EQ(M.rows(), static_cast<int>(square2D.num_vertices()));
        EXPECT_EQ(M.cols(), static_cast<int>(square2D.num_vertices()));
    }

    TEST_F(FEMAssemblersTest, MassMatrixSymmetry) {
        auto M = delta::numerical::assemble_mass_matrix(square2D, metric);
        EXPECT_TRUE(is_symmetric(M));
    }

    TEST_F(FEMAssemblersTest, MassMatrixPositiveDefinite) {
        auto M = delta::numerical::assemble_mass_matrix(square2D, metric);
        // Check that all eigenvalues are positive (crude test: diagonal entries positive)
        for (int i = 0; i < M.rows(); ++i) {
            EXPECT_GT(M.coeff(i, i), 0.0);
        }
    }

    TEST_F(FEMAssemblersTest, MassMatrixConstantFunction) {
        // For a constant function f=1, M * 1 should give vector of dual areas
        auto M = delta::numerical::assemble_mass_matrix(square2D, metric);
        Eigen::VectorXd ones = Eigen::VectorXd::Ones(M.rows());
        Eigen::VectorXd mass_times_one = M * ones;

        // Compute dual areas directly using compute_vertex_dual_areas
        auto dual_areas = delta::numerical::compute_vertex_dual_areas(square2D, metric);
        for (int i = 0; i < M.rows(); ++i) {
            EXPECT_NEAR(mass_times_one(i), dual_areas[i], 1e-12);
        }
    }

    TEST_F(FEMAssemblersTest, MassMatrixTotalSum) {
        // Sum of all entries of M should equal total area
        auto M = delta::numerical::assemble_mass_matrix(square2D, metric);
        Scalar sum = 0.0;
        for (int k = 0; k < M.outerSize(); ++k) {
            for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(M, k); it; ++it) {
                sum += it.value();
            }
        }
        EXPECT_NEAR(sum, total_area(), 1e-12);
    }

    // -----------------------------------------------------------------------------
    // Stiffness matrix (cotangent Laplacian) tests
    // -----------------------------------------------------------------------------

    TEST_F(FEMAssemblersTest, StiffnessMatrixSize) {
        auto K = delta::numerical::assemble_stiffness_matrix(square2D, metric);
        EXPECT_EQ(K.rows(), static_cast<int>(square2D.num_vertices()));
        EXPECT_EQ(K.cols(), static_cast<int>(square2D.num_vertices()));
    }

    TEST_F(FEMAssemblersTest, StiffnessMatrixSymmetry) {
        auto K = delta::numerical::assemble_stiffness_matrix(square2D, metric);
        EXPECT_TRUE(is_symmetric(K));
    }

    TEST_F(FEMAssemblersTest, StiffnessMatrixZeroRowSum) {
        auto K = delta::numerical::assemble_stiffness_matrix(square2D, metric);
        Eigen::VectorXd ones = Eigen::VectorXd::Ones(K.rows());
        Eigen::VectorXd rowsum = K * ones;
        // Row sums should be zero (Neumann boundary, interior nodes also zero because Laplacian of constant is zero)
        for (int i = 0; i < K.rows(); ++i) {
            EXPECT_NEAR(rowsum(i), 0.0, 1e-12);
        }
    }

    TEST_F(FEMAssemblersTest, StiffnessMatrixLinearFunction) {
        // For a linear function f(x,y) = x, the Laplacian is zero. So K * f should be zero (or small).
        Eigen::VectorXd f(K.rows());
        for (std::size_t i = 0; i < square2D.num_vertices(); ++i) {
            f(i) = square2D.vertex(i).x();
        }

        auto K = delta::numerical::assemble_stiffness_matrix(square2D, metric);
        Eigen::VectorXd Kf = K * f;

        double max_val = Kf.cwiseAbs().maxCoeff();
        EXPECT_LE(max_val, 1.0);
        EXPECT_NEAR(Kf.sum(), 0.0, 1e-12);
    }

    TEST_F(FEMAssemblersTest, StiffnessMatrixComparisonWithCotangentLaplacian) {
        // assemble_stiffness_matrix should be the same as build_cotangent_laplacian
        auto K = delta::numerical::assemble_stiffness_matrix(square2D, metric);
        auto L = delta::numerical::build_cotangent_laplacian(square2D, metric);

        EXPECT_EQ(K.rows(), L.rows());
        EXPECT_EQ(K.cols(), L.cols());

        Eigen::MatrixXd K_dense = Eigen::MatrixXd(K);
        Eigen::MatrixXd L_dense = Eigen::MatrixXd(L);
        EXPECT_TRUE(K_dense.isApprox(L_dense, 1e-12));
    }

    // -----------------------------------------------------------------------------
    // Lumped mass matrix tests
    // -----------------------------------------------------------------------------

    TEST_F(FEMAssemblersTest, LumpedMassMatrixSize) {
        auto M = delta::numerical::assemble_mass_matrix(square2D, metric);
        auto lumped = delta::numerical::lumped_mass_matrix(M);
        EXPECT_EQ(lumped.size(), M.rows());
    }

    TEST_F(FEMAssemblersTest, LumpedMassMatrixSumEqualsTotalArea) {
        auto M = delta::numerical::assemble_mass_matrix(square2D, metric);
        auto lumped = delta::numerical::lumped_mass_matrix(M);
        Scalar sum = lumped.sum();
        EXPECT_NEAR(sum, total_area(), 1e-12);
    }

    TEST_F(FEMAssemblersTest, LumpedMassMatrixDiagonalDominance) {
        auto M = delta::numerical::assemble_mass_matrix(square2D, metric);
        auto lumped = delta::numerical::lumped_mass_matrix(M);
        for (int i = 0; i < lumped.size(); ++i) {
            EXPECT_GT(lumped(i), 0.0);
            EXPECT_LE(lumped(i), total_area());
        }
    }

    TEST_F(FEMAssemblersTest, LumpedMassMatrixConsistencyWithRowSum) {
        auto M = delta::numerical::assemble_mass_matrix(square2D, metric);
        auto lumped = delta::numerical::lumped_mass_matrix(M);
        for (int i = 0; i < M.rows(); ++i) {
            double row_sum = 0.0;
            for (int k = 0; k < M.outerSize(); ++k) {
                for (typename Eigen::SparseMatrix<double>::InnerIterator it(M, k); it; ++it) {
                    if (it.row() == i) row_sum += it.value();
                }
            }
            EXPECT_NEAR(lumped(i), row_sum, 1e-12);
        }
    }

} // namespace delta::testing