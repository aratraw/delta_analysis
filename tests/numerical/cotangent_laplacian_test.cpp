// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/numerical/cotangent_laplacian_test.cpp
// ============================================================================
// MATHEMATICAL JUSTIFICATION FOR COTANGENT LAPLACIAN TESTS
// Last updated: 2026-04-30
// ============================================================================
//
// 1. COTANGENT LAPLACIAN: DEFINITION
// ============================================================================
//
// For a 2D simplicial complex (triangulation), the discrete Laplacian at vertex i is:
//   (L u)_i = Σ_{j∈N(i)} w_{ij} (u_i - u_j)
//
// where the weight on edge (i,j):
//   w_{ij} = (cot α_{ij} + cot β_{ij}) / 2
// α_{ij} is the angle in one adjacent triangle opposite edge (i,j),
// β_{ij} is the angle in the other triangle (for boundary edges, β = 0).
//
// The matrix L is assembled as:
//   L_{ii} = Σ_{j∈N(i)} w_{ij}
//   L_{ij} = -w_{ij}   for i≠j
//
// ============================================================================
// 2. MATRIX PROPERTIES (INDEPENDENT OF THE MESH)
// ============================================================================
//
// 2.1. Symmetry: L^T = L
//      Follows from w_{ij} = w_{ji} and symmetric assembly.
//      Tested by Symmetry for any mesh.
//
// 2.2. Row sum: Σ_j L_{ij} = 0
//      Follows from L_{ii} = Σ_{j≠i} w_{ij} and L_{ij} = -w_{ij}.
//      Tested by RowSumZero.
//
// 2.3. Constant function in the kernel: L * 1 = 0
//      Consequence of the row sum property. Tested by ConstantFunctionKernel.
//
// ============================================================================
// 3. ACTION ON FUNCTIONS (DEPENDS ON THE PRESENCE OF INTERIOR VERTICES)
// ============================================================================
//
// Let Ω be the domain of the triangulation. A vertex is called interior if all
// incident triangles lie entirely inside Ω.
//
// 3.1. Linear function u(x,y) = ax + by + c
//      For an INTERIOR vertex: (L u)_i = 0.
//      This is an exact property of the cotangent Laplacian on a closed fan of
//      triangles: for any linear function, the sum of weighted differences with
//      neighbours is zero. Verified by LinearFunctionZeroForInteriorVertex.
//
// 3.2. Quadratic function u(x,y) = x² + y²
//      At an INTERIOR vertex the exact value (L u)_i is NOT the continuous
//      Laplacian Δu = 4. The discrete operator L without mass normalisation
//      yields a value that depends on the local geometry of the mesh.
//      For the specific mesh (a square subdivided into four equal right triangles
//      around a central vertex) the analytic result is:
//        – edge weights from centre to corners are 1,
//        – u_center = 0.5, u_corners = 0,1,2,1,
//        – (L u)_center = Σ 1·(0.5 - u_corner) = -2.
//      This is exactly what QuadraticFunctionConstantLaplacianForInterior checks.
//      Under mesh refinement, L u converges to Δu multiplied by the local dual
//      cell area, and in the limit (M^{-1} L u) → 4.
//
// 3.3. The used mesh
//      make_square_with_center_mesh: square [0,1]×[0,1] with vertices (0,0),
//      (1,0), (1,1), (0,1) and centre (0.5,0.5), 4 triangles. The centre vertex
//      is interior. All cotangents are computed exactly (angles 45° and 90°).
//
// ============================================================================
// 4. WHY TESTS ON THE OLD SQUARE (TWO TRIANGLES) WERE INCORRECT
// ============================================================================
//
// In a square split by one diagonal, ALL 4 vertices lie on the boundary.
// For a boundary vertex:
//   - Linear functions are NOT required to give zero.
//   - The matrix L has diagonal entries 1, off‑diagonals -0.5, not 2 and -1.
//   - The expectations of the original test contradicted the mathematics.
//
// ============================================================================
// 5. LUMPED MASS MATRIX
// ============================================================================
//
// Diagonal mass matrix M with M_{ii} = area of the dual cell of vertex i
// (barycentric dual cell volume). For any non‑degenerate mesh, M_{ii} > 0.
// Tested by LumpedMassMatrixPositive.
//
// ============================================================================
// 6. WHAT IS NOT TESTED (AND WHY)
// ============================================================================
//
// - Exact matrix values for an arbitrary mesh – meaningless, as they depend on
//   geometry. Structural properties are sufficient.
// - Convergence to the continuous Laplacian under refinement – requires a
//   separate test with a sequence of meshes (convergence test).
// - Metric awareness: the current implementation uses the metric via edge_length;
//   correctness for non‑Euclidean metrics is not verified but assumed.
//
// ============================================================================
// 7. CONCLUSION
// ============================================================================
//
// The test suite covers:
//   ✅ Algebraic invariants (symmetry, row sum = 0).
//   ✅ Spectral property (constant in the kernel).
//   ✅ Exact behaviour on interior vertices for linear and quadratic functions,
//      with analytically computed expected values.
//   ✅ Positivity of the mass matrix.
//
// All tests strictly follow the mathematical definition of the cotangent
// Laplacian and do not depend on particular approximate implementations.
//
// ============================================================================

#include <gtest/gtest.h>
#include <Eigen/Sparse>
#include "delta/numerical/cotangent_laplacian.h"
#include "../test_fixtures_geometry_numerical.h"

namespace delta::testing {

    class CotangentLaplacianTest : public GeometryNumericalTest {
    protected:
        using Scalar = Rational;
        using Point2D = Point<2>;
        using Complex2D = Complex<2>;

        // Mesh of a square subdivided into 4 triangles (with a central vertex)
        Complex2D make_square_with_center_mesh() {
            Complex2D mesh;
            auto v0 = add_vertex(mesh, Point2D(0_r, 0_r));
            auto v1 = add_vertex(mesh, Point2D(1_r, 0_r));
            auto v2 = add_vertex(mesh, Point2D(1_r, 1_r));
            auto v3 = add_vertex(mesh, Point2D(0_r, 1_r));
            auto vc = add_vertex(mesh, Point2D(1_r / 2_r, 1_r / 2_r));

            add_edge(mesh, v0, v1); add_edge(mesh, v1, v2);
            add_edge(mesh, v2, v3); add_edge(mesh, v3, v0);
            add_edge(mesh, v0, vc); add_edge(mesh, v1, vc);
            add_edge(mesh, v2, vc); add_edge(mesh, v3, vc);

            add_triangle(mesh, v0, v1, vc);
            add_triangle(mesh, v1, v2, vc);
            add_triangle(mesh, v2, v3, vc);
            add_triangle(mesh, v3, v0, vc);
            return mesh;
        }

        // Old square mesh (only for tests that do not require interior vertices)
        Complex2D make_unit_square_triangulation() {
            Complex2D mesh;
            auto v0 = add_vertex(mesh, Point2D(0_r, 0_r));
            auto v1 = add_vertex(mesh, Point2D(1_r, 0_r));
            auto v2 = add_vertex(mesh, Point2D(1_r, 1_r));
            auto v3 = add_vertex(mesh, Point2D(0_r, 1_r));
            add_edge(mesh, v0, v1);
            add_edge(mesh, v1, v2);
            add_edge(mesh, v2, v3);
            add_edge(mesh, v3, v0);
            add_edge(mesh, v0, v2);
            add_triangle(mesh, v0, v1, v2);
            add_triangle(mesh, v0, v2, v3);
            return mesh;
        }

        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> to_dense(const Eigen::SparseMatrix<Scalar>& sparse) {
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> dense(sparse.rows(), sparse.cols());
            dense.setZero();
            for (int k = 0; k < sparse.outerSize(); ++k) {
                for (Eigen::SparseMatrix<Scalar>::InnerIterator it(sparse, k); it; ++it) {
                    dense(it.row(), it.col()) = it.value();
                }
            }
            return dense;
        }
    };

    /**
     * @test Symmetry
     * @brief Checks that the cotangent Laplacian matrix is symmetric.
     */
    TEST_F(CotangentLaplacianTest, Symmetry) {
        auto mesh = make_unit_square_triangulation();
        EuclideanMetric metric;
        auto L = build_cotangent_laplacian(mesh, metric);
        auto L_dense = to_dense(L);
        std::size_t n = mesh.num_vertices();
        Scalar eps = Rational(1, 1000000);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                EXPECT_RATIONAL_NEAR(L_dense(i, j), L_dense(j, i), eps);
            }
        }
    }
    /**
     * @test RowSumZero
     * @brief Verifies that the sum of entries in each row of L is zero.
     */
    TEST_F(CotangentLaplacianTest, RowSumZero) {
        auto mesh = make_unit_square_triangulation();
        EuclideanMetric metric;
        auto L = build_cotangent_laplacian(mesh, metric);
        auto L_dense = to_dense(L);
        std::size_t n = mesh.num_vertices();
        Scalar eps = Rational(1, 1000000);
        for (std::size_t i = 0; i < n; ++i) {
            Scalar row_sum = 0;
            for (std::size_t j = 0; j < n; ++j) row_sum += L_dense(i, j);
            EXPECT_RATIONAL_NEAR(row_sum, 0_r, eps);
        }
    }
    /**
     * @test ConstantFunctionKernel
     * @brief Checks that L * 1 = 0.
     */
    TEST_F(CotangentLaplacianTest, ConstantFunctionKernel) {
        auto mesh = make_unit_square_triangulation();
        EuclideanMetric metric;
        auto L = build_cotangent_laplacian(mesh, metric);
        std::size_t n = mesh.num_vertices();
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> ones(n);
        ones.setOnes();
        auto L_ones = L * ones;
        Scalar eps = Rational(1, 1000000);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_RATIONAL_NEAR(L_ones(i), 0_r, eps);
        }
    }
    /**
     * @test LinearFunctionZeroForInteriorVertex
     * @brief For the interior vertex of a symmetric square mesh,
     *        (L x)_center = 0 exactly.
     */
    TEST_F(CotangentLaplacianTest, LinearFunctionZeroForInteriorVertex) {
        auto mesh = make_square_with_center_mesh();
        EuclideanMetric metric;
        auto L = build_cotangent_laplacian(mesh, metric);
        std::size_t n = mesh.num_vertices();      // n = 5
        std::size_t interior = 4;                 // index of the centre vertex
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> f(n);
        for (std::size_t i = 0; i < n; ++i) f(i) = mesh.vertex(i).x(); // f(x,y)=x
        auto Lf = L * f;
        Scalar eps = Rational(1, 1000000);
        EXPECT_RATIONAL_NEAR(Lf(interior), 0_r, eps);
    }

    /**
     * @test QuadraticFunctionConstantLaplacianForInterior
     * @brief For the interior vertex of the symmetric square mesh,
     *        (L (x²+y²))_center = -2.
     *
     * This is not the continuous Laplacian (Δ=4); the discrete operator gives
     * a value that depends on the mesh size. The result -2 is analytically
     * derived for this specific mesh (weights from centre to corners are 1,
     * u_center = 0.5, u_corners = 0,1,2,1, sum = -2).
     */
    TEST_F(CotangentLaplacianTest, QuadraticFunctionConstantLaplacianForInterior) {
        auto mesh = make_square_with_center_mesh();
        EuclideanMetric metric;
        auto L = build_cotangent_laplacian(mesh, metric);
        std::size_t n = mesh.num_vertices();
        std::size_t interior = 4;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> f(n);
        for (std::size_t i = 0; i < n; ++i) {
            const auto& p = mesh.vertex(i);
            f(i) = p.x() * p.x() + p.y() * p.y();
        }
        auto Lf = L * f;
        Scalar eps = Rational(1, 1000000);
        EXPECT_RATIONAL_NEAR(Lf(interior), -2_r, eps);
    }
    /**
     * @test LumpedMassMatrixPositive
     * @brief Verifies that all diagonal entries of the lumped mass matrix are positive.
     */
    TEST_F(CotangentLaplacianTest, LumpedMassMatrixPositive) {
        auto mesh = make_unit_square_triangulation();
        auto M = build_lumped_mass_matrix(mesh);
        auto M_dense = to_dense(M);
        for (std::size_t i = 0; i < mesh.num_vertices(); ++i) {
            EXPECT_GT(M_dense(i, i), 0_r);
        }
    }

} // namespace delta::testing