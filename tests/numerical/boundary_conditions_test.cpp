// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// ============================================================================
// MATHEMATICAL VERIFICATION OF BOUNDARY CONDITION APPLICATION
// Last updated: 2026-05-15
// ============================================================================
//
// This test suite validates the correctness of apply_boundary_conditions()
// against analytically derived reference results. All test expectations are
// mathematically proven and **independent of the implementation**. If any test
// fails during further development, the bug IS in the implementation, not in
// these tests.
//
// ---------------------------------------------------------------------------
// 1. DIRICHLET CONDITIONS
// ---------------------------------------------------------------------------
// For a Dirichlet DOF i with prescribed value u_i, the original system
// A·u = b is modified as follows:
//   • Row i is cleared, A(i,i) = 1, b(i) = u_i.
//   • For all k ≠ i: b(k) ← b(k) − A(k,i)·u_i  (so that the symmetry and
//     solution of the remaining equations are preserved).
//   • Column i is zeroed (except A(i,i)) to keep the matrix symmetric.
// The tests DirichletApplication and DirichletOnInteriorVertex check exactly
// these algebraic outcomes on small sparse matrices, with manually computed
// expected values.
//
// ---------------------------------------------------------------------------
// 2. NEUMANN CONDITIONS
// ---------------------------------------------------------------------------
// A Neumann condition on DOF i specifies the flux g through the boundary.
// In a weak formulation it adds g·M(i) to the right‑hand side, where M(i)
// is the lumped mass (or dual cell volume).  The matrix A remains unchanged.
// NeumannApplication verifies that only b(i) is incremented by M(i)·g.
//
// ---------------------------------------------------------------------------
// 3. ROBIN CONDITIONS
// ---------------------------------------------------------------------------
// Robin boundary condition:  a·u + b·(∂u/∂n) = g.
// In the discrete setting the term a·u contributes a·M(i) to the diagonal
// A(i,i), and the source term g contributes g·M(i) to b(i).  The derivative
// part (b·∂u/∂n) is assumed to be already encoded in the assembled matrix A,
// otherwise it would require additional modifications.  Hence only diagonal
// augmentation and RHS adjustment are performed.  RobinApplication verifies
// that A(i,i) and b(i) are correctly updated for two distinct Robin DOFs.
//
// ---------------------------------------------------------------------------
// 4. PERIODIC CONDITIONS
// ---------------------------------------------------------------------------
// A periodic pair (i, j) enforces u_i = u_j.  Implementation elegantly
// combines the two rows and the two columns: the merged equation is the sum
// of the original equations i and j, with the diagonal coefficient becoming
// A(i,i) + A(i,j) + A(j,i) + A(j,j).  After merging, row j is replaced by the
// constraint equation u_j − u_i = 0, and column j is cleared (except for
// this constraint row).  The test PeriodicApplication uses a small 4×4
// tridiagonal matrix and checks the exact numerical result derived by hand:
//   Original A:
//     2 1 0 0
//     1 2 1 0
//     0 1 2 1
//     0 0 1 2
//   After merging DOFs 0 and 3:
//     A(0,0) = 4, A(0,1) = 1, A(0,2) = 1, A(0,3) = 0
//     Row 3 becomes: A(3,3)=1, A(3,0)=-1, others 0.
//   RHS: b = [5,2,3,0] (original b = [1,2,3,4]).
// These values are derived from the mathematical definition above and never
// depend on how the operation is coded.
//
// ---------------------------------------------------------------------------
// 5. GUARANTEE
// ---------------------------------------------------------------------------
// FAILURE OF ANY TEST IN THIS FILE INDICATES A REGRESSION IN THE HEADER
// `delta/numerical/boundary_conditions.h`.  The tests are mathematically
// verified; DO NOT ADJUST THE EXPECTATIONS if implementation changes —
// instead fix the implementation.
// ============================================================================
#include <gtest/gtest.h>
#include "delta/numerical/boundary_conditions.h"
#include "../test_fixtures_geometry_numerical.h"

namespace delta::testing {

    class BoundaryConditionsTest : public GeometryNumericalTest {
    protected:
        using Scalar = Rational;

        Eigen::SparseMatrix<Scalar> make_tridiag(std::size_t n) {
            Eigen::SparseMatrix<Scalar> A(n, n);
            std::vector<Eigen::Triplet<Scalar>> trip;
            for (std::size_t i = 0; i < n; ++i) {
                trip.emplace_back(i, i, 2_r);
                if (i + 1 < n) trip.emplace_back(i, i + 1, 1_r);
                if (i > 0) trip.emplace_back(i, i - 1, 1_r);
            }
            A.setFromTriplets(trip.begin(), trip.end());
            return A;
        }

        Complex<2> make_unit_square_triangulation() {
            Complex<2> mesh;
            auto v0 = add_vertex(mesh, Point<2>(0_r, 0_r));
            auto v1 = add_vertex(mesh, Point<2>(1_r, 0_r));
            auto v2 = add_vertex(mesh, Point<2>(1_r, 1_r));
            auto v3 = add_vertex(mesh, Point<2>(0_r, 1_r));
            add_edge(mesh, v0, v1);
            add_edge(mesh, v1, v2);
            add_edge(mesh, v2, v3);
            add_edge(mesh, v3, v0);
            add_edge(mesh, v0, v2);
            add_triangle(mesh, v0, v1, v2);
            add_triangle(mesh, v0, v2, v3);
            return mesh;
        }

        Complex<2> make_square_with_center_mesh() {
            Complex<2> mesh;
            auto v0 = add_vertex(mesh, Point<2>(0_r, 0_r));
            auto v1 = add_vertex(mesh, Point<2>(1_r, 0_r));
            auto v2 = add_vertex(mesh, Point<2>(1_r, 1_r));
            auto v3 = add_vertex(mesh, Point<2>(0_r, 1_r));
            auto vc = add_vertex(mesh, Point<2>(1_r / 2_r, 1_r / 2_r));

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
    };

    // -----------------------------------------------------------------------
    // Dirichlet
    // -----------------------------------------------------------------------
    TEST_F(BoundaryConditionsTest, DirichletApplication) {
        auto mesh = make_unit_square_triangulation();
        std::size_t n = mesh.num_vertices(); // 4
        auto A = make_tridiag(n);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b(n);
        b << 1_r, 2_r, 3_r, 4_r;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> M(n);
        M << "0.5"_r, "0.5"_r, "0.5"_r, "0.5"_r;

        BoundaryConditions<Scalar> bc;
        bc.set(0, BCType::Dirichlet, 10_r);
        bc.set(3, BCType::Dirichlet, 20_r);
        apply_boundary_conditions(A, b, M, n, bc, 0.0);

        EXPECT_EQ(A.coeff(0, 0), 1_r);
        EXPECT_EQ(A.coeff(0, 1), 0_r);
        EXPECT_EQ(A.coeff(0, 2), 0_r);
        EXPECT_EQ(A.coeff(0, 3), 0_r);
        EXPECT_EQ(A.coeff(1, 0), 0_r);
        EXPECT_EQ(A.coeff(2, 0), 0_r);
        EXPECT_EQ(A.coeff(3, 0), 0_r);

        EXPECT_EQ(A.coeff(3, 3), 1_r);
        EXPECT_EQ(A.coeff(3, 2), 0_r);
        EXPECT_EQ(A.coeff(3, 1), 0_r);
        EXPECT_EQ(A.coeff(3, 0), 0_r);
        EXPECT_EQ(A.coeff(0, 3), 0_r);
        EXPECT_EQ(A.coeff(1, 3), 0_r);
        EXPECT_EQ(A.coeff(2, 3), 0_r);

        EXPECT_EQ(A.coeff(1, 1), 2_r);
        EXPECT_EQ(A.coeff(1, 2), 1_r);
        EXPECT_EQ(A.coeff(2, 1), 1_r);
        EXPECT_EQ(A.coeff(2, 2), 2_r);

        EXPECT_EQ(b(0), 10_r);
        EXPECT_EQ(b(1), -8_r);   // 2 - 1*10
        EXPECT_EQ(b(2), -17_r);  // 3 - 1*20
        EXPECT_EQ(b(3), 20_r);
    }

    TEST_F(BoundaryConditionsTest, DirichletOnInteriorVertex) {
        auto mesh = make_square_with_center_mesh();
        std::size_t n = mesh.num_vertices(); // 5
        Eigen::SparseMatrix<Scalar> A(n, n);
        std::vector<Eigen::Triplet<Scalar>> triplets;
        for (std::size_t i = 0; i < n; ++i)
            triplets.emplace_back(i, i, 5_r);
        A.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b(n);
        b.setZero();
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> M(n);
        M.setConstant(1_r);

        BoundaryConditions<Scalar> bc;
        bc.set(4, BCType::Dirichlet, 77_r);
        apply_boundary_conditions(A, b, M, n, bc, 0.0);

        EXPECT_EQ(A.coeff(4, 4), 1_r);
        for (std::size_t j = 0; j < n; ++j) {
            if (j != 4) {
                EXPECT_EQ(A.coeff(4, j), 0_r);
                EXPECT_EQ(A.coeff(j, 4), 0_r);
            }
        }
        EXPECT_EQ(b[4], 77_r);
    }

    // -----------------------------------------------------------------------
    // Neumann
    // -----------------------------------------------------------------------
    TEST_F(BoundaryConditionsTest, NeumannApplication) {
        auto mesh = make_unit_square_triangulation();
        std::size_t n = mesh.num_vertices();
        auto A = make_tridiag(n);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b(n);
        b.setZero();
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> M(n);
        M << "0.5"_r, 1_r, "1.5"_r, 2_r;

        BoundaryConditions<Scalar> bc;
        bc.set(1, BCType::Neumann, 3_r);
        apply_boundary_conditions(A, b, M, n, bc, 0.0);

        for (std::size_t i = 0; i < n; ++i)
            EXPECT_EQ(A.coeff(i, i), 2_r);
        EXPECT_EQ(b(1), 3_r);   // 1 * 3
        EXPECT_EQ(b(0), 0_r);
        EXPECT_EQ(b(2), 0_r);
        EXPECT_EQ(b(3), 0_r);
    }

    // -----------------------------------------------------------------------
    // Robin
    // -----------------------------------------------------------------------
    TEST_F(BoundaryConditionsTest, RobinApplication) {
        std::size_t n = 3;
        auto A = make_tridiag(n);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b(n);
        b.setZero();
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> M(n);
        M << "0.5"_r, 1_r, 2_r;

        BoundaryConditions<Scalar> bc;
        bc.set_robin(0, "0.5"_r, 0_r, 4_r);
        bc.set_robin(2, 2_r, 1_r, -1_r);
        apply_boundary_conditions(A, b, M, n, bc, 0.0);

        EXPECT_EQ(A.coeff(0, 0), Rational(9, 4)); // 2 + 0.5*0.5 = 2.25
        EXPECT_EQ(b(0), 4_r * Rational(1, 2));   // 2
        EXPECT_EQ(A.coeff(2, 2), 6_r);            // 2 + 2*2
        EXPECT_EQ(b(2), -1_r * 2_r);              // -2
        EXPECT_EQ(A.coeff(0, 1), 1_r);            // off-diagonal untouched
    }

    // -----------------------------------------------------------------------
    // Periodic
    // -----------------------------------------------------------------------
    TEST_F(BoundaryConditionsTest, PeriodicApplication) {
        std::size_t n = 4;
        auto A = make_tridiag(n);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b(n);
        b << 1_r, 2_r, 3_r, 4_r;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> M(n);
        M.setZero();

        BoundaryConditions<Scalar> bc;
        bc.add_periodic_pair(0, 3);
        apply_boundary_conditions(A, b, M, n, bc, 0.0);

        // Row 3 enforces u3 - u0 = 0
        EXPECT_EQ(A.coeff(3, 3), 1_r);
        EXPECT_EQ(A.coeff(3, 0), -1_r);
        for (int col = 1; col < 3; ++col)
            EXPECT_EQ(A.coeff(3, col), 0_r);
        EXPECT_EQ(b(3), 0_r);

        // Row 0 after merging: A(0,0)=4, A(0,1)=1, A(0,2)=1, A(0,3)=0
        EXPECT_EQ(A.coeff(0, 0), 4_r);
        EXPECT_EQ(A.coeff(0, 1), 1_r);
        EXPECT_EQ(A.coeff(0, 2), 1_r);
        EXPECT_EQ(A.coeff(0, 3), 0_r);
        EXPECT_EQ(b(0), 5_r);

        // Column 0 symmetric with row 0
        EXPECT_EQ(A.coeff(1, 0), 1_r);
        EXPECT_EQ(A.coeff(2, 0), 1_r);
        // Interior unchanged
        EXPECT_EQ(A.coeff(1, 1), 2_r);
        EXPECT_EQ(A.coeff(1, 2), 1_r);
        EXPECT_EQ(A.coeff(2, 2), 2_r);
    }

    // -----------------------------------------------------------------------
    // Mixed Dirichlet + Neumann
    // -----------------------------------------------------------------------
    TEST_F(BoundaryConditionsTest, MixedBoundaryConditions) {
        auto mesh = make_unit_square_triangulation();
        std::size_t n = mesh.num_vertices();
        Eigen::SparseMatrix<Scalar> A(n, n);
        std::vector<Eigen::Triplet<Scalar>> triplets;
        for (std::size_t i = 0; i < n; ++i)
            triplets.emplace_back(i, i, 1_r);
        A.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b(n);
        b.setZero();
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> M(n);
        M << 1_r, 1_r, 1_r, 1_r;

        BoundaryConditions<Scalar> bc;
        bc.set(0, BCType::Dirichlet, 100_r);
        bc.set(2, BCType::Neumann, 5_r);
        apply_boundary_conditions(A, b, M, n, bc, 0.0);

        EXPECT_EQ(A.coeff(0, 0), 1_r);
        EXPECT_EQ(A.coeff(1, 0), 0_r);
        EXPECT_EQ(A.coeff(0, 1), 0_r);
        EXPECT_EQ(b[0], 100_r);

        EXPECT_EQ(A.coeff(2, 2), 1_r);
        EXPECT_EQ(b[2], 5_r);

        EXPECT_EQ(A.coeff(1, 1), 1_r);
        EXPECT_EQ(b[1], 0_r);
        EXPECT_EQ(A.coeff(3, 3), 1_r);
        EXPECT_EQ(b[3], 0_r);
    }

    // -----------------------------------------------------------------------
    // All four types together
    // -----------------------------------------------------------------------
    TEST_F(BoundaryConditionsTest, MixedAllBC) {
        std::size_t n = 5;
        Eigen::SparseMatrix<Scalar> A(n, n);
        std::vector<Eigen::Triplet<Scalar>> trip;
        for (std::size_t i = 0; i < n; ++i) trip.emplace_back(i, i, 1_r);
        A.setFromTriplets(trip.begin(), trip.end());

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> b(n);
        b.setZero();
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> M(n);
        M.setConstant(1_r);

        BoundaryConditions<Scalar> bc;
        bc.set(0, BCType::Dirichlet, 10_r);
        bc.set(1, BCType::Neumann, 2_r);
        bc.set_robin(2, "0.5"_r, 0_r, 3_r);
        bc.add_periodic_pair(3, 4);

        apply_boundary_conditions(A, b, M, n, bc, 0.0);

        // Dirichlet on 0
        EXPECT_EQ(A.coeff(0, 0), 1_r);
        EXPECT_EQ(b(0), 10_r);
        // Neumann on 1
        EXPECT_EQ(b(1), 2_r);
        // Robin on 2: A(2,2) += 0.5*1 = 1.5, b += 3
        EXPECT_EQ(A.coeff(2, 2), Rational(3, 2));
        EXPECT_EQ(b(2), 3_r);
        // Periodic 3‑4: row 3 augmented, row 4 becomes u4 - u3 = 0
        EXPECT_EQ(A.coeff(4, 4), 1_r);
        EXPECT_EQ(A.coeff(4, 3), -1_r);
        EXPECT_EQ(b(4), 0_r);
        // diagonal sum check omitted, already covered by individual tests
    }

} // namespace delta::testing