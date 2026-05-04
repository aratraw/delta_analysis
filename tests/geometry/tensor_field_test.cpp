// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/geometry/tensor_field_test.cpp
// ============================================================================
// TESTS FOR TENSOR FIELD (RANKS 0, 1, 2) AND TENSOR OPERATIONS
// ============================================================================
//
// This file tests the TensorField class and its operations on sparse sets of
// addresses (points). Verified features:
//   - Construction and access for scalar, vector, and matrix fields.
//   - Algebraic operations: addition, scalar multiplication.
//   - Tensor operations: tensor product, trace, symmetrisation, antisymmetrisation.
//   - Metric‑dependent operations: raising and lowering indices.
//
// All tests use 2‑dimensional points and a simple two‑point grid.
// ============================================================================

#include <gtest/gtest.h>
#include <vector>
#include <optional>
#include "delta/geometry/tensor_field.h"
#include "../test_fixtures_geometry_numerical.h"

namespace delta::testing {

    /**
     * @class TensorFieldTest
     * @brief Tests for TensorField class (ranks 0,1,2) and associated operations.
     */
    class TensorFieldTest : public GeometryNumericalTest {
    protected:
        static constexpr int DIM = 2;
        using Addr = Point<DIM>;
        using Compare = PointLess<DIM>;          // defined in base fixture
        using Grid = delta::ListGrid<Addr, Compare>;

        // Tensor types
        using Scalar0 = Scalar;                    // rank 0
        using Vector1 = Eigen::Matrix<Scalar, DIM, 1>;   // rank 1
        using Matrix2 = Eigen::Matrix<Scalar, DIM, DIM>; // rank 2

        // Tensor field with correct comparator
        template<int Rank>
        using Field = delta::geometry::TensorField<Addr, Scalar, Rank, DIM, Compare>;

        // Helper: create a simple grid with two points (0,0) and (1,0)
        Grid make_test_grid() {
            std::vector<Addr> points;
            points.push_back(make_point<DIM>(0_r, 0_r));
            points.push_back(make_point<DIM>(1_r, 0_r));
            return Grid(std::move(points), Compare());
        }

        // Check that two tensors of the same rank are close (within tolerance)
        template<int Rank>
        bool tensors_near(const typename Field<Rank>::value_type& a,
            const typename Field<Rank>::value_type& b,
            const Scalar& eps = delta::default_eps()) {
            if constexpr (Rank == 0) {
                return delta::abs(a - b) <= eps;
            }
            else {
                return matrix_near(a, b, eps);   // from base fixture
            }
        }
    };

    // =========================================================================
    // 1. Basic creation and access
    // =========================================================================

    /**
     * @test CreateAndAccess
     * @brief Verifies that fields can be constructed, values set and retrieved.
     */
    TEST_F(TensorFieldTest, CreateAndAccess) {
        Grid grid = make_test_grid();
        Field<0> scalar_field(grid);   // scalar field, initialised to zeros

        // Check that the field contains all grid points
        EXPECT_TRUE(scalar_field.contains(grid[0]));
        EXPECT_TRUE(scalar_field.contains(grid[1]));

        // Set values
        scalar_field.set(grid[0], 3_r);
        scalar_field.set(grid[1], 7_r);

        // Check access
        EXPECT_EQ(scalar_field.at(grid[0]), 3_r);
        EXPECT_EQ(scalar_field.at(grid[1]), 7_r);

        // Non‑existent point should not be contained
        Addr other = make_point<DIM>(2_r, 2_r);
        EXPECT_FALSE(scalar_field.contains(other));

        // Similarly for vector field (rank 1)
        Field<1> vector_field(grid);
        Vector1 v0; v0 << 1_r, 2_r;
        Vector1 v1; v1 << 3_r, 4_r;
        vector_field.set(grid[0], v0);
        vector_field.set(grid[1], v1);
        EXPECT_TRUE(tensors_near<1>(vector_field.at(grid[0]), v0));
        EXPECT_TRUE(tensors_near<1>(vector_field.at(grid[1]), v1));

        // For matrix field (rank 2)
        Field<2> matrix_field(grid);
        Matrix2 m0 = Matrix2::Identity() * 2_r;
        Matrix2 m1; m1 << 1_r, 2_r, 3_r, 4_r;
        matrix_field.set(grid[0], m0);
        matrix_field.set(grid[1], m1);
        EXPECT_TRUE(tensors_near<2>(matrix_field.at(grid[0]), m0));
        EXPECT_TRUE(tensors_near<2>(matrix_field.at(grid[1]), m1));
    }

    // =========================================================================
    // 2. Algebraic operations
    // =========================================================================

    /**
     * @test Addition
     * @brief Pointwise addition of two fields of the same rank.
     */
    TEST_F(TensorFieldTest, Addition) {
        Grid grid = make_test_grid();
        Field<2> A(grid), B(grid), C(grid);

        Matrix2 m1; m1 << 1_r, 2_r, 3_r, 4_r;
        Matrix2 m2; m2 << 5_r, 6_r, 7_r, 8_r;
        A.set(grid[0], m1);
        A.set(grid[1], m2);
        B.set(grid[0], m2);
        B.set(grid[1], m1);

        C = A + B;   // operator+ must be defined

        Matrix2 sum0 = m1 + m2;
        Matrix2 sum1 = m2 + m1;
        EXPECT_TRUE(tensors_near<2>(C.at(grid[0]), sum0));
        EXPECT_TRUE(tensors_near<2>(C.at(grid[1]), sum1));

        // Addition with itself
        C = A + A;
        EXPECT_TRUE(tensors_near<2>(C.at(grid[0]), m1 + m1));
    }

    /**
     * @test ScalarMultiplication
     * @brief Pointwise multiplication of a field by a scalar (left and right).
     */
    TEST_F(TensorFieldTest, ScalarMultiplication) {
        Grid grid = make_test_grid();
        Field<1> V(grid);
        Vector1 v; v << 2_r, 3_r;
        V.set(grid[0], v);
        V.set(grid[1], v);

        Field<1> W = 4_r * V;   // left scalar multiplication

        Vector1 expected = 4_r * v;
        EXPECT_TRUE(tensors_near<1>(W.at(grid[0]), expected));
        EXPECT_TRUE(tensors_near<1>(W.at(grid[1]), expected));

        // Right scalar multiplication
        Field<1> Z = V * 4_r;
        EXPECT_TRUE(tensors_near<1>(Z.at(grid[0]), expected));
    }

    /**
     * @test TensorProduct
     * @brief Outer product of two vector fields produces a matrix field.
     */
    TEST_F(TensorFieldTest, TensorProduct) {
        Grid grid = make_test_grid();
        Field<1> U(grid), V(grid);

        Vector1 u; u << 1_r, 2_r;
        Vector1 v; v << 3_r, 4_r;
        U.set(grid[0], u);
        V.set(grid[0], v);
        U.set(grid[1], u);
        V.set(grid[1], v);

        // Tensor product U ⊗ V gives a matrix field
        Field<2> T = tensor_product(U, V);   // free function

        Matrix2 expected = u * v.transpose();
        EXPECT_TRUE(tensors_near<2>(T.at(grid[0]), expected));
        EXPECT_TRUE(tensors_near<2>(T.at(grid[1]), expected));
    }

    /**
     * @test Trace
     * @brief Trace of a matrix field (rank 2 → scalar field).
     */
    TEST_F(TensorFieldTest, Trace) {
        Grid grid = make_test_grid();
        Field<2> M(grid);

        Matrix2 m; m << 1_r, 2_r, 3_r, 4_r;
        M.set(grid[0], m);
        M.set(grid[1], m * 2_r);

        Field<0> tr = trace(M);   // trace (scalar field)

        EXPECT_EQ(tr.at(grid[0]), 1_r + 4_r);   // 5
        EXPECT_EQ(tr.at(grid[1]), (1_r + 4_r) * 2_r);
    }

    /**
     * @test Symmetrize
     * @brief Symmetrisation (M + Mᵀ)/2.
     */
    TEST_F(TensorFieldTest, Symmetrize) {
        Grid grid = make_test_grid();
        Field<2> M(grid);

        Matrix2 m; m << 1_r, 2_r, 3_r, 4_r;
        M.set(grid[0], m);

        Field<2> sym = symmetrize(M);
        Matrix2 expected = (m + m.transpose()) / 2_r;
        EXPECT_TRUE(tensors_near<2>(sym.at(grid[0]), expected));
    }

    /**
     * @test Antisymmetrize
     * @brief Anti‑symmetrisation (M - Mᵀ)/2.
     */
    TEST_F(TensorFieldTest, Antisymmetrize) {
        Grid grid = make_test_grid();
        Field<2> M(grid);

        Matrix2 m; m << 1_r, 2_r, 3_r, 4_r;
        M.set(grid[0], m);

        Field<2> asym = antisymmetrize(M);
        Matrix2 expected = (m - m.transpose()) / 2_r;
        EXPECT_TRUE(tensors_near<2>(asym.at(grid[0]), expected));
    }

    // =========================================================================
    // 3. Raising and lowering indices (requires a metric field)
    // =========================================================================

    /**
     * @test RaiseLowerIndices
     * @brief Lower index with metric, then raise with inverse metric.
     */
    TEST_F(TensorFieldTest, RaiseLowerIndices) {
        Grid grid = make_test_grid();
        // Define a metric field g (diagonal, e.g., Euclidean)
        Field<2> g(grid);
        Matrix2 g_val = Matrix2::Identity();   // Euclidean metric
        g.set(grid[0], g_val);
        g.set(grid[1], g_val);

        // Inverse metric
        Field<2> g_inv(grid);
        g_inv.set(grid[0], g_val.inverse());
        g_inv.set(grid[1], g_val.inverse());

        // Vector field v
        Field<1> v(grid);
        Vector1 v_val; v_val << 2_r, 3_r;
        v.set(grid[0], v_val);
        v.set(grid[1], v_val);

        // Lower index: v_flat_i = g_ij * v^j
        Field<1> v_flat = lower_index(v, g);
        Vector1 expected_flat = g_val * v_val;
        EXPECT_TRUE(tensors_near<1>(v_flat.at(grid[0]), expected_flat));

        // Raise index: v_sharp^i = g^{ij} * v_flat_j
        Field<1> v_sharp = raise_index(v_flat, g_inv);
        EXPECT_TRUE(tensors_near<1>(v_sharp.at(grid[0]), v_val));
    }

} // namespace delta::testing