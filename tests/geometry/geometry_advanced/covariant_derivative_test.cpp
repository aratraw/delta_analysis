// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/geometry/covariant_derivative_test.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "delta/core/rational.h"
#include "delta/geometry/simplicial_complex.h"
#include "delta/geometry/connection.h"
#include "delta/geometry/tensor_field.h"
#include "delta/geometry/covariant_derivative.h"
#include "../../test_fixtures_geometry_numerical.h"

namespace delta::testing {

    class CovariantDerivativeTest : public GeometryNumericalTest {
    protected:
        using Scalar = Rational;
        using Complex2D = delta::geometry::SimplicialComplex<2, Scalar>;
        using Point2D = typename Complex2D::point_type;
        using Connection2D = delta::geometry::Connection<
            typename Complex2D::vertex_index, Scalar, 2>;
        using Matrix2 = Eigen::Matrix<Scalar, 2, 2>;
        using Vector2 = Eigen::Matrix<Scalar, 2, 1>;

        Complex2D make_unit_square_triangulation() {
            Complex2D mesh;
            auto v0 = mesh.add_vertex(Point2D(0_r, 0_r));
            auto v1 = mesh.add_vertex(Point2D(1_r, 0_r));
            auto v2 = mesh.add_vertex(Point2D(1_r, 1_r));
            auto v3 = mesh.add_vertex(Point2D(0_r, 1_r));
            mesh.add_edge(v0, v1);
            mesh.add_edge(v1, v2);
            mesh.add_edge(v2, v3);
            mesh.add_edge(v3, v0);
            mesh.add_edge(v0, v2);
            mesh.add_triangle(v0, v1, v2);
            mesh.add_triangle(v0, v2, v3);
            return mesh;
        }

        Matrix2 rot() const {
            Matrix2 R;
            R(0, 0) = 0_r; R(0, 1) = -1_r;
            R(1, 0) = 1_r; R(1, 1) = 0_r;
            return R;
        }

        void set_rotation(Connection2D& conn, const Complex2D& mesh) {
            Matrix2 R = rot();
            for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
                auto [v0, v1] = mesh.edge_at(e);
                conn.set_transport(v0, v1, R);
            }
        }
    };

    // =======================================================================
    // Scalar field on a flat mesh, trivial connection
    // =======================================================================
    TEST_F(CovariantDerivativeTest, ScalarField) {
        auto mesh = make_unit_square_triangulation();
        EuclideanMetric metric;

        // Scalar field f(x,y) = x + 2y
        TensorField<decltype(mesh)::vertex_index, Scalar, 0, 2> field;
        for (std::size_t i = 0; i < mesh.num_vertices(); ++i)
            field.set(i, mesh.vertex(i).x() + 2_r * mesh.vertex(i).y());

        // Edge from (0,0) to (1,0)
        Scalar deriv = covariant_derivative(field, Connection2D{},
            std::array<decltype(mesh)::vertex_index, 2>{0, 1},
            mesh, metric);
        EXPECT_EQ(deriv, 1_r);

        // Edge from (0,0) to (0,1)
        deriv = covariant_derivative(field, Connection2D{},
            std::array<decltype(mesh)::vertex_index, 2>{0, 3},
            mesh, metric);
        EXPECT_EQ(deriv, 2_r);
    }

    // =======================================================================
    // Vector field with constant rotation connection
    // =======================================================================
    TEST_F(CovariantDerivativeTest, VectorFieldWithRotation) {
        auto mesh = make_unit_square_triangulation();
        EuclideanMetric metric;

        Connection2D conn;
        set_rotation(conn, mesh);

        // Constant vector field v ≡ (1, 0)
        TensorField<decltype(mesh)::vertex_index, Scalar, 1, 2> field;
        for (std::size_t i = 0; i < mesh.num_vertices(); ++i)
            field.set(i, Vector2(1_r, 0_r));

        // Edge (0,0) → (1,0)
        Vector2 deriv = covariant_derivative(field, conn,
            std::array<decltype(mesh)::vertex_index, 2>{0, 1},
            mesh, metric);
        EXPECT_EQ(deriv(0), -1_r);
        EXPECT_EQ(deriv(1), -1_r);
    }

    // =======================================================================
    // Matrix field with constant rotation connection
    // =======================================================================
    TEST_F(CovariantDerivativeTest, MatrixFieldWithRotation) {
        auto mesh = make_unit_square_triangulation();
        EuclideanMetric metric;

        Connection2D conn;
        set_rotation(conn, mesh);

        // Constant matrix field M = diag(2, 3)
        TensorField<decltype(mesh)::vertex_index, Scalar, 2, 2> field;
        Matrix2 M;
        M(0, 0) = 2_r; M(0, 1) = 0_r;
        M(1, 0) = 0_r; M(1, 1) = 3_r;
        for (std::size_t i = 0; i < mesh.num_vertices(); ++i)
            field.set(i, M);

        Matrix2 deriv = covariant_derivative(field, conn,
            std::array<decltype(mesh)::vertex_index, 2>{0, 1},
            mesh, metric);
        EXPECT_EQ(deriv(0, 0), 1_r);
        EXPECT_EQ(deriv(0, 1), 0_r);
        EXPECT_EQ(deriv(1, 0), 0_r);
        EXPECT_EQ(deriv(1, 1), -1_r);
    }

} // namespace delta::testing