// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/geometry/geometry_advanced/connection_test.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "delta/core/rational.h"
#include "delta/geometry/simplicial_complex.h"
#include "delta/geometry/connection.h"
#include "../../test_fixtures_geometry_numerical.h"

namespace delta::testing {

    class ConnectionTest : public GeometryNumericalTest {
    protected:
        using Scalar = Rational;
        using Complex2D = delta::geometry::SimplicialComplex<2, Scalar>;
        using Point2D = typename Complex2D::point_type;
        using Connection2D = delta::geometry::Connection<
            typename Complex2D::vertex_index,
            Scalar,
            2>;

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

        using Matrix2 = Eigen::Matrix<Scalar, 2, 2>;
        using Vector2 = Eigen::Matrix<Scalar, 2, 1>;
    };

    TEST_F(ConnectionTest, TrivialConnectionPreservesVector) {
        auto mesh = make_unit_square_triangulation();
        Connection2D conn;
        Matrix2 I = Matrix2::Identity();

        for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
            auto [v0, v1] = mesh.edge_at(e);
            conn.set_transport(v0, v1, I);
        }

        Vector2 vec(1_r, 2_r);
        auto transported = conn.parallel_transport(0, 1, vec);
        EXPECT_EQ(transported[0], 1_r);
        EXPECT_EQ(transported[1], 2_r);

        auto back = conn.parallel_transport(1, 0, transported);
        EXPECT_EQ(back[0], 1_r);
        EXPECT_EQ(back[1], 2_r);
    }

    TEST_F(ConnectionTest, ConstantRotationTransport) {
        auto mesh = make_unit_square_triangulation();
        Matrix2 rot;
        rot(0, 0) = 0_r; rot(0, 1) = -1_r;
        rot(1, 0) = 1_r; rot(1, 1) = 0_r;

        Connection2D conn;
        for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
            auto [v0, v1] = mesh.edge_at(e);
            conn.set_transport(v0, v1, rot);
        }

        Vector2 vec(1_r, 0_r);
        auto transported = conn.parallel_transport(0, 1, vec);
        EXPECT_EQ(transported[0], 0_r);
        EXPECT_EQ(transported[1], 1_r);

        auto transported2 = conn.parallel_transport(1, 2, transported);
        EXPECT_EQ(transported2[0], -1_r);
        EXPECT_EQ(transported2[1], 0_r);
    }

    TEST_F(ConnectionTest, HolonomyAroundTriangle) {
        auto mesh = make_unit_square_triangulation();
        Matrix2 rot;
        rot(0, 0) = 0_r; rot(0, 1) = -1_r;
        rot(1, 0) = 1_r; rot(1, 1) = 0_r;

        Connection2D conn;
        // Устанавливаем матрицы для всех рёбер в соответствии с ориентацией,
        // которая будет использоваться в пути [0,1,2,0].
        // Рёбра (0,1) и (1,2) ориентированы как в пути.
        conn.set_transport(0, 1, rot);
        conn.set_transport(1, 2, rot);
        // Для ребра (0,2) нужно, чтобы transport из 2 в 0 был rot.
        // Поэтому устанавливаем rot для ориентации (2→0), тогда (0→2) станет rot^{-1}.
        conn.set_transport(2, 0, rot);

        // При правильном порядке умножения H = rot * rot * rot = rot^3.
        Matrix2 expected;
        expected(0, 0) = 0_r; expected(0, 1) = 1_r;
        expected(1, 0) = -1_r; expected(1, 1) = 0_r;

        Matrix2 hol = conn.holonomy({ 0, 1, 2, 0 });
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                EXPECT_EQ(hol(i, j), expected(i, j));
    }


    TEST_F(ConnectionTest, SubdivisionConsistency) {
        Complex2D mesh;
        auto v0 = mesh.add_vertex(Point2D(0_r, 0_r));
        auto v1 = mesh.add_vertex(Point2D(1_r, 0_r));
        mesh.add_edge(v0, v1);   // ребро с индексом 0

        Matrix2 U;
        U(0, 0) = 4_r; U(0, 1) = 0_r;
        U(1, 0) = 0_r; U(1, 1) = Rational(1, 4);
        Matrix2 U_half;
        U_half(0, 0) = 2_r; U_half(0, 1) = 0_r;
        U_half(1, 0) = 0_r; U_half(1, 1) = Rational(1, 2);

        Connection2D coarse_conn;
        coarse_conn.set_transport(v0, v1, U);

        auto [fine_mesh, subdiv_map] = mesh.barycentric_subdivide();
        // Дочерние рёбра: (v0, mid) и (v1, mid)
        const auto& children = subdiv_map.at(SimplexKey{ 1, 0 });
        ASSERT_EQ(children.size(), 2);

        Connection2D fine_conn;
        for (const auto& key : children) {
            auto [a, b] = fine_mesh.edge_at(key.index);
            // В fine_mesh ребро хранится как (min,max). Для (v0,mid) это (0,2) — ориентация v0->mid.
            // Для (v1,mid) это (1,2) — ориентация v1->mid.
            // Нам нужно, чтобы при движении по пути v0->mid->v1 матрицы были U_half.
            if (a == v0 && b > v1) { // (0,2) → v0->mid
                fine_conn.set_transport(a, b, U_half);
            }
            else if (a == v1 && b > v1) { // (1,2) → v1->mid
                // Чтобы transport из mid в v1 был U_half, устанавливаем для (v1,mid) обратную матрицу.
                fine_conn.set_transport(a, b, U_half.inverse());
            }
        }

        bool consistent = coarse_conn.is_consistent(
            fine_conn, subdiv_map, mesh, fine_mesh, Scalar(1, 1000000));
        EXPECT_TRUE(consistent);
    }
} // namespace delta::testing