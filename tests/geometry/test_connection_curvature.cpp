#define _USE_MATH_DEFINES
#include <gtest/gtest.h>
#include <cmath>
#include <numbers>
#include "delta/geometry/connection.h"
#include "delta/geometry/covariant_derivative.h"
#include "delta/geometry/curvature.h"
#include "delta/geometry/simplicial_complex.h"
#include "../test_fixtures.h"

namespace delta::testing {

    // -----------------------------------------------------------------------------
    // Test fixture for 2D connection and curvature tests
    // -----------------------------------------------------------------------------
    class Connection2DTest : public SimplicialComplexFixture<2, Rational> {
    protected:
        using Scalar = Rational;
        static constexpr int Dim = 2;
        using Conn = Connection<std::size_t, Scalar, Dim>;
        using Matrix = typename Conn::matrix_type;

        // Используем метрику, которая работает с Rational
        EuclideanMetric metric;

        // Create a trivial connection (all matrices identity)
        Conn trivial_connection() {
            Conn conn;
            for (std::size_t e = 0; e < square2D.num_edges(); ++e) {
                auto [v0, v1] = square2D.edge_at(e);
                conn.set_transport(v0, v1, Matrix::Identity());
            }
            return conn;
        }

        // Create a non-trivial connection with random rotations (SO(2))
        Conn random_connection() {
            Conn conn;
            std::mt19937 rng(53);
            // Генерируем double из диапазона [-π, π], затем преобразуем в Rational
            std::uniform_real_distribution<double> dist(-M_PI, M_PI);
            for (std::size_t e = 0; e < square2D.num_edges(); ++e) {
                auto [v0, v1] = square2D.edge_at(e);
                double angle_d = dist(rng);
                Scalar angle(angle_d);  // преобразуем в Rational
                Scalar cos_a = delta::cos(angle);
                Scalar sin_a = delta::sin(angle);
                Matrix rot;
                rot << cos_a, -sin_a,
                    sin_a, cos_a;
                conn.set_transport(v0, v1, rot);
            }
            return conn;
        }

        // Helper to create a scalar field (0-form) on square2D
        TensorField<std::size_t, Scalar, 0, Dim> scalar_field(const std::vector<Scalar>& values) {
            TensorField<std::size_t, Scalar, 0, Dim> field;
            for (std::size_t i = 0; i < square2D.num_vertices(); ++i) {
                field.set(i, values[i]);
            }
            return field;
        }

        // Helper to create a vector field on square2D
        TensorField<std::size_t, Scalar, 1, Dim> vector_field(const std::vector<Eigen::Matrix<Scalar, 2, 1>>& values) {
            TensorField<std::size_t, Scalar, 1, Dim> field;
            for (std::size_t i = 0; i < square2D.num_vertices(); ++i) {
                field.set(i, values[i]);
            }
            return field;
        }
    };

    // -----------------------------------------------------------------------------
    // Test fixture for 3D connection and curvature tests
    // -----------------------------------------------------------------------------
    class Connection3DTest : public SimplicialComplexFixture<3, Rational> {
    protected:
        using Scalar = Rational;
        static constexpr int Dim = 3;
        using Conn = Connection<std::size_t, Scalar, Dim>;
        using Matrix = typename Conn::matrix_type;

        EuclideanMetric metric;

        // Trivial connection for tetrahedron
        Conn trivial_connection() {
            Conn conn;
            for (std::size_t e = 0; e < tetrahedron3D.num_edges(); ++e) {
                auto [v0, v1] = tetrahedron3D.edge_at(e);
                conn.set_transport(v0, v1, Matrix::Identity());
            }
            return conn;
        }

        // Random rotation in 3D (SO(3))
        Conn random_connection() {
            Conn conn;
            std::mt19937 rng(54);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (std::size_t e = 0; e < tetrahedron3D.num_edges(); ++e) {
                auto [v0, v1] = tetrahedron3D.edge_at(e);
                // Create a random rotation matrix
                double theta_d = dist(rng) * 0.1;
                Scalar theta(theta_d);
                Eigen::Matrix<double, 3, 1> axis_d = Eigen::Vector3d::Random().normalized();
                Eigen::Matrix<Scalar, 3, 1> axis(axis_d(0), axis_d(1), axis_d(2));
                // Используем формулу Родрига для поворота
                Scalar cos_t = delta::cos(theta);
                Scalar sin_t = delta::sin(theta);
                Scalar one_minus_cos = 1_r - cos_t;

                Matrix rot;
                rot(0, 0) = cos_t + axis(0) * axis(0) * one_minus_cos;
                rot(0, 1) = axis(0) * axis(1) * one_minus_cos - axis(2) * sin_t;
                rot(0, 2) = axis(0) * axis(2) * one_minus_cos + axis(1) * sin_t;
                rot(1, 0) = axis(1) * axis(0) * one_minus_cos + axis(2) * sin_t;
                rot(1, 1) = cos_t + axis(1) * axis(1) * one_minus_cos;
                rot(1, 2) = axis(1) * axis(2) * one_minus_cos - axis(0) * sin_t;
                rot(2, 0) = axis(2) * axis(0) * one_minus_cos - axis(1) * sin_t;
                rot(2, 1) = axis(2) * axis(1) * one_minus_cos + axis(0) * sin_t;
                rot(2, 2) = cos_t + axis(2) * axis(2) * one_minus_cos;

                conn.set_transport(v0, v1, rot);
            }
            return conn;
        }
    };

    // =============================================================================
    // 2D Tests
    // =============================================================================

    TEST_F(Connection2DTest, TrivialConnectionParallelTransport) {
        auto conn = trivial_connection();

        auto edge = square2D.edge_at(0);
        std::size_t v0 = edge[0], v1 = edge[1];

        Eigen::Matrix<Scalar, 2, 1> vec(1_r, 2_r);
        Eigen::Matrix<Scalar, 2, 1> transported = conn.parallel_transport(v0, v1, vec);

        EXPECT_TRUE(transported.isApprox(vec, Scalar(1e-12)));

        transported = conn.parallel_transport(v1, v0, vec);
        EXPECT_TRUE(transported.isApprox(vec, Scalar(1e-12)));
    }

    TEST_F(Connection2DTest, InverseTransport) {
        auto conn = random_connection();

        auto edge = square2D.edge_at(0);
        std::size_t v0 = edge[0], v1 = edge[1];

        Matrix U = conn.get_transport(v0, v1);
        Matrix Uinv = conn.get_transport(v1, v0);

        Matrix product = U * Uinv;
        EXPECT_TRUE(product.isApprox(Matrix::Identity(), Scalar(1e-12)));

        product = Uinv * U;
        EXPECT_TRUE(product.isApprox(Matrix::Identity(), Scalar(1e-12)));
    }

    TEST_F(Connection2DTest, HolonomyAroundTriangle) {
        auto conn = trivial_connection();

        auto tri = square2D.triangle_at(0);
        std::vector<std::pair<std::size_t, std::size_t>> edges = {
            {tri[0], tri[1]},
            {tri[1], tri[2]},
            {tri[2], tri[0]}
        };

        Matrix hol = conn.holonomy(edges);
        EXPECT_TRUE(hol.isApprox(Matrix::Identity(), Scalar(1e-12)));

        auto conn_rand = random_connection();
        hol = conn_rand.holonomy(edges);
        // Проверяем, что определитель близок к 1 (с учётом Rational)
        Scalar det = hol.determinant();
        EXPECT_TRUE(delta::testing::DeltaTest::near(det, 1_r, Scalar(1e-12)));
        EXPECT_TRUE((hol * hol.transpose()).isApprox(Matrix::Identity(), Scalar(1e-12)));
    }

    TEST_F(Connection2DTest, ConsistentUnderSubdivision) {
        auto conn_coarse = random_connection();

        auto [fine_mesh, subdiv_map] = delta::geometry::barycentric_subdivide(square2D);

        auto conn_fine = trivial_connection();
        auto conn_coarse_triv = trivial_connection();

        EXPECT_TRUE(conn_coarse_triv.is_consistent(conn_fine, subdiv_map, square2D, fine_mesh, Scalar(1e-12)));

        bool found = false;
        for (const auto& [old_key, new_keys] : subdiv_map) {
            if (old_key.dim == 1) {
                for (const auto& new_key : new_keys) {
                    if (new_key.dim == 1) {
                        auto [v0, v1] = fine_mesh.edge_at(new_key.idx);
                        conn_fine.set_transport(v0, v1, Matrix::Identity() * 2_r);
                        found = true;
                        break;
                    }
                }
            }
            if (found) break;
        }
        ASSERT_TRUE(found);

        EXPECT_FALSE(conn_coarse_triv.is_consistent(conn_fine, subdiv_map, square2D, fine_mesh, Scalar(1e-12)));
    }

    TEST_F(Connection2DTest, CovariantDerivativeScalar) {
        auto conn = trivial_connection();

        std::vector<Scalar> f_vals = { 0_r, 1_r, 2_r, 3_r };
        auto f = scalar_field(f_vals);

        auto edge01 = square2D.edge_at(0);
        // Исправлено: добавлен mesh (square2D) перед metric
        Scalar df = covariant_derivative(f, conn, edge01, square2D, metric);

        EXPECT_NEAR(df.template convert_to<double>(), 1.0, 1e-12);

        auto edge12 = square2D.edge_at(1);
        df = covariant_derivative(f, conn, edge12, square2D, metric);
        Scalar len = square2D.edge_length(1, metric); // для проверки
        Scalar expected = (f_vals[2] - f_vals[1]) / len;
        EXPECT_TRUE(df == expected);
    }

    TEST_F(Connection2DTest, CovariantDerivativeVectorTrivial) {
        auto conn = trivial_connection();

        std::vector<Eigen::Matrix<Scalar, 2, 1>> v_vals = {
            {1_r, 0_r}, {2_r, 1_r}, {3_r, 2_r}, {4_r, 3_r}
        };
        auto v = vector_field(v_vals);

        auto edge01 = square2D.edge_at(0);
        auto dv = covariant_derivative(v, conn, edge01, square2D, metric);

        Scalar len = square2D.edge_length(0, metric);
        Eigen::Matrix<Scalar, 2, 1> expected = (v_vals[1] - v_vals[0]) / len;
        EXPECT_TRUE(dv.isApprox(expected, Scalar(1e-12)));

        auto edge12 = square2D.edge_at(1);
        dv = covariant_derivative(v, conn, edge12, square2D, metric);
        len = square2D.edge_length(1, metric);
        expected = (v_vals[2] - v_vals[1]) / len;
        EXPECT_TRUE(dv.isApprox(expected, Scalar(1e-12)));
    }

    TEST_F(Connection2DTest, CovariantDerivativeVectorNonTrivial) {
        auto conn = random_connection();

        std::vector<Eigen::Matrix<Scalar, 2, 1>> v_vals(4, Eigen::Matrix<Scalar, 2, 1>(1_r, 0_r));
        auto v = vector_field(v_vals);

        auto edge01 = square2D.edge_at(0);
        auto dv = covariant_derivative(v, conn, edge01, square2D, metric);

        auto conn_triv = trivial_connection();
        auto dv_triv = covariant_derivative(v, conn_triv, edge01, square2D, metric);

        EXPECT_FALSE(dv.isApprox(dv_triv, Scalar(1e-12)));
        EXPECT_LE(dv.norm().template convert_to<double>(), 10.0);
    }

    // =============================================================================
    // 3D Tests
    // =============================================================================

    TEST_F(Connection3DTest, RicciFlat) {
        auto conn = trivial_connection();

        for (std::size_t v = 0; v < tetrahedron3D.num_vertices(); ++v) {
            auto ricci = vertex_ricci_curvature_3d(tetrahedron3D, conn, v, metric);
            EXPECT_TRUE(ricci.isZero(Scalar(1e-12)));
            EXPECT_TRUE(ricci.isApprox(ricci.transpose(), Scalar(1e-12)));
        }
    }

    TEST_F(Connection3DTest, ScalarCurvatureFlat) {
        auto conn = trivial_connection();

        for (std::size_t v = 0; v < tetrahedron3D.num_vertices(); ++v) {
            Scalar scalar_curve = vertex_scalar_curvature_3d(tetrahedron3D, conn, v, metric);
            EXPECT_NEAR(scalar_curve.template convert_to<double>(), 0.0, 1e-12);
        }
    }

    TEST_F(Connection3DTest, RicciNonTrivial) {
        auto conn = random_connection();

        for (std::size_t v = 0; v < tetrahedron3D.num_vertices(); ++v) {
            auto ricci = vertex_ricci_curvature_3d(tetrahedron3D, conn, v, metric);
            EXPECT_TRUE(ricci.isApprox(ricci.transpose(), Scalar(1e-12)));
            EXPECT_FALSE(ricci.isZero(Scalar(1e-12)));
        }
    }

    TEST_F(Connection3DTest, ScalarCurvatureNonTrivial) {
        auto conn = random_connection();

        for (std::size_t v = 0; v < tetrahedron3D.num_vertices(); ++v) {
            Scalar scalar_curve = vertex_scalar_curvature_3d(tetrahedron3D, conn, v, metric);
            EXPECT_TRUE(std::isfinite(scalar_curve.template convert_to<double>()));
        }
    }

} // namespace delta::testing