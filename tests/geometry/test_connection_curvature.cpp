// tests/geometry/test_connection_curvature.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "delta/geometry/connection.h"
#include "delta/geometry/covariant_derivative.h"
#include "delta/geometry/curvature.h"
#include "delta/geometry/simplicial_complex.h"
#include "../test_fixtures.h"

namespace delta::testing {

    // -----------------------------------------------------------------------------
    // Test fixture for 2D connection and curvature tests
    // -----------------------------------------------------------------------------
    class Connection2DTest : public SimplicialComplexFixture<2, double> {
    protected:
        using Scalar = double;
        static constexpr int Dim = 2;
        using Conn = Connection<std::size_t, Scalar, Dim>;
        using Matrix = typename Conn::matrix_type;

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
            std::uniform_real_distribution<Scalar> dist(-M_PI, M_PI);
            for (std::size_t e = 0; e < square2D.num_edges(); ++e) {
                auto [v0, v1] = square2D.edge_at(e);
                Scalar angle = dist(rng);
                Matrix rot;
                rot << std::cos(angle), -std::sin(angle),
                    std::sin(angle), std::cos(angle);
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
        TensorField<std::size_t, Scalar, 1, Dim> vector_field(const std::vector<Eigen::Vector2d>& values) {
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
    class Connection3DTest : public SimplicialComplexFixture<3, double> {
    protected:
        using Scalar = double;
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

        // Random rotation in 3D (SO(3)) – not trivial, but we can use a simple
        // rotation around an axis for testing.
        Conn random_connection() {
            Conn conn;
            std::mt19937 rng(54);
            std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
            for (std::size_t e = 0; e < tetrahedron3D.num_edges(); ++e) {
                auto [v0, v1] = tetrahedron3D.edge_at(e);
                // Create a random rotation matrix (simplified: axis-angle with small angle)
                Eigen::AngleAxis<Scalar> aa(dist(rng) * 0.1, Eigen::Vector3d::Random().normalized());
                Matrix rot = aa.toRotationMatrix();
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

        // Take edge (0,1) (bottom edge of square)
        auto edge = square2D.edge_at(0); // (0,1)
        std::size_t v0 = edge.first, v1 = edge.second;

        Eigen::Vector2d vec(1.0, 2.0);
        Eigen::Vector2d transported = conn.parallel_transport(v0, v1, vec);

        // Should be unchanged
        EXPECT_TRUE(transported.isApprox(vec, 1e-12));

        // Reverse direction should also be identity
        transported = conn.parallel_transport(v1, v0, vec);
        EXPECT_TRUE(transported.isApprox(vec, 1e-12));
    }

    TEST_F(Connection2DTest, InverseTransport) {
        auto conn = random_connection();

        auto edge = square2D.edge_at(0);
        std::size_t v0 = edge.first, v1 = edge.second;

        Matrix U = conn.get_transport(v0, v1);
        Matrix Uinv = conn.get_transport(v1, v0);

        // Check that Uinv ≈ U^{-1}
        Matrix product = U * Uinv;
        EXPECT_TRUE(product.isApprox(Matrix::Identity(), 1e-12));

        product = Uinv * U;
        EXPECT_TRUE(product.isApprox(Matrix::Identity(), 1e-12));
    }

    TEST_F(Connection2DTest, HolonomyAroundTriangle) {
        // For a triangle in a flat connection, holonomy should be identity
        auto conn = trivial_connection();

        // Get edges of triangle 0 (first triangle of square: (0,1,2))
        auto tri = square2D.triangle_at(0);
        std::vector<std::pair<std::size_t, std::size_t>> edges = {
            {tri[0], tri[1]},
            {tri[1], tri[2]},
            {tri[2], tri[0]}
        };

        Matrix hol = conn.holonomy(edges);
        EXPECT_TRUE(hol.isApprox(Matrix::Identity(), 1e-12));

        // For a non-trivial connection, holonomy may not be identity, but we can check
        // that it's a rotation matrix (determinant 1, orthogonal)
        auto conn_rand = random_connection();
        hol = conn_rand.holonomy(edges);
        EXPECT_NEAR(hol.determinant(), 1.0, 1e-12);
        EXPECT_TRUE((hol * hol.transpose()).isApprox(Matrix::Identity(), 1e-12));
    }

    TEST_F(Connection2DTest, ConsistentUnderSubdivision) {
        // Create a connection on coarse mesh
        auto conn_coarse = random_connection();

        // Subdivide the mesh
        auto [fine_mesh, subdiv_map] = delta::geometry::barycentric_subdivide(square2D);

        // Create a fine connection that should be consistent: for each coarse edge,
        // the product of fine transports along its subdivision equals coarse transport.
        // We'll construct such a fine connection by setting fine transports to identity
        // and then adjusting? Instead, we can just test the consistency check on a
        // fine connection that we know is consistent (e.g., by pulling back).
        // Simpler: create fine connection as trivial, then check consistency with coarse trivial.
        auto conn_fine = trivial_connection(); // fine trivial

        // Coarse trivial connection
        auto conn_coarse_triv = trivial_connection();

        // They should be consistent (since trivial)
        EXPECT_TRUE(conn_coarse_triv.is_consistent(conn_fine, subdiv_map, square2D, fine_mesh, 1e-12));

        // If we modify one fine edge, consistency should break
        // Find a fine edge that is a descendant of some coarse edge
        bool found = false;
        for (const auto& [old_key, new_keys] : subdiv_map) {
            if (old_key.dim == 1) {
                for (const auto& new_key : new_keys) {
                    if (new_key.dim == 1) {
                        // Modify this fine edge's transport
                        auto [v0, v1] = fine_mesh.edge_at(new_key.idx);
                        conn_fine.set_transport(v0, v1, Matrix::Identity() * 2.0); // not rotation
                        found = true;
                        break;
                    }
                }
            }
            if (found) break;
        }
        ASSERT_TRUE(found);

        EXPECT_FALSE(conn_coarse_triv.is_consistent(conn_fine, subdiv_map, square2D, fine_mesh, 1e-12));
    }

    TEST_F(Connection2DTest, CovariantDerivativeScalar) {
        auto conn = trivial_connection();

        // Define scalar field f at vertices: f(0)=0, f(1)=1, f(2)=2, f(3)=3
        std::vector<Scalar> f_vals = { 0.0, 1.0, 2.0, 3.0 };
        auto f = scalar_field(f_vals);

        // Compute covariant derivative along edge (0,1)
        auto edge01 = square2D.edge_at(0);
        Scalar len = square2D.edge_length(0, metric);
        Scalar df = covariant_derivative(f, conn, edge01, metric);

        // Should equal (f(1)-f(0))/len = (1-0)/1 = 1
        EXPECT_NEAR(df, 1.0, 1e-12);

        // Edge (1,2): (2-1)/len = 1/len, len=1, so 1
        auto edge12 = square2D.edge_at(1);
        len = square2D.edge_length(1, metric);
        df = covariant_derivative(f, conn, edge12, metric);
        EXPECT_NEAR(df, (2.0 - 1.0) / len, 1e-12);
    }

    TEST_F(Connection2DTest, CovariantDerivativeVectorTrivial) {
        auto conn = trivial_connection();

        // Define vector field: v(0)= (1,0), v(1)=(2,1), v(2)=(3,2), v(3)=(4,3)
        std::vector<Eigen::Vector2d> v_vals = {
            {1,0}, {2,1}, {3,2}, {4,3}
        };
        auto v = vector_field(v_vals);

        // Edge (0,1)
        auto edge01 = square2D.edge_at(0);
        Scalar len = square2D.edge_length(0, metric);
        auto dv = covariant_derivative(v, conn, edge01, metric);

        // On trivial connection, should equal (v(1)-v(0))/len = (1,1)/1 = (1,1)
        Eigen::Vector2d expected = (v_vals[1] - v_vals[0]) / len;
        EXPECT_TRUE(dv.isApprox(expected, 1e-12));

        // Edge (1,2)
        auto edge12 = square2D.edge_at(1);
        len = square2D.edge_length(1, metric);
        dv = covariant_derivative(v, conn, edge12, metric);
        expected = (v_vals[2] - v_vals[1]) / len; // (1,1)/1
        EXPECT_TRUE(dv.isApprox(expected, 1e-12));
    }

    TEST_F(Connection2DTest, CovariantDerivativeVectorNonTrivial) {
        auto conn = random_connection();

        // Constant vector field: v = (1,0) everywhere
        std::vector<Eigen::Vector2d> v_vals(4, Eigen::Vector2d(1, 0));
        auto v = vector_field(v_vals);

        // On non-trivial connection, covariant derivative should not be zero.
        // We can't easily compute expected analytically, but we can check that it's not
        // the same as on trivial connection.
        auto edge01 = square2D.edge_at(0);
        auto dv = covariant_derivative(v, conn, edge01, metric);

        // Compute what it would be on trivial connection: zero.
        auto conn_triv = trivial_connection();
        auto dv_triv = covariant_derivative(v, conn_triv, edge01, metric);

        EXPECT_FALSE(dv.isApprox(dv_triv, 1e-12));
        // Also check that dv is not absurdly large (sanity)
        EXPECT_LE(dv.norm(), 10.0);
    }

    // =============================================================================
    // 3D Tests
    // =============================================================================

    TEST_F(Connection3DTest, RicciFlat) {
        auto conn = trivial_connection();

        // On flat tetrahedron, Ricci curvature should be zero at all vertices
        for (std::size_t v = 0; v < tetrahedron3D.num_vertices(); ++v) {
            auto ricci = vertex_ricci_curvature_3d(tetrahedron3D, conn, v, metric);
            EXPECT_TRUE(ricci.isZero(1e-12));
            // Check symmetry
            EXPECT_TRUE(ricci.isApprox(ricci.transpose(), 1e-12));
        }
    }

    TEST_F(Connection3DTest, ScalarCurvatureFlat) {
        auto conn = trivial_connection();

        for (std::size_t v = 0; v < tetrahedron3D.num_vertices(); ++v) {
            Scalar scalar_curve = vertex_scalar_curvature_3d(tetrahedron3D, conn, v, metric);
            EXPECT_NEAR(scalar_curve, 0.0, 1e-12);
        }
    }

    TEST_F(Connection3DTest, RicciNonTrivial) {
        auto conn = random_connection();

        // For non-trivial connection, Ricci should not be zero generally.
        // We'll just check that the computation runs and produces symmetric matrices.
        for (std::size_t v = 0; v < tetrahedron3D.num_vertices(); ++v) {
            auto ricci = vertex_ricci_curvature_3d(tetrahedron3D, conn, v, metric);
            // Check symmetry (should be symmetric by construction)
            EXPECT_TRUE(ricci.isApprox(ricci.transpose(), 1e-12));
            // Not zero (most likely)
            EXPECT_FALSE(ricci.isZero(1e-12));
        }
    }

    TEST_F(Connection3DTest, ScalarCurvatureNonTrivial) {
        auto conn = random_connection();

        for (std::size_t v = 0; v < tetrahedron3D.num_vertices(); ++v) {
            Scalar scalar_curve = vertex_scalar_curvature_3d(tetrahedron3D, conn, v, metric);
            // Just check it's finite
            EXPECT_TRUE(std::isfinite(scalar_curve));
        }
    }

} // namespace delta::testing