// tests/geometry/test_tensor_field_advanced.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include "delta/geometry/tensor_field.h"
#include "delta/geometry/matrix_field.h"
#include "delta/geometry/simplicial_complex.h"   // instead of barycentric_subdivision.h
#include "../test_fixtures.h"

namespace delta::testing {

    using namespace delta::geometry;

    // -----------------------------------------------------------------------------
    // Test fixture for advanced tensor field tests (2D)
    // -----------------------------------------------------------------------------
    class TensorFieldAdvancedTest : public SimplicialComplexFixture<2, double> {
    protected:
        using Point = Eigen::Vector2d;
        using Scalar = double;
        static constexpr int Dim = 2;

        EuclideanMetric metric;

        // Helper to create a random scalar field on square2D
        TensorField<Point, Scalar, 0, Dim> random_scalar_field() {
            TensorField<Point, Scalar, 0, Dim> field;
            std::mt19937 rng(50);
            std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
            for (std::size_t i = 0; i < square2D.num_vertices(); ++i) {
                field.set(square2D.vertex(i), dist(rng));
            }
            return field;
        }

        // Helper to create a random vector field on square2D
        TensorField<Point, Scalar, 1, Dim> random_vector_field() {
            TensorField<Point, Scalar, 1, Dim> field;
            std::mt19937 rng(51);
            std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
            for (std::size_t i = 0; i < square2D.num_vertices(); ++i) {
                Eigen::Vector2d v(dist(rng), dist(rng));
                field.set(square2D.vertex(i), v);
            }
            return field;
        }

        // Helper to create a random matrix field on square2D
        TensorField<Point, Scalar, 2, Dim> random_matrix_field() {
            TensorField<Point, Scalar, 2, Dim> field;
            std::mt19937 rng(52);
            std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);
            for (std::size_t i = 0; i < square2D.num_vertices(); ++i) {
                Eigen::Matrix2d m;
                m << dist(rng), dist(rng), dist(rng), dist(rng);
                field.set(square2D.vertex(i), m);
            }
            return field;
        }

        // Helper to create a metric tensor field (Euclidean metric = identity everywhere)
        TensorField<Point, Scalar, 2, Dim> euclidean_metric_field() {
            TensorField<Point, Scalar, 2, Dim> g;
            Eigen::Matrix2d identity = Eigen::Matrix2d::Identity();
            for (std::size_t i = 0; i < square2D.num_vertices(); ++i) {
                g.set(square2D.vertex(i), identity);
            }
            return g;
        }

        // Helper to create an inverse metric (also identity for Euclidean)
        TensorField<Point, Scalar, 2, Dim> euclidean_inverse_metric_field() {
            return euclidean_metric_field();
        }

        // Helper to check if two fields are approximately equal (within tolerance)
        template<int Rank>
        void expect_fields_near(const TensorField<Point, Scalar, Rank, Dim>& a,
            const TensorField<Point, Scalar, Rank, Dim>& b,
            double tol = 1e-12) {
            EXPECT_EQ(a.size(), b.size());
            for (const auto& [p, val_a] : a) {
                ASSERT_TRUE(b.contains(p));
                const auto& val_b = b.at(p);
                if constexpr (Rank == 0) {
                    EXPECT_NEAR(val_a, val_b, tol);
                }
                else if constexpr (Rank == 1) {
                    EXPECT_TRUE(val_a.isApprox(val_b, tol));
                }
                else if constexpr (Rank == 2) {
                    EXPECT_TRUE(val_a.isApprox(val_b, tol));
                }
            }
        }
    };

    // -----------------------------------------------------------------------------
    // Algebraic operations
    // -----------------------------------------------------------------------------

    TEST_F(TensorFieldAdvancedTest, ScalarFieldArithmetic) {
        auto f = random_scalar_field();
        auto g = random_scalar_field();

        auto sum = f + g;
        auto diff = f - g;
        auto scaled = f * 2.5;

        for (const auto& [p, val_f] : f) {
            const auto& val_g = g.at(p);
            EXPECT_NEAR(sum.at(p), val_f + val_g, 1e-12);
            EXPECT_NEAR(diff.at(p), val_f - val_g, 1e-12);
            EXPECT_NEAR(scaled.at(p), val_f * 2.5, 1e-12);
        }
    }

    TEST_F(TensorFieldAdvancedTest, VectorFieldArithmetic) {
        auto v = random_vector_field();
        auto w = random_vector_field();

        auto sum = v + w;
        auto diff = v - w;
        auto scaled = v * 2.5;

        for (const auto& [p, val_v] : v) {
            const auto& val_w = w.at(p);
            EXPECT_TRUE(sum.at(p).isApprox(val_v + val_w, 1e-12));
            EXPECT_TRUE(diff.at(p).isApprox(val_v - val_w, 1e-12));
            EXPECT_TRUE(scaled.at(p).isApprox(val_v * 2.5, 1e-12));
        }
    }

    TEST_F(TensorFieldAdvancedTest, MatrixFieldArithmetic) {
        auto m = random_matrix_field();
        auto n = random_matrix_field();

        auto sum = m + n;
        auto diff = m - n;
        auto scaled = m * 2.5;

        for (const auto& [p, val_m] : m) {
            const auto& val_n = n.at(p);
            EXPECT_TRUE(sum.at(p).isApprox(val_m + val_n, 1e-12));
            EXPECT_TRUE(diff.at(p).isApprox(val_m - val_n, 1e-12));
            EXPECT_TRUE(scaled.at(p).isApprox(val_m * 2.5, 1e-12));
        }
    }

    TEST_F(TensorFieldAdvancedTest, TensorProduct) {
        auto f = random_scalar_field();
        auto g = random_scalar_field();
        auto v = random_vector_field();
        auto w = random_vector_field();
        auto m = random_matrix_field();

        auto tp_00 = tensor_product(f, g);
        for (const auto& [p, val_f] : f) {
            EXPECT_NEAR(tp_00.at(p), val_f * g.at(p), 1e-12);
        }

        auto tp_01 = tensor_product(f, v);
        for (const auto& [p, val_f] : f) {
            EXPECT_TRUE(tp_01.at(p).isApprox(val_f * v.at(p), 1e-12));
        }

        auto tp_10 = tensor_product(v, f);
        for (const auto& [p, val_f] : f) {
            EXPECT_TRUE(tp_10.at(p).isApprox(v.at(p) * val_f, 1e-12));
        }

        auto tp_11 = tensor_product(v, w);
        for (const auto& [p, val_v] : v) {
            Eigen::Matrix2d expected = val_v * w.at(p).transpose();
            EXPECT_TRUE(tp_11.at(p).isApprox(expected, 1e-12));
        }

        auto tp_02 = tensor_product(f, m);
        for (const auto& [p, val_f] : f) {
            EXPECT_TRUE(tp_02.at(p).isApprox(val_f * m.at(p), 1e-12));
        }

        auto tp_20 = tensor_product(m, f);
        for (const auto& [p, val_f] : f) {
            EXPECT_TRUE(tp_20.at(p).isApprox(m.at(p) * val_f, 1e-12));
        }
    }

    // -----------------------------------------------------------------------------
    // MatrixField specific operations
    // -----------------------------------------------------------------------------

    TEST_F(TensorFieldAdvancedTest, MatrixFieldMultiplication) {
        auto m = random_matrix_field();
        auto n = random_matrix_field();

        MatrixField<Point, Scalar, Dim> mf = as_matrix_field(m);
        MatrixField<Point, Scalar, Dim> nf = as_matrix_field(n);

        auto prod = mf * nf;

        for (const auto& [p, val_m] : m) {
            EXPECT_TRUE(prod.at(p).isApprox(val_m * n.at(p), 1e-12));
        }
    }

    TEST_F(TensorFieldAdvancedTest, MatrixFieldTranspose) {
        auto m = random_matrix_field();
        MatrixField<Point, Scalar, Dim> mf = as_matrix_field(m);
        auto mt = mf.transpose();

        for (const auto& [p, val_m] : m) {
            EXPECT_TRUE(mt.at(p).isApprox(val_m.transpose(), 1e-12));
        }
    }

    TEST_F(TensorFieldAdvancedTest, MatrixFieldTrace) {
        auto m = random_matrix_field();
        MatrixField<Point, Scalar, Dim> mf = as_matrix_field(m);
        auto tr = mf.trace();

        for (const auto& [p, val_m] : m) {
            EXPECT_NEAR(tr.at(p), val_m.trace(), 1e-12);
        }
    }

    TEST_F(TensorFieldAdvancedTest, MatrixFieldDeterminant) {
        auto m = random_matrix_field();
        MatrixField<Point, Scalar, Dim> mf = as_matrix_field(m);
        auto det = mf.determinant();

        for (const auto& [p, val_m] : m) {
            EXPECT_NEAR(det.at(p), val_m.determinant(), 1e-12);
        }
    }

    TEST_F(TensorFieldAdvancedTest, MatrixFieldCommutator) {
        auto m = random_matrix_field();
        auto n = random_matrix_field();
        MatrixField<Point, Scalar, Dim> mf = as_matrix_field(m);
        MatrixField<Point, Scalar, Dim> nf = as_matrix_field(n);

        auto comm = mf.comm(nf);

        for (const auto& [p, val_m] : m) {
            Eigen::Matrix2d expected = val_m * n.at(p) - n.at(p) * val_m;
            EXPECT_TRUE(comm.at(p).isApprox(expected, 1e-12));
        }
    }

    // -----------------------------------------------------------------------------
    // Index raising and lowering with metric
    // -----------------------------------------------------------------------------

    TEST_F(TensorFieldAdvancedTest, RaiseLowerIndex) {
        auto g = euclidean_metric_field();
        auto g_inv = euclidean_inverse_metric_field();
        auto v = random_vector_field();
        auto m = random_matrix_field();

        auto v_lower = lower_index(v, g);
        expect_fields_near(v_lower, v);

        auto v_raise = raise_index(v, g_inv);
        expect_fields_near(v_raise, v);

        auto m_raised = raise_second_index(m, g_inv);
        expect_fields_near(m_raised, m);

        auto m_lowered = lower_first_index(m, g);
        expect_fields_near(m_lowered, m);
    }

    // -----------------------------------------------------------------------------
    // Refinement interpolation
    // -----------------------------------------------------------------------------

    TEST_F(TensorFieldAdvancedTest, RefineVectorField) {
        TensorField<Point, Scalar, 1, Dim> v;
        for (std::size_t i = 0; i < square2D.num_vertices(); ++i) {
            auto p = square2D.vertex(i);
            v.set(p, Eigen::Vector2d(2.0 * p.x(), 3.0 * p.y()));
        }

        auto [subdivided, subdiv_map] = barycentric_subdivide(square2D);
        auto v_refined = v.refine(square2D, subdivided, subdiv_map, metric);

        for (std::size_t i = 0; i < square2D.num_vertices(); ++i) {
            auto p = square2D.vertex(i);
            for (std::size_t j = 0; j < subdivided.num_vertices(); ++j) {
                if (subdivided.vertex(j).isApprox(p, 1e-12)) {
                    EXPECT_TRUE(v_refined.at(subdivided.vertex(j)).isApprox(v.at(p), 1e-12));
                    break;
                }
            }
        }

        bool found_mid = false;
        for (std::size_t j = 0; j < subdivided.num_vertices(); ++j) {
            auto p = subdivided.vertex(j);
            if (std::abs(p.x() - 0.5) < 1e-12 && std::abs(p.y() - 0.0) < 1e-12) {
                Eigen::Vector2d expected(1.0, 0.0);
                EXPECT_TRUE(v_refined.at(p).isApprox(expected, 1e-12));
                found_mid = true;
                break;
            }
        }
        EXPECT_TRUE(found_mid);

        bool found_center = false;
        for (std::size_t j = 0; j < subdivided.num_vertices(); ++j) {
            auto p = subdivided.vertex(j);
            if (std::abs(p.x() - 1.0 / 3.0) < 1e-12 && std::abs(p.y() - 1.0 / 3.0) < 1e-12) {
                Eigen::Vector2d expected(2.0 / 3.0, 1.0);
                EXPECT_TRUE(v_refined.at(p).isApprox(expected, 1e-12));
                found_center = true;
                break;
            }
        }
        EXPECT_TRUE(found_center);
    }

    TEST_F(TensorFieldAdvancedTest, RefineMatrixField) {
        TensorField<Point, Scalar, 2, Dim> m;
        for (std::size_t i = 0; i < square2D.num_vertices(); ++i) {
            auto p = square2D.vertex(i);
            Eigen::Matrix2d mat;
            mat << p.x(), p.y(), 2.0 * p.x(), 3.0 * p.y();
            m.set(p, mat);
        }

        auto [subdivided, subdiv_map] = barycentric_subdivide(square2D);
        auto m_refined = m.refine(square2D, subdivided, subdiv_map, metric);

        for (std::size_t i = 0; i < square2D.num_vertices(); ++i) {
            auto p = square2D.vertex(i);
            for (std::size_t j = 0; j < subdivided.num_vertices(); ++j) {
                if (subdivided.vertex(j).isApprox(p, 1e-12)) {
                    EXPECT_TRUE(m_refined.at(p).isApprox(m.at(p), 1e-12));
                    break;
                }
            }
        }

        bool found_mid = false;
        for (std::size_t j = 0; j < subdivided.num_vertices(); ++j) {
            auto p = subdivided.vertex(j);
            if (std::abs(p.x() - 0.5) < 1e-12 && std::abs(p.y() - 0.0) < 1e-12) {
                Eigen::Matrix2d expected;
                expected << 0.5, 0.0, 1.0, 0.0;
                EXPECT_TRUE(m_refined.at(p).isApprox(expected, 1e-12));
                found_mid = true;
                break;
            }
        }
        EXPECT_TRUE(found_mid);
    }

} // namespace delta::testing