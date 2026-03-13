// tests/numerical/test_cotangent_laplacian.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "delta/geometry/simplicial_complex.h"
#include "delta/numerical/cotangent_laplacian.h"
#include "delta/numerical/concepts.h"
#include "../test_fixtures.h"

// -----------------------------------------------------------------------------
// Tests for cotangent Laplacian on 2D simplicial complexes
// -----------------------------------------------------------------------------

TEST(CotangentLaplacianTest, RightTriangle) {
    delta::geometry::SimplicialComplex<2, double> mesh;
    auto v0 = mesh.add_vertex({ 0.0, 0.0 });
    auto v1 = mesh.add_vertex({ 1.0, 0.0 });
    auto v2 = mesh.add_vertex({ 0.0, 1.0 });
    mesh.add_triangle(v0, v1, v2);

    delta::geometry::EuclideanMetric metric;
    auto L = delta::numerical::build_cotangent_laplacian(mesh, metric);

    Eigen::Matrix3d expected;
    expected << 1.0, -0.5, -0.5,
        -0.5, 0.5, 0.0,
        -0.5, 0.0, 0.5;

    Eigen::Matrix3d L_dense = Eigen::Matrix3d(L);
    EXPECT_TRUE(L_dense.isApprox(expected, 1e-12));
}

TEST(CotangentLaplacianTest, EquilateralTriangle) {
    double s3 = std::sqrt(3.0);
    delta::geometry::SimplicialComplex<2, double> mesh;
    auto v0 = mesh.add_vertex({ 0.0, 0.0 });
    auto v1 = mesh.add_vertex({ 1.0, 0.0 });
    auto v2 = mesh.add_vertex({ 0.5, s3 / 2.0 });
    mesh.add_triangle(v0, v1, v2);

    delta::geometry::EuclideanMetric metric;
    auto L = delta::numerical::build_cotangent_laplacian(mesh, metric);

    double cot60 = 1.0 / s3;
    double off = -0.5 * cot60;
    double diag = cot60;

    Eigen::Matrix3d expected;
    expected << diag, off, off,
        off, diag, off,
        off, off, diag;

    Eigen::Matrix3d L_dense = Eigen::Matrix3d(L);
    EXPECT_TRUE(L_dense.isApprox(expected, 1e-12));
}

TEST(CotangentLaplacianTest, SquareMesh) {
    delta::geometry::SimplicialComplex<2, double> mesh;
    auto v0 = mesh.add_vertex({ 0.0, 0.0 });
    auto v1 = mesh.add_vertex({ 1.0, 0.0 });
    auto v2 = mesh.add_vertex({ 1.0, 1.0 });
    auto v3 = mesh.add_vertex({ 0.0, 1.0 });
    mesh.add_triangle(v0, v1, v2);
    mesh.add_triangle(v0, v2, v3);

    delta::geometry::EuclideanMetric metric;
    auto L = delta::numerical::build_cotangent_laplacian(mesh, metric);

    EXPECT_EQ(L.rows(), 4);
    EXPECT_EQ(L.cols(), 4);
    EXPECT_TRUE(L.isApprox(L.transpose()));

    Eigen::VectorXd rowsum = L * Eigen::VectorXd::Ones(4);
    EXPECT_NEAR(rowsum.norm(), 0.0, 1e-12);
}

TEST(CotangentLaplacianTest, EmptyMesh) {
    delta::geometry::SimplicialComplex<2, double> mesh;
    delta::geometry::EuclideanMetric metric;
    auto L = delta::numerical::build_cotangent_laplacian(mesh, metric);
    EXPECT_EQ(L.rows(), 0);
    EXPECT_EQ(L.cols(), 0);
}

TEST(CotangentLaplacianTest, DisconnectedComponents) {
    delta::geometry::SimplicialComplex<2, double> mesh;
    // First triangle
    auto a0 = mesh.add_vertex({ 0.0, 0.0 });
    auto a1 = mesh.add_vertex({ 1.0, 0.0 });
    auto a2 = mesh.add_vertex({ 0.0, 1.0 });
    mesh.add_triangle(a0, a1, a2);
    // Second triangle (disconnected)
    auto b0 = mesh.add_vertex({ 2.0, 2.0 });
    auto b1 = mesh.add_vertex({ 3.0, 2.0 });
    auto b2 = mesh.add_vertex({ 2.0, 3.0 });
    mesh.add_triangle(b0, b1, b2);

    delta::geometry::EuclideanMetric metric;
    auto L = delta::numerical::build_cotangent_laplacian(mesh, metric);
    EXPECT_EQ(L.rows(), 6);
    EXPECT_EQ(L.cols(), 6);

    // Check block-diagonal structure: no connection between components
    for (int i = 0; i < 3; ++i) {
        for (int j = 3; j < 6; ++j) {
            EXPECT_EQ(L.coeff(i, j), 0.0);
            EXPECT_EQ(L.coeff(j, i), 0.0);
        }
    }

    // Row sum of each block should be close to zero
    Eigen::VectorXd ones(6);
    ones << 1, 1, 1, 1, 1, 1;
    Eigen::VectorXd rowsum = L * ones;
    EXPECT_NEAR(rowsum.head(3).norm(), 0.0, 1e-12);
    EXPECT_NEAR(rowsum.tail(3).norm(), 0.0, 1e-12);
}