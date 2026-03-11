// tests/numerical/test_cotangent_laplacian.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "delta/geometry/simplicial_complex.h"
#include "delta/numerical/cotangent_laplacian.h"
#include "delta/numerical/discrete_gradient_divergence.h"

using namespace delta::geometry;
using namespace delta::numerical;

// ===================== Тесты котангенсного лапласиана =====================

TEST(CotangentLaplacianTest, RightTriangle) {
    SimplicialComplex2D<double> mesh;
    auto v0 = mesh.add_vertex({ 0.0, 0.0 });
    auto v1 = mesh.add_vertex({ 1.0, 0.0 });
    auto v2 = mesh.add_vertex({ 0.0, 1.0 });
    mesh.add_triangle(v0, v1, v2);

    auto L = build_cotangent_laplacian(mesh);

    Eigen::Matrix3d expected;
    expected << 1.0, -0.5, -0.5,
        -0.5, 0.5, 0.0,
        -0.5, 0.0, 0.5;

    Eigen::Matrix3d L_dense = Eigen::Matrix3d(L);
    EXPECT_TRUE(L_dense.isApprox(expected, 1e-12));
}

TEST(CotangentLaplacianTest, EquilateralTriangle) {
    double s3 = std::sqrt(3.0);
    SimplicialComplex2D<double> mesh;
    auto v0 = mesh.add_vertex({ 0.0, 0.0 });
    auto v1 = mesh.add_vertex({ 1.0, 0.0 });
    auto v2 = mesh.add_vertex({ 0.5, s3 / 2.0 });
    mesh.add_triangle(v0, v1, v2);

    auto L = build_cotangent_laplacian(mesh);

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
    SimplicialComplex2D<double> mesh;
    auto v0 = mesh.add_vertex({ 0.0, 0.0 });
    auto v1 = mesh.add_vertex({ 1.0, 0.0 });
    auto v2 = mesh.add_vertex({ 1.0, 1.0 });
    auto v3 = mesh.add_vertex({ 0.0, 1.0 });
    mesh.add_triangle(v0, v1, v2);
    mesh.add_triangle(v0, v2, v3);

    auto L = build_cotangent_laplacian(mesh);

    EXPECT_EQ(L.rows(), 4);
    EXPECT_EQ(L.cols(), 4);
    EXPECT_TRUE(L.isApprox(L.transpose()));

    Eigen::VectorXd rowsum = L * Eigen::VectorXd::Ones(4);
    EXPECT_NEAR(rowsum.norm(), 0.0, 1e-12);
}

TEST(CotangentLaplacianTest, EmptyMesh) {
    SimplicialComplex2D<double> mesh;
    auto L = build_cotangent_laplacian(mesh);
    EXPECT_EQ(L.rows(), 0);
    EXPECT_EQ(L.cols(), 0);
}

TEST(CotangentLaplacianTest, DisconnectedComponents) {
    SimplicialComplex2D<double> mesh;
    // Первый треугольник
    auto a0 = mesh.add_vertex({ 0.0, 0.0 });
    auto a1 = mesh.add_vertex({ 1.0, 0.0 });
    auto a2 = mesh.add_vertex({ 0.0, 1.0 });
    mesh.add_triangle(a0, a1, a2);
    // Второй треугольник (отдельная компонента)
    auto b0 = mesh.add_vertex({ 2.0, 2.0 });
    auto b1 = mesh.add_vertex({ 3.0, 2.0 });
    auto b2 = mesh.add_vertex({ 2.0, 3.0 });
    mesh.add_triangle(b0, b1, b2);

    auto L = build_cotangent_laplacian(mesh);
    EXPECT_EQ(L.rows(), 6);
    EXPECT_EQ(L.cols(), 6);

    // Проверим блочно-диагональную структуру: нет связи между компонентами
    for (int i = 0; i < 3; ++i) {
        for (int j = 3; j < 6; ++j) {
            EXPECT_EQ(L.coeff(i, j), 0.0);
            EXPECT_EQ(L.coeff(j, i), 0.0);
        }
    }

    // Сумма строк каждого блока должна быть близка к нулю
    Eigen::VectorXd ones(6);
    ones << 1, 1, 1, 1, 1, 1;
    Eigen::VectorXd rowsum = L * ones;
    EXPECT_NEAR(rowsum.head(3).norm(), 0.0, 1e-12);
    EXPECT_NEAR(rowsum.tail(3).norm(), 0.0, 1e-12);
}

// ===================== Тесты дискретных градиента и дивергенции =====================

TEST(DiscreteGradientDivergenceTest, GradientOfLinearFunction) {
    SimplicialComplex2D<double> mesh;
    auto v0 = mesh.add_vertex({ 0.0, 0.0 });
    auto v1 = mesh.add_vertex({ 1.0, 0.0 });
    auto v2 = mesh.add_vertex({ 0.0, 1.0 });
    mesh.add_triangle(v0, v1, v2);

    Eigen::VectorXd f(3);
    f << 0.0, 2.0, 3.0; // f = 2x + 3y

    auto grad = discrete_gradient(mesh, f);
    EXPECT_EQ(grad.size(), mesh.num_edges());

    auto e01 = mesh.find_edge(v0, v1);
    auto e12 = mesh.find_edge(v1, v2);
    auto e20 = mesh.find_edge(v2, v0);

    auto edge_val = [&](std::size_t e_idx) {
        const auto& edge = mesh.edges()[e_idx];
        return f[edge[1]] - f[edge[0]];
        };

    EXPECT_DOUBLE_EQ(grad[e01], edge_val(e01));
    EXPECT_DOUBLE_EQ(grad[e12], edge_val(e12));
    EXPECT_DOUBLE_EQ(grad[e20], edge_val(e20));
}

TEST(DiscreteGradientDivergenceTest, ConstantFunction) {
    SimplicialComplex2D<double> mesh;
    auto v0 = mesh.add_vertex({ 0.0, 0.0 });
    auto v1 = mesh.add_vertex({ 1.0, 0.0 });
    auto v2 = mesh.add_vertex({ 0.0, 1.0 });
    mesh.add_triangle(v0, v1, v2);

    Eigen::VectorXd f = Eigen::VectorXd::Constant(3, 5.0);
    auto grad = discrete_gradient(mesh, f);
    for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
        EXPECT_DOUBLE_EQ(grad[e], 0.0);
    }

    auto areas = compute_vertex_dual_areas(mesh);
    auto div_raw = discrete_divergence_raw(mesh, grad);
    for (std::size_t i = 0; i < mesh.num_vertices(); ++i) {
        EXPECT_DOUBLE_EQ(div_raw[i], 0.0);
    }
}

TEST(DiscreteGradientDivergenceTest, DivergenceOfConstantField) {
    SimplicialComplex2D<double> mesh;
    auto v0 = mesh.add_vertex({ 0.0, 0.0 });
    auto v1 = mesh.add_vertex({ 1.0, 0.0 });
    auto v2 = mesh.add_vertex({ 0.0, 1.0 });
    mesh.add_triangle(v0, v1, v2);

    Eigen::VectorXd edge_vals = Eigen::VectorXd::Constant(mesh.num_edges(), 1.0);
    auto div_raw = discrete_divergence_raw(mesh, edge_vals);

    double total = 0.0;
    for (const auto& val : div_raw) total += val;
    EXPECT_NEAR(total, 0.0, 1e-12);
}

TEST(DiscreteGradientDivergenceTest, DivGradOfConstantFunctionIsZero) {
    SimplicialComplex2D<double> mesh;
    auto v0 = mesh.add_vertex({ 0.0, 0.0 });
    auto v1 = mesh.add_vertex({ 1.0, 0.0 });
    auto v2 = mesh.add_vertex({ 0.0, 1.0 });
    mesh.add_triangle(v0, v1, v2);

    Eigen::VectorXd f = Eigen::VectorXd::Constant(3, 7.0);
    auto grad = discrete_gradient(mesh, f);
    auto areas = compute_vertex_dual_areas(mesh);
    auto div_grad = discrete_divergence(mesh, grad, areas);

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(div_grad[i], 0.0, 1e-12);
    }
}

TEST(DiscreteGradientDivergenceTest, EmptyMesh) {
    SimplicialComplex2D<double> mesh;
    Eigen::VectorXd f_empty(0);
    auto grad = discrete_gradient(mesh, f_empty);
    EXPECT_EQ(grad.size(), 0);
    auto div_raw = discrete_divergence_raw(mesh, grad);
    EXPECT_EQ(div_raw.size(), 0);
}

TEST(DiscreteGradientDivergenceTest, IsolatedVertex) {
    SimplicialComplex2D<double> mesh;
    auto v0 = mesh.add_vertex({ 0.0, 0.0 });          // изолирована
    auto v1 = mesh.add_vertex({ 1.0, 0.0 });
    auto v2 = mesh.add_vertex({ 0.0, 1.0 });
    auto v3 = mesh.add_vertex({ 1.0, 1.0 });
    mesh.add_triangle(v1, v2, v3);                   // треугольник из v1,v2,v3

    Eigen::VectorXd f(4);
    f << 5.0, 2.0, 3.0, 4.0;

    auto grad = discrete_gradient(mesh, f);

    auto e12 = mesh.find_edge(v1, v2);
    auto e23 = mesh.find_edge(v2, v3);
    auto e31 = mesh.find_edge(v3, v1); // ребро между v1 и v3 хранится как (min, max) = (1,3)

    EXPECT_DOUBLE_EQ(grad[e12], f[2] - f[1]); // 3-2=1
    EXPECT_DOUBLE_EQ(grad[e23], f[3] - f[2]); // 4-3=1
    EXPECT_DOUBLE_EQ(grad[e31], f[3] - f[1]); // 4-2=2
}