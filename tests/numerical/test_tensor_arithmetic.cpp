// tests/test_tensor_arithmetic.cpp
#include <gtest/gtest.h>
#include "delta/geometry/tensor_field.h"
#include "delta/geometry/matrix_field.h"
#include "delta/core/rational.h"

using namespace delta::geometry;
using Addr = int; // для простоты используем целые адреса
using Scalar = double;

TEST(TensorFieldTest, ScalarFieldArithmetic) {
    TensorField<Addr, Scalar, 0, 0> sf; // ранг 0, размерность не важна
    sf[1] = 2.5;
    sf[2] = 3.0;

    EXPECT_EQ(sf.at(1), 2.5);
    EXPECT_EQ(sf.at(2), 3.0);

    auto sum = sf + sf;
    EXPECT_EQ(sum.at(1), 5.0);
    EXPECT_EQ(sum.at(2), 6.0);

    auto scaled = sf * 2.0;
    EXPECT_EQ(scaled.at(1), 5.0);
}

TEST(TensorFieldTest, VectorFieldArithmetic) {
    TensorField<Addr, Scalar, 1, 3> vf;
    vf[1] = Eigen::Vector3d(1.0, 2.0, 3.0);
    vf[2] = Eigen::Vector3d(4.0, 5.0, 6.0);

    auto sum = vf + vf;
    EXPECT_EQ(sum.at(1), Eigen::Vector3d(2.0, 4.0, 6.0));
    EXPECT_EQ(sum.at(2), Eigen::Vector3d(8.0, 10.0, 12.0));

    auto scaled = vf * 0.5;
    EXPECT_EQ(scaled.at(1), Eigen::Vector3d(0.5, 1.0, 1.5));
}

TEST(TensorFieldTest, MatrixFieldArithmetic) {
    MatrixField<Addr, Scalar, 2> mf;
    Eigen::Matrix2d m1, m2;
    m1 << 1, 2, 3, 4;
    m2 << 5, 6, 7, 8;
    mf[1] = m1;
    mf[2] = m2;

    auto sum = mf + mf;
    EXPECT_EQ(sum.at(1), m1 * 2);
    EXPECT_EQ(sum.at(2), m2 * 2);

    auto prod = mf * mf; // поэлементное матричное умножение (определённое в MatrixField)
    EXPECT_EQ(prod.at(1), m1 * m1);
    EXPECT_EQ(prod.at(2), m2 * m2);

    auto trans = mf.transpose();
    EXPECT_EQ(trans.at(1), m1.transpose());
}

TEST(TensorFieldTest, ContainsAndSet) {
    TensorField<Addr, Scalar, 0, 0> sf;
    EXPECT_FALSE(sf.contains(10));
    sf.set(10, 123.0);
    EXPECT_TRUE(sf.contains(10));
    EXPECT_EQ(sf.at(10), 123.0);
}