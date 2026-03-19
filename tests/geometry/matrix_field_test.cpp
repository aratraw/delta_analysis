// tests/geometry/matrix_field_test.cpp
#include <gtest/gtest.h>
#include "delta/geometry/matrix_field.h"
#include "../test_fixtures_geometry_numerical.h"

namespace delta::geometry::testing {

    using delta::testing::GeometryNumericalTest;
    using delta::operator""_r;

    class MatrixFieldTest : public GeometryNumericalTest {
    protected:
        static constexpr int DIM = 2;
        using Addr = Point<DIM>;
        using Compare = PointLess<DIM>;
        using Grid = delta::ListGrid<Addr, Compare>;
        using Matrix2 = Eigen::Matrix<Scalar, DIM, DIM>;

        // Поле матриц
        using MatrixField2 = delta::geometry::MatrixField<Addr, Scalar, DIM, Compare>;

        Grid make_test_grid() {
            std::vector<Addr> points;
            points.push_back(make_point<DIM>(0_r, 0_r));
            points.push_back(make_point<DIM>(1_r, 0_r));
            return Grid(std::move(points), Compare());
        }
    };

    TEST_F(MatrixFieldTest, MatrixMultiplication) {
        Grid grid = make_test_grid();
        MatrixField2 A(grid), B(grid), C(grid);

        // Первая точка
        Matrix2 a1; a1 << 1_r, 2_r, 3_r, 4_r;
        Matrix2 b1; b1 << 5_r, 6_r, 7_r, 8_r;
        A.set(grid[0], a1);
        B.set(grid[0], b1);

        // Вторая точка
        Matrix2 a2; a2 << 2_r, 0_r, 1_r, 3_r;
        Matrix2 b2; b2 << 1_r, 2_r, 2_r, 1_r;
        A.set(grid[1], a2);
        B.set(grid[1], b2);

        C = A * B;  // оператор * должен быть определён

        Matrix2 expected1 = a1 * b1;
        Matrix2 expected2 = a2 * b2;

        EXPECT_TRUE(matrix_near(C.at(grid[0]), expected1));
        EXPECT_TRUE(matrix_near(C.at(grid[1]), expected2));
    }

    TEST_F(MatrixFieldTest, Determinant) {
        Grid grid = make_test_grid();
        MatrixField2 A(grid);

        Matrix2 a1; a1 << 1_r, 2_r, 3_r, 4_r;
        Matrix2 a2; a2 << 2_r, 0_r, 1_r, 3_r;
        A.set(grid[0], a1);
        A.set(grid[1], a2);

        auto det = A.determinant();  // должно вернуть TensorField ранга 0

        Scalar det1 = a1.determinant();  // 1*4 - 2*3 = -2
        Scalar det2 = a2.determinant();  // 2*3 - 0*1 = 6

        EXPECT_EQ(det.at(grid[0]), det1);
        EXPECT_EQ(det.at(grid[1]), det2);
    }

    TEST_F(MatrixFieldTest, Commutator) {
        Grid grid = make_test_grid();
        MatrixField2 A(grid), B(grid);

        Matrix2 a1; a1 << 1_r, 2_r, 3_r, 4_r;
        Matrix2 b1; b1 << 5_r, 6_r, 7_r, 8_r;
        A.set(grid[0], a1);
        B.set(grid[0], b1);

        Matrix2 a2; a2 << 2_r, 0_r, 1_r, 3_r;
        Matrix2 b2; b2 << 1_r, 2_r, 2_r, 1_r;
        A.set(grid[1], a2);
        B.set(grid[1], b2);

        auto comm = A.comm(B);  // [A,B] = A*B - B*A

        Matrix2 comm1 = a1 * b1 - b1 * a1;
        Matrix2 comm2 = a2 * b2 - b2 * a2;

        EXPECT_TRUE(matrix_near(comm.at(grid[0]), comm1));
        EXPECT_TRUE(matrix_near(comm.at(grid[1]), comm2));
    }

    TEST_F(MatrixFieldTest, ExponentialAndLogarithm) {
        Grid grid = make_test_grid();
        MatrixField2 A(grid);

        // Используем нильпотентную матрицу N = [[0,1],[0,0]]
        // Тогда B = I + N, log(B) = N, exp(N) = I + N = B
        Matrix2 N; N << 0_r, 1_r, 0_r, 0_r;
        Matrix2 B = Matrix2::Identity() + N;  // [[1,1],[0,1]]

        A.set(grid[0], B);
        A.set(grid[1], B);  // обе точки одинаковы для простоты

        auto logA = A.log();   // должно быть поле с матрицами N
        auto exp_logA = logA.exp();  // должно быть поле с B

        EXPECT_TRUE(matrix_near(logA.at(grid[0]), N, delta::default_eps()));
        EXPECT_TRUE(matrix_near(exp_logA.at(grid[0]), B, delta::default_eps()));

        // Проверим, что exp(logA) = A
        EXPECT_TRUE(matrix_near(exp_logA.at(grid[0]), A.at(grid[0]), delta::default_eps()));
    }

} // namespace delta::geometry::testing