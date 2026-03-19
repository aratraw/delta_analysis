//tests/geometry/tensor_field_test.cpp
#include <gtest/gtest.h>
#include <vector>
#include <optional>
#include "delta/geometry/tensor_field.h"
#include "../test_fixtures_geometry_numerical.h"

namespace delta::geometry::testing {

    using delta::testing::GeometryNumericalTest;
    using delta::operator""_r;

    /**
     * @class TensorFieldTest
     * @brief Tests for TensorField class (ranks 0,1,2) and associated operations.
     */
    class TensorFieldTest : public GeometryNumericalTest {
    protected:
        static constexpr int DIM = 2;
        using Addr = Point<DIM>;
        using Compare = PointLess<DIM>;          // теперь определён
        using Grid = delta::ListGrid<Addr, Compare>;

        // Типы тензоров
        using Scalar0 = Scalar;                    // ранг 0
        using Vector1 = Eigen::Matrix<Scalar, DIM, 1>;   // ранг 1
        using Matrix2 = Eigen::Matrix<Scalar, DIM, DIM>; // ранг 2

        // Само тензорное поле с правильным компаратором
        template<int Rank>
        using Field = delta::geometry::TensorField<Addr, Scalar, Rank, DIM, Compare>;

        // Вспомогательная функция для создания простой сетки из двух точек
        Grid make_test_grid() {
            std::vector<Addr> points;
            points.push_back(make_point<DIM>(0_r, 0_r));
            points.push_back(make_point<DIM>(1_r, 0_r));
            return Grid(std::move(points), Compare());
        }

        // Проверка, что два тензора одного ранга близки (с учётом допуска)
        template<int Rank>
        bool tensors_near(const typename Field<Rank>::value_type& a,
            const typename Field<Rank>::value_type& b,
            const Scalar& eps = delta::default_eps()) {
            if constexpr (Rank == 0) {
                return delta::abs(a - b) <= eps;
            }
            else {
                return matrix_near(a, b, eps);
            }
        }
    };
    // =========================================================================
    // 1. Basic creation and access
    // =========================================================================

    TEST_F(TensorFieldTest, CreateAndAccess) {
        Grid grid = make_test_grid();
        Field<0> scalar_field(grid);   // скалярное поле, инициализируется нулями?

        // Проверяем, что поле содержит все точки сетки
        EXPECT_TRUE(scalar_field.contains(grid[0]));
        EXPECT_TRUE(scalar_field.contains(grid[1]));

        // Устанавливаем значения
        scalar_field.set(grid[0], 3_r);
        scalar_field.set(grid[1], 7_r);

        // Проверяем доступ
        EXPECT_EQ(scalar_field.at(grid[0]), 3_r);
        EXPECT_EQ(scalar_field.at(grid[1]), 7_r);

        // Проверяем, что несуществующая точка не содержится
        Addr other = make_point<DIM>(2_r, 2_r);
        EXPECT_FALSE(scalar_field.contains(other));

        // Аналогично для векторного поля (ранг 1)
        Field<1> vector_field(grid);
        Vector1 v0; v0 << 1_r, 2_r;
        Vector1 v1; v1 << 3_r, 4_r;
        vector_field.set(grid[0], v0);
        vector_field.set(grid[1], v1);
        EXPECT_TRUE(tensors_near<1>(vector_field.at(grid[0]), v0));
        EXPECT_TRUE(tensors_near<1>(vector_field.at(grid[1]), v1));

        // Для матричного поля (ранг 2)
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

    TEST_F(TensorFieldTest, Addition) {
        Grid grid = make_test_grid();
        Field<2> A(grid), B(grid), C(grid);

        Matrix2 m1; m1 << 1_r, 2_r, 3_r, 4_r;
        Matrix2 m2; m2 << 5_r, 6_r, 7_r, 8_r;
        A.set(grid[0], m1);
        A.set(grid[1], m2);
        B.set(grid[0], m2);
        B.set(grid[1], m1);

        C = A + B;   // оператор + должен быть определён

        Matrix2 sum0 = m1 + m2;
        Matrix2 sum1 = m2 + m1;
        EXPECT_TRUE(tensors_near<2>(C.at(grid[0]), sum0));
        EXPECT_TRUE(tensors_near<2>(C.at(grid[1]), sum1));

        // Проверка сложения с самим собой
        C = A + A;
        EXPECT_TRUE(tensors_near<2>(C.at(grid[0]), m1 + m1));
    }

    TEST_F(TensorFieldTest, ScalarMultiplication) {
        Grid grid = make_test_grid();
        Field<1> V(grid);
        Vector1 v; v << 2_r, 3_r;
        V.set(grid[0], v);
        V.set(grid[1], v);

        Field<1> W = 4_r * V;   // оператор умножения скаляра на поле

        Vector1 expected = 4_r * v;
        EXPECT_TRUE(tensors_near<1>(W.at(grid[0]), expected));
        EXPECT_TRUE(tensors_near<1>(W.at(grid[1]), expected));

        // Также V * 4_r
        Field<1> Z = V * 4_r;
        EXPECT_TRUE(tensors_near<1>(Z.at(grid[0]), expected));
    }

    TEST_F(TensorFieldTest, TensorProduct) {
        Grid grid = make_test_grid();
        Field<1> U(grid), V(grid);

        Vector1 u; u << 1_r, 2_r;
        Vector1 v; v << 3_r, 4_r;
        U.set(grid[0], u);
        V.set(grid[0], v);
        U.set(grid[1], u);
        V.set(grid[1], v);

        // Тензорное произведение U ⊗ V даёт матричное поле
        Field<2> T = tensor_product(U, V);   // свободная функция

        Matrix2 expected = u * v.transpose();
        EXPECT_TRUE(tensors_near<2>(T.at(grid[0]), expected));
        EXPECT_TRUE(tensors_near<2>(T.at(grid[1]), expected));
    }

    TEST_F(TensorFieldTest, Trace) {
        Grid grid = make_test_grid();
        Field<2> M(grid);

        Matrix2 m; m << 1_r, 2_r, 3_r, 4_r;
        M.set(grid[0], m);
        M.set(grid[1], m * 2_r);

        Field<0> tr = trace(M);   // след (скалярное поле)

        EXPECT_EQ(tr.at(grid[0]), 1_r + 4_r);   // 5
        EXPECT_EQ(tr.at(grid[1]), (1_r + 4_r) * 2_r);
    }

    TEST_F(TensorFieldTest, Symmetrize) {
        Grid grid = make_test_grid();
        Field<2> M(grid);

        Matrix2 m; m << 1_r, 2_r, 3_r, 4_r;
        M.set(grid[0], m);

        Field<2> sym = symmetrize(M);
        Matrix2 expected = (m + m.transpose()) / 2_r;
        EXPECT_TRUE(tensors_near<2>(sym.at(grid[0]), expected));
    }

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

    TEST_F(TensorFieldTest, RaiseLowerIndices) {
        Grid grid = make_test_grid();
        // Задаём метрику g (диагональную, например, евклидову)
        Field<2> g(grid);
        Matrix2 g_val = Matrix2::Identity();   // евклидова метрика
        g.set(grid[0], g_val);
        g.set(grid[1], g_val);

        // Обратная метрика
        Field<2> g_inv(grid);
        g_inv.set(grid[0], g_val.inverse());
        g_inv.set(grid[1], g_val.inverse());

        // Векторное поле v
        Field<1> v(grid);
        Vector1 v_val; v_val << 2_r, 3_r;
        v.set(grid[0], v_val);
        v.set(grid[1], v_val);

        // Опускание индекса: v_flat_i = g_ij * v^j
        Field<1> v_flat = lower_index(v, g);
        Vector1 expected_flat = g_val * v_val;
        EXPECT_TRUE(tensors_near<1>(v_flat.at(grid[0]), expected_flat));

        // Поднятие индекса: v_sharp^i = g^{ij} * v_flat_j
        Field<1> v_sharp = raise_index(v_flat, g_inv);
        EXPECT_TRUE(tensors_near<1>(v_sharp.at(grid[0]), v_val));
    }

} // namespace delta::geometry::testing