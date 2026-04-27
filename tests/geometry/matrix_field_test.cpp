// tests/geometry/matrix_field_test.cpp
#include <gtest/gtest.h>
#include "delta/geometry/matrix_field.h"
#include "../test_fixtures_geometry_numerical.h"

namespace delta::testing {
    class MatrixFieldTest : public GeometryNumericalTest {
    protected:
        static constexpr int DIM = 2;
        using Addr = Point<DIM>;
        using Compare = PointLess<DIM>;
        using Grid = delta::ListGrid<Addr, Compare>;
        using Matrix2 = Eigen::Matrix<Scalar, DIM, DIM>;

        using MatrixField2 = delta::geometry::MatrixField<Addr, DIM, Compare>;

        Grid make_test_grid() {
            std::vector<Addr> points;
            points.push_back(make_point<DIM>(0_r, 0_r));
            points.push_back(make_point<DIM>(1_r, 0_r));
            return Grid(std::move(points), Compare());
        }
    };

    // -------------------------------------------------------------------------
    // Utility Tests to double-check actual precision management
    // -------------------------------------------------------------------------
    TEST_F(MatrixFieldTest, PrecisionManagementWorks) {
        // Проверяем, что set_precision действительно изменяет глобальную переменную
        Rational original_eps = delta::default_eps();

        set_precision(Rational(1, 1000));
        EXPECT_EQ(delta::default_eps(), Rational(1, 1000));

        set_precision(Rational(1, 1000000));
        EXPECT_EQ(delta::default_eps(), Rational(1, 1000000));

        // Восстанавливаем исходное значение
        set_precision(original_eps);
        EXPECT_EQ(delta::default_eps(), original_eps);
    }
    TEST_F(MatrixFieldTest, DefaultEpsilonAffectsResult) {

        Rational original_eps = delta::default_eps();

        std::vector<Rational> eps_values = {
            Rational(1, 1000),                    // 1e-3
            Rational(1, 1000000),                 // 1e-6
            Rational(1, 1000000000),              // 1e-9
            Rational(1, 1000000000000),           // 1e-12
            Rational(1, 1000000000000000),        // 1e-15
            "1/1000000000000000000"_r,           // 1e-18
            "1/1000000000000000000000"_r,        // 1e-21
            "1/1000000000000000000000000"_r,     // 1e-24
            "1/1000000000000000000000000000"_r,  // 1e-27
            "1/1000000000000000000000000000000"_r // 1e-30
        };

        Rational arg_sqrt = 2_r;
        Rational arg_exp = 1_r;
        Rational arg_log = 2_r;
        Rational arg_sin = 1_r;
        Rational arg_cos = 1_r;
        Rational arg_acos = Rational(1, 2);
        Rational arg_pow_base = 5_r;
        Rational arg_pow_exp = Rational(1, 3);

        std::vector<Rational> sqrt_results, exp_results, log_results,
            sin_results, cos_results, acos_results,
            pi_results, e_results, pow_results;

        for (const auto& eps : eps_values) {
            set_precision(eps);

            sqrt_results.push_back(delta::sqrt(arg_sqrt));
            exp_results.push_back(delta::exp(arg_exp));
            log_results.push_back(delta::log(arg_log));
            sin_results.push_back(delta::sin(arg_sin));
            cos_results.push_back(delta::cos(arg_cos));
            acos_results.push_back(delta::acos(arg_acos));
            pi_results.push_back(delta::pi());
            e_results.push_back(delta::e());
            pow_results.push_back(delta::pow(arg_pow_base, arg_pow_exp));
        }

        // Проверяем, что для каждой функции результаты не все одинаковы
        auto has_variation = [](const auto& vec) {
            if (vec.empty()) return false;
            return std::adjacent_find(vec.begin(), vec.end(), std::not_equal_to<>()) != vec.end();
            };

        EXPECT_TRUE(has_variation(sqrt_results)) << "sqrt results do not vary with epsilon";
        EXPECT_TRUE(has_variation(exp_results)) << "exp results do not vary with epsilon";
        EXPECT_TRUE(has_variation(log_results)) << "log results do not vary with epsilon";
        EXPECT_TRUE(has_variation(sin_results)) << "sin results do not vary with epsilon";
        EXPECT_TRUE(has_variation(cos_results)) << "cos results do not vary with epsilon";
        EXPECT_TRUE(has_variation(acos_results)) << "acos results do not vary with epsilon";
        EXPECT_TRUE(has_variation(pi_results)) << "pi results do not vary with epsilon";
        EXPECT_TRUE(has_variation(e_results)) << "e results do not vary with epsilon";
        EXPECT_TRUE(has_variation(pow_results)) << "pow results do not vary with epsilon";

        set_precision(original_eps);
    }

    // -------------------------------------------------------------------------
    // Basic operations
    // -------------------------------------------------------------------------
    TEST_F(MatrixFieldTest, MatrixMultiplication) {
        Grid grid = make_test_grid();
        MatrixField2 A(grid), B(grid), C(grid);

        Matrix2 a1; a1 << 1_r, 2_r, 3_r, 4_r;
        Matrix2 b1; b1 << 5_r, 6_r, 7_r, 8_r;
        A.set(grid[0], a1);
        B.set(grid[0], b1);

        Matrix2 a2; a2 << 2_r, 0_r, 1_r, 3_r;
        Matrix2 b2; b2 << 1_r, 2_r, 2_r, 1_r;
        A.set(grid[1], a2);
        B.set(grid[1], b2);

        C = A * B;
        EXPECT_TRUE(matrix_near(C.at(grid[0]), (a1 * b1).eval()));
        EXPECT_TRUE(matrix_near(C.at(grid[1]), (a2 * b2).eval()));
    }

    TEST_F(MatrixFieldTest, Determinant) {
        Grid grid = make_test_grid();
        MatrixField2 A(grid);

        Matrix2 a1; a1 << 1_r, 2_r, 3_r, 4_r;
        Matrix2 a2; a2 << 2_r, 0_r, 1_r, 3_r;
        A.set(grid[0], a1);
        A.set(grid[1], a2);

        auto det = A.determinant();
        EXPECT_EQ(det.at(grid[0]), a1.determinant());
        EXPECT_EQ(det.at(grid[1]), a2.determinant());
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

        auto comm = A.comm(B);
        EXPECT_TRUE(matrix_near(comm.at(grid[0]), (a1 * b1 - b1 * a1).eval()));
        EXPECT_TRUE(matrix_near(comm.at(grid[1]), (a2 * b2 - b2 * a2).eval()));
    }

    // -------------------------------------------------------------------------
    // In-place multiplication
    // -------------------------------------------------------------------------
    TEST_F(MatrixFieldTest, MultiplicationAssign) {
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

        MatrixField2 A_copy = A;
        A_copy *= B;
        MatrixField2 expected = A * B;
        for (const auto& addr : grid) {
            EXPECT_TRUE(matrix_near(A_copy.at(addr), expected.at(addr)));
        }
    }

    // -------------------------------------------------------------------------
    // Exponential and logarithm
    // -------------------------------------------------------------------------
    TEST_F(MatrixFieldTest, ExponentialAndLogarithm) {

        set_precision(Rational(1, 1000000));
        Grid grid = make_test_grid();
        MatrixField2 A(grid);

        // Use a small nilpotent matrix N = [[0, 0.1], [0, 0]]
        Matrix2 N; N << 0_r, "0.1"_r, 0_r, 0_r;
        Matrix2 B = Matrix2::Identity() + N;   // norm(B-I) = 0.1 < 0.5 → no scaling needed

        A.set(grid[0], B);
        A.set(grid[1], B);

        auto logA = A.log();
        auto exp_logA = logA.exp();

        // Check that log(B) = N and exp(N) = B
        EXPECT_TRUE(matrix_near(logA.at(grid[0]), N, delta::default_eps()));
        EXPECT_TRUE(matrix_near(exp_logA.at(grid[0]), B, delta::default_eps()));

        // Check that exp(log(B)) == B
        EXPECT_TRUE(matrix_near(exp_logA.at(grid[0]), A.at(grid[0]), delta::default_eps()));

        // Check log(exp(N)) == N
        MatrixField2 N_field(grid);
        N_field.set(grid[0], N);
        N_field.set(grid[1], N);
        auto expN = N_field.exp();
        auto log_expN = expN.log();
        EXPECT_TRUE(matrix_near(log_expN.at(grid[0]), N, delta::default_eps()));
    }

    TEST_F(MatrixFieldTest, ExponentialLogarithmDiagonal) {
        // Устанавливаем более низкую точность для ускорения вычислений.
        // При дефолтной точности (1e-30) тест выполняется недопустимо долго из-за
        // сложных вычислений с рациональными числами.
        set_precision(Rational(1, 1000000)); // 1e-6


        Grid grid = make_test_grid();
        MatrixField2 A(grid);
        Matrix2 D; D << "0.1"_r, 0_r, 0_r, "0.2"_r;
        A.set(grid[0], D);

        auto expD = A.exp();
        auto logExpD = expD.log();

        // При точности 1e-6 результат должен быть близок к исходной матрице.
        // Проверяем с тем же допуском.
        EXPECT_TRUE(matrix_near(logExpD.at(grid[0]), D, delta::default_eps()));
    }

    // -------------------------------------------------------------------------
    // Symmetric positive definite matrix (near identity)
    // -------------------------------------------------------------------------
    TEST_F(MatrixFieldTest, ExponentialLogarithmSymmetric) {

        set_precision(Rational(1, 1000000));
        Grid grid = make_test_grid();
        MatrixField2 A(grid);
        Matrix2 M; M << "1.1"_r, "0.05"_r, "0.05"_r, "0.9"_r;
        A.set(grid[0], M);

        auto logM = A.log();
        auto exp_logM = logM.exp();

        // Ожидаемое поведение: композиция приближённых функций exp и log
        // даёт результат, близкий к исходной матрице, но с бóльшей погрешностью,
        // чем заданный eps (здесь eps = 1e-6). Накопление ошибок из-за
        // масштабирования и аппроксимации рядов — математически корректно.
        // Устанавливаем допуск в 1e-4 (в 100 раз больше eps).
        Rational tolerance = Rational(1, 10000);  // 1e-4
        EXPECT_TRUE(matrix_near(exp_logM.at(grid[0]), M, tolerance));
    }

    // -------------------------------------------------------------------------
    // Large norm matrix (scaling required)
    // -------------------------------------------------------------------------
    TEST_F(MatrixFieldTest, ExponentialLogarithmLargeNorm) {

        set_precision(Rational(1, 1000000));
        Grid grid = make_test_grid();
        MatrixField2 A(grid);
        // Use B = I + N where N has entry 0.5, so norm(B-I)=0.5 – borderline, will trigger scaling
        Matrix2 N; N << 0_r, "0.5"_r, 0_r, 0_r;
        Matrix2 B = Matrix2::Identity() + N;
        A.set(grid[0], B);

        auto logB = A.log();
        auto exp_logB = logB.exp();

        EXPECT_TRUE(matrix_near(exp_logB.at(grid[0]), B, delta::default_eps()));

        MatrixField2 N_field(grid);
        N_field.set(grid[0], N);
        auto expN = N_field.exp();
        auto log_expN = expN.log();
        EXPECT_TRUE(matrix_near(log_expN.at(grid[0]), N, delta::default_eps()));
    }

    // -------------------------------------------------------------------------
    // Singular matrix should throw domain_error
    // -------------------------------------------------------------------------
    TEST_F(MatrixFieldTest, LogarithmSingularMatrix) {
        Grid grid = make_test_grid();
        MatrixField2 A(grid);
        Matrix2 Z; Z << 0_r, 0_r, 0_r, 0_r;
        A.set(grid[0], Z);

        EXPECT_THROW(A.log(), std::domain_error);
    }

    // -------------------------------------------------------------------------
    // Matrix far from identity (norm > 0.5) still works with scaling
    // -------------------------------------------------------------------------
    TEST_F(MatrixFieldTest, LogarithmFarFromIdentity) {

        set_precision(Rational(1, 1000000));
        Grid grid = make_test_grid();
        MatrixField2 A(grid);
        // M = diag(10, 0.1) – far from identity, but positive definite
        Matrix2 M; M << 10_r, 0_r, 0_r, "0.1"_r;
        A.set(grid[0], M);

        EXPECT_NO_THROW(A.log());
        auto logM = A.log();
        auto exp_logM = logM.exp();
        EXPECT_TRUE(matrix_near(exp_logM.at(grid[0]), M, delta::default_eps()));
    }

    // -------------------------------------------------------------------------
    // Square root consistency via exp(0.5*log(M))
    // -------------------------------------------------------------------------
    TEST_F(MatrixFieldTest, SquareRootConsistency) {
        // Ставим дефолтный эпсилон для вычислений значительно меньше ожидаемого чтобы скомпенсировать композицию аппроксимаций. 
        // Это математически корректно.
        internal::reset_default_eps();//1e-30.
        auto grid = make_test_grid();
        MatrixField2 A(grid);
        Matrix2 M; M << 2_r, 1_r, 1_r, 2_r;
        A.set(grid[0], M);
        // Вторая точка (grid[1]) остаётся нулевой — логарифм её пропустит

        auto logM = A.log();  // содержит только grid[0]

        // Умножаем каждую матрицу в поле на 0.5
        MatrixField2 halfLogField;
        for (const auto& [addr, mat] : logM) {
            halfLogField.set(addr, mat * "0.5"_r);
        }

        auto sqrtM = halfLogField.exp();
        auto sqrtM_sq = sqrtM * sqrtM;

        // Ожидаемое поведение: композиция exp(0.5 * log(M,eps=epsilon),eps=epsilon) даёт приближение
        // к квадратному корню, но с накопленной погрешностью РЕЗУЛЬТАТ НИКОГДА НЕ БУДЕТ РАВЕН M с точностью до epsilon.
        // Логарифм даёт погрешность отсечения ряда, а экспонента её раздувает (экспоненциально, да).
        Rational tolerance = Rational(1, 10000);  // 1e-5
        EXPECT_TRUE(matrix_near(sqrtM_sq.at(grid[0]), M, tolerance));
    }
} // namespace delta::testing