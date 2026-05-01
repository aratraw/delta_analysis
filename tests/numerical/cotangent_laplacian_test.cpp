// tests/numerical/cotangent_laplacian_test.cpp
// ============================================================================
// МАТЕМАТИЧЕСКОЕ ОБОСНОВАНИЕ ТЕСТОВ КОТАНГЕНСНОГО ЛАПЛАСИАНА
// Дата актуализации: 30.04.2026
// ============================================================================
//
// 1. КОТАНГЕНСНЫЙ ЛАПЛАСИАН: ОПРЕДЕЛЕНИЕ
// ============================================================================
//
// Для 2D симплициального комплекса (триангуляции) дискретный лапласиан 
// в вершине i определяется как:
//   (L u)_i = Σ_{j∈N(i)} w_{ij} (u_i - u_j)
//
// где веса на ребре (i,j):
//   w_{ij} = (cot α_{ij} + cot β_{ij}) / 2
// α_{ij} — угол в одном прилегающем треугольнике, противолежащий ребру (i,j),
// β_{ij} — во втором треугольнике (для граничного ребра β = 0).
//
// Матрица L собирается как:
//   L_{ii} = Σ_{j∈N(i)} w_{ij}
//   L_{ij} = -w_{ij}   для i≠j
//
// ============================================================================
// 2. МАТРИЧНЫЕ СВОЙСТВА (НЕ ЗАВИСЯТ ОТ СЕТКИ)
// ============================================================================
//
// 2.1. Симметричность: L^T = L
//      Следует из w_{ij} = w_{ji} и симметричной сборки.
//      Проверяется тестом Symmetry для любой сетки.
//
// 2.2. Сумма строк: Σ_j L_{ij} = 0
//      Следует из L_{ii} = Σ_{j≠i} w_{ij} и L_{ij} = -w_{ij}.
//      Проверяется тестом RowSumZero.
//
// 2.3. Константная функция в ядре: L * 1 = 0
//      Следствие суммы строк. Проверяется тестом ConstantFunctionKernel.
//
// ============================================================================
// 3. ДЕЙСТВИЕ НА ФУНКЦИЯХ (ЗАВИСИТ ОТ НАЛИЧИЯ ВНУТРЕННИХ ВЕРШИН)
// ============================================================================
//
// Пусть Ω — область, для которой построена триангуляция. Вершина называется
// внутренней, если все прилегающие к ней треугольники полностью лежат внутри Ω.
//
// 3.1. Линейная функция u(x,y) = ax + by + c
//      Для ВНУТРЕННЕЙ вершины: (L u)_i = 0.
//      Это точное свойство котангенсного лапласиана на замкнутом веере
//      треугольников: для любой линейной функции сумма взвешенных разностей
//      с соседями равна нулю. Проверено тестом LinearFunctionZeroForInteriorVertex.
//
// 3.2. Квадратичная функция u(x,y) = x² + y²
//      На ВНУТРЕННЕЙ вершине точное значение (L u)_i НЕ равно непрерывному
//      лапласиану Δu = 4. Дискретный оператор L без нормировки на массу
//      выдаёт величину, зависящую от локальной геометрии сетки.
//      Для конкретной сетки (квадрат, разбитый на 4 равных прямоугольных
//      треугольника вокруг центра) аналитически получается:
//        веса рёбер от центра к углам равны 1,
//        u_center = 0.5, u_углов = 0,1,2,1,
//        (L u)_center = Σ 1·(0.5 - u_угол) = -2.
//      Именно это значение ожидается тестом QuadraticFunctionConstantLaplacianForInterior.
//      При измельчении сетки L u будет сходиться к Δu, умноженному на локальную
//      площадь дуальной ячейки, и в пределе (M^{-1} L u) → 4.
//
// 3.3. Используемая сетка
//      Сетка make_square_with_center_mesh: квадрат [0,1]×[0,1], вершины (0,0),
//      (1,0), (1,1), (0,1) и центр (0.5,0.5), 4 треугольника. Центральная
//      вершина — внутренняя. Все котангенсы вычисляются точно (углы 45° и 90°).
//
// ============================================================================
// 4. ПОЧЕМУ ТЕСТЫ НА СТАРОЙ СЕТКЕ (КВАДРАТ ИЗ ДВУХ ТРЕУГОЛЬНИКОВ) БЫЛИ НЕВЕРНЫ
// ============================================================================
//
// В квадрате, разбитом одной диагональю, ВСЕ 4 вершины лежат на границе.
// Для граничной вершины:
//   - Линейная функция НЕ обязана давать 0.
//   - Матрица L имеет диагональные элементы = 1, внедиагональные = -0.5,
//     а не 2 и -1, как ожидал исходный тест.
//   - Ожидания теста противоречили математике.
//
// ============================================================================
// 5. МАТРИЦА МАСС (Lumped Mass Matrix)
// ============================================================================
//
// Диагональная матрица M с M_{ii} = площадь дуальной ячейки вершины i (объём
// барицентрической дуальной клетки). Для любой невырожденной сетки M_{ii} > 0.
// Проверяется тестом LumpedMassMatrixPositive.
//
// ============================================================================
// 6. ЧТО НЕ ТЕСТИРУЕТСЯ (И ПОЧЕМУ)
// ============================================================================
//
// - Явные значения матрицы для произвольной сетки — бессмысленно, так как они
//   зависят от геометрии. Достаточно проверять структурные свойства.
// - Сходимость к непрерывному лапласиану при измельчении — требует отдельного
//   теста с последовательностью сеток (convergence test).
// - Учёт метрики: текущая реализация использует метрику через edge_length;
//   корректность для неевклидовых метрик не проверяется, но предполагается.
//
// ============================================================================
// 7. ЗАКЛЮЧЕНИЕ
// ============================================================================
//
// Набор тестов покрывает:
//   ✅ Алгебраические инварианты (симметрия, сумма строк = 0).
//   ✅ Спектральное свойство (константа в ядре).
//   ✅ Точное поведение на внутренних вершинах для линейных и квадратичных
//      функций, с аналитически вычисленными ожидаемыми значениями.
//   ✅ Положительность матрицы масс.
//
// Все тесты строго следуют математическому определению котангенсного лапласиана
// и не зависят от конкретной реализации приближённых вычислений.
//
// ============================================================================

#include <gtest/gtest.h>
#include <Eigen/Sparse>
#include "delta/numerical/cotangent_laplacian.h"
#include "../test_fixtures_geometry_numerical.h"

namespace delta::testing {

    class CotangentLaplacianTest : public GeometryNumericalTest {
    protected:
        using Scalar = Rational;
        using Point2D = Point<2>;
        using Complex2D = Complex<2>;

        // Сетка квадрата, разбитого на 4 треугольника (с центральной вершиной)
        Complex2D make_square_with_center_mesh() {
            Complex2D mesh;
            auto v0 = add_vertex(mesh, Point2D(0_r, 0_r));
            auto v1 = add_vertex(mesh, Point2D(1_r, 0_r));
            auto v2 = add_vertex(mesh, Point2D(1_r, 1_r));
            auto v3 = add_vertex(mesh, Point2D(0_r, 1_r));
            auto vc = add_vertex(mesh, Point2D(1_r / 2_r, 1_r / 2_r));

            add_edge(mesh, v0, v1); add_edge(mesh, v1, v2);
            add_edge(mesh, v2, v3); add_edge(mesh, v3, v0);
            add_edge(mesh, v0, vc); add_edge(mesh, v1, vc);
            add_edge(mesh, v2, vc); add_edge(mesh, v3, vc);

            add_triangle(mesh, v0, v1, vc);
            add_triangle(mesh, v1, v2, vc);
            add_triangle(mesh, v2, v3, vc);
            add_triangle(mesh, v3, v0, vc);
            return mesh;
        }

        // Старая сетка (только для тестов, не требующих внутренних вершин)
        Complex2D make_unit_square_triangulation() {
            Complex2D mesh;
            auto v0 = add_vertex(mesh, Point2D(0_r, 0_r));
            auto v1 = add_vertex(mesh, Point2D(1_r, 0_r));
            auto v2 = add_vertex(mesh, Point2D(1_r, 1_r));
            auto v3 = add_vertex(mesh, Point2D(0_r, 1_r));
            add_edge(mesh, v0, v1);
            add_edge(mesh, v1, v2);
            add_edge(mesh, v2, v3);
            add_edge(mesh, v3, v0);
            add_edge(mesh, v0, v2);
            add_triangle(mesh, v0, v1, v2);
            add_triangle(mesh, v0, v2, v3);
            return mesh;
        }

        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> to_dense(const Eigen::SparseMatrix<Scalar>& sparse) {
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> dense(sparse.rows(), sparse.cols());
            dense.setZero();
            for (int k = 0; k < sparse.outerSize(); ++k) {
                for (Eigen::SparseMatrix<Scalar>::InnerIterator it(sparse, k); it; ++it) {
                    dense(it.row(), it.col()) = it.value();
                }
            }
            return dense;
        }
    };

    // L должна быть симметричной (для любой сетки)
    TEST_F(CotangentLaplacianTest, Symmetry) {
        auto mesh = make_unit_square_triangulation();
        EuclideanMetric metric;
        auto L = build_cotangent_laplacian(mesh, metric);
        auto L_dense = to_dense(L);
        std::size_t n = mesh.num_vertices();
        Scalar eps = Rational(1, 1000000);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                EXPECT_RATIONAL_NEAR(L_dense(i, j), L_dense(j, i), eps);
            }
        }
    }

    // Сумма элементов каждой строки = 0 (для любой сетки)
    TEST_F(CotangentLaplacianTest, RowSumZero) {
        auto mesh = make_unit_square_triangulation();
        EuclideanMetric metric;
        auto L = build_cotangent_laplacian(mesh, metric);
        auto L_dense = to_dense(L);
        std::size_t n = mesh.num_vertices();
        Scalar eps = Rational(1, 1000000);
        for (std::size_t i = 0; i < n; ++i) {
            Scalar row_sum = 0;
            for (std::size_t j = 0; j < n; ++j) row_sum += L_dense(i, j);
            EXPECT_RATIONAL_NEAR(row_sum, 0_r, eps);
        }
    }

    // Константная функция в ядре (для любой сетки)
    TEST_F(CotangentLaplacianTest, ConstantFunctionKernel) {
        auto mesh = make_unit_square_triangulation();
        EuclideanMetric metric;
        auto L = build_cotangent_laplacian(mesh, metric);
        std::size_t n = mesh.num_vertices();
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> ones(n);
        ones.setOnes();
        auto L_ones = L * ones;
        Scalar eps = Rational(1, 1000000);
        for (std::size_t i = 0; i < n; ++i) {
            EXPECT_RATIONAL_NEAR(L_ones(i), 0_r, eps);
        }
    }

    // Линейная функция – ноль только для внутренних вершин.
    // Используем сетку с центральной вершиной (внутренняя) и проверяем, что (L x)_center = 0.
    // Точное свойство для данной сетки: сумма весовых разностей с соседями равна 0.
    TEST_F(CotangentLaplacianTest, LinearFunctionZeroForInteriorVertex) {
        auto mesh = make_square_with_center_mesh();
        EuclideanMetric metric;
        auto L = build_cotangent_laplacian(mesh, metric);
        std::size_t n = mesh.num_vertices();      // n = 5
        std::size_t interior = 4;                 // индекс центральной вершины
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> f(n);
        for (std::size_t i = 0; i < n; ++i) f(i) = mesh.vertex(i).x(); // f(x,y)=x
        auto Lf = L * f;
        Scalar eps = Rational(1, 1000000);
        EXPECT_RATIONAL_NEAR(Lf(interior), 0_r, eps);
    }

    // Квадратичная функция x²+y² на внутренней вершине.
    // Для данной сетки с четырьмя симметричными треугольниками точное значение L(x²+y²) в центре = -2,
    // потому что веса рёбер центр-угол равны 1, u_center = 0.5, u_углов = 0,1,2,1 и сумма = -2.
    // Это не непрерывный лапласиан (Δ=4); дискретный оператор даёт величину, зависящую от размера ячейки.
    TEST_F(CotangentLaplacianTest, QuadraticFunctionConstantLaplacianForInterior) {
        auto mesh = make_square_with_center_mesh();
        EuclideanMetric metric;
        auto L = build_cotangent_laplacian(mesh, metric);
        std::size_t n = mesh.num_vertices();
        std::size_t interior = 4;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> f(n);
        for (std::size_t i = 0; i < n; ++i) {
            const auto& p = mesh.vertex(i);
            f(i) = p.x() * p.x() + p.y() * p.y();
        }
        auto Lf = L * f;
        Scalar eps = Rational(1, 1000000);
        EXPECT_RATIONAL_NEAR(Lf(interior), -2_r, eps);
    }

    // Lumped mass matrix – диагональные элементы положительны.
    TEST_F(CotangentLaplacianTest, LumpedMassMatrixPositive) {
        auto mesh = make_unit_square_triangulation();
        auto M = build_lumped_mass_matrix(mesh);
        auto M_dense = to_dense(M);
        for (std::size_t i = 0; i < mesh.num_vertices(); ++i) {
            EXPECT_GT(M_dense(i, i), 0_r);
        }
    }

} // namespace delta::testing