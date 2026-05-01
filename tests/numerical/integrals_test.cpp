// tests/numerical/integrals_test.cpp
// Fully specifies the mathematical contracts for integrals.h (Stage 1, block A8)
// according to the General Plan.
//
// Contracts verified:
// 1. grid_cell_volume() returns the correct measure (length/area) associated with a grid point,
//    with proper handling of boundaries (half cells at edges/corners) and product grids.
// 2. grid_integral() computes the weighted sum f(x_i) * volume_i and is exact for linear functions
//    on any grid, and converges for higher-order functions with grid refinement.
// 3. summation_by_parts_1d holds algebraically for discrete functions.
// 4. Green's first and second identities hold in 1D and 2D for arbitrary discrete functions,
//    including zero boundary conditions.

// ============================================================================
// TODO: Корректная проверка тождеств Грина в 2D (30.04.2026)
// ============================================================================
//
// ТЕКУЩЕЕ СОСТОЯНИЕ
// -----------------
// Тесты GreenFirstIdentity, GreenFirstZeroBoundary, GreenSecondIdentity,
// GreenSecondZeroBoundary в классе Integrals2DTest НЕ ВЫПОЛНЯЮТ РЕАЛЬНУЮ
// ПРОВЕРКУ тождеств Грина.
//
// Причина:
//   Функции check_green_first_2d() и check_green_second_2d() в
//   integrals.h не вычисляют граничный член независимо, а определяют
//   его как разность левой и объёмной частей, гарантируя тривиальное
//   равенство. Это делает тесты бесполезными для обнаружения ошибок
//   дискретизации.
//
//   В первом тождестве граничный член (∫ f ∇g·n ds) подгоняется под
//   уже вычисленные величины. Во втором тождестве проверка вырождается
//   в проверку симметрии матрицы жёсткости (K^T = K).
//
// НЕОБХОДИМЫЕ ИСПРАВЛЕНИЯ
// -----------------------
// 1. Реализовать независимое вычисление граничного интеграла:
//    - Для прямоугольной сетки (ProductGrid<UniformGrid,2>) получить
//      список граничных рёбер (все рёбра, принадлежащие только одной
//      ячейке, т.е. лежащие на ∂Ω).
//    - На каждом граничном ребре вычислить ∇g·n (используя значения g
//      в прилегающих узлах и, возможно, интерполяцию) и f в середине
//      ребра (или интеграл от f вдоль ребра).
//    - Просуммировать вклады с учётом ориентации.
//
// 2. Обеспечить согласованность с метрикой:
//    - Все геометрические величины (длина ребра, нормаль) должны
//      вычисляться через переданный объект Metric, а не в предположении
//      евклидовой метрики.
//
// 3. Добавить тесты сходимости:
//    - Для последовательности измельчающихся сеток (например,
//      4×4, 8×8, 16×16) вычислять невязку тождества Грина.
//    - Ожидаемый порядок сходимости для билинейных элементов должен
//      быть ~ O(h^2) для первого тождества (при условии точного
//      вычисления граничного члена).
//
// 4. Унифицировать с 1D‑реализацией:
//    - check_green_first_1d() уже делает независимый расчёт граничного
//      члена. Обобщить подход на 2D, возможно, через общий шаблонный
//      код с использованием ProductGrid или симплициальных комплексов.
//
// 5. (Опционально) Для второго тождества Грина:
//    - Кроме проверки симметрии, настоящий тест должен вычислить
//      ∫ (f Δg - g Δf) dV и соответствующий граничный член
//      ∫ (f ∇g·n - g ∇f·n) dS и убедиться в их равенстве.
//    - На равномерной сетке ожидаемое значение равно 0 и для
//      объёмной, и для граничной частей, но они должны быть вычислены
//      независимо.
//
// ПЛАН ДЕЙСТВИЙ (приоритеты)
// ---------------------------
// - [ ] Создать функцию compute_boundary_integral_first_2d(...) в integrals.h,
//       вычисляющую ∫_∂Ω f (∇g·n) ds по граничным рёбрам.
// - [ ] Изменить check_green_first_2d, чтобы она использовала эту функцию
//       и проверяла равенство левой части, объёмной правой части и
//       граничного члена с заданным допуском (tolerance).
// - [ ] Аналогично для check_green_second_2d.
// - [ ] Написать тест сходимости для первого тождества Грина (ConvergenceGreenFirst2D)
//       и добавить его в Integrals2DTest.
// - [ ] Проверить работу на неравномерных и product‑mixed сетках (если актуально).
//
// ВАЖНО:
//   До выполнения этих пунктов тесты тождеств Грина в 2D следует считать
//   ЗАГЛУШКАМИ и не использовать для верификации численных схем.
// ============================================================================
#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <cmath>
#include "delta/core/uniform_grid.h"
#include "delta/core/list_grid.h"
#include "delta/core/product_grid.h"
#include "delta/geometry/tensor_field.h"
#include "delta/numerical/integrals.h"
#include "delta/rational/literals.h"
#include "../test_fixtures_geometry_numerical.h"

namespace delta::testing {

    // -------------------------------------------------------------------------
    // 1D tests
    // -------------------------------------------------------------------------
    class Integrals1DTest : public GeometryNumericalTest {
    protected:
        using Grid1DUniform = delta::UniformGrid<Rational, std::less<Rational>>;
        using Grid1DList = delta::ListGrid<Rational, std::less<Rational>>;
        using ScalarField1D = delta::geometry::TensorField<Rational, Scalar, 0, 1, std::less<Rational>>;

        EuclideanMetric metric;

        Grid1DUniform make_uniform_grid(std::size_t n, Rational a = 0_r, Rational b = 1_r) {
            Rational step = (b - a) / (n - 1);
            return Grid1DUniform(a, step, n);
        }

        Grid1DList make_nonuniform_grid(const std::vector<Rational>& points) {
            return Grid1DList(points.begin(), points.end(), std::less<Rational>{});
        }
    };

    TEST_F(Integrals1DTest, CellVolumeUniform) {
        auto grid = make_uniform_grid(5);
        Rational h = 1_r / 4_r;

        EXPECT_EQ(grid_cell_volume(grid, 0, metric), h / 2_r);
        EXPECT_EQ(grid_cell_volume(grid, 4, metric), h / 2_r);
        for (std::size_t i = 1; i <= 3; ++i) {
            EXPECT_EQ(grid_cell_volume(grid, i, metric), h);
        }
        Rational total = 0_r;
        for (std::size_t i = 0; i < grid.size(); ++i) total += grid_cell_volume(grid, i, metric);
        EXPECT_EQ(total, 1_r);
    }

    TEST_F(Integrals1DTest, CellVolumeNonUniform) {
        std::vector<Rational> points = { 0_r, "2/10"_r, "5/10"_r, 1_r };
        auto grid = make_nonuniform_grid(points);
        EXPECT_EQ(grid_cell_volume(grid, 0, metric), "1/10"_r);
        EXPECT_EQ(grid_cell_volume(grid, 1, metric), "1/4"_r);
        EXPECT_EQ(grid_cell_volume(grid, 2, metric), "2/5"_r);
        EXPECT_EQ(grid_cell_volume(grid, 3, metric), "1/4"_r);
        Rational total = 0_r;
        for (std::size_t i = 0; i < grid.size(); ++i) total += grid_cell_volume(grid, i, metric);
        EXPECT_EQ(total, 1_r);
    }

    TEST_F(Integrals1DTest, IntegralLinearExact) {
        auto grid = make_uniform_grid(5);
        ScalarField1D f(grid);
        for (std::size_t i = 0; i < grid.size(); ++i) {
            Rational x = grid[i];
            f.set(x, x);
        }
        Rational I = grid_integral(grid, [&f](const Rational& x) { return f.at(x); }, metric);
        EXPECT_RATIONAL_NEAR(I, "1/2"_r, delta::default_eps());
    }

    TEST_F(Integrals1DTest, IntegralQuadraticConvergence) {
        std::vector<std::size_t> ns = { 5, 9, 17, 33 };
        std::vector<Rational> errors;
        Rational exact = 1_r / 3_r;

        for (std::size_t n : ns) {
            auto grid = make_uniform_grid(n);
            ScalarField1D f(grid);
            for (std::size_t i = 0; i < grid.size(); ++i) {
                Rational x = grid[i];
                f.set(x, x * x);
            }
            Rational I = grid_integral(grid, [&f](const Rational& x) { return f.at(x); }, metric);
            Rational err = delta::abs(I - exact);
            errors.push_back(err);
        }
        ASSERT_GE(errors.size(), 2);
        for (std::size_t i = 0; i < errors.size() - 1; ++i) {
            Rational ratio = errors[i] / errors[i + 1];
            EXPECT_TRUE(ratio > 3_r && ratio < 5_r);
        }
    }

    TEST_F(Integrals1DTest, SummationByParts) {
        auto grid = make_uniform_grid(5);
        ScalarField1D f(grid), g(grid);
        for (std::size_t i = 0; i < grid.size(); ++i) {
            Rational x = grid[i];
            f.set(x, x);
            g.set(x, x * x);
        }
        Rational g_right = g.at(1_r);
        bool ok = check_summation_by_parts_1d(grid, f, g, metric, g_right, "1/1000000000000"_r);
        EXPECT_TRUE(ok);
    }

    TEST_F(Integrals1DTest, SummationByPartsZeroBoundary) {
        auto grid = make_uniform_grid(5);
        ScalarField1D f(grid), g(grid);
        for (std::size_t i = 0; i < grid.size(); ++i) {
            Rational x = grid[i];
            f.set(x, x);
            g.set(x, x * (1_r - x));
        }
        Rational g_right = 0_r;
        bool ok = check_summation_by_parts_1d(grid, f, g, metric, g_right, "1/1000000000000"_r);
        EXPECT_TRUE(ok);
    }

    TEST_F(Integrals1DTest, GreenFirstIdentity) {
        auto grid = make_uniform_grid(9);
        ScalarField1D f(grid), g(grid);
        for (std::size_t i = 0; i < grid.size(); ++i) {
            Rational x = grid[i];
            f.set(x, x);
            g.set(x, x * x);
        }
        bool ok = check_green_first_1d(grid, f, g, metric, "1/1000000000000"_r);
        EXPECT_TRUE(ok);
    }

    // -------------------------------------------------------------------------
    // 2D tests
    // -------------------------------------------------------------------------
    struct MaxMetric {
        template<typename T, std::size_t N>
        auto operator()(const std::array<T, N>& a, const std::array<T, N>& b) const {
            T max_diff = 0;
            for (std::size_t i = 0; i < N; ++i) {
                T diff = a[i] - b[i];
                if (diff < 0) diff = -diff;
                if (diff > max_diff) max_diff = diff;
            }
            return max_diff;
        }
    };

    class Integrals2DTest : public GeometryNumericalTest {
    protected:
        using Grid1D = delta::UniformGrid<Rational, std::less<Rational>>;
        using Grid2D = delta::ProductGrid<Grid1D, 2>;
        using Addr2D = typename Grid2D::value_type;

        struct Addr2DCompare {
            bool operator()(const Addr2D& a, const Addr2D& b) const {
                if (a[0] < b[0]) return true;
                if (b[0] < a[0]) return false;
                return a[1] < b[1];
            }
        };

        using ScalarField2D = delta::geometry::TensorField<Addr2D, Scalar, 0, 2, Addr2DCompare>;

        Grid2D make_uniform_grid_2d(std::size_t nx, std::size_t ny,
            Rational a = 0_r, Rational b = 1_r,
            Rational c = 0_r, Rational d = 1_r) {
            Grid1D gx(a, (b - a) / (nx - 1), nx);
            Grid1D gy(c, (d - c) / (ny - 1), ny);
            return Grid2D({ gx, gy });
        }

        EuclideanMetric metric;
    };

    TEST_F(Integrals2DTest, CellVolumeUniform) {
        auto grid = make_uniform_grid_2d(4, 4);
        Rational hx = 1_r / 3_r, hy = 1_r / 3_r;
        Addr2D inner = { hx, hy };
        std::size_t inner_idx = 0;
        for (std::size_t i = 0; i < grid.size(); ++i) {
            if (grid[i] == inner) { inner_idx = i; break; }
        }
        EXPECT_EQ(grid_cell_volume(grid, inner_idx, metric), hx * hy);

        Addr2D corner = { 0_r, 0_r };
        std::size_t corner_idx = 0;
        for (std::size_t i = 0; i < grid.size(); ++i) {
            if (grid[i] == corner) { corner_idx = i; break; }
        }
        EXPECT_EQ(grid_cell_volume(grid, corner_idx, metric), (hx / 2_r) * (hy / 2_r));
    }

    TEST_F(Integrals2DTest, CellVolumeNonUniformProduct) {
        auto grid = make_uniform_grid_2d(5, 3, 0_r, 1_r, 0_r, 1_r);
        Rational hx = "1/4"_r, hy = "1/2"_r;
        Addr2D inner = { "1/2"_r, "1/2"_r };
        std::size_t inner_idx = 0;
        for (std::size_t i = 0; i < grid.size(); ++i) {
            if (grid[i] == inner) { inner_idx = i; break; }
        }
        EXPECT_EQ(grid_cell_volume(grid, inner_idx, metric), hx * hy);
        Rational total = 0_r;
        for (std::size_t i = 0; i < grid.size(); ++i) total += grid_cell_volume(grid, i, metric);
        EXPECT_EQ(total, 1_r);
    }

    TEST_F(Integrals2DTest, IntegralLinearExact) {
        auto grid = make_uniform_grid_2d(5, 5);
        ScalarField2D f(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1];
            f.set(addr, x + y);
        }
        Rational I = grid_integral(grid, [&f](const Addr2D& a) { return f.at(a); }, metric);
        EXPECT_RATIONAL_NEAR(I, 1_r, delta::default_eps());
    }

    TEST_F(Integrals2DTest, GreenFirstIdentity) {
        auto grid = make_uniform_grid_2d(9, 9);
        ScalarField2D f(grid), g(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1];
            f.set(addr, x * x + y * y);
            g.set(addr, x + y);
        }
        bool ok = check_green_first_2d(grid, f, g, metric, "1/1000000000000"_r);
        EXPECT_TRUE(ok);
    }

    TEST_F(Integrals2DTest, GreenFirstZeroBoundary) {
        auto grid = make_uniform_grid_2d(9, 9);
        ScalarField2D f(grid), g(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1];
            Rational g_val = x * (1_r - x) * y * (1_r - y);
            g.set(addr, g_val);
            f.set(addr, x * x + y * y);
        }
        bool ok = check_green_first_2d(grid, f, g, metric, "1/1000000000000"_r);
        EXPECT_TRUE(ok);
    }

    TEST_F(Integrals2DTest, GreenSecondIdentity) {
        auto grid = make_uniform_grid_2d(9, 9);
        ScalarField2D f(grid), g(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1];
            f.set(addr, x * x + y * y);
            g.set(addr, x + y);
        }
        bool ok = check_green_second_2d(grid, f, g, metric, "1/1000000000000"_r);
        EXPECT_TRUE(ok);
    }

    TEST_F(Integrals2DTest, GreenSecondZeroBoundary) {
        auto grid = make_uniform_grid_2d(9, 9);
        ScalarField2D f(grid), g(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1];
            f.set(addr, x * (1_r - x) * y * (1_r - y));
            g.set(addr, x * x + y * y);
        }
        bool ok = check_green_second_2d(grid, f, g, metric, "1/1000000000000"_r);
        EXPECT_TRUE(ok);
    }

    // -------------------------------------------------------------------------
    // Mixed grid tests
    // -------------------------------------------------------------------------
    class Integrals2DProductMixedTest : public GeometryNumericalTest {
    protected:
        using Grid1DList = delta::ListGrid<Rational, std::less<Rational>>;
        using Grid2D = delta::ProductGrid<Grid1DList, 2>;
        using Addr2D = typename Grid2D::value_type;

        struct Addr2DCompare {
            bool operator()(const Addr2D& a, const Addr2D& b) const {
                if (a[0] < b[0]) return true;
                if (b[0] < a[0]) return false;
                return a[1] < b[1];
            }
        };

        using ScalarField2D = delta::geometry::TensorField<Addr2D, Scalar, 0, 2, Addr2DCompare>;

        Grid2D make_mixed_grid() {
            std::vector<Rational> xs = { 0_r, "2/10"_r, "5/10"_r, 1_r };
            std::vector<Rational> ys = { 0_r, "3/10"_r, "7/10"_r, 1_r };
            Grid1DList gx(xs.begin(), xs.end(), std::less<Rational>{});
            Grid1DList gy(ys.begin(), ys.end(), std::less<Rational>{});
            return Grid2D({ gx, gy });
        }

        EuclideanMetric metric;
    };

    TEST_F(Integrals2DProductMixedTest, CellVolumeProduct) {
        auto grid = make_mixed_grid();
        Addr2D p = { "2/10"_r, "3/10"_r };
        std::size_t idx = 0;
        for (std::size_t i = 0; i < grid.size(); ++i) {
            if (grid[i] == p) { idx = i; break; }
        }
        Rational expected = "1/4"_r * "7/20"_r;
        EXPECT_EQ(grid_cell_volume(grid, idx, metric), expected);
        Rational total = 0_r;
        for (std::size_t i = 0; i < grid.size(); ++i) total += grid_cell_volume(grid, i, metric);
        EXPECT_EQ(total, 1_r);
    }

    TEST_F(Integrals2DProductMixedTest, IntegralQuadraticConvergence) {
        Rational exact = 2_r / 3_r;
        std::vector<std::size_t> ns = { 4, 8, 16 };
        std::vector<Rational> errors;
        for (std::size_t n : ns) {
            std::vector<Rational> xs(n);
            std::vector<Rational> ys(n);
            for (std::size_t i = 0; i < n; ++i) {
                xs[i] = Rational(static_cast<long long>(i)) / (n - 1);
                ys[i] = xs[i];
            }
            Grid1DList gx(xs.begin(), xs.end(), std::less<Rational>{});
            Grid1DList gy(ys.begin(), ys.end(), std::less<Rational>{});
            Grid2D grid({ gx, gy });
            ScalarField2D f(grid);
            for (const auto& addr : grid) {
                Rational x = addr[0], y = addr[1];
                f.set(addr, x * x + y * y);
            }
            Rational I = grid_integral(grid, [&f](const Addr2D& a) { return f.at(a); }, metric);
            Rational err = delta::abs(I - exact);
            errors.push_back(err);
        }
        ASSERT_GE(errors.size(), 2);
        for (std::size_t i = 0; i < errors.size() - 1; ++i) {
            Rational ratio = errors[i] / errors[i + 1];
            EXPECT_TRUE(ratio > 2_r && ratio < 6_r);
        }
    }

} // namespace delta::testing