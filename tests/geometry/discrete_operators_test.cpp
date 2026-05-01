// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/numerical/discrete_operators_test.cpp
#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <cmath>
#include "delta/numerical/discrete_operators.h"
#include "delta/core/uniform_grid.h"
#include "delta/core/product_grid.h"
#include "delta/geometry/tensor_field.h"
#include "../test_fixtures_geometry_numerical.h"

namespace delta::testing{
    using namespace delta::geometry;

    // Определяем метрику для массивов вне классов, чтобы она была доступна везде
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

    // -------------------------------------------------------------------------
    // 1D tests
    // -------------------------------------------------------------------------
    class DiscreteOperators1DTest : public GeometryNumericalTest {
    protected:
        using Grid1D = delta::UniformGrid<Rational, std::less<Rational>>;
        using ScalarField1D = delta::geometry::TensorField<Rational, Scalar, 0, 1, std::less<Rational>>;
        using VecField1D = delta::geometry::TensorField<Rational, Scalar, 1, 1, std::less<Rational>>;

        Grid1D make_grid_1d(std::size_t n) {
            return Grid1D(0_r, 1_r / (n - 1), n);
        }

        EuclideanMetric metric;
    };

    TEST_F(DiscreteOperators1DTest, ForwardDifference) {
        auto grid = make_grid_1d(5);
        ScalarField1D f(grid);
        for (std::size_t i = 0; i < grid.size(); ++i) {
            f.set(grid[i], grid[i]); // f(x)=x
        }
        auto df = forward_difference(grid, f, metric, "0.25"_r);
        EXPECT_RATIONAL_NEAR(df, 1_r, delta::default_eps());
        EXPECT_THROW(forward_difference(grid, f, metric, 1_r), std::out_of_range);
    }

    TEST_F(DiscreteOperators1DTest, BackwardDifference) {
        auto grid = make_grid_1d(5);
        ScalarField1D f(grid);
        for (std::size_t i = 0; i < grid.size(); ++i) {
            f.set(grid[i], grid[i]);
        }
        auto df = backward_difference(grid, f, metric, "0.25"_r);
        EXPECT_RATIONAL_NEAR(df, 1_r, delta::default_eps());
        EXPECT_THROW(backward_difference(grid, f, metric, 0_r), std::out_of_range);
    }

    TEST_F(DiscreteOperators1DTest, CentralDifference) {
        auto grid = make_grid_1d(5);
        ScalarField1D f(grid);
        for (std::size_t i = 0; i < grid.size(); ++i) {
            f.set(grid[i], grid[i]);
        }
        auto df = central_difference(grid, f, metric, "0.25"_r);
        EXPECT_RATIONAL_NEAR(df, 1_r, delta::default_eps());
        EXPECT_THROW(central_difference(grid, f, metric, 0_r), std::out_of_range);
        EXPECT_THROW(central_difference(grid, f, metric, 1_r), std::out_of_range);
    }

    TEST_F(DiscreteOperators1DTest, Gradient) {
        auto grid = make_grid_1d(5);
        ScalarField1D f(grid);
        for (std::size_t i = 0; i < grid.size(); ++i) {
            Rational x = grid[i];
            f.set(x, x * x); // f(x)=x^2
        }
        auto grad = discrete_gradient(grid, f, metric, DifferenceScheme::CENTRAL);
        for (const auto& x : grid) {
            if (x == 0_r || x == 1_r) continue; // skip boundary
            auto g = grad.at(x);
            EXPECT_RATIONAL_NEAR(g[0], 2 * x, delta::default_eps());
        }
    }

    TEST_F(DiscreteOperators1DTest, Divergence) {
        auto grid = make_grid_1d(5);
        VecField1D v(grid);
        for (std::size_t i = 0; i < grid.size(); ++i) {
            Eigen::Matrix<Scalar, 1, 1> val;
            val << 1_r;
            v.set(grid[i], val);
        }
        auto div = discrete_divergence(grid, v, metric, DifferenceScheme::CENTRAL);
        for (const auto& x : grid) {
            EXPECT_RATIONAL_NEAR(div.at(x), 0_r, delta::default_eps());
        }
    }

    // ============================================================================
// Точные полиномиальные тесты для 1D
// ============================================================================

    TEST_F(DiscreteOperators1DTest, GradientQuadraticExact) {
        auto grid = make_grid_1d(5);
        ScalarField1D f(grid);
        for (std::size_t i = 0; i < grid.size(); ++i) {
            Rational x = grid[i];
            f.set(x, x * x);
        }
        auto grad = discrete_gradient(grid, f, metric, DifferenceScheme::CENTRAL);
        for (const auto& x : grid) {
            if (x == 0_r || x == 1_r) continue;          // пропускаем границу
            auto g = grad.at(x);
            EXPECT_RATIONAL_NEAR(g[0], 2 * x, delta::default_eps());
        }
    }

    TEST_F(DiscreteOperators1DTest, LaplacianCubicExact) {
        auto grid = make_grid_1d(5);
        ScalarField1D f(grid);
        for (std::size_t i = 0; i < grid.size(); ++i) {
            Rational x = grid[i];
            f.set(x, x * x * x);
        }
        auto lap = discrete_laplacian(grid, f, metric);
        for (const auto& x : grid) {
            if (x == 0_r || x == 1_r) continue;
            EXPECT_RATIONAL_NEAR(lap.at(x), 6 * x, delta::default_eps());
        }
    }



    // -------------------------------------------------------------------------
    // 2D tests using ProductGrid
    // -------------------------------------------------------------------------
    class DiscreteOperators2DTest : public GeometryNumericalTest {
    protected:
        using Grid1D = delta::UniformGrid<Rational, std::less<Rational>>;
        using Grid2D = delta::ProductGrid<Grid1D, 2>;
        using Addr2D = typename Grid2D::value_type; // std::array<Rational,2>

        struct Addr2DCompare {
            bool operator()(const Addr2D& a, const Addr2D& b) const {
                if (a[0] < b[0]) return true;
                if (b[0] < a[0]) return false;
                return a[1] < b[1];
            }
        };

        using ScalarField2D = delta::geometry::TensorField<Addr2D, Scalar, 0, 2, Addr2DCompare>;
        using VecField2D = delta::geometry::TensorField<Addr2D, Scalar, 1, 2, Addr2DCompare>;

        Grid2D make_grid_2d(std::size_t n) {
            Grid1D gx(0_r, 1_r / (n - 1), n);
            Grid1D gy(0_r, 1_r / (n - 1), n);
            return Grid2D({ gx, gy });
        }

        MaxMetric max_metric;  // используем глобально определённую метрику
    };

    TEST_F(DiscreteOperators2DTest, GradientOfQuadratic) {
        auto grid = make_grid_2d(5); // 5x5
        ScalarField2D f(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1];
            f.set(addr, x * x + y * y);
        }
        auto grad = discrete_gradient(grid, f, max_metric, DifferenceScheme::CENTRAL);

        for (const auto& addr : grid) {
            if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r)
                continue; // skip boundary
            Rational x = addr[0], y = addr[1];
            auto g = grad.at(addr);
            EXPECT_RATIONAL_NEAR(g[0], 2 * x, delta::default_eps());
            EXPECT_RATIONAL_NEAR(g[1], 2 * y, delta::default_eps());
        }
    }

    TEST_F(DiscreteOperators2DTest, LaplacianOfQuadratic) {
        auto grid = make_grid_2d(5);
        ScalarField2D f(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1];
            f.set(addr, x * x + y * y);
        }
        auto lap = discrete_laplacian(grid, f, max_metric);
        for (const auto& addr : grid) {
            if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r)
                continue; // skip boundary
            EXPECT_RATIONAL_NEAR(lap.at(addr), 4_r, delta::default_eps());
        }
    }

    TEST_F(DiscreteOperators2DTest, CurlGradIsZero) {
        auto grid = make_grid_2d(5);
        ScalarField2D f(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1];
            f.set(addr, x * x + y * y);
        }
        auto grad = discrete_gradient(grid, f, max_metric, DifferenceScheme::CENTRAL);
        auto curl_grad = discrete_curl_2d(grid, grad, max_metric, DifferenceScheme::CENTRAL);
        for (const auto& addr : grid) {
            if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r)
                continue;
            EXPECT_RATIONAL_NEAR(curl_grad.at(addr), 0_r, delta::default_eps());
        }
    }

    // -------------------------------------------------------------------------
    // Convergence tests (order verification)
    // -------------------------------------------------------------------------
    class DiscreteOperatorsConvergenceTest : public GeometryNumericalTest {
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

        Grid2D make_grid_2d(std::size_t n) {
            Grid1D gx(0_r, 1_r / (n - 1), n);
            Grid1D gy(0_r, 1_r / (n - 1), n);
            return Grid2D({ gx, gy });
        }

        MaxMetric max_metric;
    };

    TEST_F(DiscreteOperatorsConvergenceTest, GradientSecondOrder) {
        std::vector<std::size_t> ns = { 5, 9, 17 };
        std::vector<double> errors;
        for (std::size_t n : ns) {
            auto grid = make_grid_2d(n);
            ScalarField2D f(grid);
            for (const auto& addr : grid) {
                Rational x = addr[0], y = addr[1];
                f.set(addr, x * x * x + y * y * y);   // f = x^3 + y^3
            }
            auto grad_num = discrete_gradient(grid, f, max_metric, DifferenceScheme::CENTRAL);
            double max_err = 0.0;
            for (const auto& addr : grid) {
                if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r)
                    continue;
                auto g = grad_num.at(addr);
                Rational x = addr[0], y = addr[1];
                double err_x = std::abs((g[0] - 3 * x * x).convert_to<double>());
                double err_y = std::abs((g[1] - 3 * y * y).convert_to<double>());
                max_err = std::max(max_err, std::max(err_x, err_y));
            }
            errors.push_back(max_err);
        }
        ASSERT_GE(errors.size(), 2);
        for (std::size_t i = 0; i < errors.size() - 1; ++i) {
            double ratio = errors[i] / errors[i + 1];
            EXPECT_NEAR(ratio, 4.0, 1.0); // ожидаем второй порядок
        }
    }

    TEST_F(DiscreteOperatorsConvergenceTest, LaplacianSecondOrder) {
        std::vector<std::size_t> ns = { 5, 9, 17 };
        std::vector<double> errors;
        for (std::size_t n : ns) {
            auto grid = make_grid_2d(n);
            ScalarField2D f(grid);
            for (const auto& addr : grid) {
                Rational x = addr[0], y = addr[1];
                f.set(addr, x * x * x * x + y * y * y * y);   // f = x^4 + y^4
            }
            auto lap_num = discrete_laplacian(grid, f, max_metric);
            double max_err = 0.0;
            for (const auto& addr : grid) {
                if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r)
                    continue;
                Rational x = addr[0], y = addr[1];
                double err = std::abs((lap_num.at(addr) - (12 * x * x + 12 * y * y)).convert_to<double>());
                max_err = std::max(max_err, err);
            }
            errors.push_back(max_err);
        }
        ASSERT_GE(errors.size(), 2);
        for (std::size_t i = 0; i < errors.size() - 1; ++i) {
            double ratio = errors[i] / errors[i + 1];
            EXPECT_NEAR(ratio, 4.0, 1.0);
        }
    }
    // ============================================================================
// Точные полиномиальные тесты для 2D (используют ProductGrid)
// ============================================================================

    TEST_F(DiscreteOperators2DTest, GradientQuadraticExact) {
        auto grid = make_grid_2d(5);
        ScalarField2D f(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1];
            f.set(addr, x * x + y * y);
        }
        auto grad = discrete_gradient(grid, f, max_metric, DifferenceScheme::CENTRAL);
        for (const auto& addr : grid) {
            if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r)
                continue;
            Rational x = addr[0], y = addr[1];
            auto g = grad.at(addr);
            EXPECT_RATIONAL_NEAR(g[0], 2 * x, delta::default_eps());
            EXPECT_RATIONAL_NEAR(g[1], 2 * y, delta::default_eps());
        }
    }

    TEST_F(DiscreteOperators2DTest, LaplacianCubicExact) {
        auto grid = make_grid_2d(5);
        ScalarField2D f(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1];
            f.set(addr, x * x * x + y * y * y);
        }
        auto lap = discrete_laplacian(grid, f, max_metric);
        for (const auto& addr : grid) {
            if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r)
                continue;
            Rational x = addr[0], y = addr[1];
            EXPECT_RATIONAL_NEAR(lap.at(addr), 6 * x + 6 * y, delta::default_eps());
        }
    }

    TEST_F(DiscreteOperators2DTest, LaplacianQuadraticExact) {
        auto grid = make_grid_2d(5);
        ScalarField2D f(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1];
            f.set(addr, x * x + y * y);
        }
        auto lap = discrete_laplacian(grid, f, max_metric);
        for (const auto& addr : grid) {
            if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r)
                continue;
            EXPECT_RATIONAL_NEAR(lap.at(addr), 4_r, delta::default_eps());
        }
    }

    TEST_F(DiscreteOperators2DTest, DivergenceQuadraticExact) {
        auto grid = make_grid_2d(5);
        VecField2D v(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1];
            Eigen::Matrix<Scalar, 2, 1> val;
            val << x * x, y* y;
            v.set(addr, val);
        }
        auto div = discrete_divergence(grid, v, max_metric, DifferenceScheme::CENTRAL);
        for (const auto& addr : grid) {
            if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r)
                continue;
            Rational x = addr[0], y = addr[1];
            EXPECT_RATIONAL_NEAR(div.at(addr), 2 * x + 2 * y, delta::default_eps());
        }
    }

    TEST_F(DiscreteOperators2DTest, CurlGradZero) {
        auto grid = make_grid_2d(5);
        ScalarField2D f(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1];
            f.set(addr, x * x * x + y * y * y);
        }
        auto grad = discrete_gradient(grid, f, max_metric, DifferenceScheme::CENTRAL);
        auto curl_grad = discrete_curl_2d(grid, grad, max_metric, DifferenceScheme::CENTRAL);
        for (const auto& addr : grid) {
            if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r)
                continue;
            EXPECT_RATIONAL_NEAR(curl_grad.at(addr), 0_r, delta::default_eps());
        }
    }

    TEST_F(DiscreteOperators2DTest, DivergenceZeroForSolenoidalField) {
        // Поле v = (y, -x) имеет нулевую дивергенцию (соленоидально)
        auto grid = make_grid_2d(5);
        VecField2D v(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1];
            Eigen::Matrix<Scalar, 2, 1> val;
            val << y, -x;
            v.set(addr, val);
        }
        auto div = discrete_divergence(grid, v, max_metric, DifferenceScheme::CENTRAL);
        for (const auto& addr : grid) {
            if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r)
                continue;
            EXPECT_RATIONAL_NEAR(div.at(addr), 0_r, delta::default_eps());
        }
    }
} // namespace delta::testing