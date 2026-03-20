// tests/numerical/discrete_operators_3d_4d_test.cpp
#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <cmath>
#include "delta/numerical/discrete_operators.h"
#include "delta/core/uniform_grid.h"
#include "delta/core/product_path.h"
#include "delta/geometry/tensor_field.h"
#include "../test_fixtures_geometry_numerical.h"

namespace delta::numerical::testing {

    using delta::testing::GeometryNumericalTest;
    using delta::operator""_r;
    using namespace delta::geometry;
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
    // 3D tests
    // -------------------------------------------------------------------------
    class DiscreteOperators3DTest : public GeometryNumericalTest {
    protected:
        using Grid1D = delta::UniformGrid<Rational, std::less<Rational>>;
        using Grid3D = delta::ProductGrid<Grid1D, 3>;
        using Addr3D = typename Grid3D::value_type; // std::array<Rational,3>

        struct Addr3DCompare {
            bool operator()(const Addr3D& a, const Addr3D& b) const {
                if (a[0] < b[0]) return true;
                if (b[0] < a[0]) return false;
                if (a[1] < b[1]) return true;
                if (b[1] < a[1]) return false;
                return a[2] < b[2];
            }
        };

        using ScalarField3D = delta::geometry::TensorField<Addr3D, Scalar, 0, 3, Addr3DCompare>;
        using VecField3D = delta::geometry::TensorField<Addr3D, Scalar, 1, 3, Addr3DCompare>;

        Grid3D make_grid_3d(std::size_t n) {
            Grid1D g(0_r, 1_r / (n - 1), n);
            return Grid3D({ g, g, g });
        }

        MaxMetric max_metric;
    };

    TEST_F(DiscreteOperators3DTest, GradientQuadraticExact) {
        auto grid = make_grid_3d(5);
        ScalarField3D f(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1], z = addr[2];
            f.set(addr, x * x + y * y + z * z);
        }
        auto grad = discrete_gradient(grid, f, max_metric, DifferenceScheme::CENTRAL);
        for (const auto& addr : grid) {
            if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r || addr[2] == 0_r || addr[2] == 1_r)
                continue;
            Rational x = addr[0], y = addr[1], z = addr[2];
            auto g = grad.at(addr);
            EXPECT_RATIONAL_NEAR(g[0], 2 * x, delta::default_eps());
            EXPECT_RATIONAL_NEAR(g[1], 2 * y, delta::default_eps());
            EXPECT_RATIONAL_NEAR(g[2], 2 * z, delta::default_eps());
        }
    }

    TEST_F(DiscreteOperators3DTest, LaplacianQuadraticExact) {
        auto grid = make_grid_3d(5);
        ScalarField3D f(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1], z = addr[2];
            f.set(addr, x * x + y * y + z * z);
        }
        auto lap = discrete_laplacian(grid, f, max_metric);
        for (const auto& addr : grid) {
            if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r || addr[2] == 0_r || addr[2] == 1_r)
                continue;
            EXPECT_RATIONAL_NEAR(lap.at(addr), 6_r, delta::default_eps());
        }
    }

    TEST_F(DiscreteOperators3DTest, LaplacianCubicExact) {
        auto grid = make_grid_3d(5);
        ScalarField3D f(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1], z = addr[2];
            f.set(addr, x * x * x + y * y * y + z * z * z);
        }
        auto lap = discrete_laplacian(grid, f, max_metric);
        for (const auto& addr : grid) {
            if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r || addr[2] == 0_r || addr[2] == 1_r)
                continue;
            Rational x = addr[0], y = addr[1], z = addr[2];
            EXPECT_RATIONAL_NEAR(lap.at(addr), 6 * x + 6 * y + 6 * z, delta::default_eps());
        }
    }

    TEST_F(DiscreteOperators3DTest, DivergenceQuadraticExact) {
        auto grid = make_grid_3d(5);
        VecField3D v(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1], z = addr[2];
            Eigen::Matrix<Scalar, 3, 1> val;
            val << x * x, y* y, z* z;
            v.set(addr, val);
        }
        auto div = discrete_divergence(grid, v, max_metric, DifferenceScheme::CENTRAL);
        for (const auto& addr : grid) {
            if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r || addr[2] == 0_r || addr[2] == 1_r)
                continue;
            Rational x = addr[0], y = addr[1], z = addr[2];
            EXPECT_RATIONAL_NEAR(div.at(addr), 2 * x + 2 * y + 2 * z, delta::default_eps());
        }
    }

    TEST_F(DiscreteOperators3DTest, CurlGradZero) {
        auto grid = make_grid_3d(5);
        ScalarField3D f(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1], z = addr[2];
            f.set(addr, x * x * x + y * y * y + z * z * z);
        }
        auto grad = discrete_gradient(grid, f, max_metric, DifferenceScheme::CENTRAL);
        auto curl_grad = discrete_curl_3d(grid, grad, max_metric, DifferenceScheme::CENTRAL);
        for (const auto& addr : grid) {
            if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r || addr[2] == 0_r || addr[2] == 1_r)
                continue;
            auto c = curl_grad.at(addr);
            EXPECT_RATIONAL_NEAR(c[0], 0_r, delta::default_eps());
            EXPECT_RATIONAL_NEAR(c[1], 0_r, delta::default_eps());
            EXPECT_RATIONAL_NEAR(c[2], 0_r, delta::default_eps());
        }
    }

    TEST_F(DiscreteOperators3DTest, DivCurlZero) {
        auto grid = make_grid_3d(5);
        VecField3D v(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1], z = addr[2];
            Eigen::Matrix<Scalar, 3, 1> val;
            val << y, z, x;   // произвольное поле с ненулевым ротором
            v.set(addr, val);
        }
        auto curl_v = discrete_curl_3d(grid, v, max_metric, DifferenceScheme::CENTRAL);
        auto div_curl = discrete_divergence(grid, curl_v, max_metric, DifferenceScheme::CENTRAL);
        for (const auto& addr : grid) {
            if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r || addr[2] == 0_r || addr[2] == 1_r)
                continue;
            EXPECT_RATIONAL_NEAR(div_curl.at(addr), 0_r, delta::default_eps());
        }
    }

    TEST_F(DiscreteOperators3DTest, GradientSecondOrder) {
        std::vector<std::size_t> ns = { 5, 9, 17 };
        std::vector<double> errors;
        for (std::size_t n : ns) {
            auto grid = make_grid_3d(n);
            ScalarField3D f(grid);
            for (const auto& addr : grid) {
                Rational x = addr[0], y = addr[1], z = addr[2];
                f.set(addr, x * x * x * x + y * y * y * y + z * z * z * z); // x^4 + y^4 + z^4
            }
            auto grad_num = discrete_gradient(grid, f, max_metric, DifferenceScheme::CENTRAL);
            double max_err = 0.0;
            for (const auto& addr : grid) {
                if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r || addr[2] == 0_r || addr[2] == 1_r)
                    continue;
                Rational x = addr[0], y = addr[1], z = addr[2];
                auto g = grad_num.at(addr);
                double err_x = std::abs(static_cast<double>(g[0] - 4 * x * x * x));
                double err_y = std::abs(static_cast<double>(g[1] - 4 * y * y * y));
                double err_z = std::abs(static_cast<double>(g[2] - 4 * z * z * z));
                max_err = std::max(max_err, std::max({ err_x, err_y, err_z }));
            }
            errors.push_back(max_err);
        }
        ASSERT_GE(errors.size(), 2);
        for (std::size_t i = 0; i < errors.size() - 1; ++i) {
            double ratio = errors[i] / errors[i + 1];
            EXPECT_NEAR(ratio, 4.0, 1.0);
        }
    }

    TEST_F(DiscreteOperators3DTest, LaplacianSecondOrder) {
        std::vector<std::size_t> ns = { 5, 9, 17 };
        std::vector<double> errors;
        for (std::size_t n : ns) {
            auto grid = make_grid_3d(n);
            ScalarField3D f(grid);
            for (const auto& addr : grid) {
                Rational x = addr[0], y = addr[1], z = addr[2];
                f.set(addr, x * x * x * x + y * y * y * y + z * z * z * z);
            }
            auto lap_num = discrete_laplacian(grid, f, max_metric);
            double max_err = 0.0;
            for (const auto& addr : grid) {
                if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r || addr[2] == 0_r || addr[2] == 1_r)
                    continue;
                Rational x = addr[0], y = addr[1], z = addr[2];
                double err = std::abs(static_cast<double>(lap_num.at(addr) - (12 * x * x + 12 * y * y + 12 * z * z)));
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

    // -------------------------------------------------------------------------
    // 4D tests
    // -------------------------------------------------------------------------
    class DiscreteOperators4DTest : public GeometryNumericalTest {
    protected:
        using Grid1D = delta::UniformGrid<Rational, std::less<Rational>>;
        using Grid4D = delta::ProductGrid<Grid1D, 4>;
        using Addr4D = typename Grid4D::value_type; // std::array<Rational,4>

        struct Addr4DCompare {
            bool operator()(const Addr4D& a, const Addr4D& b) const {
                if (a[0] < b[0]) return true;
                if (b[0] < a[0]) return false;
                if (a[1] < b[1]) return true;
                if (b[1] < a[1]) return false;
                if (a[2] < b[2]) return true;
                if (b[2] < a[2]) return false;
                return a[3] < b[3];
            }
        };

        using ScalarField4D = delta::geometry::TensorField<Addr4D, Scalar, 0, 4, Addr4DCompare>;
        using VecField4D = delta::geometry::TensorField<Addr4D, Scalar, 1, 4, Addr4DCompare>;

        Grid4D make_grid_4d(std::size_t n) {
            Grid1D g(0_r, 1_r / (n - 1), n);
            return Grid4D({ g, g, g, g });
        }

        MaxMetric max_metric;
    };

    TEST_F(DiscreteOperators4DTest, GradientQuadraticExact) {
        auto grid = make_grid_4d(5);
        ScalarField4D f(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1], z = addr[2], w = addr[3];
            f.set(addr, x * x + y * y + z * z + w * w);
        }
        auto grad = discrete_gradient(grid, f, max_metric, DifferenceScheme::CENTRAL);
        for (const auto& addr : grid) {
            if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r ||
                addr[2] == 0_r || addr[2] == 1_r || addr[3] == 0_r || addr[3] == 1_r)
                continue;
            Rational x = addr[0], y = addr[1], z = addr[2], w = addr[3];
            auto g = grad.at(addr);
            EXPECT_RATIONAL_NEAR(g[0], 2 * x, delta::default_eps());
            EXPECT_RATIONAL_NEAR(g[1], 2 * y, delta::default_eps());
            EXPECT_RATIONAL_NEAR(g[2], 2 * z, delta::default_eps());
            EXPECT_RATIONAL_NEAR(g[3], 2 * w, delta::default_eps());
        }
    }

    TEST_F(DiscreteOperators4DTest, LaplacianQuadraticExact) {
        auto grid = make_grid_4d(5);
        ScalarField4D f(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1], z = addr[2], w = addr[3];
            f.set(addr, x * x + y * y + z * z + w * w);
        }
        auto lap = discrete_laplacian(grid, f, max_metric);
        for (const auto& addr : grid) {
            if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r ||
                addr[2] == 0_r || addr[2] == 1_r || addr[3] == 0_r || addr[3] == 1_r)
                continue;
            EXPECT_RATIONAL_NEAR(lap.at(addr), 8_r, delta::default_eps());
        }
    }

    TEST_F(DiscreteOperators4DTest, LaplacianCubicExact) {
        auto grid = make_grid_4d(5);
        ScalarField4D f(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1], z = addr[2], w = addr[3];
            f.set(addr, x * x * x + y * y * y + z * z * z + w * w * w);
        }
        auto lap = discrete_laplacian(grid, f, max_metric);
        for (const auto& addr : grid) {
            if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r ||
                addr[2] == 0_r || addr[2] == 1_r || addr[3] == 0_r || addr[3] == 1_r)
                continue;
            Rational x = addr[0], y = addr[1], z = addr[2], w = addr[3];
            EXPECT_RATIONAL_NEAR(lap.at(addr), 6 * x + 6 * y + 6 * z + 6 * w, delta::default_eps());
        }
    }

    TEST_F(DiscreteOperators4DTest, DivergenceQuadraticExact) {
        auto grid = make_grid_4d(5);
        VecField4D v(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1], z = addr[2], w = addr[3];
            Eigen::Matrix<Scalar, 4, 1> val;
            val << x * x, y* y, z* z, w* w;
            v.set(addr, val);
        }
        auto div = discrete_divergence(grid, v, max_metric, DifferenceScheme::CENTRAL);
        for (const auto& addr : grid) {
            if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r ||
                addr[2] == 0_r || addr[2] == 1_r || addr[3] == 0_r || addr[3] == 1_r)
                continue;
            Rational x = addr[0], y = addr[1], z = addr[2], w = addr[3];
            EXPECT_RATIONAL_NEAR(div.at(addr), 2 * x + 2 * y + 2 * z + 2 * w, delta::default_eps());
        }
    }

    TEST_F(DiscreteOperators4DTest, DivGradEqualsLaplacian) {
        auto grid = make_grid_4d(5);
        ScalarField4D f(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1], z = addr[2], w = addr[3];
            f.set(addr, x * x + y * y + z * z + w * w);
        }
        auto grad = discrete_gradient(grid, f, max_metric, DifferenceScheme::CENTRAL);
        auto div_grad = discrete_divergence(grid, grad, max_metric, DifferenceScheme::CENTRAL);
        auto lap = discrete_laplacian(grid, f, max_metric);
        // Проверяем только центральную точку (0.5,0.5,0.5,0.5), так как для неё все соседи внутренние
        Addr4D center = { 0.5_r, 0.5_r, 0.5_r, 0.5_r };
        EXPECT_RATIONAL_NEAR(div_grad.at(center), lap.at(center), delta::default_eps());
        EXPECT_RATIONAL_NEAR(lap.at(center), 8_r, delta::default_eps());
    }

    TEST_F(DiscreteOperators4DTest, GradientSecondOrder) {
        set_precision(Rational(1_r / 10000000_r));
        std::vector<std::size_t> ns = { 5, 9, 17 };
        std::vector<double> errors;
        for (std::size_t n : ns) {
            auto grid = make_grid_4d(n);
            ScalarField4D f(grid);
            for (const auto& addr : grid) {
                Rational x = addr[0], y = addr[1], z = addr[2], w = addr[3];
                f.set(addr, x * x * x * x + y * y * y * y + z * z * z * z + w * w * w * w); // x^4+y^4+z^4+w^4
            }
            auto grad_num = discrete_gradient(grid, f, max_metric, DifferenceScheme::CENTRAL);
            double max_err = 0.0;
            for (const auto& addr : grid) {
                if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r ||
                    addr[2] == 0_r || addr[2] == 1_r || addr[3] == 0_r || addr[3] == 1_r)
                    continue;
                Rational x = addr[0], y = addr[1], z = addr[2], w = addr[3];
                auto g = grad_num.at(addr);
                double err_x = std::abs(static_cast<double>(g[0] - 4 * x * x * x));
                double err_y = std::abs(static_cast<double>(g[1] - 4 * y * y * y));
                double err_z = std::abs(static_cast<double>(g[2] - 4 * z * z * z));
                double err_w = std::abs(static_cast<double>(g[3] - 4 * w * w * w));
                max_err = std::max(max_err, std::max({ err_x, err_y, err_z, err_w }));
            }
            errors.push_back(max_err);
        }
        ASSERT_GE(errors.size(), 2);
        for (std::size_t i = 0; i < errors.size() - 1; ++i) {
            double ratio = errors[i] / errors[i + 1];
            EXPECT_NEAR(ratio, 4.0, 1.0);
        }
    }

    TEST_F(DiscreteOperators4DTest, LaplacianSecondOrder) {
        std::vector<std::size_t> ns = { 5, 9, 17 };
        std::vector<double> errors;
        for (std::size_t n : ns) {
            auto grid = make_grid_4d(n);
            ScalarField4D f(grid);
            for (const auto& addr : grid) {
                Rational x = addr[0], y = addr[1], z = addr[2], w = addr[3];
                f.set(addr, x * x * x * x + y * y * y * y + z * z * z * z + w * w * w * w);
            }
            auto lap_num = discrete_laplacian(grid, f, max_metric);
            double max_err = 0.0;
            for (const auto& addr : grid) {
                if (addr[0] == 0_r || addr[0] == 1_r || addr[1] == 0_r || addr[1] == 1_r ||
                    addr[2] == 0_r || addr[2] == 1_r || addr[3] == 0_r || addr[3] == 1_r)
                    continue;
                Rational x = addr[0], y = addr[1], z = addr[2], w = addr[3];
                double err = std::abs(static_cast<double>(lap_num.at(addr) - (12 * x * x + 12 * y * y + 12 * z * z + 12 * w * w)));
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

} // namespace delta::numerical::testing