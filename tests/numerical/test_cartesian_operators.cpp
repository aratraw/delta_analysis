// tests/numerical/test_cartesian_operators.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "delta/numerical/grid_ops_1d.h"
#include "delta/numerical/grid_curl.h"
#include "delta/numerical/grid_laplacian.h"
#include "delta/numerical/grid_ops_tensor_1d.h"
#include "delta/numerical/concepts.h"
#include "delta/geometry/tensor_field.h"
#include "../test_fixtures.h"

namespace delta::testing {

    // -----------------------------------------------------------------------------
    // Test fixture for Cartesian operators (1D, 2D, 3D)
    // -----------------------------------------------------------------------------
    class CartesianOperatorsTest : public CartesianGridFixture {
    protected:
        using Point1D = double;
        using Point2D = Eigen::Vector2d;
        using Point3D = Eigen::Vector3d;

        EuclideanMetric metric;

        // Helper to create a scalar field on 1D grid
        OperationalFunction<Point1D, double, Grid1D> scalar_field_1d(std::function<double(double)> func) {
            return OperationalFunction<Point1D, double, Grid1D>(grid1D, func);
        }

        // Helper to create a scalar field on 3D grid (as a map from point to value)
        TensorField<Point3D, double, 0, 3> scalar_field_3d(std::function<double(const Point3D&)> func) {
            TensorField<Point3D, double, 0, 3> field;
            for (std::size_t i = 0; i < grid3D.x_grid().size(); ++i) {
                for (std::size_t j = 0; j < grid3D.y_grid().size(); ++j) {
                    for (std::size_t k = 0; k < grid3D.z_grid().size(); ++k) {
                        Point3D p = grid3D.point_at(i, j, k);
                        field.set(p, func(p));
                    }
                }
            }
            return field;
        }

        // Helper to create a vector field on 3D grid
        TensorField<Point3D, double, 1, 3> vector_field_3d(std::function<Eigen::Vector3d(const Point3D&)> func) {
            TensorField<Point3D, double, 1, 3> field;
            for (std::size_t i = 0; i < grid3D.x_grid().size(); ++i) {
                for (std::size_t j = 0; j < grid3D.y_grid().size(); ++j) {
                    for (std::size_t k = 0; k < grid3D.z_grid().size(); ++k) {
                        Point3D p = grid3D.point_at(i, j, k);
                        field.set(p, func(p));
                    }
                }
            }
            return field;
        }
    };

    // -----------------------------------------------------------------------------
    // 1D grid operators (grid_ops_1d.h)
    // -----------------------------------------------------------------------------

    TEST_F(CartesianOperatorsTest, ForwardDifference1D) {
        auto f = [](double x) { return x * x; };
        double x = 0.5;
        double h = grid1D.step();
        double fd = forward_diff(f, x, h);
        double exact = 2.0 * x; // 1.0
        EXPECT_NEAR(fd, exact, 0.11); // O(h) error
    }

    TEST_F(CartesianOperatorsTest, BackwardDifference1D) {
        auto f = [](double x) { return x * x; };
        double x = 0.5;
        double h = grid1D.step();
        double bd = backward_diff(f, x, h);
        double exact = 2.0 * x;
        EXPECT_NEAR(bd, exact, 0.11);
    }

    TEST_F(CartesianOperatorsTest, CentralDifference1D) {
        auto f = [](double x) { return x * x; };
        double x = 0.5;
        double h = grid1D.step();
        double cd = central_diff(f, x, h);
        double exact = 2.0 * x;
        EXPECT_NEAR(cd, exact, 1e-12); // exact for quadratic
    }

    TEST_F(CartesianOperatorsTest, LaplacianGeneral1D) {
        auto f = scalar_field_1d([](double x) { return x * x * x; }); // f''' = 6, f'' = 6x
        double x = 0.5;
        double lap = laplacian_general(grid1D, [&](double p) { return f(p); }, metric, x);
        double exact = 6.0 * x; // 3.0
        EXPECT_NEAR(lap, exact, 0.1); // O(h) error on non-uniform? Actually grid is uniform.
    }

    // -----------------------------------------------------------------------------
    // 3D curl and divergence identities (grid_curl.h)
    // -----------------------------------------------------------------------------

    TEST_F(CartesianOperatorsTest, CurlGradZero) {
        // Scalar field f(x,y,z) = x*y*z
        auto f = scalar_field_3d([](const Point3D& p) {
            return p.x() * p.y() * p.z();
            });

        // Compute gradient manually using central differences
        TensorField<Point3D, double, 1, 3> grad;
        double hx = grid3D.x_grid().step();
        double hy = grid3D.y_grid().step();
        double hz = grid3D.z_grid().step();

        for (std::size_t i = 0; i < grid3D.x_grid().size(); ++i) {
            for (std::size_t j = 0; j < grid3D.y_grid().size(); ++j) {
                for (std::size_t k = 0; k < grid3D.z_grid().size(); ++k) {
                    Point3D p = grid3D.point_at(i, j, k);
                    double dfdx = 0.0, dfdy = 0.0, dfdz = 0.0;

                    if (i > 0 && i + 1 < grid3D.x_grid().size())
                        dfdx = (f(grid3D.point_at(i + 1, j, k)) - f(grid3D.point_at(i - 1, j, k))) / (2.0 * hx);
                    else if (i == 0)
                        dfdx = (f(grid3D.point_at(i + 1, j, k)) - f(p)) / hx;
                    else if (i == grid3D.x_grid().size() - 1)
                        dfdx = (f(p) - f(grid3D.point_at(i - 1, j, k))) / hx;

                    if (j > 0 && j + 1 < grid3D.y_grid().size())
                        dfdy = (f(grid3D.point_at(i, j + 1, k)) - f(grid3D.point_at(i, j - 1, k))) / (2.0 * hy);
                    else if (j == 0)
                        dfdy = (f(grid3D.point_at(i, j + 1, k)) - f(p)) / hy;
                    else if (j == grid3D.y_grid().size() - 1)
                        dfdy = (f(p) - f(grid3D.point_at(i, j - 1, k))) / hy;

                    if (k > 0 && k + 1 < grid3D.z_grid().size())
                        dfdz = (f(grid3D.point_at(i, j, k + 1)) - f(grid3D.point_at(i, j, k - 1))) / (2.0 * hz);
                    else if (k == 0)
                        dfdz = (f(grid3D.point_at(i, j, k + 1)) - f(p)) / hz;
                    else if (k == grid3D.z_grid().size() - 1)
                        dfdz = (f(p) - f(grid3D.point_at(i, j, k - 1))) / hz;

                    grad.set(p, Eigen::Vector3d(dfdx, dfdy, dfdz));
                }
            }
        }

        // Compute curl of grad
        auto curl_grad = discrete_curl(grid3D, grad, metric);

        // Check that curl_grad is zero everywhere
        double max_error = 0.0;
        for (const auto& [p, vec] : curl_grad) {
            double err = vec.norm();
            if (err > max_error) max_error = err;
        }
        EXPECT_NEAR(max_error, 0.0, 1e-10);
    }

    TEST_F(CartesianOperatorsTest, DivCurlZero) {
        // Vector field v = (sin y, sin z, sin x) (div curl should be zero)
        auto v = vector_field_3d([](const Point3D& p) {
            return Eigen::Vector3d(std::sin(p.y()), std::sin(p.z()), std::sin(p.x()));
            });

        auto curl_v = discrete_curl(grid3D, v, metric);

        // Compute divergence of curl_v
        double hx = grid3D.x_grid().step();
        double hy = grid3D.y_grid().step();
        double hz = grid3D.z_grid().step();

        double max_error = 0.0;
        for (std::size_t i = 0; i < grid3D.x_grid().size(); ++i) {
            for (std::size_t j = 0; j < grid3D.y_grid().size(); ++j) {
                for (std::size_t k = 0; k < grid3D.z_grid().size(); ++k) {
                    Point3D p = grid3D.point_at(i, j, k);
                    const auto& w = curl_v.at(p);

                    double dwx_dx = 0.0, dwy_dy = 0.0, dwz_dz = 0.0;

                    if (i > 0 && i + 1 < grid3D.x_grid().size())
                        dwx_dx = (curl_v.at(grid3D.point_at(i + 1, j, k))(0) - curl_v.at(grid3D.point_at(i - 1, j, k))(0)) / (2.0 * hx);
                    else if (i == 0)
                        dwx_dx = (curl_v.at(grid3D.point_at(i + 1, j, k))(0) - w(0)) / hx;
                    else if (i == grid3D.x_grid().size() - 1)
                        dwx_dx = (w(0) - curl_v.at(grid3D.point_at(i - 1, j, k))(0)) / hx;

                    if (j > 0 && j + 1 < grid3D.y_grid().size())
                        dwy_dy = (curl_v.at(grid3D.point_at(i, j + 1, k))(1) - curl_v.at(grid3D.point_at(i, j - 1, k))(1)) / (2.0 * hy);
                    else if (j == 0)
                        dwy_dy = (curl_v.at(grid3D.point_at(i, j + 1, k))(1) - w(1)) / hy;
                    else if (j == grid3D.y_grid().size() - 1)
                        dwy_dy = (w(1) - curl_v.at(grid3D.point_at(i, j - 1, k))(1)) / hy;

                    if (k > 0 && k + 1 < grid3D.z_grid().size())
                        dwz_dz = (curl_v.at(grid3D.point_at(i, j, k + 1))(2) - curl_v.at(grid3D.point_at(i, j, k - 1))(2)) / (2.0 * hz);
                    else if (k == 0)
                        dwz_dz = (curl_v.at(grid3D.point_at(i, j, k + 1))(2) - w(2)) / hz;
                    else if (k == grid3D.z_grid().size() - 1)
                        dwz_dz = (w(2) - curl_v.at(grid3D.point_at(i, j, k - 1))(2)) / hz;

                    double div = dwx_dx + dwy_dy + dwz_dz;
                    double err = std::abs(div);
                    if (err > max_error) max_error = err;
                }
            }
        }
        EXPECT_NEAR(max_error, 0.0, 1e-10);
    }

    // -----------------------------------------------------------------------------
    // Laplacian matrix (grid_laplacian.h)
    // -----------------------------------------------------------------------------

    TEST_F(CartesianOperatorsTest, LaplacianMatrix1D) {
        auto L = build_laplacian_matrix(grid1D, metric);
        EXPECT_EQ(L.rows(), static_cast<int>(grid1D.size()));
        EXPECT_EQ(L.cols(), static_cast<int>(grid1D.size()));

        // Check symmetry
        EXPECT_TRUE(L.isApprox(L.transpose(), 1e-12));

        Eigen::VectorXd ones = Eigen::VectorXd::Ones(L.rows());
        Eigen::VectorXd rowsum = L * ones;

        // For interior nodes (not boundaries), row sum should be zero.
        // In a 1D grid with 11 points, indices 1..9 are interior.
        for (int i = 1; i < L.rows() - 1; ++i) {
            EXPECT_NEAR(rowsum(i), 0.0, 1e-12);
        }
    }

    // -----------------------------------------------------------------------------
    // 1D tensor field operators (grid_ops_tensor_1d.h)
    // -----------------------------------------------------------------------------

    TEST_F(CartesianOperatorsTest, Gradient1D) {
        // Scalar field f(x) = x^2
        auto f = scalar_field_1d([](double x) { return x * x; });

        auto grad = discrete_gradient_1d(grid1D, [&](double x) { return f(x); }, metric, DifferenceScheme::CENTRAL);

        // At interior point x=0.5, exact derivative = 1.0
        double x_test = 0.5;
        EXPECT_NEAR(grad.at(x_test)(0), 1.0, 1e-12);

        // At left boundary using forward scheme
        auto grad_forward = discrete_gradient_1d(grid1D, [&](double x) { return f(x); }, metric, DifferenceScheme::FORWARD);
        double x_left = grid1D[0];
        EXPECT_NEAR(grad_forward.at(x_left)(0), 0.1, 1e-12);
    }

    TEST_F(CartesianOperatorsTest, Divergence1D) {
        // Vector field v(x) = (x^2) as a 1D vector (only one component)
        TensorField<double, double, 1, 1> v;
        for (std::size_t i = 0; i < grid1D.size(); ++i) {
            double x = grid1D[i];
            v.set(x, Eigen::Matrix<double, 1, 1>(x * x));
        }

        auto div = discrete_divergence_1d(grid1D, v, metric, DifferenceScheme::BACKWARD);
        // For v(x)=x^2, divergence = derivative = 2x
        double x_test = 0.5;
        EXPECT_NEAR(div.at(x_test), 1.0, 1e-12); // 2*0.5=1.0

        // At right boundary, backward gives derivative
        double x_right = grid1D[grid1D.size() - 1];
        EXPECT_NEAR(div.at(x_right), 2.0 * x_right, 0.11); // O(h) error
    }

} // namespace delta::testing