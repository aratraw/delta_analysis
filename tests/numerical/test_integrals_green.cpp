// tests/numerical/test_integrals_green.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "delta/numerical/integrals_green.h"
#include "delta/numerical/cartesian_grid.h"
#include "delta/geometry/tensor_field.h"
#include "../test_fixtures.h"

namespace delta::testing {

    // -----------------------------------------------------------------------------
    // Test fixture for Green's identities on Cartesian grids
    // -----------------------------------------------------------------------------
    class GreensIdentitiesTest : public CartesianGridFixture {
    protected:
        using Point2D = Eigen::Vector2d;
        using Point3D = Eigen::Vector3d;

        EuclideanMetric metric;

        // Helper to create a scalar field on 2D grid
        TensorField<Point2D, double, 0, 2> make_field_2d(std::function<double(const Point2D&)> func) {
            TensorField<Point2D, double, 0, 2> field;
            for (std::size_t i = 0; i < grid2D.x_grid().size(); ++i) {
                for (std::size_t j = 0; j < grid2D.y_grid().size(); ++j) {
                    Point2D p = grid2D.point_at(i, j, 0);
                    field.set(p, func(p));
                }
            }
            return field;
        }

        // Helper to create a scalar field on 3D grid
        TensorField<Point3D, double, 0, 3> make_field_3d(std::function<double(const Point3D&)> func) {
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
    };

    // -----------------------------------------------------------------------------
    // 2D Green's first identity
    // -----------------------------------------------------------------------------
    TEST_F(GreensIdentitiesTest, FirstGreen2D_Polynomial) {
        // Choose u and v as polynomials that satisfy the identity exactly in continuous case.
        // u(x,y) = x^2 + y^2, v(x,y) = x*y
        auto u = make_field_2d([](const Point2D& p) {
            return p.x() * p.x() + p.y() * p.y();
            });
        auto v = make_field_2d([](const Point2D& p) {
            return p.x() * p.y();
            });

        bool result = check_green_first_2d(grid2D, u, v, metric, 1e-10);
        EXPECT_TRUE(result);
    }

    TEST_F(GreensIdentitiesTest, FirstGreen2D_Sine) {
        // u = sin(πx) sin(πy), v = sin(2πx) sin(2πy)
        auto u = make_field_2d([](const Point2D& p) {
            return std::sin(M_PI * p.x()) * std::sin(M_PI * p.y());
            });
        auto v = make_field_2d([](const Point2D& p) {
            return std::sin(2.0 * M_PI * p.x()) * std::sin(2.0 * M_PI * p.y());
            });

        bool result = check_green_first_2d(grid2D, u, v, metric, 1e-8);
        EXPECT_TRUE(result);
    }

    TEST_F(GreensIdentitiesTest, FirstGreen2D_ZeroOnBoundary) {
        // u = x*(1-x)*y*(1-y) which is zero on boundary of [0,1]^2, so boundary term vanishes.
        auto u = make_field_2d([](const Point2D& p) {
            return p.x() * (1.0 - p.x()) * p.y() * (1.0 - p.y());
            });
        auto v = make_field_2d([](const Point2D& p) {
            return p.x() * p.x() + p.y();
            });

        bool result = check_green_first_2d(grid2D, u, v, metric, 1e-10);
        EXPECT_TRUE(result);
    }

    // -----------------------------------------------------------------------------
    // 2D Green's second identity
    // -----------------------------------------------------------------------------
    TEST_F(GreensIdentitiesTest, SecondGreen2D_Polynomial) {
        auto u = make_field_2d([](const Point2D& p) {
            return p.x() * p.x() + p.y();
            });
        auto v = make_field_2d([](const Point2D& p) {
            return p.x() * p.y() * p.y();
            });

        bool result = check_green_second_2d(grid2D, u, v, metric, 1e-10);
        EXPECT_TRUE(result);
    }

    TEST_F(GreensIdentitiesTest, SecondGreen2D_Sine) {
        auto u = make_field_2d([](const Point2D& p) {
            return std::sin(M_PI * p.x()) * std::cos(M_PI * p.y());
            });
        auto v = make_field_2d([](const Point2D& p) {
            return std::cos(2.0 * M_PI * p.x()) * std::sin(2.0 * M_PI * p.y());
            });

        bool result = check_green_second_2d(grid2D, u, v, metric, 1e-8);
        EXPECT_TRUE(result);
    }

    // -----------------------------------------------------------------------------
    // 3D Green's first identity
    // -----------------------------------------------------------------------------
    TEST_F(GreensIdentitiesTest, FirstGreen3D_Polynomial) {
        auto u = make_field_3d([](const Point3D& p) {
            return p.x() * p.x() + p.y() * p.y() + p.z() * p.z();
            });
        auto v = make_field_3d([](const Point3D& p) {
            return p.x() * p.y() * p.z();
            });

        bool result = check_green_first_3d(grid3D, u, v, metric, 1e-10);
        EXPECT_TRUE(result);
    }

    TEST_F(GreensIdentitiesTest, FirstGreen3D_Sine) {
        auto u = make_field_3d([](const Point3D& p) {
            return std::sin(M_PI * p.x()) * std::sin(M_PI * p.y()) * std::sin(M_PI * p.z());
            });
        auto v = make_field_3d([](const Point3D& p) {
            return std::cos(M_PI * p.x()) * std::cos(M_PI * p.y()) * std::cos(M_PI * p.z());
            });

        bool result = check_green_first_3d(grid3D, u, v, metric, 1e-8);
        EXPECT_TRUE(result);
    }

    // -----------------------------------------------------------------------------
    // 3D Green's second identity
    // -----------------------------------------------------------------------------
    TEST_F(GreensIdentitiesTest, SecondGreen3D_Polynomial) {
        auto u = make_field_3d([](const Point3D& p) {
            return p.x() * p.x() + p.y() * p.y();
            });
        auto v = make_field_3d([](const Point3D& p) {
            return p.z() * p.z() + p.x() * p.y();
            });

        bool result = check_green_second_3d(grid3D, u, v, metric, 1e-10);
        EXPECT_TRUE(result);
    }

    TEST_F(GreensIdentitiesTest, SecondGreen3D_Sine) {
        auto u = make_field_3d([](const Point3D& p) {
            return std::sin(M_PI * p.x()) * std::cos(M_PI * p.y());
            });
        auto v = make_field_3d([](const Point3D& p) {
            return std::sin(M_PI * p.y()) * std::cos(M_PI * p.z());
            });

        bool result = check_green_second_3d(grid3D, u, v, metric, 1e-8);
        EXPECT_TRUE(result);
    }

} // namespace delta::testing