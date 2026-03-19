// tests/numerical/discrete_operators_test.cpp
#include <gtest/gtest.h>
#include "delta/numerical/discrete_operators.h"
#include "delta/core/uniform_grid.h"
#include "delta/geometry/tensor_field.h"   // для TensorField
#include "../test_fixtures_geometry_numerical.h"

namespace delta::numerical::testing {

    using delta::testing::GeometryNumericalTest;
    using delta::operator""_r;
    using namespace delta::geometry;  // для TensorField

    // Test fixture for 1D and 2D discrete operators
    class DiscreteOperatorsTest : public GeometryNumericalTest {
    protected:
        // 1D uniform grid on [0,1] with N points
        using Grid1D = delta::UniformGrid<Rational, std::less<Rational>>;
        using ScalarField1D = delta::geometry::TensorField<Rational, Scalar, 0, 1, std::less<Rational>>;
        using VecField1D = delta::geometry::TensorField<Rational, Scalar, 1, 1, std::less<Rational>>;

        Grid1D make_grid_1d(std::size_t n) {
            return Grid1D(0_r, 1_r / (n - 1), n);
        }

        EuclideanMetric metric;
    };

    TEST_F(DiscreteOperatorsTest, ForwardDifference1D) {
        auto grid = make_grid_1d(5); // points: 0, 0.25, 0.5, 0.75, 1
        ScalarField1D f(grid);
        f.set(0_r, 0_r);
        f.set(0.25_r, 0.25_r);
        f.set(0.5_r, 0.5_r);
        f.set(0.75_r, 0.75_r);
        f.set(1_r, 1_r);

        // forward difference at interior points
        auto df = forward_difference(grid, f, metric, 0.25_r);
        EXPECT_RATIONAL_NEAR(df, 1_r, delta::default_eps());
        df = forward_difference(grid, f, metric, 0.5_r);
        EXPECT_RATIONAL_NEAR(df, 1_r, delta::default_eps());
        df = forward_difference(grid, f, metric, 0.75_r);
        EXPECT_RATIONAL_NEAR(df, 1_r, delta::default_eps());

        // at left boundary, forward difference works
        df = forward_difference(grid, f, metric, 0_r);
        EXPECT_RATIONAL_NEAR(df, 1_r, delta::default_eps());

        // at right boundary, forward difference should throw (no neighbour)
        EXPECT_THROW(forward_difference(grid, f, metric, 1_r), std::out_of_range);
    }

    TEST_F(DiscreteOperatorsTest, BackwardDifference1D) {
        auto grid = make_grid_1d(5);
        ScalarField1D f(grid);
        f.set(0_r, 0_r);
        f.set(0.25_r, 0.25_r);
        f.set(0.5_r, 0.5_r);
        f.set(0.75_r, 0.75_r);
        f.set(1_r, 1_r);

        auto df = backward_difference(grid, f, metric, 0.25_r);
        EXPECT_RATIONAL_NEAR(df, 1_r, delta::default_eps());
        df = backward_difference(grid, f, metric, 0.5_r);
        EXPECT_RATIONAL_NEAR(df, 1_r, delta::default_eps());
        df = backward_difference(grid, f, metric, 0.75_r);
        EXPECT_RATIONAL_NEAR(df, 1_r, delta::default_eps());

        EXPECT_THROW(backward_difference(grid, f, metric, 0_r), std::out_of_range);
        df = backward_difference(grid, f, metric, 1_r);
        EXPECT_RATIONAL_NEAR(df, 1_r, delta::default_eps());
    }

    TEST_F(DiscreteOperatorsTest, CentralDifference1D) {
        auto grid = make_grid_1d(5);
        ScalarField1D f(grid);
        f.set(0_r, 0_r);
        f.set(0.25_r, 0.25_r);
        f.set(0.5_r, 0.5_r);
        f.set(0.75_r, 0.75_r);
        f.set(1_r, 1_r);

        auto df = central_difference(grid, f, metric, 0.25_r);
        EXPECT_RATIONAL_NEAR(df, 1_r, delta::default_eps());
        df = central_difference(grid, f, metric, 0.5_r);
        EXPECT_RATIONAL_NEAR(df, 1_r, delta::default_eps());
        df = central_difference(grid, f, metric, 0.75_r);
        EXPECT_RATIONAL_NEAR(df, 1_r, delta::default_eps());

        EXPECT_THROW(central_difference(grid, f, metric, 0_r), std::out_of_range);
        EXPECT_THROW(central_difference(grid, f, metric, 1_r), std::out_of_range);
    }

    // For 2D, we would need a product grid. We'll implement that later.
    // For now, we can test gradient and divergence in 1D, where they reduce to differences.

    TEST_F(DiscreteOperatorsTest, Gradient1D) {
        auto grid = make_grid_1d(5);
        ScalarField1D f(grid);
        f.set(0_r, 0_r);
        f.set(0.25_r, 0.25_r);
        f.set(0.5_r, 0.5_r);
        f.set(0.75_r, 0.75_r);
        f.set(1_r, 1_r);

        auto grad = discrete_gradient(grid, f, metric, DifferenceScheme::CENTRAL);
        // gradient returns a vector field of dimension 1
        for (const auto& x : grid) {
            auto g = grad.at(x);
            if (x == 0_r || x == 1_r) {
                // boundary: central fails, so we use fallback (forward/backward)
                EXPECT_RATIONAL_NEAR(g[0], 1_r, delta::default_eps());
            }
            else {
                EXPECT_RATIONAL_NEAR(g[0], 1_r, delta::default_eps());
            }
        }
    }

    TEST_F(DiscreteOperatorsTest, Divergence1D) {
        auto grid = make_grid_1d(5);
        // vector field in 1D is just a scalar field but stored as 1D vector
        VecField1D v(grid);
        v.set(0_r, (Eigen::Matrix<Scalar, 1, 1>() << 1_r).finished());
        v.set(0.25_r, (Eigen::Matrix<Scalar, 1, 1>() << 1_r).finished());
        v.set(0.5_r, (Eigen::Matrix<Scalar, 1, 1>() << 1_r).finished());
        v.set(0.75_r, (Eigen::Matrix<Scalar, 1, 1>() << 1_r).finished());
        v.set(1_r, (Eigen::Matrix<Scalar, 1, 1>() << 1_r).finished());

        auto div = discrete_divergence(grid, v, metric, DifferenceScheme::CENTRAL);
        // div of constant vector field should be zero
        for (const auto& x : grid) {
            EXPECT_RATIONAL_NEAR(div.at(x), 0_r, delta::default_eps());
        }
    }

} // namespace delta::numerical::testing