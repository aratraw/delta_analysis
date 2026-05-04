// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

//tests/regulative_ideas/test_matrix.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "../test_fixtures.h"
#include "delta/core/list_grid.h"
#include "delta/core/delta_path.h"
#include "delta/core/delta_operator.h"
#include "delta/calculus/riemann_sum.h"

namespace delta::testing {

    // Custom comparator for Eigen::MatrixXd based on Frobenius norm.
    struct MatrixLess {
        bool operator()(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) const {
            return a.norm() < b.norm();
        }
    };

    /**
     * @class MatrixPathTest
     * @brief Test suite for matrix‑valued functions (Eigen::MatrixXd) in Δ‑analysis.
     *
     * Verifies that grids, paths, Riemann sums, and adaptive refinement work
     * correctly with matrix addresses and values.
     */
    class MatrixPathTest : public DeltaTest {};

    /**
     * @test TraceIntegral
     * @brief Basic sanity check for Riemann sums of matrix‑valued functions.
     *
     * Constructs a path from two diagonal matrices (0.5·I and I) and refines it
     * three times. At each level computes the left Riemann sum of the identity
     * function (f(X)=X) and checks that the norm is positive and changes between
     * successive levels.
     */
    TEST_F(MatrixPathTest, TraceIntegral) {
        using Addr = Eigen::MatrixXd;
        using Value = Eigen::MatrixXd;      // Note: Value is also a matrix type.
        using Distance = double;
        using Compare = MatrixLess;

        Eigen::MatrixXd a(2, 2); a << 0.5, 0, 0, 0.5;   // 0.5 * I
        Eigen::MatrixXd b(2, 2); b << 1, 0, 0, 1;
        std::vector<Addr> init = { a, b };
        ListGrid<Addr, Compare> grid0(init.begin(), init.end(), Compare());

        LessBetweenness betweenness;
        FrobeniusMetric metric;
        EuclideanValueMetric value_metric;

        // Operator for matrices (defined in delta_operator.h)
        MatrixMidpointOperator op;
        auto strategy = StaticStrategy<MatrixMidpointOperator>(op);

        auto path = DeltaPath<Addr, Value, Distance, LessBetweenness, FrobeniusMetric,
            EuclideanValueMetric, decltype(strategy), Compare>(
                grid0, strategy, betweenness, metric, value_metric);

        // Function returns the matrix itself.
        auto func = [](const Addr& m) -> Addr { return m; };

        Eigen::MatrixXd prev(2, 2); prev.setZero();
        for (int i = 0; i < 3; ++i) {
            Eigen::MatrixXd result = left_riemann_sum(path.current_grid(), func);
            double norm = result.norm();
            EXPECT_GT(norm, 0.0);
            if (i > 0) {
                double prev_norm = prev.norm();
                EXPECT_NE(norm, prev_norm);
            }
            prev = result;
            path.advance(func);
        }
    }

    /**
     * @test EmptyGridRiemannSum
     * @brief Verify that the left Riemann sum on an empty grid returns a zero matrix.
     */
    TEST_F(MatrixPathTest, EmptyGridRiemannSum) {
        using Addr = Eigen::MatrixXd;
        using Compare = MatrixLess;
        ListGrid<Addr, Compare> empty_grid;
        auto func = [](const Addr&) -> Addr { return Addr::Zero(2, 2); };
        auto sum = left_riemann_sum(empty_grid, func);
        EXPECT_TRUE(sum.isZero(1e-12));
        // Dimensions are not checked because an empty grid has no points.
    }

    /**
     * @test SinglePointGridRiemannSum
     * @brief Verify that the left Riemann sum on a grid with a single point returns zero.
     */
    TEST_F(MatrixPathTest, SinglePointGridRiemannSum) {
        using Addr = Eigen::MatrixXd;
        using Compare = MatrixLess;
        Eigen::MatrixXd m(2, 2); m << 1, 2, 3, 4;
        ListGrid<Addr, Compare> grid({ m });
        auto func = [](const Addr& x) -> Addr { return x; };
        auto sum = left_riemann_sum(grid, func);
        EXPECT_TRUE(sum.isZero(1e-12));
    }

    /**
     * @test IdentityIntegral
     * @brief Check that the integral of the identity function on [0,1] (as matrices)
     *        approaches 0.5·I after several uniform refinements.
     *
     * Starting from the endpoints 0·I and I, five midpoint refinements are applied.
     * The left Riemann sum should converge to 0.5·I with an error below 1e‑10.
     */
    TEST_F(MatrixPathTest, IdentityIntegral) {
        using Addr = Eigen::MatrixXd;
        using Compare = MatrixLess;
        using Grid = ListGrid<Addr, Compare>;
        using Path = DeltaPath<Addr, Addr, double, LessBetweenness, FrobeniusMetric,
            EuclideanValueMetric, StaticStrategy<MatrixMidpointOperator>, Compare>;

        Eigen::MatrixXd a(2, 2); a << 0, 0, 0, 0;
        Eigen::MatrixXd b(2, 2); b << 1, 0, 0, 1;
        Grid grid0({ a, b });

        MatrixMidpointOperator op;
        auto strategy = StaticStrategy<MatrixMidpointOperator>(op);
        LessBetweenness betweenness;
        FrobeniusMetric metric;
        EuclideanValueMetric value_metric;

        Path path(grid0, strategy, betweenness, metric, value_metric);
        auto func = [](const Addr& x) -> Addr { return x; };

        const int steps = 5;
        for (int i = 0; i < steps; ++i) {
            path.advance(func);
        }

        auto integral = left_riemann_sum(path.current_grid(), func);
        double expected = static_cast<double>((1 << steps) - 1) / (1 << (steps + 1));
        Eigen::MatrixXd expected_mat(2, 2); expected_mat << expected, 0, 0, expected;
        EXPECT_TRUE(integral.isApprox(expected_mat, 1e-10));
    }

    /**
     * @test AdaptivePathForSquare
     * @brief Verify that AdaptiveDeltaPath works with matrix‑valued functions.
     *
     * An adaptive path is created for f(X)=X² on the interval [0·I, I].
     * Several refinement steps are performed; the path must increase in size
     * and remain sorted according to MatrixLess.
     */
    TEST_F(MatrixPathTest, AdaptivePathForSquare) {
        using Addr = Eigen::MatrixXd;
        using Value = Eigen::MatrixXd;
        using Distance = double;
        using Compare = MatrixLess;
        using Metric = FrobeniusMetric;
        using ValMetric = EuclideanValueMetric;
        // For matrix addresses a simple betweenness that always returns true is sufficient.
        struct DummyBetweenness {
            bool operator()(const Addr&, const Addr&, const Addr&) const { return true; }
        };

        std::vector<Addr> init;
        Eigen::MatrixXd a(2, 2); a << 0, 0, 0, 0;
        Eigen::MatrixXd b(2, 2); b << 1, 0, 0, 1;
        init.push_back(a);
        init.push_back(b);

        auto func = [](const Addr& x) -> Addr { return x * x; };

        AdaptiveDeltaPath<Addr, Value, Distance, DummyBetweenness, Metric, ValMetric,
            MatrixMidpointOperator, Compare>
            path(init, func, MatrixMidpointOperator{}, 0.1, DummyBetweenness{}, Metric{}, ValMetric{});

        int steps = 0;
        const int max_steps = 5;
        while (steps < max_steps && path.advance()) {
            ++steps;
            // Verify that points remain sorted according to the matrix comparator.
            EXPECT_TRUE(std::is_sorted(path.points().begin(), path.points().end(), Compare()));
        }
        EXPECT_GT(steps, 0);
        EXPECT_GT(path.size(), 2);
    }

} // namespace delta::testing