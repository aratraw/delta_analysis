// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/test_fixtures.h
#pragma once

#include <gtest/gtest.h>
#include <vector>
#include <set>
#include <random>
#include "delta/core/rational.h"
#include "delta/core/regulative_idea.h"
#include "delta/core/value_metric.h"
#include "delta/core/list_grid.h"
#include "delta/core/uniform_grid.h"
#include "delta/core/grid_refine.h"
#include "delta/core/delta_operator.h"
#include "delta/core/delta_strategy.h"
#include "delta/core/delta_path.h"
#include "delta/core/adaptive_delta_path.h"
#include "delta/core/operational_function.h"
#include "delta/core/completion.h"
#include "delta/calculus/modulus.h"
#include "delta/calculus/continuity.h"
#include "delta/calculus/differentiability.h"

namespace delta::testing {
    using namespace delta;
    using namespace delta::calculus;
    using Addr = Rational;
    using Val = Rational;
    using Dist = Rational;
    using Between = LessBetweenness;
    using AddrMetric = EuclideanMetric;
    using ValMetric = EuclideanValueMetric;
    using Compare = std::less<Addr>;

    /**
     * @class DeltaTest
     * @brief Base test fixture for all Δ‑analysis tests.
     *
     * Provides common type aliases (Addr, Val, Dist, etc.) and utility methods
     * for checking grid sortedness, bounds, rational near‑equality, and for
     * creating IntervalInfo objects. It also includes factory functions for
     * constructing various paths and strategies.
     */
    class DeltaTest : public ::testing::Test {
    protected:
        // -------------------------------------------------------------------------
        // Precision management (inherit from DeltaTest, but we add convenience)
        // -------------------------------------------------------------------------
        void SetUp() override {
            old_precision_ = default_eps();
        }

        void TearDown() override {
            set_default_eps(old_precision_);
        }

        static void set_precision(const Rational& eps) {
            set_default_eps(eps);
        }
        //if something breaks miserably - blame these usings.
        using Addr = testing::Addr;
        using Val = testing::Val;
        using Dist = testing::Dist;
        using Between = testing::Between;
        using AddrMetric = testing::AddrMetric;
        using ValMetric = testing::ValMetric;
        using Compare = testing::Compare;



        /**
         * @brief Checks that a grid is sorted according to its comparator.
         * @tparam Grid A type satisfying GridConcept.
         * @param grid The grid to check.
         * @return true if the grid is strictly increasing.
         */
        template<typename Grid>
        bool is_sorted(const Grid& grid) const {
            return std::is_sorted(grid.begin(), grid.end(), grid.comparator());
        }

        /**
         * @brief Checks that a boost::container::flat_set is strictly increasing.
         * @tparam Set A flat_set type.
         * @param s The set to check.
         * @return true if elements are in increasing order.
         */
        template<typename Set>
        bool is_sorted_set(const Set& s) const {
            if (s.empty()) return true;
            auto it = s.begin();
            auto next = std::next(it);
            while (next != s.end()) {
                if (!(*it < *next)) return false;
                ++it;
                ++next;
            }
            return true;
        }

        /**
         * @brief Verifies that the first and last elements of a grid match the expected bounds.
         * @tparam Grid A type satisfying GridConcept.
         * @param grid The grid to check.
         * @param start Expected first element.
         * @param end Expected last element.
         * @return true if the grid is non‑empty and its endpoints match.
         */
        template<typename Grid>
        bool bounds_match(const Grid& grid, const Addr& start, const Addr& end) const {
            return grid.size() > 0 && grid[0] == start && grid[grid.size() - 1] == end;
        }

        /**
         * @brief Compares two rational numbers with a tolerance.
         * @param a First number.
         * @param b Second number.
         * @param eps Allowed absolute difference (default 1e-6).
         * @return true if |a - b| ≤ eps.
         */
        static bool near(const Rational& a, const Rational& b, const Rational& eps = Rational(1, 1000000)) {
            Rational diff = a - b;
            if (diff < 0) diff = -diff;
            return diff <= eps;
        }

        /**
         * @brief Creates an IntervalInfo object for testing operators.
         * @tparam ValType Type of function values (default Val = Rational).
         * @param left Left endpoint.
         * @param right Right endpoint.
         * @param f_left Value at left.
         * @param f_right Value at right.
         * @param max_osc Maximum oscillation on the current level.
         * @param level Current refinement level (default 0).
         * @return IntervalInfo populated with the given data and default betweenness/metrics.
         */
        template<typename ValType = Val>
        auto make_info(const Addr& left, const Addr& right,
            const ValType& f_left, const ValType& f_right,
            const Dist& max_osc, std::size_t level = 0) const {
            return IntervalInfo<Addr, ValType, Dist, Between, AddrMetric, ValMetric>(
                left, right, level, f_left, f_right, max_osc,
                Between{}, AddrMetric{}, ValMetric{});
        }

    private:
        Rational old_precision_= default_eps();;
    };

    // -------------------------------------------------------------------------
    // Factory functions for constructing paths and strategies
    // -------------------------------------------------------------------------

    /**
     * @brief Creates a static strategy that always uses the MidpointOperator.
     */
    inline auto make_midpoint_strategy() {
        return StaticStrategy<MidpointOperator>(MidpointOperator{});
    }

    /**
     * @brief Creates a static strategy that uses a FixedLambdaOperator with the given lambda.
     * @param lambda Fraction of the interval where the new point is placed (must be in (0,1)).
     */
    inline auto make_lambda_strategy(const Rational& lambda) {
        return StaticStrategy<FixedLambdaOperator>(FixedLambdaOperator(lambda));
    }

    /**
     * @brief Creates a dynamic strategy from a vector of operators.
     * @tparam Op Operator type (e.g., FixedLambdaOperator).
     * @param ops Vector of operators; for level i, operator ops[i] is used if i < ops.size(),
     *            otherwise the last operator is repeated.
     */
    template<typename Op>
    inline auto make_dynamic_strategy(const std::vector<Op>& ops) {
        return DynamicStrategy<Op>(ops);
    }

    /**
     * @brief Creates a DeltaPath with the given strategy.
     * @tparam Strategy Type of strategy (e.g., StaticStrategy<MidpointOperator>).
     * @param grid Initial grid.
     * @param strategy Strategy to use for refinement.
     * @return DeltaPath object.
     */
    template<typename Strategy>
    inline auto make_path(const ListGrid<Addr, Compare>& grid, Strategy&& strategy) {
        return DeltaPath<Addr, Val, Dist, Between, AddrMetric, ValMetric, std::decay_t<Strategy>, Compare>(
            grid, std::forward<Strategy>(strategy), Between{}, AddrMetric{}, ValMetric{});
    }

    /**
     * @brief Shortcut to create a DeltaPath that uses the midpoint operator.
     * @param grid Initial grid.
     * @return DeltaPath with midpoint strategy.
     */
    inline auto make_midpoint_path(const ListGrid<Addr, Compare>& grid) {
        return make_path(grid, make_midpoint_strategy());
    }

    /**
     * @brief Creates an AdaptiveDeltaPath.
     * @tparam Op Operator type.
     * @param init Initial addresses (must be sorted).
     * @param func Function to compute values.
     * @param op Delta operator.
     * @param threshold Priority threshold; intervals with priority ≤ threshold are not refined.
     * @return AdaptiveDeltaPath object.
     */
    template<typename Op>
    inline auto make_adaptive_path(const std::vector<Addr>& init,
        std::function<Val(const Addr&)> func,
        Op&& op,
        const Dist& threshold = Dist(0)) {
        return AdaptiveDeltaPath<Addr, Val, Dist, Between, AddrMetric, ValMetric, std::decay_t<Op>, Compare>(
            init, func, std::forward<Op>(op), threshold, Between{}, AddrMetric{}, ValMetric{});
    }

    /**
     * @brief Computes 2^n for small n (used in tests).
     * @param n Exponent.
     * @return Rational representing 2^n.
     */
    static inline Rational pow2(std::size_t n) {
        Rational result = 1;
        for (std::size_t i = 0; i < n; ++i) result *= 2;
        return result;
    }

    // -------------------------------------------------------------------------
    // Macro for approximate rational comparison
    // -------------------------------------------------------------------------
#define EXPECT_RATIONAL_NEAR(val, expected, eps) \
    EXPECT_LE(delta::abs((val) - (expected)), (eps))
    // -------------------------------------------------------------------------
    // Additional fixtures for core and calculus tests
    // -------------------------------------------------------------------------

    /**
     * @brief Fixture providing instances of various metrics.
     */
    class MetricFixture : public DeltaTest {
    protected:
        EuclideanMetric euclidean;
        DiscreteMetric discrete;
        StringUltrametric stringUltra;
        PAdicMetric<2> pAdic2;
        PAdicMetric<3> pAdic3;
    };

    /**
     * @brief Generator for fields on a grid.
     * @tparam Grid Type of grid (must provide value_type as point type).
     * @tparam Value Type of field values (e.g., double, Eigen::VectorXd).
     */
    template<typename Grid, typename Value>
    class FieldGenerator {
    public:
        using Point = typename Grid::value_type;
        using Scalar = typename Point::Scalar;

        /// Constant field
        static std::function<Value(const Point&)> constant(Value c) {
            return [c](const Point&) { return c; };
        }

        /// Linear field: value = coeff·point (for scalar or vector values)
        static std::function<Value(const Point&)> linear(const Eigen::Matrix<Scalar, Point::RowsAtCompileTime, 1>& coeff) {
            return [coeff](const Point& p) -> Value {
                return coeff.dot(p.template cast<Scalar>());
                };
        }

        /// Quadratic field: value = sum coeff_i * (p_i)^2
        static std::function<Value(const Point&)> quadratic(const Eigen::Matrix<Scalar, Point::RowsAtCompileTime, 1>& coeff) {
            return [coeff](const Point& p) -> Value {
                return (p.array() * p.array()).matrix().dot(coeff);
                };
        }

        /// Sine field: sin(kx·x) * sin(ky·y) * sin(kz·z) (works for 1D,2D,3D)
        static std::function<Value(const Point&)> sine(double kx, double ky = 0, double kz = 0) {
            return [kx, ky, kz](const Point& p) -> Value {
                double v = 1.0;
                if constexpr (Point::RowsAtCompileTime >= 1) v *= std::sin(kx * p.x());
                if constexpr (Point::RowsAtCompileTime >= 2) v *= std::sin(ky * p.y());
                if constexpr (Point::RowsAtCompileTime >= 3) v *= std::sin(kz * p.z());
                return v;
                };
        }

        /// Random field (uniform in [-scale, scale])
        static std::function<Value(const Point&)> random(Scalar scale = 1.0) {
            static std::mt19937 rng(42);
            static std::uniform_real_distribution<Scalar> dist(-scale, scale);
            return [scale](const Point&) { return dist(rng); };
        }
    };

} // namespace delta::testing