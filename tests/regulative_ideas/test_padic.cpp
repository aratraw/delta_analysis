// tests/regulative_ideas/test_padic.cpp
#include <gtest/gtest.h>
#include "../test_fixtures.h"
#include "delta/calculus/continuity.h"
#include "delta/calculus/modulus.h"
#include "delta/calculus/riemann_sum.h"

namespace delta::testing {

    /**
     * @class PAdicPathTest
     * @brief Template test fixture for checking Δ‑analysis concepts with p‑adic metric.
     *
     * This fixture sets up a dyadic path on [0,1] with the p‑adic metric (PAdicMetric<p>)
     * and Euclidean metric for values. It is used to test continuity, differentiability,
     * and integration in the context of a regulative idea based on p‑adic numbers.
     *
     * @tparam p The prime defining the p‑adic metric.
     */
    template<int p>
    class PAdicPathTest : public DeltaTest {
    protected:
        using Addr = Rational;
        using Value = Rational;
        using Distance = Rational;
        using Compare = std::less<Addr>;
        // For the p‑adic path, the betweenness relation is not crucial.
        using Betweenness = LessBetweenness;
        using Metric = PAdicMetric<p>;
        using ValueMetric = EuclideanValueMetric;

        void SetUp() override {
            ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
            betweenness_ = Betweenness{};
            metric_ = Metric{};
            value_metric_ = ValueMetric{};
            op_ = MidpointOperator{};
            strategy_ = std::make_unique<StaticStrategy<MidpointOperator>>(op_);
            path_ = std::make_unique<DeltaPath<Addr, Value, Distance, Betweenness, Metric, ValueMetric,
                StaticStrategy<MidpointOperator>, Compare>>(
                    grid0, *strategy_, betweenness_, metric_, value_metric_);
        }

        Betweenness betweenness_;
        Metric metric_;
        ValueMetric value_metric_;
        MidpointOperator op_;
        std::unique_ptr<StaticStrategy<MidpointOperator>> strategy_;
        std::unique_ptr<DeltaPath<Addr, Value, Distance, Betweenness, Metric, ValueMetric,
            StaticStrategy<MidpointOperator>, Compare>> path_;
    };

    // Explicit instantiations for p=2 and p=3.
    using PAdicPathTest2 = PAdicPathTest<2>;
    using PAdicPathTest3 = PAdicPathTest<3>;

    /**
     * @test ConstantFunction (p=2)
     * @brief Verify that a constant function satisfies continuity with a zero modulus
     *        on the dyadic path under the 2‑adic metric.
     *
     * For any constant function, the oscillation is zero, so the continuity condition
     * holds regardless of the metric.
     */
    TEST_F(PAdicPathTest2, ConstantFunction) {
        auto func = [](const Addr&) { return Rational(5); };
        PowerModulus<Rational> modulus(0_r, 1_r);
        for (int n = 0; n < 5; ++n) {
            const auto& grid = path_->current_grid();
            bool ok = check_continuity_level(grid, func, value_metric_, modulus, Rational(1, 1000000000000));
            EXPECT_TRUE(ok) << "Failed at level " << n;
            if (n < 4) path_->advance(func);
        }
    }

    /**
     * @test ConstantFunction (p=3)
     * @brief Same as above but for the 3‑adic metric.
     */
    TEST_F(PAdicPathTest3, ConstantFunction) {
        auto func = [](const Addr&) { return Rational(5); };
        PowerModulus<Rational> modulus(0_r, 1_r);
        for (int n = 0; n < 5; ++n) {
            const auto& grid = path_->current_grid();
            bool ok = check_continuity_level(grid, func, value_metric_, modulus, Rational(1, 1000000000000));
            EXPECT_TRUE(ok) << "Failed at level " << n;
            if (n < 4) path_->advance(func);
        }
    }

    /**
     * @test DivisibilityFunction
     * @brief A sanity check for a function that depends on divisibility by p.
     *
     * The function returns 1 if the address is an integer divisible by p, otherwise 0.
     * Continuity in the p‑adic metric is non‑trivial; here we only verify that the
     * code runs without errors (no assertions or exceptions). The modulus used is
     * intentionally simple; the exact continuity properties are not checked.
     */
    TEST_F(PAdicPathTest2, DivisibilityFunction) {
        // Function: 1 if x is an integer divisible by p, else 0.
        auto func = [](const Addr& x) -> Rational {
            int num = x.numerator().convert_to<int>();
            int den = x.denominator().convert_to<int>();
            // For simplicity, consider only integers (denominator == 1).
            if (den == 1) {
                return (num % 2 == 0) ? Rational(1) : Rational(0);
            }
            return Rational(0);
            };
        // The continuity in p‑adic metric is not obvious; we just check that the call
        // does not crash.
        PowerModulus<Rational> modulus(1_r, 1_r);
        for (int n = 0; n < 3; ++n) {
            const auto& grid = path_->current_grid();
            check_continuity_level(grid, func, value_metric_, modulus, Rational(1, 1000000000000));
            path_->advance(func);
        }
        // Reaching this point means no exception was thrown.
        SUCCEED();
    }

    /**
     * @test EmptyGridRiemannSum
     * @brief Verify that the left Riemann sum on an empty grid returns zero.
     */
    TEST_F(PAdicPathTest2, EmptyGridRiemannSum) {
        ListGrid<Addr, Compare> empty_grid;
        auto func = [](const Addr&) { return Rational(0); };
        auto sum = left_riemann_sum(empty_grid, func);
        EXPECT_EQ(sum, 0_r);
    }

    /**
     * @test SinglePointGridRiemannSum
     * @brief Verify that the left Riemann sum on a grid with a single point returns zero.
     */
    TEST_F(PAdicPathTest2, SinglePointGridRiemannSum) {
        ListGrid<Addr, Compare> grid({ 5_r });
        auto func = [](const Addr& x) { return x; };
        auto sum = left_riemann_sum(grid, func);
        EXPECT_EQ(sum, 0_r);
    }

    /**
     * @test IdentityDifferentiability
     * @brief Check that the identity function f(x)=x is differentiable at x=1/2
     *        with derivative 1, using a zero modulus (exact equality).
     */
    TEST_F(PAdicPathTest2, IdentityDifferentiability) {
        // Build a sequence of grids as in the calculus tests.
        ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
        auto path = make_midpoint_path(grid0);
        auto func = [](const Addr& x) { return x; };

        using Grid = ListGrid<Addr, Compare>;
        std::vector<Grid> grids;
        grids.push_back(path.current_grid());
        for (int i = 0; i < 5; ++i) {
            path.advance(func);
            grids.push_back(path.current_grid());
        }

        Addr x = 1_r / 2_r;
        Distance D = 1_r;
        PowerModulus<Rational> modulus(0_r, 1_r); // zero modulus because error is zero
        Rational tolerance = Rational(1, 1000000000000); // 1e-12 as Rational

        std::size_t first_level = 0;
        for (; first_level < grids.size(); ++first_level) {
            if (find_address_index(grids[first_level], x) >= 0) break;
        }
        ASSERT_LT(first_level, grids.size());

        bool diff = check_differentiability(grids, x, func, D, modulus, first_level, tolerance);
        EXPECT_TRUE(diff);
    }

    /**
     * @test IdentityIntegral
     * @brief Verify that the left Riemann sum of f(x)=x on [0,1] converges to 1/2
     *        under the p‑adic metric (the metric does not affect the integral value,
     *        only the grid construction; here we use the usual dyadic refinement).
     */
    TEST_F(PAdicPathTest2, IdentityIntegral) {
        ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
        auto path = make_midpoint_path(grid0);
        auto func = [](const Addr& x) { return x; };

        const int steps = 10;
        for (int i = 0; i < steps; ++i) {
            path.advance(func);
        }

        auto integral = left_riemann_sum(path.current_grid(), func);
        EXPECT_RATIONAL_NEAR(integral, 1_r / 2_r, Rational(1, 1000));
    }

    /**
     * @test AdaptivePathForSquare
     * @brief Test that AdaptiveDeltaPath works with the p‑adic metric for f(x)=x².
     *
     * An adaptive path is created with a small threshold. After several refinements,
     * the point set must be strictly increasing and the size must have increased.
     */
    TEST_F(PAdicPathTest2, AdaptivePathForSquare) {
        std::vector<Addr> init = { 0_r, 1_r };
        auto func = [](const Addr& x) { return x * x; };
        AdaptiveDeltaPath<Addr, Value, Distance, Betweenness, Metric, ValueMetric, MidpointOperator, Compare>
            path(init, func, MidpointOperator{}, Rational(1, 100), betweenness_, metric_, value_metric_);

        int steps = 0;
        const int max_steps = 10;
        while (steps < max_steps && path.advance()) {
            ++steps;
            EXPECT_TRUE(is_sorted_set(path.points()));
        }
        EXPECT_GT(steps, 0);
        EXPECT_GT(path.size(), 2);
    }

} // namespace delta::testing