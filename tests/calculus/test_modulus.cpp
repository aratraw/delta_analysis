// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/calculus/test_modulus.cpp
#include <gtest/gtest.h>
#include "test_fixtures.h"
#include "delta/rational/transcendentals.h"

namespace delta::testing {

    /**
     * @class ModulusTest
     * @brief Tests for modulus of continuity classes (PowerModulus, LogarithmicModulus).
     */
    class ModulusTest : public DeltaTest {};

    /**
     * @test Verify the PowerModulus for both double and Rational types.
     */
    TEST_F(ModulusTest, PowerModulus) {
        // Double version
        PowerModulus<double> mod_d(2.0, 1.5);
        EXPECT_DOUBLE_EQ(mod_d(0.0), 0.0);
        EXPECT_DOUBLE_EQ(mod_d(4.0), 2.0 * std::pow(4.0, 1.5));
        EXPECT_NEAR(mod_d(0.25), 2.0 * std::pow(0.25, 1.5), 1e-12);

        // Rational version using exact rational arithmetic
        PowerModulus<Rational> mod_r(2_r, Rational(3, 2));
        // 2 * 4^{3/2} = 2 * (sqrt(4)^3) = 2 * 8 = 16
        // Use approximate comparison because delta::pow is approximate for non-integer exponents
        EXPECT_RATIONAL_NEAR(mod_r(4_r), 16_r, Rational(1) / 1000000000000_r);
        // 2 * (1/4)^{3/2} = 2 * (1/8) = 1/4
        EXPECT_RATIONAL_NEAR(mod_r(Rational(1, 4)), Rational(1, 4), Rational(1) / 1000000000000_r);
    }

    /**
     * @test Verify the LogarithmicModulus for double and Rational.
     */
    TEST_F(ModulusTest, LogarithmicModulus) {
        // Double version
        LogarithmicModulus<double> mod_d(1.0, 2.0);
        double delta = 0.1;
        double expected = 1.0 / std::pow(std::abs(std::log(delta)), 2.0);
        EXPECT_NEAR(mod_d(delta), expected, 1e-12);
        EXPECT_TRUE(std::isinf(mod_d(0.0)));

        // Rational version
        LogarithmicModulus<Rational> mod_r(1_r, 2_r);
        Rational delta_r = Rational(1, 10); // 0.1
        // Compute expected: 1 / (ln(0.1))^2 using exact rational functions
        Rational log_delta = delta::log(delta_r);
        Rational expected_r = 1_r / (log_delta * log_delta);
        Rational result = mod_r(delta_r);
        // Allow small tolerance due to series approximations
        EXPECT_RATIONAL_NEAR(result, expected_r, Rational(1, 1000000000000));

        // Test exception for non-positive delta
        EXPECT_THROW(mod_r(Rational(0)), std::domain_error);
        EXPECT_THROW(mod_r(Rational(-1)), std::domain_error);
    }

    /**
     * @test Check that the modulus classes satisfy the Modulus concept.
     */
    TEST_F(ModulusTest, ModulusConcept) {
        static_assert(Modulus<PowerModulus<double>, double>);
        static_assert(Modulus<LogarithmicModulus<double>, double>);
        static_assert(Modulus<PowerModulus<Rational>, Rational>);
        static_assert(Modulus<LogarithmicModulus<Rational>, Rational>);
    }

    // -------------------------------------------------------------------------
    // Testing check_continuity_level with different moduli (Rational)
    // -------------------------------------------------------------------------

    /**
     * @class ContinuityModulusTest
     * @brief Tests continuity checks using various moduli on a dyadic path.
     */
    class ContinuityModulusTest : public DeltaTest {
    protected:
        void SetUp() override {
            path_ = std::make_unique<DeltaPath<Addr, Rational, Dist,
                Between, AddrMetric, ValMetric,
                decltype(make_midpoint_strategy()), Compare>>(
                    ListGrid<Addr, Compare>({ 0_r, 1_r }),
                    make_midpoint_strategy(),
                    Between{}, AddrMetric{}, ValMetric{}
                    );
        }

        std::unique_ptr<DeltaPath<Addr, Rational, Dist,
            Between, AddrMetric, ValMetric,
            decltype(make_midpoint_strategy()), Compare>> path_;
    };

    /**
     * @test Identity function f(x)=x, for which |Δf| equals the grid step.
     */
    TEST_F(ContinuityModulusTest, IdentityWithPowerModulus) {
        auto func = [](const Addr& x) { return x; };
        ValMetric vm;

        PowerModulus<Rational> mod(1_r, 1_r);

        for (int n = 0; n < 5; ++n) {
            const auto& grid = path_->current_grid();
            bool ok = check_continuity_level(grid, func, vm, mod, Rational(1, 1000000000000));
            EXPECT_TRUE(ok) << "Failed at level " << n;
            if (n < 4) path_->advance(func);
        }
    }

    /**
     * @test Quadratic function f(x)=x².
     */
    TEST_F(ContinuityModulusTest, QuadraticWithPowerModulus) {
        auto func = [](const Addr& x) { return x * x; };
        ValMetric vm;

        PowerModulus<Rational> mod(2_r, 1_r);

        for (int n = 0; n < 5; ++n) {
            const auto& grid = path_->current_grid();
            bool ok = check_continuity_level(grid, func, vm, mod, Rational(1, 1000000000000));
            EXPECT_TRUE(ok) << "Failed at level " << n;
            if (n < 4) path_->advance(func);
        }
    }

    /**
     * @test Square root function f(x)=√x (Hölder with α=0.5).
     */
    TEST_F(ContinuityModulusTest, SqrtWithHolderModulus) {

        internal::reset_default_eps();
        auto func = [](const Addr& x) -> Rational {
            return delta::sqrt(x);
            };
        ValMetric vm;

        PowerModulus<Rational> mod(1_r, Rational(1, 2));

        for (int n = 0; n < 10; ++n) {
            const auto& grid = path_->current_grid();
            bool ok = check_continuity_level(grid, func, vm, mod, Rational(1, 1000000));
            EXPECT_TRUE(ok) << "Failed at level " << n;
            if (n < 9) path_->advance(func);
        }
    }

    /**
     * @test Square root with a linear modulus should fail.
     */
    TEST_F(ContinuityModulusTest, SqrtFailsWithLinearModulus) {

        internal::reset_default_eps();
        auto func = [](const Addr& x) -> Rational {
            return delta::sqrt(x);
            };
        ValMetric vm;

        PowerModulus<Rational> mod(1_r, 1_r);

        bool all_ok = true;
        for (int n = 0; n < 10; ++n) {
            const auto& grid = path_->current_grid();
            bool ok = check_continuity_level(grid, func, vm, mod, Rational(1, 1000000));
            if (!ok) all_ok = false;
            if (n < 9) path_->advance(func);
        }
        EXPECT_FALSE(all_ok);
    }

    // -------------------------------------------------------------------------
    // Testing check_differentiability with moduli
    // -------------------------------------------------------------------------

    /**
     * @class DifferentiabilityModulusTest
     * @brief Tests differentiability checks using various moduli.
     */
    class DifferentiabilityModulusTest : public DeltaTest {
    protected:
        void SetUp() override {

            internal::reset_default_eps();
            ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
            auto path = make_midpoint_path(grid0);
            auto func = [](const Addr& x) { return x; };

            grids_.push_back(path.current_grid());
            for (int i = 0; i < 5; ++i) {
                path.advance(func);
                grids_.push_back(path.current_grid());
            }
        }

        std::vector<ListGrid<Addr, Compare>> grids_;
    };

    /**
     * @test Identity function f(x)=x, derivative 1, error = 0.
     */
    TEST_F(DifferentiabilityModulusTest, Identity) {
        auto func = [](const Addr& x) { return x; };
        Addr x = 1_r / 2_r;
        Dist D = 1_r;
        PowerModulus<Rational> mod(0_r, 1_r);
        Rational tolerance = Rational(1, 1000000000000);
        bool diff = check_differentiability(grids_, x, func, D, mod, 1, tolerance);
        EXPECT_TRUE(diff);
    }

    /**
     * @test Quadratic f(x)=x², derivative 2x, error ≤ grid step.
     */
    TEST_F(DifferentiabilityModulusTest, Quadratic) {
        auto func = [](const Addr& x) { return x * x; };
        Addr x = 1_r / 2_r;
        Dist D = 1_r; // 2*0.5 = 1
        PowerModulus<Rational> mod(1_r, 1_r);
        Rational tolerance = Rational(1, 1000000000000);
        bool diff = check_differentiability(grids_, x, func, D, mod, 1, tolerance);
        EXPECT_TRUE(diff);
    }

    /**
     * @test Absolute value at zero — not differentiable.
     */
    TEST_F(DifferentiabilityModulusTest, AbsoluteValue) {
        ListGrid<Addr, Compare> grid0({ -1_r, 0_r, 1_r });
        auto path = make_midpoint_path(grid0);
        auto func = [](const Addr& x) -> Rational {
            return delta::abs(x);
            };

        std::vector<ListGrid<Addr, Compare>> grids;
        grids.push_back(path.current_grid());
        for (int i = 0; i < 5; ++i) {
            path.advance(func);
            grids.push_back(path.current_grid());
        }

        Addr x = 0_r;
        Dist D = 0_r;
        PowerModulus<Rational> mod(1_r, 1_r);
        Rational tolerance = Rational(1, 1000000000000);
        bool diff = check_differentiability(grids, x, func, D, mod, 0, tolerance);
        EXPECT_FALSE(diff);
    }

} // namespace delta::testing