// tests/calculus/test_modulus.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "test_fixtures.h"

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
        // test the double version
        PowerModulus<double> mod_d(2.0, 1.5);
        EXPECT_DOUBLE_EQ(mod_d(0.0), 0.0);
        EXPECT_DOUBLE_EQ(mod_d(4.0), 2.0 * std::pow(4.0, 1.5));
        EXPECT_NEAR(mod_d(0.25), 2.0 * std::pow(0.25, 1.5), 1e-12);

        // test the Rational version
        PowerModulus<Rational> mod_r(2_r, Rational(3, 2)); // 2 * delta^1.5
        // we expect approximate equality
        EXPECT_NEAR(mod_r(4_r).convert_to<double>(), 2.0 * std::pow(4.0, 1.5), 1e-12);
    }

    /**
     * @test Verify the LogarithmicModulus for double.
     */
    TEST_F(ModulusTest, LogarithmicModulus) {
        LogarithmicModulus<double> mod(1.0, 2.0);
        double delta = 0.1;
        double expected = 1.0 / std::pow(abs(std::log(delta)), 2.0);
        EXPECT_NEAR(mod(delta), expected, 1e-12);
        EXPECT_TRUE(std::isinf(mod(0.0)));
    }

    /**
     * @test Check that the modulus classes satisfy the Modulus concept.
     */
    TEST_F(ModulusTest, ModulusConcept) {
        static_assert(Modulus<PowerModulus<double>, double>);
        static_assert(Modulus<LogarithmicModulus<double>, double>);
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
        auto func = [](const Addr& x) { return x; }; // returns Rational
        ValMetric vm; // EuclideanValueMetric for Rational

        PowerModulus<Rational> mod(1_r, 1_r);

        for (int n = 0; n < 5; ++n) {
            const auto& grid = path_->current_grid();
            bool ok = check_continuity_level(grid, func, vm, mod, 1e-12);
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
            bool ok = check_continuity_level(grid, func, vm, mod, 1e-12);
            EXPECT_TRUE(ok) << "Failed at level " << n;
            if (n < 4) path_->advance(func);
        }
    }

    /**
     * @test Square root function f(x)=√x (Hölder with α=0.5).
     */
    TEST_F(ContinuityModulusTest, SqrtWithHolderModulus) {
        auto func = [](const Addr& x) -> Rational {
            double val = std::sqrt(x.convert_to<double>());
            return Rational(static_cast<int64_t>(val * 1e12), 1e12); // approximation
            };
        ValMetric vm;

        PowerModulus<Rational> mod(1_r, Rational(1, 2)); // alpha = 0.5

        for (int n = 0; n < 10; ++n) {
            const auto& grid = path_->current_grid();
            bool ok = check_continuity_level(grid, func, vm, mod, 1e-6);
            EXPECT_TRUE(ok) << "Failed at level " << n;
            if (n < 9) path_->advance(func);
        }
    }

    /**
     * @test Square root with a linear modulus should fail.
     */
    TEST_F(ContinuityModulusTest, SqrtFailsWithLinearModulus) {
        auto func = [](const Addr& x) -> Rational {
            double val = std::sqrt(x.convert_to<double>());
            return Rational(static_cast<int64_t>(val * 1e12), 1e12);
            };
        ValMetric vm;

        PowerModulus<Rational> mod(1_r, 1_r); // linear

        bool all_ok = true;
        for (int n = 0; n < 10; ++n) {
            const auto& grid = path_->current_grid();
            bool ok = check_continuity_level(grid, func, vm, mod, 1e-6);
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
            ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
            auto path = make_midpoint_path(grid0);
            auto func = [](const Addr& x) { return x; }; // identity, Rational

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
        bool diff = check_differentiability(grids_, x, func, D, mod, 1);
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
        bool diff = check_differentiability(grids_, x, func, D, mod, 1);
        EXPECT_TRUE(diff);
    }

    /**
     * @test Absolute value at zero — not differentiable.
     */
    TEST_F(DifferentiabilityModulusTest, AbsoluteValue) {
        ListGrid<Addr, Compare> grid0({ -1_r, 0_r, 1_r });
        auto path = make_midpoint_path(grid0);
        auto func = [](const Addr& x) -> Rational {
            double xd = x.convert_to<double>();
            return Rational(static_cast<int64_t>(abs(xd) * 1e12), 1e12);
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
        bool diff = check_differentiability(grids, x, func, D, mod, 0);
        EXPECT_FALSE(diff);
    }

} // namespace delta::testing