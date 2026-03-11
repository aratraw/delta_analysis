// tests/calculus/test_modulus_continuity.cpp
#include <gtest/gtest.h>
#include "test_fixtures.h"
#include "delta/calculus/modulus.h"

namespace delta::testing {

    /**
     * @class ModulusContinuityTest
     * @brief Tests verifying that functions satisfy a given modulus of continuity
     *        on each interval of a refined grid.
     */
    class ModulusContinuityTest : public DeltaTest {};

    /**
     * @test Verify that the square root function satisfies the Hölder condition
     *       with exponent 0.5 on a dyadic path.
     */
    TEST_F(ModulusContinuityTest, SqrtFunctionHasHolderExponentHalf) {
        ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
        auto path = make_midpoint_path(grid0);

        auto func = [](const Addr& x) -> Rational {
            double val = std::sqrt(x.convert_to<double>());
            // Return Rational approximating the double value.
            // For modulus comparison we will use double.
            return Rational(val);
            };

        const int MAX_LEVEL = 10;
        double M = 1.0;
        double gamma = 0.5;
        calculus::PowerModulus<double> mod(M, gamma);

        for (int n = 0; n <= MAX_LEVEL; ++n) {
            const auto& grid = path.current_grid();
            for (std::size_t i = 0; i + 1 < grid.size(); ++i) {
                double left = grid[i].convert_to<double>();
                double right = grid[i + 1].convert_to<double>();
                double dx = right - left;
                double df = abs(std::sqrt(right) - std::sqrt(left));
                double bound = mod(dx);
                EXPECT_LE(df, bound + 1e-12) << "Failed at level " << n << " interval [" << left << "," << right << "]";
            }
            if (n < MAX_LEVEL) path.advance(func);
        }
    }

    /**
     * @test Verify that the identity function satisfies the linear modulus
     *       ω(δ)=δ on a dyadic path.
     */
    TEST_F(ModulusContinuityTest, ExponentialModulusForIdentity) {
        ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
        auto path = make_midpoint_path(grid0);

        auto func = [](const Addr& x) { return x; };

        double M = 1.0;
        double gamma = 1.0;
        calculus::PowerModulus<double> mod(M, gamma);

        for (int n = 0; n <= 5; ++n) {
            const auto& grid = path.current_grid();
            for (std::size_t i = 0; i + 1 < grid.size(); ++i) {
                double left = grid[i].convert_to<double>();
                double right = grid[i + 1].convert_to<double>();
                double dx = right - left;
                double df = (func(right) - func(left)).convert_to<double>();
                double bound = mod(dx);
                EXPECT_LE(df, bound + 1e-12);
            }
            if (n < 5) path.advance(func);
        }
    }

} // namespace delta::testing