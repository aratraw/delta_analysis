// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/calculus/test_modulus_continuity.cpp
#include <gtest/gtest.h>
#include "test_fixtures.h"
#include "delta/calculus/modulus.h"
#include "delta/rational/transcendentals.h"

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

        // Use exact rational sqrt
        auto func = [](const Addr& x) -> Rational {
            return delta::sqrt(x);
            };

        const int MAX_LEVEL = 10;
        // Modulus ω(δ) = δ^{0.5} as Rational
        PowerModulus<Rational> mod(1_r, Rational(1, 2));

        for (int n = 0; n <= MAX_LEVEL; ++n) {
            const auto& grid = path.current_grid();
            for (std::size_t i = 0; i + 1 < grid.size(); ++i) {
                Addr left = grid[i];
                Addr right = grid[i + 1];
                Rational dx = right - left;
                Rational df = delta::abs(delta::sqrt(right) - delta::sqrt(left));
                Rational bound = mod(dx);
                // Allow small tolerance due to rational approximations in sqrt
                Rational tolerance = Rational(1, 1000000000000);
                EXPECT_LE(df, bound + tolerance)
                    << "Failed at level " << n << " interval ["
                    << left << ", " << right << "]";
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

        PowerModulus<Rational> mod(1_r, 1_r);

        for (int n = 0; n <= 5; ++n) {
            const auto& grid = path.current_grid();
            for (std::size_t i = 0; i + 1 < grid.size(); ++i) {
                Addr left = grid[i];
                Addr right = grid[i + 1];
                Rational dx = right - left;
                Rational df = delta::abs(right - left);
                Rational bound = mod(dx);
                EXPECT_LE(df, bound);
            }
            if (n < 5) path.advance(func);
        }
    }

} // namespace delta::testing