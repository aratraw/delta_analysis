// tests/calculus/test_modulus_continuity.cpp
#include <gtest/gtest.h>
#include "test_fixtures.h"
#include "delta/calculus/modulus.h"
#include "delta/rational/transcendentals.h"  // for delta::sqrt

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
        ScopedEagerEval eager; // принудительно вычислять всё сразу без ленивых узлов
        //ТРАНСЦЕНДЕНТНЫЕ ФУНКЦИИ ВЫЗЫВАЮТ ЛЕНИВОСТЬ. ЛЕНИВЫЕ УЗЛЫ КОПЯТСЯ В ПУЛ. ОЧИСТКУ ПУЛА МЫ ЕЩЁ НЕ ЗАВЕЗЛИ.
        ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
        auto path = make_midpoint_path(grid0);

        // функция возвращает вычисленное значение (уже не ленивое)
        auto func = [](const Addr& x) -> Rational {
            return delta::sqrt(x).eval();
            };

        const int MAX_LEVEL = 10;
        double M = 1.0;
        double gamma = 0.5;
        calculus::PowerModulus<double> mod(M, gamma);

        for (int n = 0; n <= MAX_LEVEL; ++n) {
            const auto& grid = path.current_grid();
            for (std::size_t i = 0; i + 1 < grid.size(); ++i) {
                Addr left = grid[i];
                Addr right = grid[i + 1];
                // явно вычисляем адреса, чтобы они были простыми числами
                Rational left_val = left.eval();
                Rational right_val = right.eval();
                Rational dx = right_val - left_val;
                Rational df = delta::abs(delta::sqrt(right_val) - delta::sqrt(left_val));
                double bound = mod(dx.to_double());
                EXPECT_LE(df.to_double(), bound + 1e-12)
                    << "Failed at level " << n << " interval ["
                    << left_val.to_double() << "," << right_val.to_double() << "]";
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
                Addr left = grid[i];
                Addr right = grid[i + 1];
                Rational dx = right - left;
                Rational df = delta::abs(right - left);
                double bound = mod(dx.to_double());
                EXPECT_LE(df.to_double(), bound + 1e-12);
            }
            if (n < 5) path.advance(func);
        }
    }

} // namespace delta::testing