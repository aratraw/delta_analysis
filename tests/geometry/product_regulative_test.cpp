// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/geometry/product_regulative_test.cpp
// ============================================================================
// TESTS FOR PRODUCT REGULATIVE IDEAS AND PRODUCT DELTA PATHS
// ============================================================================
//
// This file tests the product structures defined in product_regulative.h:
//   - ProductRegulativeIdea – product of two regulative ideas (same type).
//   - ProductDeltaPath – Cartesian product of two delta paths.
//   - Fundamental sequences and real number construction (ℝⁿ approximation).
//
// The tests verify:
//   - Betweenness and metric on product addresses (pairs of rationals).
//   - Construction and refinement of product paths (dyadic grids in ℝ²).
//   - Fundamental sequences for π and e (Leibniz series and exponential series).
//   - RealNumber construction via fundamental sequences.
//   - Dyadic grid approximation of ℝ² (density of dyadic rationals).
//
// All tests use Euclidean metric and dyadic refinement strategies.
// ============================================================================

#include <gtest/gtest.h>
#include <vector>
#include <utility>
#include "delta/geometry/product_regulative.h"
#include "delta/core/delta_path.h"
#include "delta/core/uniform_grid.h"
#include "delta/core/completion.h"
#include "../test_fixtures_geometry_numerical.h"

namespace delta::testing {
    // Base regulative idea for tests
    using BaseIdea = delta::RegulativeIdea<
        Rational,
        delta::LessBetweenness,
        delta::EuclideanMetric
    >;

    /**
     * @class ProductRegulativeTest
     * @brief Tests for product regulative ideas and product paths.
     */
    class ProductRegulativeTest : public GeometryNumericalTest {
    protected:
        using RI1D = BaseIdea;
        using PathAddr = std::array<Addr, 2>;   // use inherited Addr (Rational)

        // Types for 2D product (uses std::pair because ProductIdea expects a pair)
        using ProductAddr = std::pair<Addr, Addr>;
        using ProductBetweenness = delta::geometry::detail::ProductBetweenness<
            delta::LessBetweenness,
            delta::LessBetweenness
        >;
        using ProductMetric = delta::geometry::detail::ProductMetric<
            delta::EuclideanMetric,
            delta::EuclideanMetric
        >;
        using ProductIdea = delta::geometry::ProductRegulativeIdea<RI1D, RI1D>;

        // Path types – using inherited types Compare, Between, AddrMetric, ValMetric
        using Grid1D = delta::ListGrid<Addr, Compare>;
        using Path1D = delta::DeltaPath<
            Addr,                                   // Addr
            Val,                                    // Value
            Dist,                                   // Distance
            Between,                                // Betweenness
            AddrMetric,                             // Metric
            ValMetric,                              // ValueMetric
            delta::StaticStrategy<delta::MidpointOperator>, // Strategy
            Compare                                  // Compare
        >;

        using ProductPath = delta::geometry::ProductDeltaPath<Path1D, Path1D>;
        using Path2DFunc = typename ProductPath::Func;  // Function type for product

        // Helper: create a dyadic path from start to end
        Path1D make_dyadic_path(Addr start, Addr end) {
            std::vector<Addr> points = { start, end };
            Grid1D grid0(std::move(points), Compare());
            auto strategy = delta::testing::make_midpoint_strategy();  // fixed
            return Path1D(grid0,
                std::move(strategy),
                Between(),
                AddrMetric(),
                ValMetric());
        }

        // Check if a number is dyadic (denominator is a power of two)
        bool is_dyadic(const Addr& x) {
            if (x == 0) return false;

            auto den = x.denominator().convert_to<long long>();

            if (den == 0) return false;
            while (den % 2 == 0) {
                den /= 2;
            }
            return den == 1;
        }

        // Helper: identity function for the product path
        Path2DFunc make_identity_function() {
            return [](const std::array<Addr, 2>& addrs) -> std::array<Addr, 2> {
                return addrs;
                };
        }
    };

    // =========================================================================
    // Test group 1: Product of two 1D regulative ideas
    // =========================================================================

    /**
     * @test ProductBetweenness
     * @brief Verifies coordinate‑wise betweenness on product addresses.
     */
    TEST_F(ProductRegulativeTest, ProductBetweenness) {
        RI1D idea1;
        RI1D idea2;
        ProductIdea product_idea(idea1, idea2);
        const auto& betweenness = product_idea.betweenness();

        // All coordinates strictly increasing → true
        ProductAddr a1(1_r, 1_r);
        ProductAddr a2(2_r, 2_r);
        ProductAddr a3(3_r, 3_r);
        EXPECT_TRUE(betweenness(a1, a2, a3));

        // Opposite monotonicity in coordinates → false
        ProductAddr b1_addr(1_r, 3_r);
        ProductAddr b2_addr(2_r, 2_r);
        ProductAddr b3_addr(3_r, 1_r);
        EXPECT_FALSE(betweenness(b1_addr, b2_addr, b3_addr));

        // Inconsistent along first coordinate → false
        ProductAddr c1(1_r, 1_r);
        ProductAddr c2(1_r, 2_r);
        ProductAddr c3(2_r, 3_r);
        EXPECT_FALSE(betweenness(c1, c2, c3));

        // First coordinate equal (not strictly increasing) → false (needs strict betweenness)
        ProductAddr d1(1_r, 1_r);
        ProductAddr d2(1_r, 2_r);
        ProductAddr d3(1_r, 3_r);
        EXPECT_FALSE(betweenness(d1, d2, d3));

        // Second coordinate non‑monotonic → false
        ProductAddr e1(1_r, 1_r);
        ProductAddr e2(2_r, 0_r);
        ProductAddr e3(3_r, -1_r);
        EXPECT_FALSE(betweenness(e1, e2, e3));
    }

    /**
     * @test ProductMetric
     * @brief Checks max‑metric (Chebyshev distance) on product addresses.
     */
    TEST_F(ProductRegulativeTest, ProductMetric) {
        RI1D idea1;
        RI1D idea2;
        ProductIdea product_idea(idea1, idea2);
        const auto& metric = product_idea.metric();

        ProductAddr a(1_r, 2_r);
        ProductAddr b(4_r, 6_r);
        EXPECT_EQ(metric(a, b), 4_r);

        ProductAddr c(5_r, 5_r);
        ProductAddr d(2_r, 7_r);
        EXPECT_EQ(metric(c, d), 3_r);

        ProductAddr e(-3_r, -5_r);
        ProductAddr f(2_r, 1_r);
        EXPECT_EQ(metric(e, f), 6_r);

        EXPECT_EQ(metric(a, a), 0_r);
    }

    // Beware fellow traveller, here be dragons.
    // =========================================================================
    // Test group 2: ProductDeltaPath behaviour
    // =========================================================================

    /**
     * @test ProductPathConstruction
     * @brief Checks that product path initialises correctly (level 0, 2×2 grid).
     */
    TEST_F(ProductRegulativeTest, ProductPathConstruction) {
        Path1D path1 = make_dyadic_path(0_r, 1_r);
        Path1D path2 = make_dyadic_path(0_r, 1_r);

        ProductPath product_path(path1, path2);

        EXPECT_EQ(product_path.level(), 0);
        auto grid = product_path.current_grid();
        EXPECT_EQ(grid.size(), 4);
    }

    /**
     * @test ProductPathAdvance
     * @brief Verifies refinement of the product grid (dyadic points) and that all
     *        non‑zero coordinates are dyadic rationals.
     */
    TEST_F(ProductRegulativeTest, ProductPathAdvance) {
        Path1D path1 = make_dyadic_path(0_r, 1_r);
        Path1D path2 = make_dyadic_path(0_r, 1_r);

        ProductPath product_path(path1, path2);
        auto identity_func = make_identity_function();

        auto grid0 = product_path.current_grid();
        EXPECT_EQ(grid0.size(), 4);

        product_path.advance(identity_func);
        EXPECT_EQ(product_path.level(), 1);

        auto grid1 = product_path.current_grid();
        EXPECT_EQ(grid1.size(), 9);

        // Check that all non‑zero coordinates are dyadic
        for (std::size_t i = 0; i < grid1.size(); ++i) {
            PathAddr addr = grid1[i];
            if (addr[0] != 0) EXPECT_TRUE(is_dyadic(addr[0]));
            if (addr[1] != 0) EXPECT_TRUE(is_dyadic(addr[1]));
        }

        product_path.advance(identity_func);
        EXPECT_EQ(product_path.level(), 2);

        auto grid2 = product_path.current_grid();
        EXPECT_EQ(grid2.size(), 25);

        bool found_0_05 = false;
        for (std::size_t i = 0; i < grid2.size(); ++i) {
            auto addr = grid2[i];
            if (addr[0] == 0_r && addr[1] == "0.5"_r) {
                found_0_05 = true;
                break;
            }
        }
        EXPECT_TRUE(found_0_05);

        bool found_05_075 = false;
        for (std::size_t i = 0; i < grid2.size(); ++i) {
            auto addr = grid2[i];
            if (addr[0] == "0.5"_r && addr[1] == "0.75"_r) {
                found_05_075 = true;
                break;
            }
        }
        EXPECT_TRUE(found_05_075);
    }

    // =========================================================================
    // Test group 3: Fundamental sequences and ℝⁿ invariant
    // =========================================================================

    /**
     * @test FundamentalSequenceToPi
     * @brief Approximates π using Leibniz series and checks fundamental property.
     */
    TEST_F(ProductRegulativeTest, FundamentalSequenceToPi) {
        using namespace delta;

        auto pi_seq_gen = [](std::size_t n) -> Rational {
            Rational sum = 0_r;
            for (std::size_t k = 0; k <= n; ++k) {
                Rational term = 1_r / (2 * k + 1);
                if (k % 2 == 0) {
                    sum += term;
                }
                else {
                    sum -= term;
                }
            }
            return 4_r * sum;
            };

        // Power decay modulus for Leibniz series: |π - π_n| ≤ 4/(2n+1) ≈ 2/n, so C=4, α=1
        PowerDecayModulus modulus(4_r, 1_r);
        FundamentalSequence<PowerDecayModulus> pi_seq(pi_seq_gen, modulus, 0);

        Rational pi_exact = pi("0.000000000000000000000000000001"_r);
        std::size_t n = 1000;
        Rational pi_approx = pi_seq(n);
        Rational error = delta::abs(pi_approx - pi_exact);

        EXPECT_LT(error, "0.01"_r);

        // Fundamental property: |x_n - x_{n+100}| ≤ modulus(n)
        Rational diff = delta::abs(pi_seq(n) - pi_seq(n + 100));
        Rational bound = pi_seq.modulus()(n);  // estimate via modulus

        // For n=1000, modulus(1000) ≈ 4/1000 = 0.004
        // Add a small tolerance for approximate computations
        EXPECT_LE(diff, bound + "0.0001"_r);
    }

    /**
     * @test FundamentalSequenceToE
     * @brief Approximates e using exponential series and checks fundamental property.
     */
    TEST_F(ProductRegulativeTest, FundamentalSequenceToE) {
        using namespace delta;

        auto e_seq_gen = [](std::size_t n) -> Rational {
            Rational sum = 0_r;
            Rational fact = 1_r;
            for (std::size_t k = 0; k <= n; ++k) {
                if (k > 0) fact *= k;
                sum += 1_r / fact;
            }
            return sum;
            };

        FundamentalSequence e_seq(e_seq_gen, 3_r, "0.5"_r, 0);

        Rational e_exact = e("0.000000000000000000000000000001"_r);
        std::size_t n = 20;
        Rational e_approx = e_seq(n);
        Rational error = delta::abs(e_approx - e_exact);

        EXPECT_LT(error, "0.0000000001"_r);

        Rational diff = delta::abs(e_seq(n) - e_seq(n + 5));
        Rational bound = e_seq.bound() * delta::pow(e_seq.rate(), static_cast<int>(n));
        EXPECT_LE(diff, bound + "0.0000000001"_r);
    }

    /**
     * @test RealNumberConstruction
     * @brief Tests RealNumber from fundamental sequences and approximate equality.
     */
    TEST_F(ProductRegulativeTest, RealNumberConstruction) {
        using namespace delta;

        auto pi_seq_gen = [](std::size_t n) -> Rational {
            return pi("0.000000000000000000000000000001"_r);
            };

        auto pi_seq = std::make_shared<FundamentalSequence<ExponentialModulus>>(
            pi_seq_gen, 1_r, "0.5"_r, 0
        );

        RealNumber pi_real(pi_seq);
        Rational pi_approx = pi_real.approximate(10);
        Rational pi_exact = pi("0.000000000000000000000000000001"_r);

        EXPECT_RATIONAL_NEAR(pi_approx, pi_exact, Rational(1, 1000000000000));

        RealNumber half_real("0.5"_r);
        EXPECT_EQ(half_real.approximate(0), "0.5"_r);

        RealNumber half_real2("0.5"_r);
        EXPECT_TRUE(half_real == half_real2);

        // Use string literal for exact specification
        RealNumber almost_half("0.5000000001"_r);
        EXPECT_TRUE(half_real.approx_equal(almost_half, Rational(1, 1000000)));
        EXPECT_FALSE(half_real.approx_equal(almost_half, Rational(1, 1000000000000)));
    }

    /**
     * @test ProductGridApproximatesR2
     * @brief Demonstrates that the product of two dyadic paths produces a grid
     *        that approximates ℝ² and that dyadic rationals are dense.
     */
    TEST_F(ProductRegulativeTest, ProductGridApproximatesR2) {
        Path1D path1 = make_dyadic_path(0_r, 1_r);
        Path1D path2 = make_dyadic_path(0_r, 1_r);
        ProductPath product_path(path1, path2);
        auto identity_func = make_identity_function();

        for (int level = 0; level <= 3; ++level) {
            while (product_path.level() < static_cast<std::size_t>(level)) {
                product_path.advance(identity_func);
            }

            auto grid = product_path.current_grid();
            std::size_t expected_size = (1 << level) + 1;
            expected_size = expected_size * expected_size;
            EXPECT_EQ(grid.size(), expected_size);

            if (level == 3) {
                bool found = false;
                for (std::size_t i = 0; i < grid.size(); ++i) {
                    auto addr = grid[i];
                    if (addr[0] == 3_r / 8_r && addr[1] == 5_r / 8_r) {
                        found = true;
                        break;
                    }
                }
                EXPECT_TRUE(found);
            }
        }

        Rational a = 1_r / 3_r;
        Rational b = 2_r / 3_r;

        Rational a_approx5 = 11_r / 32_r;
        Rational b_approx5 = 21_r / 32_r;

        Rational error_a = delta::abs(a - a_approx5);
        Rational error_b = delta::abs(b - b_approx5);

        EXPECT_LE(error_a, 1_r / 32_r);
        EXPECT_LE(error_b, 1_r / 32_r);

        Rational a_approx6 = 21_r / 64_r;
        Rational error_a6 = delta::abs(a - a_approx6);
        EXPECT_LT(error_a6, error_a);
    }

} // namespace delta::testing