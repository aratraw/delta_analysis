// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/numerical/integrals_test.cpp
// ============================================================================
// TESTS FOR INTEGRATION UTILITIES AND GREEN'S IDENTITIES (STAGE 1, BLOCK A8)
// ============================================================================
//
// This file tests the basic integration facilities:
//   - cell_volume() for uniform, list, and product grids (1D and 2D)
//   - integral() – weighted sum of field values
//   - Summation‑by‑parts and Green's first identity in 1D
//   - Green's first and second identities in 2D (using FEM stiffness matrix)
//
// The 2D tests currently rely on check_green_first_2d() and check_green_second_2d()
// which, at the time of writing, do NOT independently compute the boundary integral.
// Instead they derive the boundary term from the identity itself, making the test
// trivial. See the large TODO below for the necessary corrections.
//
// ============================================================================
// TODO: CORRECT GREEN'S IDENTITY TESTS IN 2D (30.04.2026)
// ============================================================================
//
// CURRENT STATE
// -------------
// The tests GreenFirstIdentity, GreenFirstZeroBoundary, GreenSecondIdentity,
// GreenSecondZeroBoundary in the Integrals2DTest class DO NOT PERFORM A REAL
// VERIFICATION of Green's identities.
//
// Reason:
//   The functions check_green_first_2d() and check_green_second_2d() in
//   integrals.h do not compute the boundary term independently. Instead they
//   define it as the difference between the left and volume terms, ensuring
//   a trivial equality. This makes the tests useless for detecting discretisation
//   errors.
//
//   In the first identity, the boundary term ∫ f ∇g·n ds is adjusted to fit the
//   already computed quantities. In the second identity, the check degenerates
//   into a verification of stiffness matrix symmetry (K^T = K).
//
// NECESSARY CORRECTIONS
// ---------------------
// 1. Implement an independent computation of the boundary integral:
//    - For a rectangular grid (ProductGrid<UniformGrid,2>) obtain the list of
//      boundary edges (edges belonging to only one cell, i.e. lying on ∂Ω).
//    - On each boundary edge, compute ∇g·n using adjacent node values and
//      possibly interpolation, and f at the edge midpoint (or integrate f along
//      the edge).
//    - Sum the contributions with proper orientation.
//
// 2. Ensure consistency with the metric:
//    - All geometric quantities (edge length, normal) must be computed through
//      the supplied Metric object, not under the Euclidean assumption.
//
// 3. Add convergence tests:
//    - For a sequence of refined grids (e.g. 4×4, 8×8, 16×16) compute the
//      residual of Green's identity.
//    - The expected convergence order for bilinear elements should be ~ O(h²)
//      for the first identity (provided the boundary term is computed accurately).
//
// 4. Unify with the 1D implementation:
//    - check_green_first_1d() already performs an independent boundary term
//      calculation. Generalise that approach to 2D, possibly through a common
//      template code using ProductGrid or simplicial complexes.
//
// 5. (Optional) For the second Green identity:
//    - Beyond symmetry, a proper test should compute ∫ (f Δg - g Δf) dV and the
//      corresponding boundary term ∫ (f ∇g·n - g ∇f·n) dS and verify their equality.
//    - On a uniform grid both the volume and boundary parts are zero, but they
//      must be computed independently.
//
// ACTION PLAN (priorities)
// -------------------------
// - [ ] Create compute_boundary_integral_first_2d(...) in integrals.h that
//       computes ∫_∂Ω f (∇g·n) ds over boundary edges.
// - [ ] Modify check_green_first_2d to use that function and verify the equality
//       left = volume + boundary within a given tolerance.
// - [ ] Similarly for check_green_second_2d.
// - [ ] Write a convergence test for the first Green identity (ConvergenceGreenFirst2D)
//       and add it to Integrals2DTest.
// - [ ] Test on non‑uniform and product‑mixed grids if relevant.
//
// IMPORTANT:
//   Until these points are addressed, the 2D Green identity tests should be
//   considered STUBS and not be used to validate numerical schemes.
//
// ============================================================================

#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <cmath>
#include "delta/core/uniform_grid.h"
#include "delta/core/list_grid.h"
#include "delta/core/product_grid.h"
#include "delta/geometry/tensor_field.h"
#include "delta/numerical/integrals.h"
#include "delta/rational/literals.h"
#include "../test_fixtures_geometry_numerical.h"

namespace delta::testing {

    // -------------------------------------------------------------------------
    // 1D tests
    // -------------------------------------------------------------------------
    class Integrals1DTest : public GeometryNumericalTest {
    protected:
        using Grid1DUniform = delta::UniformGrid<Rational, std::less<Rational>>;
        using Grid1DList = delta::ListGrid<Rational, std::less<Rational>>;
        using ScalarField1D = delta::geometry::TensorField<Rational, Scalar, 0, 1, std::less<Rational>>;

        EuclideanMetric metric;

        Grid1DUniform make_uniform_grid(std::size_t n, Rational a = 0_r, Rational b = 1_r) {
            Rational step = (b - a) / (n - 1);
            return Grid1DUniform(a, step, n);
        }

        Grid1DList make_nonuniform_grid(const std::vector<Rational>& points) {
            return Grid1DList(points.begin(), points.end(), std::less<Rational>{});
        }
    };

    /**
     * @test CellVolumeUniform (1D)
     * @brief Verifies cell volumes on a uniform 1D grid (half cells at boundaries).
     */
    TEST_F(Integrals1DTest, CellVolumeUniform) {
        auto grid = make_uniform_grid(5);
        Rational h = 1_r / 4_r;

        EXPECT_EQ(grid_cell_volume(grid, 0, metric), h / 2_r);
        EXPECT_EQ(grid_cell_volume(grid, 4, metric), h / 2_r);
        for (std::size_t i = 1; i <= 3; ++i) {
            EXPECT_EQ(grid_cell_volume(grid, i, metric), h);
        }
        Rational total = 0_r;
        for (std::size_t i = 0; i < grid.size(); ++i) total += grid_cell_volume(grid, i, metric);
        EXPECT_EQ(total, 1_r);
    }

    /**
     * @test CellVolumeNonUniform (1D)
     * @brief Checks cell volumes on a non‑uniform 1D grid.
     */
    TEST_F(Integrals1DTest, CellVolumeNonUniform) {
        std::vector<Rational> points = { 0_r, "2/10"_r, "5/10"_r, 1_r };
        auto grid = make_nonuniform_grid(points);
        EXPECT_EQ(grid_cell_volume(grid, 0, metric), "1/10"_r);
        EXPECT_EQ(grid_cell_volume(grid, 1, metric), "1/4"_r);
        EXPECT_EQ(grid_cell_volume(grid, 2, metric), "2/5"_r);
        EXPECT_EQ(grid_cell_volume(grid, 3, metric), "1/4"_r);
        Rational total = 0_r;
        for (std::size_t i = 0; i < grid.size(); ++i) total += grid_cell_volume(grid, i, metric);
        EXPECT_EQ(total, 1_r);
    }

    /**
     * @test IntegralLinearExact (1D)
     * @brief Integrates a linear function exactly on a uniform grid.
     */
    TEST_F(Integrals1DTest, IntegralLinearExact) {
        auto grid = make_uniform_grid(5);
        ScalarField1D f(grid);
        for (std::size_t i = 0; i < grid.size(); ++i) {
            Rational x = grid[i];
            f.set(x, x);
        }
        Rational I = grid_integral(grid, [&f](const Rational& x) { return f.at(x); }, metric);
        EXPECT_RATIONAL_NEAR(I, "1/2"_r, delta::default_eps());
    }

    /**
     * @test IntegralQuadraticConvergence (1D)
     * @brief Checks that the trapezoidal rule for x² converges with second order.
     */
    TEST_F(Integrals1DTest, IntegralQuadraticConvergence) {
        std::vector<std::size_t> ns = { 5, 9, 17, 33 };
        std::vector<Rational> errors;
        Rational exact = 1_r / 3_r;

        for (std::size_t n : ns) {
            auto grid = make_uniform_grid(n);
            ScalarField1D f(grid);
            for (std::size_t i = 0; i < grid.size(); ++i) {
                Rational x = grid[i];
                f.set(x, x * x);
            }
            Rational I = grid_integral(grid, [&f](const Rational& x) { return f.at(x); }, metric);
            Rational err = delta::abs(I - exact);
            errors.push_back(err);
        }
        ASSERT_GE(errors.size(), 2);
        for (std::size_t i = 0; i < errors.size() - 1; ++i) {
            Rational ratio = errors[i] / errors[i + 1];
            EXPECT_TRUE(ratio > 3_r && ratio < 5_r);
        }
    }

    /**
     * @test SummationByParts (1D)
     * @brief Verifies the discrete summation‑by‑parts identity.
     */
    TEST_F(Integrals1DTest, SummationByParts) {
        auto grid = make_uniform_grid(5);
        ScalarField1D f(grid), g(grid);
        for (std::size_t i = 0; i < grid.size(); ++i) {
            Rational x = grid[i];
            f.set(x, x);
            g.set(x, x * x);
        }
        Rational g_right = g.at(1_r);
        bool ok = check_summation_by_parts_1d(grid, f, g, metric, g_right, "1/1000000000000"_r);
        EXPECT_TRUE(ok);
    }

    /**
     * @test SummationByPartsZeroBoundary (1D)
     * @brief Checks summation‑by‑parts with zero boundary condition on the right.
     */
    TEST_F(Integrals1DTest, SummationByPartsZeroBoundary) {
        auto grid = make_uniform_grid(5);
        ScalarField1D f(grid), g(grid);
        for (std::size_t i = 0; i < grid.size(); ++i) {
            Rational x = grid[i];
            f.set(x, x);
            g.set(x, x * (1_r - x));
        }
        Rational g_right = 0_r;
        bool ok = check_summation_by_parts_1d(grid, f, g, metric, g_right, "1/1000000000000"_r);
        EXPECT_TRUE(ok);
    }

    /**
     * @test GreenFirstIdentity (1D)
     * @brief Validates Green's first identity in 1D.
     */
    TEST_F(Integrals1DTest, GreenFirstIdentity) {
        auto grid = make_uniform_grid(9);
        ScalarField1D f(grid), g(grid);
        for (std::size_t i = 0; i < grid.size(); ++i) {
            Rational x = grid[i];
            f.set(x, x);
            g.set(x, x * x);
        }
        bool ok = check_green_first_1d(grid, f, g, metric, "1/1000000000000"_r);
        EXPECT_TRUE(ok);
    }

    // -------------------------------------------------------------------------
    // 2D tests
    // -------------------------------------------------------------------------
    struct MaxMetric {
        template<typename T, std::size_t N>
        auto operator()(const std::array<T, N>& a, const std::array<T, N>& b) const {
            T max_diff = 0;
            for (std::size_t i = 0; i < N; ++i) {
                T diff = a[i] - b[i];
                if (diff < 0) diff = -diff;
                if (diff > max_diff) max_diff = diff;
            }
            return max_diff;
        }
    };

    class Integrals2DTest : public GeometryNumericalTest {
    protected:
        using Grid1D = delta::UniformGrid<Rational, std::less<Rational>>;
        using Grid2D = delta::ProductGrid<Grid1D, 2>;
        using Addr2D = typename Grid2D::value_type;

        struct Addr2DCompare {
            bool operator()(const Addr2D& a, const Addr2D& b) const {
                if (a[0] < b[0]) return true;
                if (b[0] < a[0]) return false;
                return a[1] < b[1];
            }
        };

        using ScalarField2D = delta::geometry::TensorField<Addr2D, Scalar, 0, 2, Addr2DCompare>;

        Grid2D make_uniform_grid_2d(std::size_t nx, std::size_t ny,
            Rational a = 0_r, Rational b = 1_r,
            Rational c = 0_r, Rational d = 1_r) {
            Grid1D gx(a, (b - a) / (nx - 1), nx);
            Grid1D gy(c, (d - c) / (ny - 1), ny);
            return Grid2D({ gx, gy });
        }

        EuclideanMetric metric;
    };

    /**
     * @test CellVolumeUniform (2D)
     * @brief Checks cell volumes on a uniform 2D grid (inner cell vs. corner cell).
     */
    TEST_F(Integrals2DTest, CellVolumeUniform) {
        auto grid = make_uniform_grid_2d(4, 4);
        Rational hx = 1_r / 3_r, hy = 1_r / 3_r;
        Addr2D inner = { hx, hy };
        std::size_t inner_idx = 0;
        for (std::size_t i = 0; i < grid.size(); ++i) {
            if (grid[i] == inner) { inner_idx = i; break; }
        }
        EXPECT_EQ(grid_cell_volume(grid, inner_idx, metric), hx * hy);

        Addr2D corner = { 0_r, 0_r };
        std::size_t corner_idx = 0;
        for (std::size_t i = 0; i < grid.size(); ++i) {
            if (grid[i] == corner) { corner_idx = i; break; }
        }
        EXPECT_EQ(grid_cell_volume(grid, corner_idx, metric), (hx / 2_r) * (hy / 2_r));
    }

    /**
     * @test CellVolumeNonUniformProduct (2D)
     * @brief Verifies product grid cell volumes on a non‑uniform mesh.
     */
    TEST_F(Integrals2DTest, CellVolumeNonUniformProduct) {
        auto grid = make_uniform_grid_2d(5, 3, 0_r, 1_r, 0_r, 1_r);
        Rational hx = "1/4"_r, hy = "1/2"_r;
        Addr2D inner = { "1/2"_r, "1/2"_r };
        std::size_t inner_idx = 0;
        for (std::size_t i = 0; i < grid.size(); ++i) {
            if (grid[i] == inner) { inner_idx = i; break; }
        }
        EXPECT_EQ(grid_cell_volume(grid, inner_idx, metric), hx * hy);
        Rational total = 0_r;
        for (std::size_t i = 0; i < grid.size(); ++i) total += grid_cell_volume(grid, i, metric);
        EXPECT_EQ(total, 1_r);
    }

    /**
     * @test IntegralLinearExact (2D)
     * @brief Integrates f(x,y)=x+y exactly over the unit square.
     */
    TEST_F(Integrals2DTest, IntegralLinearExact) {
        auto grid = make_uniform_grid_2d(5, 5);
        ScalarField2D f(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1];
            f.set(addr, x + y);
        }
        Rational I = grid_integral(grid, [&f](const Addr2D& a) { return f.at(a); }, metric);
        EXPECT_RATIONAL_NEAR(I, 1_r, delta::default_eps());
    }

    /**
     * @test GreenFirstIdentity (2D)
     * @brief Invokes the 2D Green's first identity check.
     * @note Currently a stub – see the large TODO above.
     */
    TEST_F(Integrals2DTest, GreenFirstIdentity) {
        auto grid = make_uniform_grid_2d(9, 9);
        ScalarField2D f(grid), g(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1];
            f.set(addr, x * x + y * y);
            g.set(addr, x + y);
        }
        bool ok = check_green_first_2d(grid, f, g, metric, "1/1000000000000"_r);
        EXPECT_TRUE(ok);
    }

    /**
     * @test GreenFirstZeroBoundary (2D)
     * @brief Tests the first identity with a function vanishing on the boundary.
     * @note Stub – see TODO.
     */
    TEST_F(Integrals2DTest, GreenFirstZeroBoundary) {
        auto grid = make_uniform_grid_2d(9, 9);
        ScalarField2D f(grid), g(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1];
            Rational g_val = x * (1_r - x) * y * (1_r - y);
            g.set(addr, g_val);
            f.set(addr, x * x + y * y);
        }
        bool ok = check_green_first_2d(grid, f, g, metric, "1/1000000000000"_r);
        EXPECT_TRUE(ok);
    }

    /**
     * @test GreenSecondIdentity (2D)
     * @brief Invokes the 2D Green's second identity check.
     * @note Stub – see TODO.
     */
    TEST_F(Integrals2DTest, GreenSecondIdentity) {
        auto grid = make_uniform_grid_2d(9, 9);
        ScalarField2D f(grid), g(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1];
            f.set(addr, x * x + y * y);
            g.set(addr, x + y);
        }
        bool ok = check_green_second_2d(grid, f, g, metric, "1/1000000000000"_r);
        EXPECT_TRUE(ok);
    }

    /**
     * @test GreenSecondZeroBoundary (2D)
     * @brief Tests the second identity with a function vanishing on the boundary.
     * @note Stub – see TODO.
     */
    TEST_F(Integrals2DTest, GreenSecondZeroBoundary) {
        auto grid = make_uniform_grid_2d(9, 9);
        ScalarField2D f(grid), g(grid);
        for (const auto& addr : grid) {
            Rational x = addr[0], y = addr[1];
            f.set(addr, x * (1_r - x) * y * (1_r - y));
            g.set(addr, x * x + y * y);
        }
        bool ok = check_green_second_2d(grid, f, g, metric, "1/1000000000000"_r);
        EXPECT_TRUE(ok);
    }

    // -------------------------------------------------------------------------
    // Mixed grid tests
    // -------------------------------------------------------------------------
    class Integrals2DProductMixedTest : public GeometryNumericalTest {
    protected:
        using Grid1DList = delta::ListGrid<Rational, std::less<Rational>>;
        using Grid2D = delta::ProductGrid<Grid1DList, 2>;
        using Addr2D = typename Grid2D::value_type;

        struct Addr2DCompare {
            bool operator()(const Addr2D& a, const Addr2D& b) const {
                if (a[0] < b[0]) return true;
                if (b[0] < a[0]) return false;
                return a[1] < b[1];
            }
        };

        using ScalarField2D = delta::geometry::TensorField<Addr2D, Scalar, 0, 2, Addr2DCompare>;

        Grid2D make_mixed_grid() {
            std::vector<Rational> xs = { 0_r, "2/10"_r, "5/10"_r, 1_r };
            std::vector<Rational> ys = { 0_r, "3/10"_r, "7/10"_r, 1_r };
            Grid1DList gx(xs.begin(), xs.end(), std::less<Rational>{});
            Grid1DList gy(ys.begin(), ys.end(), std::less<Rational>{});
            return Grid2D({ gx, gy });
        }

        EuclideanMetric metric;
    };

    /**
     * @test CellVolumeProduct (mixed grid)
     * @brief Cell volume in a product of two non‑uniform 1D grids.
     */
    TEST_F(Integrals2DProductMixedTest, CellVolumeProduct) {
        auto grid = make_mixed_grid();
        Addr2D p = { "2/10"_r, "3/10"_r };
        std::size_t idx = 0;
        for (std::size_t i = 0; i < grid.size(); ++i) {
            if (grid[i] == p) { idx = i; break; }
        }
        Rational expected = "1/4"_r * "7/20"_r;
        EXPECT_EQ(grid_cell_volume(grid, idx, metric), expected);
        Rational total = 0_r;
        for (std::size_t i = 0; i < grid.size(); ++i) total += grid_cell_volume(grid, i, metric);
        EXPECT_EQ(total, 1_r);
    }

    /**
     * @test IntegralQuadraticConvergence (mixed grid)
     * @brief Checks convergence of the integral of x²+y² on a sequence of refined
     *        product grids built from non‑uniform points.
     */
    TEST_F(Integrals2DProductMixedTest, IntegralQuadraticConvergence) {
        Rational exact = 2_r / 3_r;
        std::vector<std::size_t> ns = { 4, 8, 16 };
        std::vector<Rational> errors;
        for (std::size_t n : ns) {
            std::vector<Rational> xs(n);
            std::vector<Rational> ys(n);
            for (std::size_t i = 0; i < n; ++i) {
                xs[i] = Rational(static_cast<long long>(i)) / (n - 1);
                ys[i] = xs[i];
            }
            Grid1DList gx(xs.begin(), xs.end(), std::less<Rational>{});
            Grid1DList gy(ys.begin(), ys.end(), std::less<Rational>{});
            Grid2D grid({ gx, gy });
            ScalarField2D f(grid);
            for (const auto& addr : grid) {
                Rational x = addr[0], y = addr[1];
                f.set(addr, x * x + y * y);
            }
            Rational I = grid_integral(grid, [&f](const Addr2D& a) { return f.at(a); }, metric);
            Rational err = delta::abs(I - exact);
            errors.push_back(err);
        }
        ASSERT_GE(errors.size(), 2);
        for (std::size_t i = 0; i < errors.size() - 1; ++i) {
            Rational ratio = errors[i] / errors[i + 1];
            EXPECT_TRUE(ratio > 2_r && ratio < 6_r);
        }
    }

} // namespace delta::testing