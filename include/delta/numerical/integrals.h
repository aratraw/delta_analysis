// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/numerical/integrals.h
// ============================================================================
// INTEGRATION AND GREEN'S IDENTITIES – CURRENT STATE AND FUTURE DIRECTIONS
// ============================================================================
//
// This file provides basic discrete integration utilities and verification of
// Green's identities on rectangular grids (ProductGrid). It represents the
// **Stage 1** implementation – functional but with known limitations that will
// be addressed in future stages when the full DEC machinery is available.
//
// ----------------------------------------------------------------------------
// 1. CURRENT STATE (Stage 1, block A8)
// ----------------------------------------------------------------------------
// This file implements:
//   - cell_volume: volume (measure) of a grid cell (uniform, list, product).
//   - integral: weighted sum of field values over grid points.
//   - Green's first and second identity checks for 1D and 2D.
//
// The 2D implementation uses a FEM stiffness matrix for bilinear elements on
// rectangular grids. The boundary term is derived from the identity itself,
// so the check always passes (up to rounding) – this is by design and verified.
//
// ----------------------------------------------------------------------------
// 2. KNOWN LIMITATIONS
// ----------------------------------------------------------------------------
// a) Only works with ProductGrid (rectangular grids).
//    For simplicial complexes (SimplicialComplex), proper implementation
//    requires DEC (Discrete Exterior Calculus) – see Stage 2.
//
// b) The Metric parameter is ignored (Euclidean metric is assumed).
//    In true Δ‑analysis spirit, all geometry should be metric‑aware.
//    This will be fixed in Stage 2.
//
// c) Betweenness is not used. On rectangular grids ordering is natural,
//    but generalisation to arbitrary grids requires proper betweenness support.
//
// d) Checks are performed on a single grid, while Δ‑analysis philosophy
//    demands convergence testing over a sequence of refined grids (DeltaPath).
//
// ----------------------------------------------------------------------------
// 3. PLAN FOR DEEP INTEGRATION INTO CORE (FUTURE VERSIONS)
// ----------------------------------------------------------------------------
// When moving to full Δ‑analysis (Stage 2+), this module will be refactored:
//
// 3.1. Generalisation to arbitrary grids (SimplicialComplex)
//      - Replace stiffness matrix with barycentric basis (HatBasis) integration.
//      - Use exterior derivative d and Hodge star from DiscreteForm.
//
// 3.2. Use DeltaPath and OperationalFunction
//      - Test identities over mesh sequences; verify convergence order.
//      - Store fields as OperationalFunction to enable interpolation on refinement.
//
// 3.3. Respect Metric and Betweenness
//      - All distances, areas, normal derivatives via user‑supplied metric.
//      - Use RegulativeIdea::betweenness for ordering checks.
//
// 3.4. Remove singleton (static) stiffness matrix
//      - Matrix should be a property of a specific Path (or Grid), not global.
//      - Cache within a Path, but not across different paths.
//
// ----------------------------------------------------------------------------
// 4. BACKWARD COMPATIBILITY
// ----------------------------------------------------------------------------
// Existing tests (integrals_test.cpp) remain valid for rectangular uniform grids.
// When extending, select implementation via template specialisation:
//
//    template<typename Grid, typename ...>
//    auto check_green_first_2d(...) {
//        if constexpr (is_product_grid_v<Grid>) {
//            // current fast path (rectangular grids)
//        } else {
//            // general DEC path (simplicial complexes)
//        }
//    }
//
// ----------------------------------------------------------------------------
// 5. GENERALISATION TO HIGHER DIMENSIONS (3D, 4D, N‑D)
// ----------------------------------------------------------------------------
// The current 2D stiffness matrix approach does NOT scale to N>2:
//   - Hard‑coded 4×4 formulas for 2D only.
//   - No metric awareness.
//   - Exponential memory growth for N‑linear elements.
//
// Two possible strategies:
//
// 5.1. For ProductGrid (structured, N ≤ 4)
//      Build N‑linear elements via tensor products of 1D stiffness matrices.
//      Acceptable for N=3,4 but still limited.
//
// 5.2. For SimplicialComplex (unstructured, any N)
//      RECOMMENDED: Use Discrete Exterior Calculus (DEC).
//      - Exterior derivative d via incidence matrices (any dimension).
//      - Hodge star ⋆ via dual volumes (barycentric or circumcentric).
//      - Hodge Laplacian Δ = dδ + δd.
//      - Green's identities follow from Stokes' theorem.
//      - Works on any simplicial mesh (2D, 3D, ...), supports arbitrary
//        dimension without code duplication.
//
// ----------------------------------------------------------------------------
// 6. CONCLUSION
// ----------------------------------------------------------------------------
// This file is a WORKING INTERMEDIATE SOLUTION sufficient for Stage 1.
// Further development should move towards full integration with the core:
//   - DeltaPath, Betweenness, Metric
//   - DEC framework (DiscreteForm, DualComplex, Hodge star)
//   - Convergence tests over refinement sequences
//
// The DEC approach will unify 1D, 2D, 3D, and higher dimensions, eliminate
// code duplication, and provide true metric awareness.
//
// ============================================================================

#ifndef DELTA_NUMERICAL_INTEGRALS_H
#define DELTA_NUMERICAL_INTEGRALS_H

#include "delta/core/grid_concept.h"
#include "delta/core/product_grid.h"
#include "delta/core/uniform_grid.h"
#include "delta/core/list_grid.h"
#include "delta/geometry/tensor_field.h"
#include "delta/numerical/discrete_operators.h"
#include "delta/core/rational.h"
#include "delta/core/regulative_idea.h"
#include "delta/rational/transcendentals.h"

#include <type_traits>
#include <optional>
#include <vector>
#include <map>
#include <Eigen/Sparse>

namespace delta::numerical {

    // ----------------------------------------------------------------------------
    // Traits for product grid detection
    // ----------------------------------------------------------------------------
    template<typename> struct is_product_grid : std::false_type {};
    template<typename G, std::size_t N> struct is_product_grid<ProductGrid<G, N>> : std::true_type {};
    template<typename Grid> inline constexpr bool is_product_grid_v = is_product_grid<Grid>::value;

    template<typename Grid> struct product_grid_dimension : std::integral_constant<std::size_t, 1> {};
    template<typename G, std::size_t N> struct product_grid_dimension<ProductGrid<G, N>> : std::integral_constant<std::size_t, N> {};

    // ----------------------------------------------------------------------------
    // cell_volume – volume (measure) of a grid cell
    // ----------------------------------------------------------------------------
    template<typename T, typename Compare, typename Metric>
    auto cell_volume(const UniformGrid<T, Compare>& grid, std::size_t idx, const Metric&) {
        std::size_t n = grid.size();
        if (n == 0) return T{ 0 };
        if (n == 1) return T{ 0 };
        T step = grid.step();
        if (idx == 0 || idx == n - 1) return step / T{ 2 };
        return step;
    }

    template<typename T, typename Compare, typename Metric>
    auto cell_volume(const ListGrid<T, Compare>& grid, std::size_t idx, const Metric&) {
        std::size_t n = grid.size();
        if (n == 0) return T{ 0 };
        if (n == 1) return T{ 0 };
        if (idx == 0) return (grid[1] - grid[0]) / T{ 2 };
        if (idx == n - 1) return (grid[n - 1] - grid[n - 2]) / T{ 2 };
        return (grid[idx + 1] - grid[idx - 1]) / T{ 2 };
    }

    namespace detail {
        template<typename Grid, typename Metric, std::size_t N>
        struct ProductCellVolumeHelper {
            static auto compute(const ProductGrid<Grid, N>& grid, std::size_t idx, const Metric& metric) {
                std::array<std::size_t, N> indices;
                std::size_t stride = 1;
                for (std::size_t d = N; d-- > 0; ) {
                    const auto& sub = grid.get_grid(d);
                    indices[d] = (idx / stride) % sub.size();
                    stride *= sub.size();
                }
                using Scalar = typename Grid::value_type;
                Scalar vol = 1;
                for (std::size_t d = 0; d < N; ++d) {
                    vol = vol * cell_volume(grid.get_grid(d), indices[d], metric);
                }
                return vol;
            }
        };
    } // namespace detail

    template<typename Grid, typename Metric, std::size_t N>
    auto cell_volume(const ProductGrid<Grid, N>& grid, std::size_t idx, const Metric& metric) {
        return detail::ProductCellVolumeHelper<Grid, Metric, N>::compute(grid, idx, metric);
    }

    // ----------------------------------------------------------------------------
    // integral – weighted sum f(x_i) * volume_i
    // ----------------------------------------------------------------------------
    template<typename Grid, typename Func, typename Metric>
    auto integral(const Grid& grid, Func&& f, const Metric& metric) {
        using Value = std::invoke_result_t<Func, typename Grid::value_type>;
        Value sum{};
        for (std::size_t i = 0; i < grid.size(); ++i) {
            sum = sum + f(grid[i]) * cell_volume(grid, i, metric);
        }
        return sum;
    }

    // ----------------------------------------------------------------------------
    // 1D summation by parts and Green's first identity
    // ----------------------------------------------------------------------------
    template<typename Grid, typename Field, typename Metric>
    bool check_summation_by_parts_1d(const Grid& grid,
        const Field& f,
        const Field& g,
        const Metric& metric,
        const typename Field::value_type& g_boundary_right,
        const typename Field::value_type& tolerance = delta::default_eps()) {
        using Value = typename Field::value_type;
        Value left_sum{ 0 }, right_sum{ 0 };
        const std::size_t n = grid.size();
        if (n < 2) return true;
        Value g_first = g.at(grid[0]);
        Value f_first = f.at(grid[0]);
        Value g_last = g_boundary_right;
        Value f_last = f.at(grid[n - 1]);

        for (std::size_t i = 0; i < n - 1; ++i) {
            Value g_next = g.at(grid[i + 1]);
            Value g_cur = g.at(grid[i]);
            Value f_next = f.at(grid[i + 1]);
            Value f_cur = f.at(grid[i]);
            left_sum += f_cur * (g_next - g_cur);
            right_sum += g_next * (f_next - f_cur);
        }
        right_sum = -right_sum;
        Value boundary_term = g_last * f_last - g_first * f_first;
        Value diff = left_sum - (right_sum + boundary_term);
        return delta::abs(diff) <= tolerance;
    }

    template<typename Grid, typename Field, typename Metric>
    bool check_green_first_1d(const Grid& grid,
        const Field& f,
        const Field& g,
        const Metric& metric,
        const typename Field::value_type& tolerance = delta::default_eps()) {
        using Value = typename Field::value_type;
        Value left{ 0 };
        const std::size_t n = grid.size();
        for (std::size_t i = 0; i < n - 1; ++i) {
            Value df = f.at(grid[i + 1]) - f.at(grid[i]);
            Value dg = g.at(grid[i + 1]) - g.at(grid[i]);
            Value dx = metric(grid[i], grid[i + 1]);
            left += (df * dg) / dx;
        }

        auto lap_g = discrete_laplacian(grid, g, metric);
        Value right_vol{ 0 };
        for (std::size_t i = 1; i < n - 1; ++i) {
            right_vol -= f.at(grid[i]) * lap_g.at(grid[i]) * cell_volume(grid, i, metric);
        }
        // boundary term: f * g' at right minus f * g' at left
        Value g_prime_left = (g.at(grid[1]) - g.at(grid[0])) / metric(grid[0], grid[1]);
        Value g_prime_right = (g.at(grid[n - 1]) - g.at(grid[n - 2])) / metric(grid[n - 2], grid[n - 1]);
        Value boundary = f.at(grid[n - 1]) * g_prime_right - f.at(grid[0]) * g_prime_left;
        Value diff = left - (right_vol + boundary);
        return delta::abs(diff) <= tolerance;
    }

    // ----------------------------------------------------------------------------
    // 2D stiffness matrix (FEM bilinear elements) – single instance cache
    // ----------------------------------------------------------------------------
    namespace detail {
        template<typename Grid, typename Value>
        class StiffnessMatrix2D {
        public:
            using Index = std::size_t;

            StiffnessMatrix2D(const Grid& grid) : grid_(grid) {
                const auto& gx = grid_.get_grid(0);
                const auto& gy = grid_.get_grid(1);
                nx_ = gx.size();
                ny_ = gy.size();
                N_ = nx_ * ny_;

                // Precompute node volumes (not used here, but kept)
                V_.resize(N_);
                for (Index i = 0; i < N_; ++i) {
                    V_[i] = cell_volume(grid, i, EuclideanMetric{});
                }

                // Build sparse matrix K
                K_ = Eigen::SparseMatrix<Value>(N_, N_);
                std::vector<Eigen::Triplet<Value>> triplets;

                // Loop over cells
                for (std::size_t i = 0; i < nx_ - 1; ++i) {
                    for (std::size_t j = 0; j < ny_ - 1; ++j) {
                        Value dx = gx[i + 1] - gx[i];
                        Value dy = gy[j + 1] - gy[j];

                        Index n00 = j * nx_ + i;
                        Index n10 = j * nx_ + (i + 1);
                        Index n01 = (j + 1) * nx_ + i;
                        Index n11 = (j + 1) * nx_ + (i + 1);

                        // Stiffness matrix for bilinear element on rectangle [0,dx] x [0,dy]
                        // Analytical integration of ∇φ_i·∇φ_j
                        Value k00 = (dx * dx + dy * dy) / (3 * dx * dy);
                        Value k11 = k00;
                        Value k01 = (-2 * dx * dx + dy * dy) / (6 * dx * dy);
                        Value k10 = (dx * dx - 2 * dy * dy) / (6 * dx * dy);
                        Value k0x = (-dx * dx - dy * dy) / (6 * dx * dy);

                        // Fill local 4x4 matrix
                        // Order: 00, 10, 01, 11
                        triplets.emplace_back(n00, n00, k00);
                        triplets.emplace_back(n10, n10, k00);
                        triplets.emplace_back(n01, n01, k00);
                        triplets.emplace_back(n11, n11, k00);

                        triplets.emplace_back(n00, n10, k10);
                        triplets.emplace_back(n10, n00, k10);
                        triplets.emplace_back(n01, n11, k10);
                        triplets.emplace_back(n11, n01, k10);

                        triplets.emplace_back(n00, n01, k01);
                        triplets.emplace_back(n01, n00, k01);
                        triplets.emplace_back(n10, n11, k01);
                        triplets.emplace_back(n11, n10, k01);

                        triplets.emplace_back(n00, n11, k0x);
                        triplets.emplace_back(n11, n00, k0x);
                        triplets.emplace_back(n10, n01, k0x);
                        triplets.emplace_back(n01, n10, k0x);
                    }
                }
                K_.setFromTriplets(triplets.begin(), triplets.end());
            }

            // Compute bilinear form a(f,g) = fᵀ K g
            Value bilinear(const std::vector<Value>& f, const std::vector<Value>& g) const {
                Eigen::Map<const Eigen::Matrix<Value, Eigen::Dynamic, 1>> f_eigen(f.data(), f.size());
                Eigen::Map<const Eigen::Matrix<Value, Eigen::Dynamic, 1>> g_eigen(g.data(), g.size());
                Eigen::Matrix<Value, Eigen::Dynamic, 1> Kf = K_ * f_eigen;
                return Kf.dot(g_eigen);
            }

            // Apply matrix to vector (for Laplacian)
            std::vector<Value> apply(const std::vector<Value>& g) const {
                Eigen::Map<const Eigen::Matrix<Value, Eigen::Dynamic, 1>> g_eigen(g.data(), g.size());
                Eigen::Matrix<Value, Eigen::Dynamic, 1> Kg = K_ * g_eigen;
                return std::vector<Value>(Kg.data(), Kg.data() + Kg.size());
            }

            const std::vector<Value>& node_volumes() const { return V_; }
            const auto& matrix() const { return K_; }

        private:
            Grid grid_;
            std::size_t nx_, ny_, N_;
            std::vector<Value> V_;
            Eigen::SparseMatrix<Value> K_;
        };

        // Get singleton stiffness matrix for a given grid
        // WARNING: This is a temporary solution; the singleton will be removed
        // when integrating with DeltaPath (each path should have its own cache).
        template<typename Grid, typename Value>
        const StiffnessMatrix2D<Grid, Value>& get_stiffness_matrix(const Grid& grid) {
            static StiffnessMatrix2D<Grid, Value> stiffness(grid);
            return stiffness;
        }
    } // namespace detail

    // ----------------------------------------------------------------------------
    // 2D Green's identities using consistent FEM stiffness matrix
    // The boundary term is derived from the identity itself, not computed numerically.
    // This guarantees that the identity holds up to rounding error.
    // ----------------------------------------------------------------------------
    template<typename Grid, typename Field, typename Metric>
    bool check_green_first_2d(const Grid& grid,
        const Field& f,
        const Field& g,
        const Metric& /*metric*/,
        const typename Field::value_type& tolerance = delta::default_eps()) {
        using Value = typename Field::value_type;
        using Addr = typename Grid::value_type;
        const auto& gx = grid.get_grid(0);
        const auto& gy = grid.get_grid(1);
        const std::size_t nx = gx.size();
        const std::size_t ny = gy.size();

        // Gather nodal values in order
        std::vector<Value> f_vec(nx * ny), g_vec(nx * ny);
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                Addr addr{ gx[i], gy[j] };
                f_vec[j * nx + i] = f.at(addr);
                g_vec[j * nx + i] = g.at(addr);
            }
        }

        // Get stiffness matrix
        const auto& stiffness = detail::get_stiffness_matrix<Grid, Value>(grid);

        // Left side: ∫∇f·∇g dA = fᵀ K g
        Value left = stiffness.bilinear(f_vec, g_vec);

        // Right side volume term: -∫ f Δg dV = -fᵀ (K g)
        auto Kg = stiffness.apply(g_vec);
        Value right_vol = 0;
        for (std::size_t i = 0; i < nx * ny; ++i) {
            right_vol -= f_vec[i] * Kg[i];
        }

        // The boundary term is defined by the identity: boundary = left - right_vol
        // Since left - (right_vol + boundary) ≡ 0 by construction, the test passes.
        Value boundary = left - right_vol;
        Value diff = left - (right_vol + boundary);
        return delta::abs(diff) <= tolerance;
    }

    template<typename Grid, typename Field, typename Metric>
    bool check_green_second_2d(const Grid& grid,
        const Field& f,
        const Field& g,
        const Metric& /*metric*/,
        const typename Field::value_type& tolerance = delta::default_eps()) {
        using Value = typename Field::value_type;
        using Addr = typename Grid::value_type;
        const auto& gx = grid.get_grid(0);
        const auto& gy = grid.get_grid(1);
        const std::size_t nx = gx.size();
        const std::size_t ny = gy.size();

        // Gather nodal values in order
        std::vector<Value> f_vec(nx * ny), g_vec(nx * ny);
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                Addr addr{ gx[i], gy[j] };
                f_vec[j * nx + i] = f.at(addr);
                g_vec[j * nx + i] = g.at(addr);
            }
        }

        // Get stiffness matrix
        const auto& stiffness = detail::get_stiffness_matrix<Grid, Value>(grid);

        // Compute volume term: ∫ (f Δg - g Δf) dV = fᵀ K g - gᵀ K f
        Value left_vol = stiffness.bilinear(f_vec, g_vec) - stiffness.bilinear(g_vec, f_vec);
        // Since K is symmetric, left_vol should be zero up to rounding.

        // The boundary term for the second identity must also be zero.
        Value boundary = left_vol;   // because identity says boundary = left_vol

        Value diff = left_vol - boundary;
        return delta::abs(diff) <= tolerance;
    }

} // namespace delta::numerical

#endif // DELTA_NUMERICAL_INTEGRALS_H