// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/geometry/product_regulative.h
// ============================================================================
// PRODUCT OF REGULATIVE IDEAS AND DELTA PATHS
// ============================================================================
//
// This header provides tools for constructing higher‑dimensional regulative
// structures from 1‑D components.  It defines:
//   - ProductRegulativeIdea   : product of two regulative ideas of the SAME type.
//   - PowerRegulativeIdea     : N copies of the same regulative idea (for ℝⁿ).
//   - ProductDeltaPath        : product of several delta paths (all of the same type).
//
// ----------------------------------------------------------------------------
// MATHEMATICAL BACKGROUND
// ----------------------------------------------------------------------------
// A regulative idea consists of an address set, a betweenness relation, and a
// metric.  The product of two such ideas (with identical underlying types) is
// defined on the Cartesian product of addresses.  Betweenness is required to
// hold coordinate‑wise, and the metric is the max‑norm (Chebyshev distance).
//
// Similarly, a delta path is a refinement sequence over a one‑dimensional
// grid.  The product path yields a regular product grid in higher dimensions.
// All component paths must be of the same type – mixing, e.g., matrix‑valued
// paths with binary‑string paths is mathematically unsound and forbidden.
//
// ----------------------------------------------------------------------------
// ⚠️ CRITICAL REQUIREMENT ⚠️
//   - ProductRegulativeIdea requires both operand ideas to be of the SAME type.
//   - ProductDeltaPath requires all component paths to be of the SAME type.
//   - Mixing different regulative idea types or path types is not allowed.
//
// ============================================================================

// ============================================================================
// TODO: REFACTORING WITH PHILOSOPHICAL TURN FOR FIRST-CLASS CITIZENSHIP 
// OF NON-STANDARD ANALYSES - UNIFY PRODUCT AND POWER, LIFT TYPE RESTRICTIONS
// ============================================================================
//
// 1. EXTEND PRODUCT TO N DIMENSIONS (VARIADIC)
//    - Generalise `ProductRegulativeIdea` to accept any number of template
//      arguments (RI1, RI2, ..., RIn).
//    - Address type becomes a `std::tuple<Addr1, Addr2, ..., Addrn>.
//    - Betweenness holds iff it holds for every coordinate (pack expansion).
//    - Metric becomes the max‑norm over all coordinates:
//        metric(tuple a, tuple b) = max_i( metric_i( get<i>(a), get<i>(b) ) ).
//    - Remove the `static_assert` that forces all ideas to be of the same type.
//
// 2. KEEP `PowerRegulativeIdea` FOR THE HOMOGENEOUS CASE
//    - When all dimensions share the EXACT SAME regulative idea, use
//      `PowerRegulativeIdea<RI, N>` as an optimisation (address = std::array).
//    - This is more efficient (stack‑allocated array, contiguous memory) and
//      convenient for building ℝⁿ from ℝ.
//
// 3. RULES OF USE
//    - For homogeneous products (identical idea per coordinate):
//        PowerRegulativeIdea<MyIdea, 3>
//    - For heterogeneous products (different ideas per coordinate):
//        ProductRegulativeIdea<IdeaX, IdeaY, IdeaZ>
//    - Both can be nested:
//        ProductRegulativeIdea< PowerRegulativeIdea<A,2>, B, PowerRegulativeIdea<C,3> >
// 4. PRIORITY: MEDIUM / LOW
// 5. PATCH ALL THE USE-CASES ACCORDINGLY TO REFACTOR IF NEEDED.
// ============================================================================

#pragma once

#include <tuple>
#include <array>
#include <utility>
#include <algorithm>
#include <functional>
#include <type_traits>
#include "delta/core/regulative_idea.h"
#include "delta/core/product_grid.h"

namespace delta::geometry {

    // -------------------------------------------------------------------------
    // Helper templates for combining betweenness and metric
    // -------------------------------------------------------------------------

    namespace detail {

        /**
         * @brief Combined betweenness relation for the product of two regulative ideas.
         *
         * A point y lies between x and z iff on each coordinate separately it lies
         * between the corresponding coordinates.
         *
         * @tparam B1 Betweenness type of the first idea.
         * @tparam B2 Betweenness type of the second idea.
         */
        template<typename B1, typename B2>
        struct ProductBetweenness {
            ProductBetweenness() = default;
            ProductBetweenness(const B1& b1, const B2& b2) : b1_(b1), b2_(b2) {}

            /**
             * @brief Evaluate betweenness on a pair of addresses.
             * @tparam T1 First address type (must have .first, .second)
             * @tparam T2 Second address type
             * @tparam T3 Third address type
             * @return true if the middle address lies between the other two in both coordinates.
             */
            template<typename T1, typename T2, typename T3>
            bool operator()(const T1& x, const T2& y, const T3& z) const {
                return b1_(x.first, y.first, z.first) &&
                    b2_(x.second, y.second, z.second);
            }

        private:
            B1 b1_;
            B2 b2_;
        };

        /**
         * @brief Combined metric for the product of two regulative ideas.
         *
         * Uses the max metric: distance = max(distance1, distance2).
         *
         * @tparam M1 Metric type of the first idea.
         * @tparam M2 Metric type of the second idea.
         */
        template<typename M1, typename M2>
        struct ProductMetric {
            ProductMetric() = default;
            ProductMetric(const M1& m1, const M2& m2) : m1_(m1), m2_(m2) {}

            /**
             * @brief Compute distance between two product addresses.
             * @tparam T1 First address type (with .first, .second)
             * @tparam T2 Second address type
             * @return max( metric1(a1,b1), metric2(a2,b2) )
             */
            template<typename T1, typename T2>
            auto operator()(const T1& a, const T2& b) const {
                auto d1 = m1_(a.first, b.first);
                auto d2 = m2_(a.second, b.second);
                return (d1 > d2) ? d1 : d2;
            }

        private:
            M1 m1_;
            M2 m2_;
        };

        /**
         * @brief Combined betweenness relation for N copies of the same regulative idea.
         * @tparam B Base betweenness type.
         * @tparam N Number of dimensions.
         */
        template<typename B, std::size_t N>
        struct PowerBetweenness {
            using address_type = std::array<typename B::first_argument_type, N>;
            using result_type = bool;

            PowerBetweenness() = default;
            explicit PowerBetweenness(const B& b) : b_(b) {}

            /**
             * @brief Evaluate betweenness for arrays of addresses.
             * @return true iff betweenness holds for every coordinate.
             */
            bool operator()(const address_type& x,
                const address_type& y,
                const address_type& z) const {
                for (std::size_t i = 0; i < N; ++i) {
                    if (!b_(x[i], y[i], z[i])) {
                        return false;
                    }
                }
                return true;
            }

            B b_;
        };

        /**
         * @brief Combined metric for N copies of the same regulative idea.
         * Uses max metric over all coordinates.
         * @tparam M Base metric type.
         * @tparam N Number of dimensions.
         */
        template<typename M, std::size_t N>
        struct PowerMetric {
            using address_type = std::array<typename M::first_argument_type, N>;
            using result_type = typename M::result_type;

            PowerMetric() = default;
            explicit PowerMetric(const M& m) : m_(m) {}

            /**
             * @brief Compute max distance over all coordinates.
             */
            result_type operator()(const address_type& a,
                const address_type& b) const {
                result_type max_dist = 0;
                for (std::size_t i = 0; i < N; ++i) {
                    auto dist = m_(a[i], b[i]);
                    if (dist > max_dist) {
                        max_dist = dist;
                    }
                }
                return max_dist;
            }

            M m_;
        };

    } // namespace detail

    // -------------------------------------------------------------------------
    // ProductRegulativeIdea – product of two regulative ideas
    // -------------------------------------------------------------------------

    /**
     * @class ProductRegulativeIdea
     * @brief Product (Cartesian product) of two regulative ideas of the same type.
     *
     * The resulting idea works on addresses that are std::pair<Addr1, Addr2>.
     * Betweenness is evaluated coordinate‑wise; metric is the max metric.
     *
     * @tparam RI1 First regulative idea type.
     * @tparam RI2 Second regulative idea type.
     * @note Both RI1 and RI2 must be the same type.
     */
    template<typename RI1, typename RI2>
    class ProductRegulativeIdea {
        static_assert(std::is_same_v<RI1, RI2>,
            "ProductRegulativeIdea: both ideas must be of the same type");
    public:
        using idea1_type = RI1;
        using idea2_type = RI2;
        using address1_type = typename RI1::address_type;
        using address2_type = typename RI2::address_type;
        using betweenness1_type = typename RI1::betweenness_type;
        using betweenness2_type = typename RI2::betweenness_type;
        using metric1_type = typename RI1::metric_type;
        using metric2_type = typename RI2::metric_type;

        using address_type = std::pair<address1_type, address2_type>;
        using betweenness_type = detail::ProductBetweenness<betweenness1_type, betweenness2_type>;
        using metric_type = detail::ProductMetric<metric1_type, metric2_type>;

        /**
         * @brief Construct from two regulative ideas.
         * @param ri1 First idea (default‑constructed if omitted).
         * @param ri2 Second idea (default‑constructed if omitted).
         */
        ProductRegulativeIdea(const RI1& ri1 = RI1(), const RI2& ri2 = RI2())
            : ri1_(ri1), ri2_(ri2)
            , betweenness_(betweenness_type(ri1_.betweenness, ri2_.betweenness))
            , metric_(metric_type(ri1_.metric, ri2_.metric)) {
        }

        /// @brief Access the combined betweenness relation.
        const betweenness_type& betweenness() const { return betweenness_; }

        /// @brief Access the combined metric.
        const metric_type& metric() const { return metric_; }

        /// @brief Access the first regulative idea.
        const RI1& idea1() const { return ri1_; }

        /// @brief Access the second regulative idea.
        const RI2& idea2() const { return ri2_; }

    private:
        RI1 ri1_;
        RI2 ri2_;
        betweenness_type betweenness_;
        metric_type metric_;
    };

    // -------------------------------------------------------------------------
    // PowerRegulativeIdea – N copies of a regulative idea
    // -------------------------------------------------------------------------

    /**
     * @class PowerRegulativeIdea
     * @brief Regular power of a regulative idea (N copies).
     *
     * Builds an idea for ℝⁿ from an idea for ℝ. The address type becomes
     * std::array<Addr, N>; betweenness and metric extend coordinate‑wise
     * with the max metric.
     *
     * @tparam RI Base regulative idea (for 1D).
     * @tparam N Number of dimensions (positive, ≤10).
     */
    template<typename RI, int N>
    class PowerRegulativeIdea {
        static_assert(N > 0, "PowerRegulativeIdea requires positive N");
        static_assert(N <= 10, "PowerRegulativeIdea supports up to 10 dimensions");

    public:
        using base_idea_type = RI;
        using base_address_type = typename RI::address_type;
        using base_betweenness_type = typename RI::betweenness_type;
        using base_metric_type = typename RI::metric_type;

        using address_type = std::array<base_address_type, N>;
        using betweenness_type = detail::PowerBetweenness<base_betweenness_type, N>;
        using metric_type = detail::PowerMetric<base_metric_type, N>;

        /**
         * @brief Construct from a base regulative idea.
         * @param ri Base idea (default‑constructed if omitted).
         */
        explicit PowerRegulativeIdea(const RI& ri = RI())
            : ri_(ri)
            , betweenness_(ri_.betweenness())
            , metric_(ri_.metric()) {
        }

        /// @brief Access the combined betweenness relation.
        const betweenness_type& betweenness() const { return betweenness_; }

        /// @brief Access the combined metric.
        const metric_type& metric() const { return metric_; }

        /// @brief Access the base regulative idea.
        const RI& base_idea() const { return ri_; }

    private:
        RI ri_;
        betweenness_type betweenness_;
        metric_type metric_;
    };

    // -------------------------------------------------------------------------
    // ProductDeltaPath – product of several delta paths
    // -------------------------------------------------------------------------

    /**
     * @class ProductDeltaPath
     * @brief Product (Cartesian product) of several delta paths.
     *
     * All component paths must be of the same type. The resulting grid is the
     * Cartesian product of the individual grids (ProductGrid).  The `advance`
     * method simultaneously advances each component path using a function that
     * maps an array of addresses (one from each path) to an array of new values.
     *
     * @tparam Paths Types of the component delta paths (all must be identical).
     */
    template<typename... Paths>
    class ProductDeltaPath {
        static_assert(sizeof...(Paths) > 0, "ProductDeltaPath requires at least one path");

        // Ensure all paths are of the same type.
        using FirstPath = std::tuple_element_t<0, std::tuple<Paths...>>;
        static_assert((std::is_same_v<FirstPath, Paths> && ...),
            "All paths in ProductDeltaPath must have the same type");

    public:
        using Addr = typename FirstPath::addr_type;
        using Value = typename FirstPath::value_type;

        /// Function type for advancing the product: takes an array of addresses,
        /// returns an array of corresponding new values.
        using Func = std::function<std::array<Value, sizeof...(Paths)>(const std::array<Addr, sizeof...(Paths)>&)>;

        using path_types = std::tuple<Paths...>;
        static constexpr std::size_t num_paths = sizeof...(Paths);

        /// Grid type is the product of the component grids (all of the same kind).
        using grid_type = delta::ProductGrid<typename FirstPath::grid_type, num_paths>;

        /// Metric type is taken from the first path (all metrics identical).
        using metric_type = typename FirstPath::metric_type;

        /**
         * @brief Construct from a list of delta paths.
         * @param paths The component paths (forwarded).
         */
        explicit ProductDeltaPath(Paths... paths)
            : paths_(std::move(paths)...) {
        }

        /**
         * @brief Construct from a tuple of delta paths.
         * @param paths A tuple containing the component paths.
         */
        explicit ProductDeltaPath(std::tuple<Paths...> paths)
            : paths_(std::move(paths)) {
        }

        /**
         * @brief Perform one refinement step for all component paths simultaneously.
         *
         * The supplied function `func` receives an array of current addresses
         * (one from each component path) and must produce an array of new values
         * (one for each path).  Each component path is then advanced using the
         * respective element as the new value for its next grid point.
         *
         * @param func The function that maps addresses to values.
         */
        void advance(const Func& func) {
            auto current_addrs = current_addresses();

            std::apply([&](auto&... p) {
                [&] <std::size_t... Is>(std::index_sequence<Is...>) {
                    (p.advance([&](const Addr& new_addr) -> Value {
                        std::array<Addr, num_paths> addrs = current_addrs;
                        addrs[Is] = new_addr;
                        return func(addrs)[Is];
                        }), ...);
                }(std::index_sequence_for<Paths...>{});
                }, paths_);
        }

        /**
         * @brief Return the current product grid.
         * @return A ProductGrid built from the current grids of the component paths.
         */
        grid_type current_grid() const {
            auto grids = std::apply([](const auto&... p) {
                return std::array<typename FirstPath::grid_type, num_paths>{ p.current_grid()... };
                }, paths_);
            return grid_type(std::move(grids));
        }

        /**
         * @brief Return the current addresses (the last grid points of each component path).
         */
        std::array<Addr, num_paths> current_addresses() const {
            return std::apply([](const auto&... p) {
                return std::array<Addr, num_paths>{
                    p.current_grid()[p.current_grid().size() - 1]...
                };
                }, paths_);
        }

        /// @brief Return the current refinement level (same for all paths).
        std::size_t level() const {
            return std::get<0>(paths_).level();
        }

        /**
         * @brief Compute the maximum gap between consecutive grid points
         *        according to the given metric.
         * @tparam ExtMetric Type of the metric (must be callable on two grid points).
         * @param metric The metric to use.
         * @return The maximum distance between neighbours.
         */
        template<typename ExtMetric>
        auto max_gap(const ExtMetric& metric) const {
            auto grid = current_grid();
            const std::size_t n = grid.size();
            if (n < 2) {
                using Distance = decltype(metric(grid[0], grid[0]));
                return Distance{ 0 };
            }

            auto max_d = metric(grid[0], grid[0]);
            for (std::size_t i = 0; i + 1 < n; ++i) {
                auto d = metric(grid[i], grid[i + 1]);
                if (d > max_d) max_d = d;
            }
            return max_d;
        }

        /// @brief Access the tuple of component paths.
        const std::tuple<Paths...>& paths() const { return paths_; }

        /// @brief Access the metric (from the first component path).
        const metric_type& metric() const { return std::get<0>(paths_).metric(); }

    private:
        std::tuple<Paths...> paths_;
    };

    // -------------------------------------------------------------------------
    // Helper functions for creating ProductDeltaPath
    // -------------------------------------------------------------------------

    /**
     * @brief Create a ProductDeltaPath from two delta paths.
     */
    template<typename Path1, typename Path2>
    ProductDeltaPath<Path1, Path2> make_product_path(Path1 p1, Path2 p2) {
        static_assert(std::is_same_v<Path1, Path2>,
            "make_product_path: paths must be of the same type");
        return ProductDeltaPath<Path1, Path2>(std::move(p1), std::move(p2));
    }

    /**
     * @brief Create a ProductDeltaPath from three delta paths.
     */
    template<typename Path1, typename Path2, typename Path3>
    ProductDeltaPath<Path1, Path2, Path3> make_product_path(Path1 p1, Path2 p2, Path3 p3) {
        static_assert(std::is_same_v<Path1, Path2> && std::is_same_v<Path1, Path3>,
            "make_product_path: all paths must be of the same type");
        return ProductDeltaPath<Path1, Path2, Path3>(std::move(p1), std::move(p2), std::move(p3));
    }

    // -------------------------------------------------------------------------
    // Traits for determining product types
    // -------------------------------------------------------------------------

    /**
     * @brief Trait to obtain the address type of a product of regulative ideas.
     */
    template<typename... RIs>
    struct product_address_type;

    template<typename RI1, typename RI2>
    struct product_address_type<RI1, RI2> {
        using type = std::pair<typename RI1::address_type, typename RI2::address_type>;
    };

    template<typename RI, int N>
    struct product_address_type<PowerRegulativeIdea<RI, N>> {
        using type = std::array<typename RI::address_type, N>;
    };

    /**
     * @brief Trait to obtain the metric type of a product of regulative ideas.
     */
    template<typename... RIs>
    struct product_metric_type;

    template<typename RI1, typename RI2>
    struct product_metric_type<RI1, RI2> {
        using type = detail::ProductMetric<
            typename RI1::metric_type,
            typename RI2::metric_type
        >;
    };

    template<typename RI, int N>
    struct product_metric_type<PowerRegulativeIdea<RI, N>> {
        using type = detail::PowerMetric<typename RI::metric_type, N>;
    };

} // namespace delta::geometry