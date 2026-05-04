// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/core/grid_refine.h
#pragma once

#include <type_traits>
#include <vector>
#include "list_grid.h"
#include "uniform_grid.h"

namespace delta {

    namespace detail {
        template<typename G>
        struct is_list_grid : std::false_type {};

        template<typename T, typename C>
        struct is_list_grid<ListGrid<T, C>> : std::true_type {};

        template<typename G>
        struct is_uniform_grid : std::false_type {};

        template<typename T, typename C>
        struct is_uniform_grid<UniformGrid<T, C>> : std::true_type {};
    }

    /**
     * @brief Generic refinement function.
     *
     * For ListGrid, calls its refine method.
     * For UniformGrid, generates a ListGrid (since result may not be uniform).
     *
     * @tparam Grid Type of grid.
     * @tparam RefineOp Callable with signature Addr(const Addr&, const Addr&)
     * @param grid The grid to refine.
     * @param refine The refinement operator.
     * @return A new ListGrid (always ListGrid for now, for simplicity).
     */
    template<typename Grid, typename RefineOp>
    auto refine_grid(const Grid& grid, RefineOp&& refine) {
        using Addr = typename Grid::value_type;
        using Compare = std::remove_cvref_t<decltype(grid.comparator())>;

        if constexpr (detail::is_list_grid<Grid>::value) {
            return grid.refine(std::forward<RefineOp>(refine));
        }
        else {
            // For any other grid type (including UniformGrid), generate ListGrid
            std::size_t n = grid.size();
            if (n == 0) return ListGrid<Addr, Compare>(std::vector<Addr>{}, grid.comparator());
            if (n == 1) return ListGrid<Addr, Compare>(std::vector<Addr>{grid[0]}, grid.comparator());

            std::vector<Addr> next;
            next.reserve(2 * n - 1);

            for (std::size_t i = 0; i < n - 1; ++i) {
                Addr left = grid[i];
                Addr right = grid[i + 1];
                next.push_back(left);
                Addr mid = refine(left, right);
                next.push_back(std::move(mid));
            }
            next.push_back(grid[n - 1]);

            return ListGrid<Addr, Compare>(std::move(next), grid.comparator());
        }
    }

} // namespace delta