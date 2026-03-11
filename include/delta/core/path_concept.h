// include/delta/core/path_concept.h
#pragma once

#include "grid_concept.h"
#include <concepts>

namespace delta {

    /**
     * @concept Path
     * @brief Requirements for a Δ‑path.
     *
     * A path provides a sequence of refined grids. It must be able to advance
     * to the next level, return the current grid, and provide the current level.
     * Additionally, it must be able to compute the maximum gap of the current grid
     * using a given metric.
     */
    template<typename P, typename Metric>
    concept Path = requires(P p, const P cp, Metric m) {
        { p.advance() } -> std::same_as<void>;
        { cp.current_grid() } -> SimpleGrid;
        { cp.level() } -> std::convertible_to<std::size_t>;
        { cp.max_gap(m) } -> std::regular;   // возвращаемый тип должен быть регулярным
    };

} // namespace delta