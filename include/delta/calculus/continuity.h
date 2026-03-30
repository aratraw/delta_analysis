// include/delta/calculus/continuity.h
#pragma once

#include <cstddef>
#include <vector>
#include <algorithm>
#include <type_traits>
#include "delta/core/rational.h"

namespace delta::calculus {

    /**
     * @brief Compute the maximum oscillation of a function on a grid.
     *
     * The oscillation is defined as the maximum distance between values at consecutive
     * addresses, measured by the given value metric.
     *
     * @tparam Grid        A type satisfying GridConcept (must provide size() and operator[]).
     * @tparam Func        Callable with signature Value(const Addr&).
     * @tparam ValueMetric A metric on values (must satisfy ValueMetric<ValueMetric, Value, Distance>).
     * @param grid The grid over which to compute the oscillation.
     * @param func The function.
     * @param vm   The value metric.
     * @return The maximum distance between f(x_i) and f(x_{i+1}) over all consecutive pairs.
     *         Returns a default‑constructed Distance (zero) if grid.size() < 2.
     */
    template<typename Grid, typename Func, typename ValueMetric>
    auto max_oscillation(const Grid& grid, Func&& func, const ValueMetric& vm) {
        using Distance = Rational; // принудительно используем Rational, чтобы избежать проблем с expression templates
        Distance max_dist{ 0 };
        const std::size_t n = grid.size();
        if (n < 2) return max_dist;

        for (std::size_t i = 0; i + 1 < n; ++i) {
            Distance d = vm(func(grid[i + 1]), func(grid[i]));
            if (d > max_dist) max_dist = d;
        }
        return max_dist;
    }

    /**
     * @brief Check whether a function satisfies a given modulus of continuity on a grid.
     *
     * For all consecutive addresses x_i, x_{i+1} in the grid, verifies that
     *   vm(f(x_{i+1}), f(x_i)) ≤ modulus(δ) + tolerance,
     * where δ is the maximum gap of the grid (max_gap(grid)).
     *
     * This is a generalised version of Definition 7.5.2 from the Δ‑analysis theory.
     *
     * @tparam Grid        A type satisfying GridConcept.
     * @tparam Func        Callable with signature Value(const Addr&).
     * @tparam ValueMetric A metric on values.
     * @tparam Mod         A modulus type (must satisfy Modulus<Mod, Distance>).
     * @param grid      The grid.
     * @param func      The function.
     * @param vm        The value metric.
     * @param modulus   The modulus of continuity (callable with the maximum gap).
     * @param tolerance Additional tolerance (will be converted to Distance).
     * @return true if the inequality holds for every consecutive pair.
     */
    template<typename Grid, typename Func, typename ValueMetric, typename Mod, typename T = Rational>
    bool check_continuity_level(const Grid& grid, Func&& func, const ValueMetric& vm,
        const Mod& modulus, const T& tolerance = Rational(0)) {
        using Distance = Rational; // все расстояния приводим к Rational
        Distance max_osc = max_oscillation(grid, std::forward<Func>(func), vm);
        Distance delta_n = max_gap(grid);
        Distance bound = modulus(delta_n);
        Distance tol = tolerance;
        return max_osc <= bound + tol;
    }

} // namespace delta::calculus