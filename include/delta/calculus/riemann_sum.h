// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/calculus/riemann_sum.h
#pragma once

#include <cstddef>
#include <functional>
#include <cmath>
#include <type_traits>
#include "delta/core/regulative_idea.h"
#include <Eigen/Core>   // for Eigen::MatrixBase

namespace delta::calculus::detail {
    /**
     * @brief Default materialisation: forwards the argument unchanged.
     *
     * For most types, this simply returns the argument as is.
     */
    template<typename T>
    decltype(auto) materialize(T&& x) {
        return std::forward<T>(x);
    }

    /**
     * @brief Specialisation for Eigen expression templates.
     *
     * Evaluates the expression to an actual matrix, preventing expensive
     * re-evaluations in loops.
     */
    template<typename Derived>
    auto materialize(const Eigen::MatrixBase<Derived>& x) {
        return x.eval();
    }
}

namespace delta::calculus {

    /**
     * @brief Compute the left Riemann sum of a function over a grid.
     *
     * For a grid with points x0, x1, ..., x_{n-1}, the left Riemann sum is
     * Σ_{i=0}^{n-2} f(x_i) * (x_{i+1} - x_i).
     *
     * @tparam Grid A type satisfying GridConcept; its value_type must satisfy SubtractableAddress.
     * @tparam Func Callable with signature Value(const Addr&).
     * @param grid The grid over which to integrate.
     * @param func The function to integrate.
     * @return The left Riemann sum. If grid.size() < 2, returns a default‑constructed Value.
     */
    template<typename Grid, typename Func>
        requires SubtractableAddress<typename Grid::value_type>
    auto left_riemann_sum(const Grid& grid, Func&& func) {
        using Addr = typename Grid::value_type;
        using RawValue = std::invoke_result_t<Func, Addr>;
        using Value = std::decay_t<RawValue>;
        const std::size_t n = grid.size();
        if (n < 2) return Value{};
        Value sum = detail::materialize(func(grid[0]) * (grid[1] - grid[0]));
        for (std::size_t i = 1; i + 1 < n; ++i) {
            sum = detail::materialize(sum + func(grid[i]) * (grid[i + 1] - grid[i]));
        }
        return sum;
    }

    /**
     * @brief Compute the right Riemann sum of a function over a grid.
     *
     * For a grid with points x0, x1, ..., x_{n-1}, the right Riemann sum is
     * Σ_{i=0}^{n-2} f(x_{i+1}) * (x_{i+1} - x_i).
     *
     * @tparam Grid A type satisfying GridConcept; its value_type must satisfy SubtractableAddress.
     * @tparam Func Callable with signature Value(const Addr&).
     * @param grid The grid over which to integrate.
     * @param func The function to integrate.
     * @return The right Riemann sum. If grid.size() < 2, returns a default‑constructed Value.
     */
    template<typename Grid, typename Func>
        requires SubtractableAddress<typename Grid::value_type>
    auto right_riemann_sum(const Grid& grid, Func&& func) {
        using Addr = typename Grid::value_type;
        using RawValue = std::invoke_result_t<Func, Addr>;
        using Value = std::decay_t<RawValue>;
        const std::size_t n = grid.size();
        if (n < 2) return Value{};
        Value sum = detail::materialize(func(grid[1]) * (grid[1] - grid[0]));
        for (std::size_t i = 1; i + 1 < n; ++i) {
            sum = detail::materialize(sum + func(grid[i + 1]) * (grid[i + 1] - grid[i]));
        }
        return sum;
    }

    /**
     * @brief Compute a tagged Riemann sum with a user‑supplied tagger.
     *
     * For each interval [x_i, x_{i+1}], the point at which to evaluate the function
     * is given by `tagger(x_i, x_{i+1})`. The sum is Σ f(tagger(x_i, x_{i+1})) * (x_{i+1} - x_i).
     *
     * @tparam Grid A type satisfying GridConcept; its value_type must satisfy SubtractableAddress.
     * @tparam Func Callable with signature Value(const Addr&).
     * @tparam Tagger Callable with signature Addr(const Addr& left, const Addr& right).
     * @param grid The grid over which to integrate.
     * @param func The function to integrate.
     * @param tagger A function that returns a tag point inside each interval.
     * @return The tagged Riemann sum. If grid.size() < 2, returns a default‑constructed Value.
     */
    template<typename Grid, typename Func, typename Tagger>
        requires SubtractableAddress<typename Grid::value_type>
    auto tagged_riemann_sum(const Grid& grid, Func&& func, Tagger&& tagger) {
        using Addr = typename Grid::value_type;
        using RawValue = std::invoke_result_t<Func, Addr>;
        using Value = std::decay_t<RawValue>;
        const std::size_t n = grid.size();
        if (n < 2) return Value{};
        Value sum = detail::materialize(func(tagger(grid[0], grid[1])) * (grid[1] - grid[0]));
        for (std::size_t i = 1; i + 1 < n; ++i) {
            sum = detail::materialize(sum + func(tagger(grid[i], grid[i + 1])) * (grid[i + 1] - grid[i]));
        }
        return sum;
    }

    /**
     * @brief Compute a Riemann‑like sum on a tree‑structured grid (binary strings).
     *
     * This function is tailored for a TreeDeltaPath (or any type providing `current_grid()`
     * that returns a TreeGrid). For each leaf at the current level, the contribution is
     * `func(addr) * 2^{-level}`, i.e., the uniform measure on the tree.
     *
     * @tparam Path A type that models a tree path (must provide `current_grid()` and `level()`).
     * @tparam Func Callable with signature double(const std::string&).
     * @param path The tree path.
     * @param func The function to integrate.
     * @return The approximate integral over the tree.
     */
    template<typename Path, typename Func>
    Rational tree_riemann_sum(const Path& path, Func&& func) {
        Rational sum = 0_r;
        const auto& grid = path.current_grid();
        std::size_t level = grid.level();
        Rational weight = Rational(1) / delta::pow(Rational(2), static_cast<int>(level));
        for (const auto& addr : grid) {
            if (addr.size() == level) {
                sum += func(addr) * weight;
            }
        }
        return sum;
    }

} // namespace delta::calculus