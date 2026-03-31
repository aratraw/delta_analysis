// include/delta/calculus/differentiability.h
#pragma once

#include <cstddef>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "modulus.h"
#include "delta/core/regulative_idea.h"
#include "delta/core/rational.h"

namespace delta::calculus {
    using delta::abs;
    /**
     * @brief Find the index of an address in a grid.
     *
     * @tparam Grid A type satisfying GridConcept.
     * @tparam Addr Address type (must be equality‑comparable).
     * @param grid The grid to search.
     * @param addr The address to locate.
     * @return Index of the address if found, otherwise -1.
     */
    template<typename Grid, typename Addr>
        requires SimpleGrid<Grid>
    std::ptrdiff_t find_address_index(const Grid& grid, const Addr& addr) {
        for (std::size_t i = 0; i < grid.size(); ++i) {
            if (grid[i] == addr) return static_cast<std::ptrdiff_t>(i);
        }
        return -1;
    }

    /**
     * @brief Compute the left difference quotient at an interior address.
     *
     * For a grid with points ..., x_{i-1}, x_i, x_{i+1}, ... and a function f,
     * the left difference quotient is (f(x_i) - f(x_{i-1})) / (x_i - x_{i-1}).
     *
     * @tparam Grid A type satisfying GridConcept; its value_type must be SubtractableAddress.
     * @tparam Func Callable with signature Value(const Addr&).
     * @param grid The grid containing the address.
     * @param addr The address at which the quotient is computed (must be interior).
     * @param func The function.
     * @return The left difference quotient.
     * @throws std::invalid_argument if addr is not found or is an endpoint.
     */
    template<typename Grid, typename Func>
        requires SimpleGrid<Grid>&& SubtractableAddress<typename Grid::value_type>
    auto left_difference_quotient(const Grid& grid, const typename Grid::value_type& addr,
        Func&& func) {
        std::ptrdiff_t idx = find_address_index(grid, addr);
        if (idx <= 0 || idx >= static_cast<std::ptrdiff_t>(grid.size()) - 1) {
            throw std::invalid_argument("Address not found or is endpoint");
        }
        const auto& left = grid[static_cast<std::size_t>(idx - 1)];
        auto f_left = func(left);
        auto f_right = func(addr);
        auto dx = addr - left;
        return (f_right - f_left) / dx;
    }

    /**
     * @brief Compute the right difference quotient at an interior address.
     *
     * For a grid with points ..., x_{i-1}, x_i, x_{i+1}, ... and a function f,
     * the right difference quotient is (f(x_{i+1}) - f(x_i)) / (x_{i+1} - x_i).
     *
     * @tparam Grid A type satisfying GridConcept; its value_type must be SubtractableAddress.
     * @tparam Func Callable with signature Value(const Addr&).
     * @param grid The grid containing the address.
     * @param addr The address at which the quotient is computed (must be interior).
     * @param func The function.
     * @return The right difference quotient.
     * @throws std::invalid_argument if addr is not found or is an endpoint.
     */
    template<typename Grid, typename Func>
        requires SimpleGrid<Grid>&& SubtractableAddress<typename Grid::value_type>
    auto right_difference_quotient(const Grid& grid, const typename Grid::value_type& addr,
        Func&& func) {
        std::ptrdiff_t idx = find_address_index(grid, addr);
        if (idx <= 0 || idx >= static_cast<std::ptrdiff_t>(grid.size()) - 1) {
            throw std::invalid_argument("Address not found or is endpoint");
        }
        const auto& right = grid[static_cast<std::size_t>(idx + 1)];
        auto f_left = func(addr);
        auto f_right = func(right);
        auto dx = right - addr;
        return (f_right - f_left) / dx;
    }

    /**
     * @brief Check differentiability at a point using a modulus of convergence.
     *
     * For all refinement levels n >= first_level, verifies that both the left and right
     * difference quotients differ from the expected derivative D by at most
     * modulus(δ_n) + tolerance, where δ_n is the maximum gap of the grid at level n.
     *
     * This implements a generalised version of Definition 3.4 from the Δ‑analysis theory.
     *
     * @tparam Grid    A type satisfying GridConcept; its value_type must be SubtractableAddress.
     * @tparam Func    Callable with signature Value(const Addr&).
     * @tparam Distance Scalar type for distances (must support convert_to<double> and arithmetic).
     * @tparam Addr    Address type (must be SubtractableAddress).
     * @tparam Mod     Modulus type (must satisfy Modulus<Mod, Distance>).
     * @param grids      Vector of grids for successive refinement levels.
     * @param addr       The address to test (must be interior from first_level onward).
     * @param func       The function.
     * @param D          Expected derivative value.
     * @param modulus    Modulus of convergence (called with the maximum gap of each grid).
     * @param first_level The first level at which addr appears (inclusive).
     * @param tolerance  Additional tolerance for comparisons (default 1e-12).
     * @return true if the differentiability condition holds for all levels n >= first_level.
     */
    template<typename Grid, typename Func, typename Distance, typename Addr, typename Mod>
        requires SubtractableAddress<Addr>
    bool check_differentiability(const std::vector<Grid>& grids, const Addr& addr,
        Func&& func, const Distance& D, const Mod& modulus,
        std::size_t first_level, const Rational& tolerance = Rational(1, 1000000000000)) {
        for (std::size_t n = first_level; n < grids.size(); ++n) {
            const auto& grid = grids[n];
            std::ptrdiff_t idx = find_address_index(grid, addr);
            if (idx < 0) return false;
            if (idx == 0 || idx == static_cast<std::ptrdiff_t>(grid.size()) - 1) return false;

            auto left_dq = left_difference_quotient(grid, addr, func);
            auto right_dq = right_difference_quotient(grid, addr, func);

            Distance delta_n = max_gap(grid);
            Distance bound = modulus(delta_n);   // modulus must return a Distance
            // Теперь сравниваем рациональные числа напрямую
            if (delta::abs(left_dq - D) > bound + tolerance ||
                delta::abs(right_dq - D) > bound + tolerance) {
                return false;
            }
        }
        return true;
    }

} // namespace delta::calculus