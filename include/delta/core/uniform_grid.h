// include/delta/core/uniform_grid.h
#pragma once

#include <cstddef>
#include <cassert>
#include <functional>
#include <type_traits>
#include "grid_concept.h"
#include "regulative_idea.h"

namespace delta {

    /**
     * @class UniformGrid
     * @brief Grid for uniformly spaced addresses.
     *
     * Stores only the start, step, and number of points. Addresses are computed on the fly,
     * making this grid type extremely lightweight and suitable for dyadic and other regular grids.
     *
     * @tparam T Address type. Must satisfy LinearAddress<T> (i.e., supports addition and scalar multiplication).
     * @tparam Compare Comparison functor (must be consistent with the natural order induced by the arithmetic).
     *         Defaults to std::less<T>.
     */
    template<typename T, typename Compare = std::less<T>>
        requires LinearAddress<T, T>
    class UniformGrid {
    public:
        using value_type = T;
        using size_type = std::size_t;
        using comparator_type = Compare;

        /// Forward iterator that generates addresses on the fly.
        class const_iterator;

        // -------------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------------

        /// Default constructor – creates an empty grid (count = 0, start and step default‑constructed).
        UniformGrid() = default;

        /**
         * @brief Construct a uniform grid.
         * @param start First address (e.g., left endpoint).
         * @param step  Constant spacing between consecutive addresses.
         * @param count Number of points (must be ≥ 1).
         * @param comp  Comparator to use (should be consistent with the order of the generated points).
         */
        UniformGrid(T start, T step, size_type count, Compare comp = Compare())
            : start_(std::move(start)), step_(std::move(step)), count_(count), comp_(std::move(comp)) {
            assert(count_ >= 1 && "UniformGrid must have at least one point");
        }

        // -------------------------------------------------------------------------
        // Accessors required by GridConcept
        // -------------------------------------------------------------------------

        /// Returns the number of points in the grid.
        size_type size() const noexcept { return count_; }

        /**
         * @brief Returns the address at the given index.
         * @param index Must be less than size().
         * @return start_ + step_ * index.
         */
        T operator[](size_type index) const noexcept {
            assert(index < count_);
            // Multiplication of step_ by integer works for any LinearAddress type.
            return start_ + step_ * static_cast<int64_t>(index);
        }

        /// Returns an iterator to the first point (start_).
        const_iterator begin() const noexcept { return const_iterator(this, 0); }

        /// Returns an iterator past the last point.
        const_iterator end() const noexcept { return const_iterator(this, count_); }

        /// Returns the start value (first address).
        T start() const noexcept { return start_; }

        /// Returns the step (spacing between addresses).
        T step() const noexcept { return step_; }

        /// Returns the number of points (same as size()).
        size_type count() const noexcept { return count_; }

        /// Returns the comparator used by the grid.
        const Compare& comparator() const noexcept { return comp_; }

        // -------------------------------------------------------------------------
        // Refinement is not provided directly – use the free function refine_grid
        // -------------------------------------------------------------------------

    private:
        T start_;        ///< First address of the grid.
        T step_;         ///< Uniform spacing.
        size_type count_; ///< Number of points.
        Compare comp_;    ///< Comparator (needed for GridConcept compatibility).

    public:
        /**
         * @class const_iterator
         * @brief Forward iterator for UniformGrid.
         *
         * Stores a pointer to the parent grid and the current index.
         * Dereferencing computes the address on the fly.
         */
        class const_iterator {
        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = T;
            using difference_type = std::ptrdiff_t;
            using pointer = const T*;
            using reference = const T&;

            const_iterator() = default;

            const_iterator(const UniformGrid* grid, size_type idx) : grid_(grid), idx_(idx) {}

            /// Returns the address at the current index.
            T operator*() const { return (*grid_)[idx_]; }

            /// Pre‑increment.
            const_iterator& operator++() { ++idx_; return *this; }

            /// Post‑increment.
            const_iterator operator++(int) { auto tmp = *this; ++*this; return tmp; }

            /// Equality comparison.
            bool operator==(const const_iterator& other) const { return idx_ == other.idx_; }

            /// Inequality comparison.
            bool operator!=(const const_iterator& other) const { return idx_ != other.idx_; }

        private:
            const UniformGrid* grid_ = nullptr;
            size_type idx_ = 0;
        };
    };
} // namespace delta