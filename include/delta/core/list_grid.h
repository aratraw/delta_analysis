// include/delta/core/list_grid.h
#pragma once

#include <vector>
#include <algorithm>
#include <cassert>
#include <functional>
#include "grid_concept.h"

namespace delta {

    /**
     * @class ListGrid
     * @brief Grid implementation based on std::vector (for general use).
     *
     * Stores addresses in a sorted vector. Suitable for arbitrary grids
     * where addresses are not uniformly spaced. Satisfies GridConcept.
     *
     * @tparam T Address type.
     * @tparam Compare Comparison functor (strict weak ordering).
     */
    template<typename T, typename Compare = std::less<T>>
    class ListGrid {
    public:
        using value_type = T;
        using size_type = typename std::vector<T>::size_type;
        using const_iterator = typename std::vector<T>::const_iterator;
        using comparator_type = Compare;

        // -------------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------------

        /// Default constructor – creates an empty grid.
        ListGrid() = default;

        /**
         * @brief Construct from an initializer list.
         * @param init Initializer list of addresses (must be sorted according to comp).
         * @param comp Comparator to use.
         */
        ListGrid(std::initializer_list<T> init, Compare comp = Compare())
            : data_(init), comp_(std::move(comp)) {
            assert(std::is_sorted(data_.begin(), data_.end(), comp_) &&
                "Initial grid must be strictly increasing according to comparator");
        }

        /**
         * @brief Construct by moving a vector of addresses.
         * @param vec Vector of addresses (must be sorted according to comp).
         * @param comp Comparator to use.
         */
        explicit ListGrid(std::vector<T>&& vec, Compare comp = Compare())
            : data_(std::move(vec)), comp_(std::move(comp)) {
            assert(std::is_sorted(data_.begin(), data_.end(), comp_) &&
                "Grid must be strictly increasing");
        }

        /**
         * @brief Construct from an iterator range.
         * @tparam InputIt Iterator type.
         * @param first Beginning of range.
         * @param last End of range.
         * @param comp Comparator to use.
         */
        template<typename InputIt>
        ListGrid(InputIt first, InputIt last, Compare comp = Compare())
            : data_(first, last), comp_(std::move(comp)) {
            assert(std::is_sorted(data_.begin(), data_.end(), comp_) &&
                "Grid must be strictly increasing");
        }

        // -------------------------------------------------------------------------
        // Accessors
        // -------------------------------------------------------------------------

        /// Returns the number of addresses in the grid.
        size_type size() const noexcept { return data_.size(); }

        /// Returns the address at the given index (no bounds checking in release mode).
        const T& operator[](size_type index) const noexcept { return data_[index]; }

        /// Returns an iterator to the first address.
        const_iterator begin() const noexcept { return data_.begin(); }

        /// Returns an iterator past the last address.
        const_iterator end() const noexcept { return data_.end(); }

        /// Returns a const reference to the underlying vector.
        const std::vector<T>& data() const noexcept { return data_; }

        /// Returns the comparator used by the grid.
        const Compare& comparator() const noexcept { return comp_; }

        std::vector<value_type> collect_points() const {
            return data_;   // data_ is std::vector<T>
        }

        // -------------------------------------------------------------------------
        // Refinement
        // -------------------------------------------------------------------------

        /**
         * @brief Generate the next refinement level.
         *
         * For every consecutive pair (x, y) in this grid, computes z = refine(x, y)
         * and inserts z between them. The resulting grid uses the same comparator.
         *
         * @tparam RefineOp Callable with signature T(const T&, const T&)
         * @param refine The refinement operator (Δₙ).
         * @return A new ListGrid containing the refined list.
         */
        template<typename RefineOp>
        ListGrid refine(RefineOp&& refine) const {
            size_type n = size();
            if (n == 0) return ListGrid(std::vector<T>{}, comp_);
            if (n == 1) return ListGrid(std::vector<T>{data_.front()}, comp_);

            std::vector<T> next;
            next.reserve(2 * n - 1);

            for (size_type i = 0; i < n - 1; ++i) {
                const T& left = data_[i];
                const T& right = data_[i + 1];
                next.push_back(left);
                T mid = refine(left, right);
                next.push_back(std::move(mid));
            }
            next.push_back(data_.back());

            return ListGrid(std::move(next), comp_);
        }

        /**
         * @brief Equality comparison (compares underlying data).
         */
        bool operator==(const ListGrid& other) const noexcept {
            return data_ == other.data_;
        }

    private:
        std::vector<T> data_;
        Compare comp_;

        /**
         * @brief Private constructor that assumes the vector is already sorted.
         * @param vec Sorted vector of addresses.
         * @param comp Comparator.
         * @param trusted Unused tag to distinguish from public constructors.
         */
        ListGrid(std::vector<T>&& vec, Compare comp, bool /*trusted*/)
            : data_(std::move(vec)), comp_(std::move(comp)) {
        }
    };

    // Ensure ListGrid satisfies the GridConcept.
    static_assert(OrderedGrid<ListGrid<int>>);

} // namespace delta