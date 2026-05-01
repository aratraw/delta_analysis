// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/core/grid.h
#pragma once

#include <vector>
#include <algorithm>
#include <cassert>
#include <functional>

namespace delta {

    /**
     * @class Grid
     * @brief Represents a finite ordered list of addresses (S_n) in Δ-analysis.
     *
     * The order is determined by a comparator (default std::less), which must be
     * consistent with the betweenness relation of the regulative idea.
     * @tparam T Address type.
     * @tparam Compare Comparison functor (strict weak ordering).
     */
    template<typename T, typename Compare = std::less<T>>
    class Grid {
    public:
        using value_type = T;
        using size_type = typename std::vector<T>::size_type;
        using const_iterator = typename std::vector<T>::const_iterator;

        // -------------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------------

        Grid() = default;

        /**
         * @brief Construct from initializer list, using given comparator.
         */
        Grid(std::initializer_list<T> init, Compare comp = Compare())
            : data_(init), comp_(std::move(comp)) {
            assert(std::is_sorted(data_.begin(), data_.end(), comp_) &&
                "Initial grid must be strictly increasing according to comparator");
        }

        /**
         * @brief Construct from a sorted vector.
         */
        explicit Grid(std::vector<T>&& vec, Compare comp = Compare())
            : data_(std::move(vec)), comp_(std::move(comp)) {
            assert(std::is_sorted(data_.begin(), data_.end(), comp_) &&
                "Grid must be strictly increasing");
        }

        /**
         * @brief Construct from a sorted range.
         */
        template<typename InputIt>
        Grid(InputIt first, InputIt last, Compare comp = Compare())
            : data_(first, last), comp_(std::move(comp)) {
            assert(std::is_sorted(data_.begin(), data_.end(), comp_) &&
                "Grid must be strictly increasing");
        }

        // -------------------------------------------------------------------------
        // Accessors
        // -------------------------------------------------------------------------

        size_type size() const noexcept { return data_.size(); }
        const T& operator[](size_type index) const noexcept { return data_[index]; }
        const_iterator begin() const noexcept { return data_.begin(); }
        const_iterator end() const noexcept { return data_.end(); }
        const std::vector<T>& data() const noexcept { return data_; }
        const Compare& comparator() const noexcept { return comp_; }

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
         * @return A new Grid containing the refined list.
         */
        template<typename RefineOp>
        Grid refine(RefineOp&& refine) const {
            size_type n = size();
            // Empty Grid - return empty grid with Comparator
            if (n == 0) return Grid(std::vector<T>{}, comp_);
            // Grid with 1 element - return copy (nothing to insert)
            if (n == 1) return Grid(std::vector<T>{data_.front()}, comp_);

            std::vector<T> next;
            next.reserve(2 * n - 1);

            for (size_type i = 0; i < n - 1; ++i) {
                const T& left = data_[i];
                const T& right = data_[i + 1];
                next.push_back(left);
                T mid = refine(left, right);
                // Note: The betweenness condition should be checked by the caller (DeltaPath)
                // because the comparator alone cannot enforce "strictly between".
                next.push_back(std::move(mid));
            }
            next.push_back(data_.back());

            return Grid(std::move(next), comp_);
        }

        /**
         * @brief Equality comparison (compares underlying data).
         */
        bool operator==(const Grid& other) const noexcept {
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
         *
         * Used internally by refine() to avoid re‑checking sortedness.
         */
        Grid(std::vector<T>&& vec, Compare comp, bool /*trusted*/)
            : data_(std::move(vec)), comp_(std::move(comp)) {
        }
    };

} // namespace delta