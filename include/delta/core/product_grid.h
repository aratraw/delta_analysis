// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/core/product_grid.h
#pragma once

#include <array>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <vector>
#include "grid_concept.h"

namespace delta {

    // -----------------------------------------------------------------------------
    // ProductGrid – декартово произведение N сеток одного типа
    // -----------------------------------------------------------------------------
    template<typename Grid, std::size_t N>
    class ProductGrid {
        static_assert(N > 0, "ProductGrid requires at least one grid");
        static_assert(SimpleGrid<Grid>, "Grid must satisfy SimpleGrid concept");

    public:
        using value_type = std::array<typename Grid::value_type, N>;
        using size_type = std::size_t;
        class const_iterator;

        // -------------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------------
        explicit ProductGrid(std::array<Grid, N> grids)
            : grids_(std::move(grids)) {
            for (std::size_t i = 0; i < N; ++i) {
                sizes_[i] = grids_[i].size();
            }
        }

        // -------------------------------------------------------------------------
        // Accessors required by GridConcept
        // -------------------------------------------------------------------------
        size_type size() const noexcept {
            size_type total = 1;
            for (std::size_t i = 0; i < N; ++i) {
                total *= sizes_[i];
            }
            return total;
        }

        value_type operator[](size_type idx) const {
            if (idx >= size()) throw std::out_of_range("ProductGrid::operator[]");
            return compute_tuple(idx);
        }

        const_iterator begin() const noexcept { return const_iterator(this, 0); }
        const_iterator end() const noexcept { return const_iterator(this, size()); }

        // Additional method for discrete operators
        const Grid& get_grid(std::size_t i) const {
            if (i >= N) throw std::out_of_range("ProductGrid::get_grid");
            return grids_[i];
        }

        // -------------------------------------------------------------------------
        // Utility for parallel processing: returns a flat vector of all points
        // -------------------------------------------------------------------------
        std::vector<value_type> collect_points() const {
            std::vector<value_type> result;
            result.reserve(size());
            for (const auto& addr : *this) {
                result.push_back(addr);
            }
            return result;
        }

        // -------------------------------------------------------------------------
        // Iterator
        // -------------------------------------------------------------------------
        class const_iterator {
        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = ProductGrid::value_type;
            using difference_type = std::ptrdiff_t;
            using pointer = const value_type*;
            using reference = const value_type&;

            const_iterator() = default;
            const_iterator(const ProductGrid* grid, size_type idx) : grid_(grid), idx_(idx) {}

            value_type operator*() const { return grid_->compute_tuple(idx_); }
            const_iterator& operator++() { ++idx_; return *this; }
            const_iterator operator++(int) { auto tmp = *this; ++*this; return tmp; }
            bool operator==(const const_iterator& other) const { return idx_ == other.idx_; }
            bool operator!=(const const_iterator& other) const { return idx_ != other.idx_; }

        private:
            const ProductGrid* grid_ = nullptr;
            size_type idx_ = 0;
        };

    private:
        std::array<Grid, N> grids_;
        std::array<size_type, N> sizes_;

        value_type compute_tuple(size_type idx) const {
            value_type result;
            size_type remaining = idx;
            for (std::size_t i = N; i-- > 0; ) {
                size_type local_idx = remaining % sizes_[i];
                remaining /= sizes_[i];
                result[i] = grids_[i][local_idx];
            }
            return result;
        }
    };

} // namespace delta