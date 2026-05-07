// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0
// =============================================================================
// DIFFERENCE BETWEEN Betweenness AND COMPARATOR (LESS) IN MULTIDIMENSIONAL
// Δ‑ANALYSIS
// =============================================================================
//
// In one dimension, a strict total order (less) on points is unambiguous:
// given two distinct rational numbers, one is smaller and lies “to the left”
// of the other.  This order is intimately linked to the geometric betweenness
// relation:  x < y < z  ⟺  betweenness(x, y, z) holds.
//
// In two or more dimensions the situation is fundamentally different.
// Consider points A = {1, 2, 3} and B = {3, 1, 2} in ℝ³.
//   • Lexicographic “less” says A < B because the first coordinate 1 < 3.
//     This ordering lacks any deeper meaning: it treats coordinates as
//     independent keys, ignoring the fact that a coordinate tuple represents
//     a single geometric location.  Any statement of the form “A < B” is
//     arbitrary – the relation depends on which axis we compare first, and
//     it changes under rotations or translations of the coordinate system.
//     Even comparing distances from the origin would be coordinate‑dependent
//     unless the metric is radial.
//   • The betweenness predicate between(A, B, C), on the other hand, remains
//     well‑defined in any regulative idea, no matter how complicated the
//     address space or the metric tensor.  between(A, B, C) means “B lies
//     between A and C according to the chosen betweenness relation”.
//     This relation respects the geometric structure of the space:
//     for a linear order (1D) it is the familiar ordering; for a tree
//     (binary strings) it is the unique ancestor relation; for a product
//     of regulative ideas it holds component‑wise.  It never depends on
//     an artificial numbering of axes.
//
// IMPLICATION FOR THE LIBRARY:
//   • A comparator (less) is required only to satisfy the OrderedGrid
//     concept, because classes like OperationalFunction internally use
//     std::map.  Any strict weak ordering is sufficient; the library
//     therefore adopts the natural lexicographic order derived from the
//     comparators of the component 1‑D grids.  This order is canonical,
//     deterministic, and sufficient for container storage / lookup.
//   • Betweenness is the geometric primitive that guides refinement.
//     Delta operators use betweenness to decide whether a proposed new
//     point actually lies inside an interval.  Adaptive paths check
//     betweenness to guarantee that inserted points stay within bounds.
//     Betweenness must NOT be replaced by a simple less comparison.
//
// CRUCIAL RULE:  Never confuse the two.  Use the comparator (less) when you
// need a total order for containers; use the betweenness predicate when you
// need to make a geometric statement about points in the address space.
// 
// NOTE: So maybe OperationalFunction should not demand an OrderedGrid concept in the sense of lexicography (less)
// only in the sense of an actual meaningful geometry (Betweenness)? 
// =============================================================================
// include/delta/core/product_grid.h
#pragma once

#include <array>
#include <cstddef>
#include <functional>          // required for std::less (used as fallback)
#include <iterator>
#include <stdexcept>
#include <vector>
#include "grid_concept.h"

namespace delta {

    // -----------------------------------------------------------------------------
    // ProductGrid – Cartesian product of N grids of the same type
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
        // Nested comparator: delegates to comparators of the component grids
        // -------------------------------------------------------------------------
        // This comparator is constructed from the component grids' comparators,
        // providing a strict weak ordering consistent with the ordering of each
        // 1‑D axis.  It is NOT a geometric distance‑based ordering; it is intended
        // for use in ordered associative containers (std::map) inside
        // OperationalFunction, where any canonical total order suffices.
        //
        // Design rationale:
        //   - ProductGrid is an OrderedGrid, therefore it must provide a comparator.
        //   - The natural choice is lexicographic order using the comparators of
        //     the 1‑D grids.  This preserves the per‑axis ordering and results in
        //     a strict weak ordering.
        //   - Future specialisations could use a space‑filling curve or other
        //     metric‑based comparator without changing the interface.
        class product_comparator {
            // Убираем ссылки с типа, возвращаемого comparator() (который может быть const Compare&)
            using CompType = std::decay_t<decltype(std::declval<Grid>().comparator())>;
            std::array<CompType, N> comps_;
        public:
            product_comparator(const std::array<Grid, N>& grids) {
                for (std::size_t i = 0; i < N; ++i)
                    comps_[i] = grids[i].comparator();   // копируем компаратор по значению
            }

            bool operator()(const value_type& a, const value_type& b) const {
                for (std::size_t i = 0; i < N; ++i) {
                    if (comps_[i](a[i], b[i])) return true;
                    if (comps_[i](b[i], a[i])) return false;
                }
                return false; // equal
            }
        };
        // Satisfy OrderedGrid concept
        using comparator_type = product_comparator;
        comparator_type comparator() const noexcept { return comparator_type(grids_); }

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
            for (std::size_t i = 0; i < N; ++i) total *= sizes_[i];
            return total;
        }

        value_type operator[](size_type idx) const {
            if (idx >= size()) throw std::out_of_range("ProductGrid::operator[]");
            return compute_tuple(idx);
        }

        const_iterator begin() const noexcept { return const_iterator(this, 0); }
        const_iterator end() const noexcept { return const_iterator(this, size()); }

        const Grid& get_grid(std::size_t i) const {
            if (i >= N) throw std::out_of_range("ProductGrid::get_grid");
            return grids_[i];
        }

        std::vector<value_type> collect_points() const {
            std::vector<value_type> result;
            result.reserve(size());
            for (const auto& addr : *this) result.push_back(addr);
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

    // Verify ProductGrid satisfies OrderedGrid
    //template<typename T>
    //using UniformGrid_ = UniformGrid<T>; // help deduction
    //static_assert(OrderedGrid<ProductGrid<UniformGrid<int>, 2>>,
    //    "ProductGrid must satisfy OrderedGrid");

} // namespace delta