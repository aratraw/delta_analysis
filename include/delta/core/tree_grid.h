// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/core/tree_grid.h
#pragma once

#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <stdexcept>
#include "grid_concept.h"
#include "regulative_idea.h"

namespace delta {

    /**
     * @class TreeGrid
     * @brief Grid for binary tree addresses (strings of '0' and '1').
     *
     * Stores all nodes of a complete binary tree up to a given level.
     * Nodes are stored in a sorted vector (lexicographic order) to provide
     * random access and iterators required by GridConcept.
     *
     * @note Generating a grid for level L creates 2^(L+1)-1 nodes, which grows
     *       exponentially. For large L, this class may become memory‑intensive.
     *       It is intended for testing and demonstration purposes.
     *
     * @tparam Compare Comparison functor for ordering (must be consistent with
     *                 lexicographic order of binary strings). Default std::less<std::string>.
     */
    template<typename Compare = std::less<std::string>>
    class TreeGrid {
    public:
        using value_type = std::string;
        using size_type = std::size_t;
        using const_iterator = typename std::vector<value_type>::const_iterator;
        using comparator_type = Compare;

        // -------------------------------------------------------------------------
        // Construction
        // -------------------------------------------------------------------------

        /**
         * @brief Construct a grid containing all binary strings of length ≤ level.
         * @param level The maximum depth (0 = only the root node "").
         * @param comp  Comparison functor (must be consistent with lexicographic order).
         */
        explicit TreeGrid(std::size_t level = 0, Compare comp = Compare())
            : level_(level), comp_(std::move(comp)) {
            generate_nodes();
        }

        // -------------------------------------------------------------------------
        // Accessors required by GridConcept
        // -------------------------------------------------------------------------

        /// Returns the number of nodes in the grid (2^(level+1) - 1).
        size_type size() const noexcept { return nodes_.size(); }

        /// Returns the node at the given index (lexicographic order).
        const value_type& operator[](size_type index) const {
            if (index >= nodes_.size()) throw std::out_of_range("TreeGrid index out of range");
            return nodes_[index];
        }

        /// Returns an iterator to the first node (lexicographically smallest).
        const_iterator begin() const noexcept { return nodes_.begin(); }

        /// Returns an iterator past the last node.
        const_iterator end() const noexcept { return nodes_.end(); }

        /// Returns the comparator used for ordering.
        const Compare& comparator() const noexcept { return comp_; }

        /// Returns a flat vector of all nodes (all addresses in the tree).
        std::vector<value_type> collect_points() const {
            return nodes_;   // nodes_ is std::vector<std::string>
        }

        // -------------------------------------------------------------------------
        // Tree-specific methods
        // -------------------------------------------------------------------------

        /// Returns the current level (max depth) of the grid.
        std::size_t level() const noexcept { return level_; }

        /// Returns the parent of a node, or empty string for the root.
        static value_type parent(const value_type& node) {
            if (node.empty()) return "";
            return node.substr(0, node.size() - 1);
        }

        /// Returns the left child (node + "0").
        static value_type left_child(const value_type& node) {
            return node + "0";
        }

        /// Returns the right child (node + "1").
        static value_type right_child(const value_type& node) {
            return node + "1";
        }

        /// Returns true if the node is a leaf at current level (i.e., has no children in grid).
        bool is_leaf(const value_type& node) const {
            return node.size() == level_;
        }

        // -------------------------------------------------------------------------
        // Mutators (for path advancement)
        // -------------------------------------------------------------------------

        /**
         * @brief Increase the grid level by one, generating all new nodes.
         * @note This operation rebuilds the entire grid from scratch; complexity O(2^level).
         */
        void advance() {
            ++level_;
            generate_nodes();
        }

        /**
         * @brief Set the grid to a specific level.
         * @param level New level (≥ 0).
         * @note Rebuilds the grid if the level changes; complexity O(2^level).
         */
        void set_level(std::size_t level) {
            if (level != level_) {
                level_ = level;
                generate_nodes();
            }
        }

    private:
        /**
         * @brief Generate all nodes up to the current level and store them in lexicographic order.
         *
         * Uses a depth‑first traversal (stack) to produce all binary strings of length ≤ level_,
         * then sorts the resulting vector according to comp_. This approach is simple but
         * creates an exponential number of strings; use with caution for large levels.
         */
        void generate_nodes() {
            nodes_.clear();
            if (level_ == 0) {
                nodes_.push_back("");
                return;
            }
            // Generate all strings of length 0..level_ using DFS (stack)
            std::vector<value_type> stack;
            stack.push_back("");
            while (!stack.empty()) {
                value_type current = stack.back();
                stack.pop_back();
                nodes_.push_back(current);
                if (current.size() < level_) {
                    stack.push_back(current + "0");
                    stack.push_back(current + "1");
                }
            }
            // DFS order is not lexicographic; sort according to comparator.
            std::sort(nodes_.begin(), nodes_.end(), comp_);
        }

        std::size_t level_;
        Compare comp_;
        std::vector<value_type> nodes_;
    };

    // Verify that TreeGrid satisfies the GridConcept with std::string addresses.
    static_assert(OrderedGrid<TreeGrid<>>);
} // namespace delta