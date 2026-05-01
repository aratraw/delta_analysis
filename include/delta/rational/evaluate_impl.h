// (c) 2026 Timofey Ishmisev.
// Licensed under PolyForm Small Business License 1.0.0

// evaluate_impl.h
// -----------------------------------------------------------------------------
// Evaluation of lazy expression trees (both dirty and clean nodes).
//
// This file provides the core tree evaluation machinery used by LazyRational.
// It traverses a directed acyclic graph (DAG) of nodes (SUM, PRODUCT, unary
// ops, constants, etc.) and computes the resulting Value.
//
// Key features:
//   - Post‑order traversal with caching (each node evaluated once)
//   - Two summation strategies:
//       * Standard: copies leaf values, uses pyramidal compact reduction (PCR)
//       * Inplace: moves leaf values, modifies the input vector for speed
//   - Flat product evaluation via sequential multiplication
//   - Dispatch to eager transcendentals (sqrt, sin, cos, etc.) from evaluation_core.h
//
// The summation uses PCR (pyramidal_compact_reduce) which groups terms into
// batches of BATCH_SIZE (default 32) and reduces hierarchically. This minimizes
// intermediate expression swell and improves performance for large sums.
// -----------------------------------------------------------------------------

#pragma once

#include "node_types.h"
#include "lazy_nodes.h"
#include "evaluation_core.h"
#include "reduce.h"
#include "utils.h"

#include <stack>
#include <vector>
#include <optional>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

namespace delta::internal {

    // ------------------------------------------------------------------------
    // Summation strategies (handling of SUM nodes)
    // ------------------------------------------------------------------------
    // Standard strategy: copies leaf values, uses PCR without modifying input.
    struct SumStrategy_Standard {
        static constexpr bool allows_inplace = false;
        Value operator()(const std::vector<Value>& values) const {
            return pyramidal_compact_reduce_copy(values);
        }
    };

    // Inplace strategy: moves leaf values, modifies the input vector.
    // This reduces memory allocations but destroys the original leaf_values.
    struct SumStrategy_Inplace {
        static constexpr bool allows_inplace = true;
        Value operator()(std::vector<Value>& values) const {
            pyramidal_compact_reduce_inplace(values);
            return std::move(values[0]);
        }
    };

    // ------------------------------------------------------------------------
    // Multiplication strategy (sequential, no batching)
    // ------------------------------------------------------------------------
    // Multiplies a mix of leaf constants and child node results.
    // Leaf values are moved, child values are copied.
    struct ProdStrategy_Sequential {
        Value operator()(std::vector<Value> leaf_values, const std::vector<Value>& child_values) const {
            if (leaf_values.empty() && child_values.empty()) {
                return Value(1);
            }
            Value result = !leaf_values.empty() ? leaf_values[0] : child_values[0];
            size_t start_leaf = !leaf_values.empty() ? 1 : 0;
            size_t start_child = !leaf_values.empty() ? 0 : 1;

            for (size_t i = start_leaf; i < leaf_values.size(); ++i) {
                result *= leaf_values[i];
            }
            for (size_t i = start_child; i < child_values.size(); ++i) {
                result *= child_values[i];
            }
            return result;
        }
    };

    // ------------------------------------------------------------------------
    // Unified template function for evaluating an expression tree
    // ------------------------------------------------------------------------
    // NodeType can be DirtyNode or Node (from node_pool.h).
    // ValueAccessor provides const_value(node) and eps_value(node).
    // SumStrategy and ProdStrategy are policy classes for reduction.
    // ------------------------------------------------------------------------
    template<typename NodeType, typename ValueAccessor, typename SumStrategy, typename ProdStrategy>
    Value evaluate_tree(int root,
        const std::vector<NodeType>& nodes,
        ValueAccessor&& value_accessor,
        SumStrategy sum_strategy,
        ProdStrategy prod_strategy)
    {
        const size_t n = nodes.size();
        std::vector<std::optional<Value>> cache(n);
        std::stack<int> st;
        st.push(root);

        while (!st.empty()) {
            int idx = st.top();
            if (cache[idx].has_value()) {
                st.pop();
                continue;
            }

            const NodeType& node = nodes[idx];
            bool children_ready = true;

            // Check if all children have been evaluated
            for (int child : node.children) {
                if (!cache[child].has_value()) {
                    st.push(child);
                    children_ready = false;
                }
            }

            if (!children_ready) continue;

            Value result;
            switch (node.op) {
            case LazyOp::CONST: {
                result = value_accessor.const_value(node);
                break;
            }

            case LazyOp::SUM: {
                std::vector<Value> to_reduce;
                if constexpr (SumStrategy::allows_inplace) {
                    to_reduce = std::move(const_cast<NodeType&>(node).leaf_values);
                }
                else {
                    to_reduce = node.leaf_values;
                }
                to_reduce.reserve(to_reduce.size() + node.children.size());
                for (int child : node.children) {
                    to_reduce.push_back(cache[child].value());
                }
                result = sum_strategy(to_reduce);
                break;
            }

            case LazyOp::PRODUCT: {
                std::vector<Value> leaf_vals;
                if constexpr (SumStrategy::allows_inplace) {
                    leaf_vals = std::move(const_cast<NodeType&>(node).leaf_values);
                }
                else {
                    leaf_vals = node.leaf_values;
                }
                std::vector<Value> child_vals;
                child_vals.reserve(node.children.size());
                for (int child : node.children) {
                    child_vals.push_back(cache[child].value());
                }
                result = prod_strategy(std::move(leaf_vals), child_vals);
                break;
            }

            case LazyOp::NEG:
                result = -cache[node.children[0]].value();
                break;
            case LazyOp::RECIP:
                result = Value(1) / cache[node.children[0]].value();
                break;
            case LazyOp::SQRT: {
                Value eps = value_accessor.eps_value(node);
                result = eager_sqrt(cache[node.children[0]].value(), eps);
                break;
            }
            case LazyOp::EXP: {
                Value eps = value_accessor.eps_value(node);
                result = eager_exp(cache[node.children[0]].value(), eps);
                break;
            }
            case LazyOp::LOG: {
                Value eps = value_accessor.eps_value(node);
                result = eager_log(cache[node.children[0]].value(), eps);
                break;
            }
            case LazyOp::SIN: {
                Value eps = value_accessor.eps_value(node);
                result = eager_sin(cache[node.children[0]].value(), eps);
                break;
            }
            case LazyOp::COS: {
                Value eps = value_accessor.eps_value(node);
                result = eager_cos(cache[node.children[0]].value(), eps);
                break;
            }
            case LazyOp::ACOS: {
                Value eps = value_accessor.eps_value(node);
                result = eager_acos(cache[node.children[0]].value(), eps);
                break;
            }
            case LazyOp::PI: {
                Value eps = value_accessor.eps_value(node);
                result = eager_pi(eps);
                break;
            }
            case LazyOp::E: {
                Value eps = value_accessor.eps_value(node);
                result = eager_e(eps);
                break;
            }
            case LazyOp::POW: {
                Value eps = value_accessor.eps_value(node);
                result = eager_pow(cache[node.children[0]].value(),
                    cache[node.children[1]].value(),
                    eps);
                break;
            }
            default:
                throw std::logic_error("evaluate_tree: unknown LazyOp");
            }

            cache[idx] = std::move(result);
            st.pop();
        }

        return std::move(cache[root].value());
    }

    // ------------------------------------------------------------------------
    // Public APIs for dirty tree (DirtyNode)
    // ------------------------------------------------------------------------

    // evaluate_dirty – evaluates the tree without destroying it
    inline Value evaluate_dirty(const std::vector<DirtyNode>& nodes,
        const std::vector<Value>& constants,
        int root) {
        struct Accessor {
            const std::vector<Value>& constants;
            Value const_value(const DirtyNode& node) const {
                return constants[node.value_idx];
            }
            Value eps_value(const DirtyNode& node) const {
                return (node.eps_idx != -1) ? constants[node.eps_idx] : Value{};
            }
        };
        SumStrategy_Standard sum_strategy;
        ProdStrategy_Sequential prod_strategy;
        return evaluate_tree<DirtyNode>(root, nodes, Accessor{ constants }, sum_strategy, prod_strategy);
    }

    // evaluate_dirty_inplace – evaluates the tree, moving leaf_values (optimization)
    inline Value evaluate_dirty_inplace(std::vector<DirtyNode>& nodes,
        std::vector<Value>& constants,
        int root) {
        struct Accessor {
            std::vector<Value>& constants;
            Value const_value(const DirtyNode& node) const {
                return constants[node.value_idx];
            }
            Value eps_value(const DirtyNode& node) const {
                return (node.eps_idx != -1) ? constants[node.eps_idx] : Value{};
            }
        };
        SumStrategy_Inplace sum_strategy;
        ProdStrategy_Sequential prod_strategy;
        return evaluate_tree<DirtyNode>(root, nodes, Accessor{ constants }, sum_strategy, prod_strategy);
    }

} // namespace delta::internal