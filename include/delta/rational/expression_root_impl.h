#pragma once

#include "expression_root.h"
#include "evaluation_core.h"
#include "simplify.h"
#include "rational_class.h" 

#include <absl/hash/hash.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <unordered_map>
#include <stack>

namespace delta::internal {

    // ============================================================================
    // Constructors
    // ============================================================================

    inline ExpressionRoot::ExpressionRoot(Value val) {
        root_index_ = add_node(Node(LazyOp::CONST, -1, -1, add_value(val), Interval(to_double(val)), 0));
        hash_ = compute_hash(nodes_, values_, root_index_);
    }

    inline ExpressionRoot::ExpressionRoot(LazyOp op, int c0, int c1, Interval approx, int depth) {
        root_index_ = add_node(Node(op, c0, c1, -1, approx, depth));
        hash_ = compute_hash(nodes_, values_, root_index_);
    }

    inline ExpressionRoot::ExpressionRoot(std::vector<Node> nodes, std::vector<Value> values, int root_index)
        : nodes_(std::move(nodes)), values_(std::move(values)), root_index_(root_index)
    {
        hash_ = compute_hash(nodes_, values_, root_index_);
    }

    // ============================================================================
    // Internal helpers
    // ============================================================================

    inline int ExpressionRoot::add_node(Node node) {
        int idx = static_cast<int>(nodes_.size());
        nodes_.push_back(node);
        return idx;
    }

    inline int ExpressionRoot::add_value(Value val) {
        int idx = static_cast<int>(values_.size());
        values_.push_back(std::move(val));
        return idx;
    }

    // ============================================================================
    // Hash computation
    // ============================================================================

    inline std::size_t ExpressionRoot::compute_hash(const std::vector<Node>& nodes,
        const std::vector<Value>& values,
        int root) {
        absl::Hash<std::size_t> hasher;
        std::size_t h = 0;
        std::function<void(int)> hash_node = [&](int idx) {
            const Node& n = nodes[idx];
            h = hasher(h ^ static_cast<std::size_t>(n.op));
            if (n.op == LazyOp::CONST) {
                const Value& v = values[n.value_idx];
                if (const auto* s = std::get_if<SmallStorage>(&v)) {
                    SmallStorage norm = *s;
                    norm.normalize();
                    h = hasher(h ^ absl::Hash<absl::int128>{}(norm.num));
                    h = hasher(h ^ absl::Hash<absl::uint128>{}(norm.den));
                }
                else {
                    const auto& b = std::get<BigStorage>(v);
                    std::string repr = b.num().str() + "/" + b.den().str();
                    h = hasher(h ^ std::hash<std::string>{}(repr));
                }
            }
            else {
                if (n.child0 != -1) hash_node(n.child0);
                if (n.child1 != -1) hash_node(n.child1);
            }
            };
        hash_node(root);
        return h;
    }

    // ============================================================================
    // Helper to copy a subtree from a source ExpressionRoot into new vectors
    // ============================================================================
    static int copy_subtree(const ExpressionRoot& src,
        int src_idx,
        std::vector<Node>& new_nodes,
        std::vector<Value>& new_values,
        std::unordered_map<int, int>& idx_map) {
        // Итеративный DFS с явным стеком
        struct Frame {
            int src_idx;
            int state; // 0 = children not processed, 1 = ready to create node
        };
        std::stack<Frame> st;
        st.push({ src_idx, 0 });

        while (!st.empty()) {
            Frame& f = st.top();
            auto it = idx_map.find(f.src_idx);
            if (it != idx_map.end()) {
                // Уже скопирован
                st.pop();
                continue;
            }

            const Node& n = src.nodes()[f.src_idx];
            if (f.state == 0) {
                // Проверяем детей, которые ещё не скопированы
                bool need_children = false;
                if (n.child0 != -1 && idx_map.find(n.child0) == idx_map.end()) {
                    st.push({ n.child0, 0 });
                    need_children = true;
                }
                if (n.child1 != -1 && idx_map.find(n.child1) == idx_map.end()) {
                    st.push({ n.child1, 0 });
                    need_children = true;
                }
                if (need_children) {
                    f.state = 1; // после возврата из детей будем создавать узел
                    continue;
                }
                else {
                    // Нет детей или все уже скопированы – создаём узел сразу
                    f.state = 1;
                }
            }

            if (f.state == 1) {
                // Все дети скопированы, создаём новый узел
                int new_child0 = (n.child0 != -1) ? idx_map[n.child0] : -1;
                int new_child1 = (n.child1 != -1) ? idx_map[n.child1] : -1;

                int new_val_idx = -1;
                if (n.op == LazyOp::CONST) {
                    new_val_idx = static_cast<int>(new_values.size());
                    new_values.push_back(src.values()[n.value_idx]);
                }
                else if (n.op == LazyOp::SQRT || n.op == LazyOp::EXP || n.op == LazyOp::LOG ||
                    n.op == LazyOp::SIN || n.op == LazyOp::COS || n.op == LazyOp::ACOS ||
                    n.op == LazyOp::PI || n.op == LazyOp::E) {
                    new_val_idx = static_cast<int>(new_values.size());
                    new_values.push_back(src.values()[n.value_idx]);
                }

                new_nodes.emplace_back(n.op, new_child0, new_child1, new_val_idx, n.approx, n.depth);
                int new_idx = static_cast<int>(new_nodes.size()) - 1;
                idx_map[f.src_idx] = new_idx;
                st.pop();
            }
        }
        return idx_map[src_idx];
    }
    // ============================================================================
    // Public building methods (immutable)
    // ============================================================================

    inline ExpressionRoot ExpressionRoot::add(const ExpressionRoot& other) const {
        // Check depth and simplify if needed
        if (1 + std::max(nodes_[root_index_].depth, other.nodes_[other.root_index_].depth) > MAX_LAZY_DEPTH) {
            return simplify().add(other.simplify());
        }

        std::vector<Node> new_nodes;
        std::vector<Value> new_values;
        std::unordered_map<int, int> idx_map_this, idx_map_other;

        int new_root0 = copy_subtree(*this, root_index_, new_nodes, new_values, idx_map_this);
        int new_root1 = copy_subtree(other, other.root_index_, new_nodes, new_values, idx_map_other);

        int new_depth = 1 + std::max(new_nodes[new_root0].depth, new_nodes[new_root1].depth);
        Interval approx = compute_interval(LazyOp::ADD, new_nodes[new_root0].approx, new_nodes[new_root1].approx);
        new_nodes.emplace_back(LazyOp::ADD, new_root0, new_root1, -1, approx, new_depth);
        int new_root = static_cast<int>(new_nodes.size()) - 1;
        return ExpressionRoot(std::move(new_nodes), std::move(new_values), new_root);
    }

    inline ExpressionRoot ExpressionRoot::sub(const ExpressionRoot& other) const {
        if (1 + std::max(nodes_[root_index_].depth, other.nodes_[other.root_index_].depth) > MAX_LAZY_DEPTH) {
            return simplify().sub(other.simplify());
        }
        std::vector<Node> new_nodes;
        std::vector<Value> new_values;
        std::unordered_map<int, int> idx_map_this, idx_map_other;

        int new_root0 = copy_subtree(*this, root_index_, new_nodes, new_values, idx_map_this);
        int new_root1 = copy_subtree(other, other.root_index_, new_nodes, new_values, idx_map_other);

        int new_depth = 1 + std::max(new_nodes[new_root0].depth, new_nodes[new_root1].depth);
        Interval approx = compute_interval(LazyOp::SUB, new_nodes[new_root0].approx, new_nodes[new_root1].approx);
        new_nodes.emplace_back(LazyOp::SUB, new_root0, new_root1, -1, approx, new_depth);
        int new_root = static_cast<int>(new_nodes.size()) - 1;
        return ExpressionRoot(std::move(new_nodes), std::move(new_values), new_root);
    }

    inline ExpressionRoot ExpressionRoot::mul(const ExpressionRoot& other) const {
        if (1 + std::max(nodes_[root_index_].depth, other.nodes_[other.root_index_].depth) > MAX_LAZY_DEPTH) {
            return simplify().mul(other.simplify());
        }
        std::vector<Node> new_nodes;
        std::vector<Value> new_values;
        std::unordered_map<int, int> idx_map_this, idx_map_other;

        int new_root0 = copy_subtree(*this, root_index_, new_nodes, new_values, idx_map_this);
        int new_root1 = copy_subtree(other, other.root_index_, new_nodes, new_values, idx_map_other);

        int new_depth = 1 + std::max(new_nodes[new_root0].depth, new_nodes[new_root1].depth);
        Interval approx = compute_interval(LazyOp::MUL, new_nodes[new_root0].approx, new_nodes[new_root1].approx);
        new_nodes.emplace_back(LazyOp::MUL, new_root0, new_root1, -1, approx, new_depth);
        int new_root = static_cast<int>(new_nodes.size()) - 1;
        return ExpressionRoot(std::move(new_nodes), std::move(new_values), new_root);
    }

    inline ExpressionRoot ExpressionRoot::div(const ExpressionRoot& other) const {
        if (1 + std::max(nodes_[root_index_].depth, other.nodes_[other.root_index_].depth) > MAX_LAZY_DEPTH) {
            return simplify().div(other.simplify());
        }
        std::vector<Node> new_nodes;
        std::vector<Value> new_values;
        std::unordered_map<int, int> idx_map_this, idx_map_other;

        int new_root0 = copy_subtree(*this, root_index_, new_nodes, new_values, idx_map_this);
        int new_root1 = copy_subtree(other, other.root_index_, new_nodes, new_values, idx_map_other);

        int new_depth = 1 + std::max(new_nodes[new_root0].depth, new_nodes[new_root1].depth);
        Interval approx = compute_interval(LazyOp::DIV, new_nodes[new_root0].approx, new_nodes[new_root1].approx);
        new_nodes.emplace_back(LazyOp::DIV, new_root0, new_root1, -1, approx, new_depth);
        int new_root = static_cast<int>(new_nodes.size()) - 1;
        return ExpressionRoot(std::move(new_nodes), std::move(new_values), new_root);
    }

    inline ExpressionRoot ExpressionRoot::neg() const {
        if (1 + nodes_[root_index_].depth > MAX_LAZY_DEPTH) {
            return simplify().neg();
        }
        std::vector<Node> new_nodes;
        std::vector<Value> new_values;
        std::unordered_map<int, int> idx_map;

        int new_child = copy_subtree(*this, root_index_, new_nodes, new_values, idx_map);

        int new_depth = 1 + new_nodes[new_child].depth;
        Interval approx = compute_interval(LazyOp::NEG, new_nodes[new_child].approx);
        new_nodes.emplace_back(LazyOp::NEG, new_child, -1, -1, approx, new_depth);
        int new_root = static_cast<int>(new_nodes.size()) - 1;
        return ExpressionRoot(std::move(new_nodes), std::move(new_values), new_root);
    }

    // Transcendental functions (unary with eps)
    inline ExpressionRoot ExpressionRoot::sqrt(const Rational& eps) const {
        if (1 + nodes_[root_index_].depth > MAX_LAZY_DEPTH) {
            return simplify().sqrt(eps);
        }
        std::vector<Node> new_nodes;
        std::vector<Value> new_values;
        std::unordered_map<int, int> idx_map;

        int new_child = copy_subtree(*this, root_index_, new_nodes, new_values, idx_map);

        int new_depth = 1 + new_nodes[new_child].depth;
        Interval approx = compute_interval(LazyOp::SQRT, new_nodes[new_child].approx);
        Value eps_val = eps.to_value();
        int eps_idx = static_cast<int>(new_values.size());
        new_values.push_back(eps_val);
        new_nodes.emplace_back(LazyOp::SQRT, new_child, -1, eps_idx, approx, new_depth);
        int new_root = static_cast<int>(new_nodes.size()) - 1;
        return ExpressionRoot(std::move(new_nodes), std::move(new_values), new_root);
    }

    inline ExpressionRoot ExpressionRoot::exp(const Rational& eps) const {
        if (1 + nodes_[root_index_].depth > MAX_LAZY_DEPTH) {
            return simplify().exp(eps);
        }
        std::vector<Node> new_nodes;
        std::vector<Value> new_values;
        std::unordered_map<int, int> idx_map;

        int new_child = copy_subtree(*this, root_index_, new_nodes, new_values, idx_map);

        int new_depth = 1 + new_nodes[new_child].depth;
        Interval approx = compute_interval(LazyOp::EXP, new_nodes[new_child].approx);
        Value eps_val = eps.to_value();
        int eps_idx = static_cast<int>(new_values.size());
        new_values.push_back(eps_val);
        new_nodes.emplace_back(LazyOp::EXP, new_child, -1, eps_idx, approx, new_depth);
        int new_root = static_cast<int>(new_nodes.size()) - 1;
        return ExpressionRoot(std::move(new_nodes), std::move(new_values), new_root);
    }

    inline ExpressionRoot ExpressionRoot::log(const Rational& eps) const {
        if (1 + nodes_[root_index_].depth > MAX_LAZY_DEPTH) {
            return simplify().log(eps);
        }
        std::vector<Node> new_nodes;
        std::vector<Value> new_values;
        std::unordered_map<int, int> idx_map;

        int new_child = copy_subtree(*this, root_index_, new_nodes, new_values, idx_map);

        int new_depth = 1 + new_nodes[new_child].depth;
        Interval approx = compute_interval(LazyOp::LOG, new_nodes[new_child].approx);
        Value eps_val = eps.to_value();
        int eps_idx = static_cast<int>(new_values.size());
        new_values.push_back(eps_val);
        new_nodes.emplace_back(LazyOp::LOG, new_child, -1, eps_idx, approx, new_depth);
        int new_root = static_cast<int>(new_nodes.size()) - 1;
        return ExpressionRoot(std::move(new_nodes), std::move(new_values), new_root);
    }

    inline ExpressionRoot ExpressionRoot::sin(const Rational& eps) const {
        if (1 + nodes_[root_index_].depth > MAX_LAZY_DEPTH) {
            return simplify().sin(eps);
        }
        std::vector<Node> new_nodes;
        std::vector<Value> new_values;
        std::unordered_map<int, int> idx_map;

        int new_child = copy_subtree(*this, root_index_, new_nodes, new_values, idx_map);

        int new_depth = 1 + new_nodes[new_child].depth;
        Interval approx = compute_interval(LazyOp::SIN, new_nodes[new_child].approx);
        Value eps_val = eps.to_value();
        int eps_idx = static_cast<int>(new_values.size());
        new_values.push_back(eps_val);
        new_nodes.emplace_back(LazyOp::SIN, new_child, -1, eps_idx, approx, new_depth);
        int new_root = static_cast<int>(new_nodes.size()) - 1;
        return ExpressionRoot(std::move(new_nodes), std::move(new_values), new_root);
    }

    inline ExpressionRoot ExpressionRoot::cos(const Rational& eps) const {
        if (1 + nodes_[root_index_].depth > MAX_LAZY_DEPTH) {
            return simplify().cos(eps);
        }
        std::vector<Node> new_nodes;
        std::vector<Value> new_values;
        std::unordered_map<int, int> idx_map;

        int new_child = copy_subtree(*this, root_index_, new_nodes, new_values, idx_map);

        int new_depth = 1 + new_nodes[new_child].depth;
        Interval approx = compute_interval(LazyOp::COS, new_nodes[new_child].approx);
        Value eps_val = eps.to_value();
        int eps_idx = static_cast<int>(new_values.size());
        new_values.push_back(eps_val);
        new_nodes.emplace_back(LazyOp::COS, new_child, -1, eps_idx, approx, new_depth);
        int new_root = static_cast<int>(new_nodes.size()) - 1;
        return ExpressionRoot(std::move(new_nodes), std::move(new_values), new_root);
    }

    inline ExpressionRoot ExpressionRoot::acos(const Rational& eps) const {
        if (1 + nodes_[root_index_].depth > MAX_LAZY_DEPTH) {
            return simplify().acos(eps);
        }
        std::vector<Node> new_nodes;
        std::vector<Value> new_values;
        std::unordered_map<int, int> idx_map;

        int new_child = copy_subtree(*this, root_index_, new_nodes, new_values, idx_map);

        int new_depth = 1 + new_nodes[new_child].depth;
        Interval approx = compute_interval(LazyOp::ACOS, new_nodes[new_child].approx);
        Value eps_val = eps.to_value();
        int eps_idx = static_cast<int>(new_values.size());
        new_values.push_back(eps_val);
        new_nodes.emplace_back(LazyOp::ACOS, new_child, -1, eps_idx, approx, new_depth);
        int new_root = static_cast<int>(new_nodes.size()) - 1;
        return ExpressionRoot(std::move(new_nodes), std::move(new_values), new_root);
    }

    // Static constants
    inline ExpressionRoot ExpressionRoot::pi(const Rational& eps) {
        Value eps_val = eps.to_value();
        std::vector<Node> nodes;
        std::vector<Value> values;
        int eps_idx = static_cast<int>(values.size());
        values.push_back(eps_val);
        Node node(LazyOp::PI, -1, -1, eps_idx, Interval(M_PI), 0);
        nodes.push_back(node);
        return ExpressionRoot(std::move(nodes), std::move(values), 0);
    }

    inline ExpressionRoot ExpressionRoot::e(const Rational& eps) {
        Value eps_val = eps.to_value();
        std::vector<Node> nodes;
        std::vector<Value> values;
        int eps_idx = static_cast<int>(values.size());
        values.push_back(eps_val);
        Node node(LazyOp::E, -1, -1, eps_idx, Interval(M_E), 0);
        nodes.push_back(node);
        return ExpressionRoot(std::move(nodes), std::move(values), 0);
    }

    // ============================================================================
    // Public methods that use evaluation_core and simplify
    // ============================================================================

    inline ExpressionRoot ExpressionRoot::simplify() const {
        return delta::internal::simplify(*this);
    }

    inline Rational ExpressionRoot::eval() const {
        return Rational(evaluate(*this));
    }

} // namespace delta::internal