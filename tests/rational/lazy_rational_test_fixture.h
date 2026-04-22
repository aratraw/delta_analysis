#pragma once

#include <gtest/gtest.h>
#include "test_utils.h"
#include "delta/core/rational.h"
#include "delta/rational/node_pool.h"
#include "delta/rational/storage.h"
#include "absl/container/inlined_vector.h"

namespace delta::testing {

    class LazyRationalTestFixture : public delta::testing::RationalTest {
    protected:

        size_t total_operands(const LazyRational& lr) const {
            assert(is_dirty(lr));
            int root = dirty_root_index(lr);
            return dirty_node_leaf_count(lr, root) + dirty_node_complex_count(lr, root);
        }
        // ------------------------------------------------------------------------
        // Проверка состояния
        // ------------------------------------------------------------------------
        bool is_dirty(const LazyRational& lr) const {
            return lr.state_ == LazyRational::State::Dirty;
        }

        bool is_clean(const LazyRational& lr) const {
            return lr.state_ == LazyRational::State::Clean;
        }

        // ------------------------------------------------------------------------
        // Доступ к грязному дереву
        // ------------------------------------------------------------------------
        size_t dirty_node_count(const LazyRational& lr) const {
            assert(is_dirty(lr));
            return lr.nodes_.size();
        }

        size_t dirty_constant_count(const LazyRational& lr) const {
            assert(is_dirty(lr));
            return lr.constants_.size();
        }

        int dirty_root_index(const LazyRational& lr) const {
            assert(is_dirty(lr));
            return lr.root_;
        }

        const internal::DirtyNode& dirty_node(const LazyRational& lr, int idx) const {
            assert(is_dirty(lr));
            assert(idx >= 0 && static_cast<size_t>(idx) < lr.nodes_.size());
            return lr.nodes_[idx];
        }

        internal::LazyOp dirty_root_op(const LazyRational& lr) const {
            assert(is_dirty(lr));
            return lr.nodes_[lr.root_].op;
        }

        absl::InlinedVector<int32_t, 4> dirty_root_children(const LazyRational& lr) const {
            assert(is_dirty(lr));
            return lr.nodes_[lr.root_].children;
        }

        int dirty_root_value_idx(const LazyRational& lr) const {
            assert(is_dirty(lr));
            assert(lr.nodes_[lr.root_].op == internal::LazyOp::CONST);
            return lr.nodes_[lr.root_].value_idx;
        }

        int dirty_root_eps_idx(const LazyRational& lr) const {
            assert(is_dirty(lr));
            return lr.nodes_[lr.root_].eps_idx;
        }

        internal::Value dirty_constant(const LazyRational& lr, int idx) const {
            assert(is_dirty(lr));
            return lr.constants_[idx];
        }

        internal::LazyOp dirty_node_op(const LazyRational& lr, int node_idx) const {
            assert(is_dirty(lr));
            return lr.nodes_[node_idx].op;
        }

        absl::InlinedVector<int32_t, 4> dirty_node_children(const LazyRational& lr, int node_idx) const {
            assert(is_dirty(lr));
            return lr.nodes_[node_idx].children;
        }

        int dirty_node_value_idx(const LazyRational& lr, int node_idx) const {
            assert(is_dirty(lr));
            assert(lr.nodes_[node_idx].op == internal::LazyOp::CONST);
            return lr.nodes_[node_idx].value_idx;
        }

        int dirty_node_eps_idx(const LazyRational& lr, int node_idx) const {
            assert(is_dirty(lr));
            return lr.nodes_[node_idx].eps_idx;
        }

        // Доступ к гетерогенным полям SUM/PRODUCT
        size_t dirty_node_leaf_count(const LazyRational& lr, int node_idx) const {
            assert(is_dirty(lr));
            return lr.nodes_[node_idx].leaf_values.size();
        }

        size_t dirty_node_complex_count(const LazyRational& lr, int node_idx) const {
            assert(is_dirty(lr));
            return lr.nodes_[node_idx].complex_children.size();
        }

        const internal::Value& dirty_node_leaf_value(const LazyRational& lr, int node_idx, size_t i) const {
            assert(is_dirty(lr));
            return lr.nodes_[node_idx].leaf_values[i];
        }

        int dirty_node_complex_child(const LazyRational& lr, int node_idx, size_t i) const {
            assert(is_dirty(lr));
            return lr.nodes_[node_idx].complex_children[i];
        }

        // ------------------------------------------------------------------------
        // Доступ к чистому дереву
        // ------------------------------------------------------------------------
        int clean_root_index(const LazyRational& lr) const {
            assert(is_clean(lr));
            return lr.clean_index_;
        }

        size_t clean_node_refcount(const LazyRational& lr, int node_idx) const {
            (void)lr;
            if (node_idx < 0 || static_cast<size_t>(node_idx) >= internal::pool.nodes.size())
                return 0;
            return internal::pool.refcount[node_idx];
        }

        // ------------------------------------------------------------------------
        // Проверка каноничности SUM
        // ------------------------------------------------------------------------
        bool is_canonical_sum(const LazyRational& lr) const {
            if (!is_clean(lr)) return false;
            const auto& node = internal::pool.nodes[lr.clean_index_];
            if (node.op != internal::LazyOp::SUM) return false;

            for (const auto& v : node.leaf_values) {
                if (internal::is_zero(v)) return false;
            }

            for (int32_t child : node.complex_children) {
                const auto& child_node = internal::pool.nodes[child];
                if (child_node.op == internal::LazyOp::CONST &&
                    internal::is_zero(internal::pool.values[child_node.value_idx]))
                    return false;
            }

            const auto& cc = node.complex_children;
            for (size_t i = 1; i < cc.size(); ++i) {
                uint64_t hash_prev = internal::pool.nodes[cc[i - 1]].hash;
                uint64_t hash_cur = internal::pool.nodes[cc[i]].hash;
                if (hash_prev > hash_cur) return false;
                if (hash_prev == hash_cur && cc[i - 1] >= cc[i]) return false;
            }
            return true;
        }

        // ------------------------------------------------------------------------
        // Проверка каноничности PRODUCT
        // ------------------------------------------------------------------------
        bool is_canonical_product(const LazyRational& lr) const {
            if (!is_clean(lr)) return false;
            const auto& node = internal::pool.nodes[lr.clean_index_];
            if (node.op != internal::LazyOp::PRODUCT) return false;

            for (const auto& v : node.leaf_values) {
                if (internal::is_one(v)) return false;
            }

            for (int32_t child : node.complex_children) {
                const auto& child_node = internal::pool.nodes[child];
                if (child_node.op == internal::LazyOp::CONST &&
                    internal::is_one(internal::pool.values[child_node.value_idx]))
                    return false;
            }

            const auto& cc = node.complex_children;
            for (size_t i = 1; i < cc.size(); ++i) {
                uint64_t hash_prev = internal::pool.nodes[cc[i - 1]].hash;
                uint64_t hash_cur = internal::pool.nodes[cc[i]].hash;
                if (hash_prev > hash_cur) return false;
                if (hash_prev == hash_cur && cc[i - 1] >= cc[i]) return false;
            }
            return true;
        }

        // ------------------------------------------------------------------------
        // Вспомогательные обходы грязного дерева
        // ------------------------------------------------------------------------
        bool has_node_with_op(const LazyRational& lr, internal::LazyOp op) const {
            if (is_dirty(lr)) {
                for (const auto& node : lr.nodes_) {
                    if (node.op == op) return true;
                }
                return false;
            }
            else {
                // Рекурсивный обход чистого дерева из пула
                std::stack<int> st;
                st.push(lr.clean_index_);
                while (!st.empty()) {
                    int idx = st.top(); st.pop();
                    const auto& node = internal::pool.nodes[idx];
                    if (node.op == op) return true;
                    if (node.op == internal::LazyOp::SUM || node.op == internal::LazyOp::PRODUCT) {
                        for (int child : node.complex_children) st.push(child);
                    }
                    else {
                        for (int child : node.children) st.push(child);
                    }
                }
                return false;
            }
        }

        // ------------------------------------------------------------------------
        // Сброс глобального пула
        // ------------------------------------------------------------------------
        void reset_global_pool() {
            internal::reset_pool();
        }

        // ------------------------------------------------------------------------
        // Доступ к чистому индексу
        // ------------------------------------------------------------------------
        int clean_index(const LazyRational& lr) const {
            assert(is_clean(lr));
            return lr.clean_index_;
        }

        // ------------------------------------------------------------------------
        // Доступ к узлам чистого дерева
        // ------------------------------------------------------------------------
        const internal::Node& clean_node(const LazyRational& lr, int node_idx) const {
            assert(is_clean(lr));
            return internal::pool.nodes[node_idx];
        }

        const absl::InlinedVector<int32_t, 4>& clean_node_children(const LazyRational& lr, int node_idx) const {
            assert(is_clean(lr));
            const auto& node = internal::pool.nodes[node_idx];
            if (node.op != internal::LazyOp::SUM && node.op != internal::LazyOp::PRODUCT) {
                return node.children;
            }
            throw std::logic_error("Node is SUM/PRODUCT, use complex_children or leaf_values");
        }

        const absl::InlinedVector<int32_t, 4>& clean_node_complex_children(const LazyRational& lr, int node_idx) const {
            assert(is_clean(lr));
            const auto& node = internal::pool.nodes[node_idx];
            if (node.op == internal::LazyOp::SUM || node.op == internal::LazyOp::PRODUCT) {
                return node.complex_children;
            }
            throw std::logic_error("Node is not SUM/PRODUCT");
        }

        // ------------------------------------------------------------------------
        // Проверка, является ли чистый узел константой с заданным значением
        // ------------------------------------------------------------------------
        bool clean_node_is_constant(const LazyRational& lr, int node_idx, const Rational& expected) const {
            assert(is_clean(lr));
            const auto& node = internal::pool.nodes[node_idx];
            if (node.op != internal::LazyOp::CONST) return false;
            const auto& val = internal::pool.values[node.value_idx];
            return Rational(val) == expected;
        }

        // ------------------------------------------------------------------------
        // Получение refcount
        // ------------------------------------------------------------------------
        size_t refcount(int node_idx) const {
            if (node_idx < 0 || static_cast<size_t>(node_idx) >= internal::pool.refcount.size()) return 0;
            return internal::pool.refcount[node_idx];
        }
    };

} // namespace delta::test