// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// lazy_rational_test_fixture.h
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

        // dirty_root_children и dirty_node_children удалены – вместо них используйте
        // dirty_node_children с соответствующей проверкой (например,
        // dirty_node_children(lr, dirty_root_index(lr)) для корневых детей).

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
            return lr.nodes_[node_idx].children.size();
        }

        const internal::Value& dirty_node_leaf_value(const LazyRational& lr, int node_idx, size_t i) const {
            assert(is_dirty(lr));
            return lr.nodes_[node_idx].leaf_values[i];
        }

        int dirty_node_complex_child(const LazyRational& lr, int node_idx, size_t i) const {
            assert(is_dirty(lr));
            return lr.nodes_[node_idx].children[i];
        }

        // Единый метод доступа к children для любого грязного узла
        const absl::InlinedVector<int32_t, 2>& dirty_node_children(const LazyRational& lr, int node_idx) const {
            assert(is_dirty(lr));
            return lr.nodes_[node_idx].children;
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

            for (int32_t child : node.children) {
                const auto& child_node = internal::pool.nodes[child];
                if (child_node.op == internal::LazyOp::CONST &&
                    internal::is_zero(internal::pool.values[child_node.value_idx]))
                    return false;
            }

            const auto& cc = node.children;
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

            for (int32_t child : node.children) {
                const auto& child_node = internal::pool.nodes[child];
                if (child_node.op == internal::LazyOp::CONST &&
                    internal::is_one(internal::pool.values[child_node.value_idx]))
                    return false;
            }

            const auto& cc = node.children;
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
                // Рекурсивный обход чистого дерева из пула – теперь все дети в children
                std::stack<int> st;
                st.push(lr.clean_index_);
                while (!st.empty()) {
                    int idx = st.top(); st.pop();
                    const auto& node = internal::pool.nodes[idx];
                    if (node.op == op) return true;
                    for (int child : node.children) st.push(child);
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

        // Единый метод доступа к children для любого чистого узла
        const absl::InlinedVector<int32_t, 2>& clean_node_children(const LazyRational& lr, int node_idx) const {
            assert(is_clean(lr));
            return internal::pool.nodes[node_idx].children;
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


        // Утилиты для печати
        void print_node(const internal::Node& node, const std::vector<internal::Value>& values, int idx) {
            std::cout << "  Node[" << idx << "] op=" << static_cast<int>(node.op);
            if (node.op == internal::LazyOp::CONST) {
                std::cout << " value_idx=" << node.value_idx;
                if (node.value_idx >= 0 && node.value_idx < (int)values.size())
                    std::cout << " value=" << internal::to_string(values[node.value_idx]);
                else
                    std::cout << " value=INVALID";
            }
            if (node.eps_idx != -1) {
                std::cout << " eps_idx=" << node.eps_idx;
                if (node.eps_idx >= 0 && node.eps_idx < (int)values.size())
                    std::cout << " eps=" << internal::to_string(values[node.eps_idx]);
            }
            if (!node.leaf_values.empty()) {
                std::cout << " leaf_values: ";
                for (const auto& v : node.leaf_values)
                    std::cout << internal::to_string(v) << " ";
            }
            if (!node.children.empty()) {
                std::cout << " children: ";
                for (int c : node.children) std::cout << c << " ";
            }
            std::cout << std::endl;
        }

        void print_pool(const std::string& label) {
            std::cout << "\n=== " << label << " ===" << std::endl;
            std::cout << "pool.nodes.size()=" << internal::pool.nodes.size()
                << " next_free_index=" << internal::pool.next_free_index
                << " max_size=" << internal::pool.max_size
                << " gc_threshold=" << internal::pool.gc_threshold << std::endl;

            for (size_t i = 0; i < internal::pool.nodes.size(); ) {
                const auto& node = internal::pool.nodes[i];
                if (internal::pool.is_occupied(node)) {
                    print_node(node, internal::pool.values, i);
                    ++i;
                }
                else {
                    size_t start = i;
                    while (i < internal::pool.nodes.size() && !internal::pool.is_occupied(internal::pool.nodes[i])) {
                        ++i;
                    }
                    size_t end = i - 1;
                    if (start == end) {
                        std::cout << "  Node[" << start << "] empty" << std::endl;
                    }
                    else {
                        std::cout << "  Nodes[" << start << ".." << end << "] empty (" << (end - start + 1) << " in a row)" << std::endl;
                    }
                }
            }
        }
        void print_clean_registry() {
            std::cout << "Clean registry (" << internal::g_clean_rationals.size() << " objects):" << std::endl;
            for (auto* obj : internal::g_clean_rationals) {
                std::cout << "  " << (void*)obj << " clean_index=" << obj->clean_index_ << std::endl;
            }
        }
        void print_lazy(const LazyRational& lr, const std::string& name) {
            std::cout << "LazyRational " << name << ": state=" << (lr.is_clean() ? "Clean" : "Dirty");
            if (lr.is_clean()) {
                std::cout << " clean_index=" << lr.clean_index_;
                std::cout << std::endl;
                if (lr.clean_index_ >= 0 && lr.clean_index_ < (int)internal::pool.nodes.size()) {
                    const auto& node = internal::pool.nodes[lr.clean_index_];
                    std::cout << "  clean node: ";
                    print_node(node, internal::pool.values, lr.clean_index_);
                }
            }
            else {
                std::cout << " root=" << lr.root_ << " nodes_.size=" << lr.nodes_.size()
                    << " constants_.size=" << lr.constants_.size();
                std::cout << std::endl;
                for (size_t i = 0; i < lr.nodes_.size(); ++i) {
                    const auto& dn = lr.nodes_[i];
                    std::cout << "  dirty Node[" << i << "] op=" << static_cast<int>(dn.op);
                    if (dn.value_idx != -1) {
                        std::cout << " value_idx=" << dn.value_idx;
                        if (dn.value_idx < (int)lr.constants_.size())
                            std::cout << " value=" << internal::to_string(lr.constants_[dn.value_idx]);
                    }
                    if (dn.eps_idx != -1) {
                        std::cout << " eps_idx=" << dn.eps_idx;
                        if (dn.eps_idx < (int)lr.constants_.size())
                            std::cout << " eps=" << internal::to_string(lr.constants_[dn.eps_idx]);
                    }
                    if (!dn.leaf_values.empty()) {
                        std::cout << " leaf_values: ";
                        for (const auto& v : dn.leaf_values)
                            std::cout << internal::to_string(v) << " ";
                    }
                    if (!dn.children.empty()) {
                        std::cout << " children: ";
                        for (int c : dn.children) std::cout << c << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }
    };

} // namespace delta::testing