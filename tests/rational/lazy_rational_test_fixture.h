#pragma once

#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "delta/rational/node_pool.h"
#include "delta/rational/storage.h"
#include "absl/container/inlined_vector.h"

namespace delta::test {

    // Фикстура для доступа к внутренностям LazyRational (через friend)
    class LazyRationalTestFixture : public ::testing::Test {
    protected:
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
        // Доступ к грязному дереву (должно быть в состоянии Dirty)
        // ------------------------------------------------------------------------
        size_t dirty_node_count(const LazyRational& lr) const {
            assert(is_dirty(lr));
            return lr.nodes_.size();
        }

        size_t dirty_constant_count(const LazyRational& lr) const {
            assert(is_dirty(lr));
            return lr.constants_.size();
        }

        internal::LazyOp dirty_root_op(const LazyRational& lr) const {
            assert(is_dirty(lr));
            return lr.nodes_[lr.root_].op;
        }

        // Возвращаем InlinedVector (константную ссылку или копию – решайте сами)
        // Для удобства тестов возвращаем копию, чтобы не зависеть от времени жизни
        absl::InlinedVector<int, 2> dirty_root_children(const LazyRational& lr) const {
            assert(is_dirty(lr));
            return lr.nodes_[lr.root_].children;   // копия
        }

        int dirty_root_const_index(const LazyRational& lr) const {
            assert(is_dirty(lr));
            return lr.nodes_[lr.root_].const_index;
        }

        internal::Value dirty_constant(const LazyRational& lr, int idx) const {
            assert(is_dirty(lr));
            return lr.constants_[idx];
        }

        internal::LazyOp dirty_node_op(const LazyRational& lr, int node_idx) const {
            assert(is_dirty(lr));
            return lr.nodes_[node_idx].op;
        }

        absl::InlinedVector<int, 2> dirty_node_children(const LazyRational& lr, int node_idx) const {
            assert(is_dirty(lr));
            return lr.nodes_[node_idx].children;   // копия
        }

        int dirty_node_const_index(const LazyRational& lr, int node_idx) const {
            assert(is_dirty(lr));
            return lr.nodes_[node_idx].const_index;
        }

        // ------------------------------------------------------------------------
        // Доступ к чистому дереву (должно быть в состоянии Clean)
        // ------------------------------------------------------------------------
        int clean_root_index(const LazyRational& lr) const {
            assert(is_clean(lr));
            return lr.clean_index_;
        }

        size_t clean_node_refcount(const LazyRational& lr, int node_idx) const {
            (void)lr; // не используется, но сохраняем сигнатуру
            if (node_idx < 0 || static_cast<size_t>(node_idx) >= internal::pool.nodes.size())
                return 0;
            return internal::pool.refcount[node_idx];
        }

        // ------------------------------------------------------------------------
        // Проверка каноничности чистого SUM
        // ------------------------------------------------------------------------
        bool is_canonical_sum(const LazyRational& lr) const {
            if (!is_clean(lr)) return false;
            const auto& node = internal::pool.nodes[lr.clean_index_];
            if (node.op != internal::LazyOp::SUM || !node.children) return false;
            const auto& children = *node.children;

            // 1. Нет нулевых констант
            for (int32_t child : children) {
                const auto& child_node = internal::pool.nodes[child];
                if (child_node.op == internal::LazyOp::CONST &&
                    internal::is_zero(internal::pool.values[child_node.value_idx]))
                    return false;
            }

            // 2. Дети отсортированы по хэшу (канонический порядок)
            for (size_t i = 1; i < children.size(); ++i) {
                uint64_t hash_prev = internal::pool.nodes[children[i - 1]].hash;
                uint64_t hash_cur = internal::pool.nodes[children[i]].hash;
                if (hash_prev > hash_cur) return false;
                if (hash_prev == hash_cur && children[i - 1] >= children[i]) return false;
            }
            return true;
        }

        // ------------------------------------------------------------------------
        // Проверка каноничности чистого PRODUCT
        // ------------------------------------------------------------------------
        bool is_canonical_product(const LazyRational& lr) const {
            if (!is_clean(lr)) return false;
            const auto& node = internal::pool.nodes[lr.clean_index_];
            if (node.op != internal::LazyOp::PRODUCT || !node.children) return false;
            const auto& children = *node.children;

            // 1. Нет единичных констант
            for (int32_t child : children) {
                const auto& child_node = internal::pool.nodes[child];
                if (child_node.op == internal::LazyOp::CONST &&
                    internal::is_one(internal::pool.values[child_node.value_idx]))
                    return false;
            }

            // 2. Дети отсортированы по хэшу
            for (size_t i = 1; i < children.size(); ++i) {
                uint64_t hash_prev = internal::pool.nodes[children[i - 1]].hash;
                uint64_t hash_cur = internal::pool.nodes[children[i]].hash;
                if (hash_prev > hash_cur) return false;
                if (hash_prev == hash_cur && children[i - 1] >= children[i]) return false;
            }
            return true;
        }

        // ------------------------------------------------------------------------
        // Вспомогательные обходы грязного дерева
        // ------------------------------------------------------------------------
        bool has_node_with_op(const LazyRational& lr, internal::LazyOp op) const {
            if (!is_dirty(lr)) return false;
            for (const auto& node : lr.nodes_) {
                if (node.op == op) return true;
            }
            return false;
        }

        // ------------------------------------------------------------------------
        // Сброс глобального пула (для изоляции тестов)
        // ------------------------------------------------------------------------
        void reset_global_pool() {
            internal::reset_pool();
        }

        // ------------------------------------------------------------------------
        // Доступ к чистому индексу (обёртка над clean_index_)
        // ------------------------------------------------------------------------
        int clean_index(const LazyRational& lr) const {
            assert(is_clean(lr));
            return lr.clean_index_;
        }

        // ------------------------------------------------------------------------
        // Доступ к узлам чистого дерева (через глобальный пул)
        // ------------------------------------------------------------------------
        const internal::Node& clean_node(const LazyRational& lr, int node_idx) const {
            assert(is_clean(lr));
            return internal::pool.nodes[node_idx];
        }

        // ------------------------------------------------------------------------
        // Доступ к полю children чистого узла (как ссылка на вектор)
        // ------------------------------------------------------------------------
        const std::vector<int32_t>& clean_node_children(const LazyRational& lr, int node_idx) const {
            assert(is_clean(lr));
            const auto& node = internal::pool.nodes[node_idx];
            if (!node.children) throw std::logic_error("Node has no children");
            return *node.children;
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
        // Получение refcount для чистого узла (уже есть clean_node_refcount, но можно переименовать)
        // ------------------------------------------------------------------------
        size_t refcount(int node_idx) const {
            if (node_idx < 0 || static_cast<size_t>(node_idx) >= internal::pool.refcount.size()) return 0;
            return internal::pool.refcount[node_idx];
        }
    };

} // namespace delta::test