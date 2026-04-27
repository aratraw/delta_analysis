// lazy_nodes.h
// Версия 3.2 – удалены поля approx и depth из TempNode; DirtyNode без изменений.
// ----------------------------------------------------------------------------

#pragma once

#include "node_types.h"
#include "storage.h"
#include "interval.h"
#include <absl/container/inlined_vector.h>
#include <vector>
#include <cstdint>

namespace delta::internal {

    // ------------------------------------------------------------------------
    // DirtyNode – узел грязного дерева (без approx/depth изначально)
    // ------------------------------------------------------------------------
    struct DirtyNode {
        LazyOp op;
        int32_t value_idx = -1;
        int32_t eps_idx = -1;
        std::vector<Value> leaf_values;               // только для SUM/PRODUCT
        absl::InlinedVector<int32_t, 2> children;     // все дочерние узлы

        DirtyNode() = default;

        explicit DirtyNode(LazyOp op_, int32_t val_idx)
            : op(op_), value_idx(val_idx) {
        }

        DirtyNode(LazyOp op_, absl::InlinedVector<int32_t, 2> children_, int32_t eps_idx_)
            : op(op_), children(std::move(children_)), eps_idx(eps_idx_) {
        }

        DirtyNode(LazyOp op_,
            std::vector<Value> leaf_values_,
            absl::InlinedVector<int32_t, 2> children_)
            : op(op_), leaf_values(std::move(leaf_values_)), children(std::move(children_)) {
        }

        DirtyNode(DirtyNode&&) noexcept = default;
        DirtyNode& operator=(DirtyNode&&) noexcept = default;
        DirtyNode(const DirtyNode&) = default;
        DirtyNode& operator=(const DirtyNode&) = default;
    };

    // ------------------------------------------------------------------------
    // TempNode – временный узел для локального построения и упрощения
    // ------------------------------------------------------------------------
    struct TempNode {
        LazyOp op;
        int value_idx = -1;
        int eps_idx = -1;
        std::vector<Value> leaf_values;      // только для SUM/PRODUCT
        std::vector<int> children;           // все дочерние узлы
        uint64_t hash;

        // Конструктор для всех операций, кроме SUM/PRODUCT
        TempNode(LazyOp op_,
            std::vector<int> children_,
            int value_idx_,
            int eps_idx_,
            uint64_t hash_)
            : op(op_),
            children(std::move(children_)),
            value_idx(value_idx_),
            eps_idx(eps_idx_),
            hash(hash_) {
        }

        // Конструктор для SUM/PRODUCT с гетерогенным хранением
        TempNode(LazyOp op_,
            std::vector<Value> leaf_values_,
            std::vector<int> children_,
            int value_idx_,
            int eps_idx_,
            uint64_t hash_)
            : op(op_),
            value_idx(value_idx_),
            eps_idx(eps_idx_),
            leaf_values(std::move(leaf_values_)),
            children(std::move(children_)),
            hash(hash_) {
        }

        TempNode(TempNode&&) noexcept = default;
        TempNode& operator=(TempNode&&) noexcept = default;
        TempNode(const TempNode&) = default;
        TempNode& operator=(const TempNode&) = default;
    };

} // namespace delta::internal