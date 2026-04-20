// lazy_nodes.h
// Версия 3.0 – унифицированные грязные узлы (DirtyNode) и временные узлы (TempNode)
// ----------------------------------------------------------------------------
// Изменения:
//   - DirtyNode структурно идентичен чистому Node:
//       * value_idx, eps_idx, leaf_values, complex_children
//       * единый children: absl::InlinedVector<int32_t, 4>
//   - TempNode также обновлён: children вместо child0/child1, leaf_values для упрощения
//   - Добавлены реализации всех конструкторов (inline)
// ----------------------------------------------------------------------------

#pragma once

#include "node_types.h"   // вместо node_pool.h
#include "storage.h"       // для Value (уже есть, но для ясности)
#include "interval.h"      // для Interval
#include <absl/container/inlined_vector.h>
#include <vector>
#include <cstdint>

namespace delta::internal {

    // ------------------------------------------------------------------------
    // DirtyNode – узел грязного дерева, move-only, без глобального состояния
    // ------------------------------------------------------------------------
    struct DirtyNode {
        LazyOp op;
        absl::InlinedVector<int32_t, 4> children;
        int32_t value_idx = -1;
        int32_t eps_idx = -1;
        std::vector<Value> leaf_values;
        absl::InlinedVector<int32_t, 4> complex_children;

        DirtyNode() = default;

        // CONST
        explicit DirtyNode(LazyOp op_, int32_t val_idx)
            : op(op_), value_idx(val_idx) {
        }

        // Унарные / бинарные / PI / E (без leaf_values)
        DirtyNode(LazyOp op_, absl::InlinedVector<int32_t, 4> children_, int32_t eps_idx_)
            : op(op_), children(std::move(children_)), eps_idx(eps_idx_) {
        }

        // SUM / PRODUCT с гетерогенным хранением
        DirtyNode(LazyOp op_,
            std::vector<Value> leaf_values_,
            absl::InlinedVector<int32_t, 4> complex_children_)
            : op(op_), leaf_values(std::move(leaf_values_)), complex_children(std::move(complex_children_)) {
        }

        // Move-конструктор / оператор = default
        DirtyNode(DirtyNode&&) noexcept = default;
        DirtyNode& operator=(DirtyNode&&) noexcept = default;

        // Копирование разрешено (требуется для std::vector<DirtyNode>)
        DirtyNode(const DirtyNode&) = default;
        DirtyNode& operator=(const DirtyNode&) = default;
    };

    // ------------------------------------------------------------------------
    // TempNode – временный узел для локального построения и упрощения
    // ------------------------------------------------------------------------
    struct TempNode {
        LazyOp op;
        std::vector<int> children;
        int value_idx = -1;
        int eps_idx = -1;
        std::vector<Value> leaf_values;
        std::vector<int> complex_children;
        uint64_t hash;
        Interval approx;
        int32_t depth;

        // Конструктор для обычных операций (не SUM/PRODUCT)
        TempNode(LazyOp op_,
            std::vector<int> children_,
            int value_idx_,
            int eps_idx_,
            uint64_t hash_,
            Interval approx_,
            int32_t depth_)
            : op(op_),
            children(std::move(children_)),
            value_idx(value_idx_),
            eps_idx(eps_idx_),
            hash(hash_),
            approx(approx_),
            depth(depth_) {
        }

        // Конструктор для SUM/PRODUCT с гетерогенным хранением
        TempNode(LazyOp op_,
            std::vector<Value> leaf_values_,
            std::vector<int> complex_children_,
            int value_idx_,
            int eps_idx_,
            uint64_t hash_,
            Interval approx_,
            int32_t depth_)
            : op(op_),
            value_idx(value_idx_),
            eps_idx(eps_idx_),
            leaf_values(std::move(leaf_values_)),
            complex_children(std::move(complex_children_)),
            hash(hash_),
            approx(approx_),
            depth(depth_) {
        }

        // Move-конструктор / оператор
        TempNode(TempNode&&) noexcept = default;
        TempNode& operator=(TempNode&&) noexcept = default;

        // Копирование разрешено (используется при упрощении)
        TempNode(const TempNode&) = default;
        TempNode& operator=(const TempNode&) = default;
    };

} // namespace delta::internal