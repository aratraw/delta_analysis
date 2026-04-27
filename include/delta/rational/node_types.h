// node_types.h
#pragma once

#include "storage.h"
#include "interval.h"
#include "utils.h"
#include <absl/container/inlined_vector.h>
#include <vector>
#include <cstdint>

namespace delta::internal {

    enum class LazyOp : uint8_t {
        CONST, SUM, PRODUCT, NEG, RECIP, SQRT, EXP, LOG, SIN, COS, ACOS, PI, E, POW
    };

    struct Node {
        LazyOp op;
        uint64_t hash;

        absl::InlinedVector<int32_t, 2> children;

        int32_t value_idx = -1;
        int32_t eps_idx = -1;

        std::vector<Value> leaf_values;   // только для SUM и PRODUCT

        Node() = default;

        // Конструктор для CONST (только значение)
        Node(LazyOp op, int32_t val_idx, uint64_t hash);

        // Обобщённый конструктор для унарных, бинарных операций, PI, E и т.д.
        // Принимает children и eps_idx.
        Node(LazyOp op, absl::InlinedVector<int32_t, 2> children,
            int32_t eps_idx, uint64_t hash);

        // Конструктор для SUM и PRODUCT с листовыми значениями и дочерними узлами
        Node(LazyOp op, std::vector<Value> leaf_values,
            absl::InlinedVector<int32_t, 2> children,
            uint64_t hash);

        // Move-only
        Node(Node&&) noexcept = default;
        Node& operator=(Node&&) noexcept = default;
        Node(const Node&) = delete;
        Node& operator=(const Node&) = delete;
    };

    // Вспомогательные функции метаданных (реализации будут в node_pool.h)
    uint64_t compute_hash_const(const Value& v);
    uint64_t combine_hash(LazyOp op, uint64_t h0, uint64_t h1 = 0, int64_t extra = 0);

} // namespace delta::internal