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
        int32_t depth;
        Interval approx;
        uint64_t hash;

        absl::InlinedVector<int32_t, 4> children;
        int32_t value_idx = -1;
        int32_t eps_idx = -1;

        std::vector<Value> leaf_values;
        absl::InlinedVector<int32_t, 4> complex_children;

        Node() = default;
        Node(LazyOp op, int32_t val_idx, int32_t depth, Interval approx, uint64_t hash);
        Node(LazyOp op, absl::InlinedVector<int32_t, 4> children, int32_t eps_idx, int32_t depth, Interval approx, uint64_t hash);
        Node(LazyOp op, std::vector<Value> leaf_values, absl::InlinedVector<int32_t, 4> complex_children, int32_t depth, Interval approx, uint64_t hash);

        Node(Node&&) noexcept = default;
        Node& operator=(Node&&) noexcept = default;
        Node(const Node&) = delete;
        Node& operator=(const Node&) = delete;
    };

    // Вспомогательные функции метаданных (реализации будут в node_pool.h)
    uint64_t compute_hash_const(const Value& v);
    uint64_t combine_hash(LazyOp op, uint64_t h0, uint64_t h1 = 0, int64_t extra = 0);
    Interval interval_from_value(const Value& v);
    Interval compute_sum_interval(const std::vector<Value>& leaf_values, const std::vector<Interval>& child_intervals);
    Interval compute_product_interval(const std::vector<Value>& leaf_values, const std::vector<Interval>& child_intervals);
    Interval compute_interval(LazyOp op, const Interval& a, const Interval& b = Interval());

} // namespace delta::internal