// lazy_nodes.h
#pragma once

#include "node_pool.h"          // для delta::internal::LazyOp
#include "absl/container/inlined_vector.h"
#include <vector>
#include <cstdint>

namespace delta::internal {

    // ----------------------------------------------------------------------------
    // DirtyNode – узел грязного дерева.
    // Использует LazyOp из node_pool.h (чистые узлы и грязные используют одни и те же операции).
    // ----------------------------------------------------------------------------
    struct DirtyNode {
        LazyOp op;
        absl::InlinedVector<int, 2> children;   // индексы в LazyRational::nodes_
        int const_index = -1;                   // индекс в LazyRational::constants_

        DirtyNode(LazyOp op, absl::InlinedVector<int, 2> children, int const_index = -1)
            : op(op), children(std::move(children)), const_index(const_index) {
        }
    };

    // ----------------------------------------------------------------------------
    // TempNode – временный узел для локального построения и упрощения
    // (используется только в LazyRational::canonicalize())
    // ----------------------------------------------------------------------------
    struct TempNode {
        LazyOp op;
        std::vector<int> children;      // индексы в том же векторе TempNode
        int value_idx = -1;             // индекс в локальном векторе Value (для CONST)
        int eps_idx = -1;               // индекс в локальном векторе Value (для eps)
        uint64_t hash;
        Interval approx;
        int32_t depth;

        TempNode(LazyOp op,
            std::vector<int> children,
            int value_idx,
            int eps_idx,
            uint64_t hash,
            Interval approx,
            int32_t depth)
            : op(op)
            , children(std::move(children))
            , value_idx(value_idx)
            , eps_idx(eps_idx)
            , hash(hash)
            , approx(approx)
            , depth(depth) {
        }
    };

} // namespace delta::internal