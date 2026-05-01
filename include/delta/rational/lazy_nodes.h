// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// lazy_nodes.h
// -----------------------------------------------------------------------------
// NODE STRUCTURES FOR LAZY EXPRESSION TREES
// -----------------------------------------------------------------------------
// This file defines two node types used for representing lazy rational
// expressions:
//
//   - DirtyNode: used by LazyRational when it is in "dirty" (mutable) state.
//                Nodes are stored in std::vector and can be mutated in place.
//                No hashing or global uniquification.
//
//   - TempNode: used during canonicalization (simplify_impl.h) for building
//               a temporary representation before converting to the global
//               immutable pool. Contains hashes for equality detection.
//
// These structures are separate to avoid pulling in simplif_impl.h dependencies
// into the core mutable representation.
//
// TODO: could/should merge with node_types.h for clarity but beware circular
// dependencies. The general include-architecture of the whole folder is very
// tight and interwoven. Keeping them separate is a deliberate choice to
// minimise header dependencies.
// -----------------------------------------------------------------------------

#pragma once

#include "node_types.h"
#include "storage.h"
#include "interval.h"
#include <absl/container/inlined_vector.h>
#include <vector>
#include <cstdint>

namespace delta::internal {

    // ------------------------------------------------------------------------
    // DirtyNode – node of the dirty (mutable) expression tree
    // Originally had approx and depth fields, but they were removed as they
    // were unused and only added complexity.
    // ------------------------------------------------------------------------
    struct DirtyNode {
        LazyOp op;
        int32_t value_idx = -1;      // index into constants_ vector (for CONST nodes)
        int32_t eps_idx = -1;        // index into constants_ vector (epsilon for transcendentals)
        std::vector<Value> leaf_values;               // for SUM/PRODUCT only (constant literals)
        absl::InlinedVector<int32_t, 2> children;     // child node indices

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
    // TempNode – temporary node used during simplification/canonicalization
    // Stores hash for equality detection. Children are indices into a temporary
    // vector (not the global pool). This structure is used exclusively in
    // simplify_impl.h and is destroyed after canonisation.
    // ------------------------------------------------------------------------
    struct TempNode {
        LazyOp op;
        int value_idx = -1;      // temporary value index (for CONST)
        int eps_idx = -1;        // temporary epsilon index
        std::vector<Value> leaf_values;      // for SUM/PRODUCT only
        std::vector<int> children;           // child TempNode indices (not DirtyNode indices!)
        uint64_t hash;                       // precomputed hash for fast equality

        // Constructor for all operations except SUM/PRODUCT
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

        // Constructor for SUM/PRODUCT with heterogeneous storage
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