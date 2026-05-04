// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// node_types.h
// -----------------------------------------------------------------------------
// CORE NODE STRUCTURE FOR THE GLOBAL IMMUTABLE POOL
// -----------------------------------------------------------------------------
//
// This file defines the Node structure used in the global hash‑consed pool
// (node_pool.h). Nodes represent immutable sub‑expressions in canonicalised
// (clean) lazy rational trees.
//
// The design is minimal – only the fields necessary for evaluation and hashing
// are stored. No caching of approximations, no depths, no auxiliary data.
//
// -----------------------------------------------------------------------------
// NODE TYPES (LazyOp)
// -----------------------------------------------------------------------------
//
// - CONST: leaf node containing a constant rational Value.
// - SUM:   sum of constants (leaf_values) and child nodes.
// - PRODUCT: product of constants (leaf_values) and child nodes.
// - NEG:   unary negation.
// - RECIP: reciprocal (1/x).
// - SQRT:  square root.
// - EXP:   exponential.
// - LOG:   natural logarithm.
// - SIN:   sine.
// - COS:   cosine.
// - ACOS:  arccosine.
// - PI:    constant π (no children, epsilon in eps_idx).
// - E:     constant e (no children, epsilon in eps_idx).
// - POW:   power (base, exponent) – binary, stored as two children.
//
// -----------------------------------------------------------------------------
// STORAGE STRATEGY
// -----------------------------------------------------------------------------
//
// Each node stores:
//   - op: the operation type.
//   - hash: precomputed hash for fast equality (used by caches).
//   - children: inlined vector of child node indices (max 2 – most ops are
//     unary or binary). Inlined storage avoids heap allocation for typical cases.
//   - value_idx / eps_idx: indices into the pool's values array (constants).
//   - leaf_values: only used for SUM and PRODUCT – stores constant factors
//     or terms directly as Value (not as child nodes). This reduces node count
//     and improves cache locality.
//
// Why separate leaf_values from children?
//   - SUM(1, x, 2, y) becomes leaf_values=[1,2], children=[x_node, y_node].
//   - Without this, we would need extra CONST nodes for 1 and 2, increasing
//     memory usage and traversal time.
//   - It also simplifies algebraic simplifications (e.g., grouping constants).
//
// -----------------------------------------------------------------------------
// MOVE‑ONLY SEMANTICS
// -----------------------------------------------------------------------------
//
// Node is move‑only (copy constructor deleted). This is intentional – nodes
// are created in the pool and never copied. Moving is used during pool
// reallocation (GC) where nodes are transferred from the old pool to the new
// one. Copying would be expensive and unnecessary.
//
// -----------------------------------------------------------------------------
// HASHING FUNCTIONS (declared here, implemented in node_pool.h)
// -----------------------------------------------------------------------------
//
// - compute_hash_const(v): returns a hash for a constant Value.
// - combine_hash(op, h0, h1, extra): combines operation code and child hashes
//   into a single hash for the node. Used by the pool's caching mechanism.
//
// The hash is precomputed at node construction time and stored in the node.
// This makes equality checks O(1) in most cases (compare hashes first, then
// if hashes match, compare structurally).
//
// -----------------------------------------------------------------------------

#pragma once

#include "storage.h"
#include "interval.h"
#include "utils.h"
#include <absl/container/inlined_vector.h>
#include <vector>
#include <cstdint>

namespace delta::internal {

    // Operation type for lazy expression nodes.
    // Stored as uint8_t to reduce memory footprint.
    enum class LazyOp : uint8_t {
        CONST,      // constant value
        SUM,        // sum of constants and/or child nodes
        PRODUCT,    // product of constants and/or child nodes
        NEG,        // unary minus
        RECIP,      // reciprocal (1/x)
        SQRT,       // square root
        EXP,        // exponential
        LOG,        // natural logarithm
        SIN,        // sine
        COS,        // cosine
        ACOS,       // arccosine
        PI,         // constant π
        E,          // constant e
        POW         // power (base, exponent)
    };

    // Node in the global immutable pool.
    // Move‑only, no copying.
    struct Node {
        LazyOp op;                          // operation type
        uint64_t hash;                      // precomputed hash (for fast equality)

        absl::InlinedVector<int32_t, 2> children; // child node indices (inlined, max 2)

        int32_t value_idx = -1;             // index into pool.values (for CONST)
        int32_t eps_idx = -1;               // index into pool.values (epsilon for transcendentals)

        std::vector<Value> leaf_values;     // constant values – only for SUM and PRODUCT

        Node() = default;

        // Constructor for CONST nodes.
        Node(LazyOp op, int32_t val_idx, uint64_t hash);

        // Constructor for unary/binary ops, PI, E, etc.
        // Takes children and epsilon index (if needed).
        Node(LazyOp op, absl::InlinedVector<int32_t, 2> children,
            int32_t eps_idx, uint64_t hash);

        // Constructor for SUM and PRODUCT – with leaf constants and child nodes.
        Node(LazyOp op, std::vector<Value> leaf_values,
            absl::InlinedVector<int32_t, 2> children,
            uint64_t hash);

        // Move‑only semantics
        Node(Node&&) noexcept = default;
        Node& operator=(Node&&) noexcept = default;
        Node(const Node&) = delete;
        Node& operator=(const Node&) = delete;
    };

    // -------------------------------------------------------------------------
    // Hashing helpers (implemented in node_pool.h)
    // -------------------------------------------------------------------------

    // Compute hash for a constant Value.
    uint64_t compute_hash_const(const Value& v);

    // Combine operation and child hashes into a single node hash.
    // Arguments:
    //   op    – operation code
    //   h0    – hash of the first child (or 0 if none)
    //   h1    – hash of the second child (or 0 if none)
    //   extra – extra value (e.g., eps_idx) to mix in
    uint64_t combine_hash(LazyOp op, uint64_t h0, uint64_t h1 = 0, int64_t extra = 0);

} // namespace delta::internal