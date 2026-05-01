// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// node_pool.h
// -----------------------------------------------------------------------------
// GLOBAL HASH‑CONSED POOL FOR IMMUTABLE (CLEAN) EXPRESSION NODES
// -----------------------------------------------------------------------------
//
// This file implements the global node pool that stores all canonicalised
// (clean) nodes. The pool is thread‑local – each thread has its own independent
// pool, avoiding locks at the cost of memory duplication across threads.
//
// Key concepts:
//   - Nodes are immutable and unique: the pool guarantees that no two distinct
//     nodes represent the same expression (hash‑consing). This is achieved by
//     caching: constant_cache, sum_product_cache, unary_cache.
//   - Reference counting (refcount vector) tracks how many clean LazyRational
//     objects currently point to each node. When the last reference is removed,
//     the node's children have their refcounts decremented (the node itself
//     remains in the pool until the next GC).
//   - Garbage collection (collect_garbage()) is triggered when the number of
//     allocated nodes reaches gc_threshold (90% of max_size). GC evaluates
//     every live root (referenced from a clean LazyRational) into a constant
//     and replaces the entire subtree with that constant – the tree collapses
//     into a single CONST node at the same index.
//
// -----------------------------------------------------------------------------
// ARCHITECTURE OVERVIEW
// -----------------------------------------------------------------------------
//
// 1. Node structure (defined in node_types.h)
//    - op: LazyOp (CONST, SUM, PRODUCT, NEG, RECIP, SQRT, EXP, LOG, SIN, …)
//    - hash: precomputed hash for fast equality
//    - children: inlined vector of child indices (max 2 for most ops)
//    - leaf_values: vector of constant Values (only for SUM and PRODUCT)
//    - value_idx / eps_idx: indices into pool.values array (constants)
//
// 2. Key types for caching
//    - SumProductKey: used to index SUM and PRODUCT nodes. Contains op,
//      leaf_values (sorted, canonicalised), and children (sorted by hash/index).
//    - UnaryKey: used for all other ops (NEG, RECIP, SQRT, EXP, LOG, SIN,
//      COS, ACOS, PI, E, POW). Contains op, children (ordered, typically 1 or 2),
//      and eps_idx (index of epsilon constant if needed).
//
// 3. NodePool struct (thread_local)
//    - nodes: vector of Node (the actual nodes)
//    - values: vector of constants (Value) shared among CONST nodes and epsilons
//    - refcount: reference count per node (0 = no live references)
//    - next_free_index: hint for the next free slot to allocate
//    - max_size, gc_threshold: limits for automatic GC
//    - Various caches: constant_cache, sum_product_cache, unary_cache
//
// 4. Allocation policy (allocate_node) – APPEND‑ONLY POOL DESIGN
//    - The pool is append‑only between GC cycles. Slots are never reused
//      individually – only a full GC creates a brand new pool.
//    - If next_free_index >= gc_threshold and GC not disabled → collect_garbage()
//    - Scan forward from next_free_index for an unoccupied slot.
//    - Invariant: everything before next_free_index is guaranteed occupied.
//    - Amortised O(1) search for a free slot (not the insertion itself).
//    - next_free_index is not guaranteed to point to a free slot, but it is
//      highly probable. Slots after next_free_index may be occupied or free
//      – no strong guarantee, but in practice they are free.
//    - If no free slot found, expand the vector (but never exceed max_size
//      if GC is enabled; GC must have already failed if we reach this point).
//
// 5. Memory allocation strategy
//    - The pool does NOT pre‑allocate the full max_size upfront.
//    - Instead, it grows in dynamic chunks of 4096 nodes at a time.
//    - This keeps memory usage proportional to actual needs, not the configured
//      limit. The max_size acts as a soft cap – expansion stops when this
//      limit is reached, after which only GC can free space.
//    - Chunk size 4096 was chosen empirically as a balance between reducing
//      reallocations and avoiding excessive memory waste.
//
// 6. Reference counting (increment_ref / decrement_ref)
//    - increment_ref: increases refcount of a node (called when a clean
//      LazyRational starts referencing it, e.g., in clone() or canonicalize()).
//    - decrement_ref: decreases refcount; when it reaches zero, recursively
//      decrements children's refcounts.
//    - IMPORTANT: The node's fields are NOT cleared when refcount reaches zero.
//      This is intentional. The pool is append‑only; individual slots are never
//      reused until the next full GC. Therefore clearing is unnecessary.
//    - Dead nodes (refcount == 0) remain in the pool as garbage until the
//      next garbage collection, at which point the entire pool is replaced.
//
// 7. Garbage collection (collect_garbage)
//    - Triggered automatically when next_free_index >= gc_threshold.
//    - Also can be called manually via force_garbage_collect().
//    - Algorithm:
//      a) Take a snapshot of all clean LazyRational objects (roots) from the
//         global clean object registry (g_clean_rationals).
//      b) If no roots exist → reset the entire pool to empty.
//      c) Otherwise, collect the set of live root indices and find the maximum.
//      d) Create a brand new NodePool sized to (max_root_index + 1).
//      e) For each live root index, evaluate the entire subtree to a Value
//         (using evaluate() which traverses and computes the rational result).
//      f) Store the resulting Value as a constant in the new pool, and place
//         a CONST node at the SAME INDEX as the original root.
//      g) The new pool now contains ONLY CONST nodes – every live tree has
//         been collapsed (folded) into a single constant.
//      h) Replace the old pool with the new one (swap/move).
//    - Important invariant: after GC, the pool contains exactly as many nodes
//      as there are root indices (one CONST per live root). No other nodes exist.
//    - The pool is NOT compacted in the sense that indices remain the same
//      as before GC. If root indices were sparse (e.g., 0, 1000, 50000), the
//      new pool will have size 50001 but only three slots are occupied (0, 1000,
//      50000). The rest are free (unoccupied) slots.
//
// 8. reset_pool – full global state cleanup
//    - Called when the user wants to completely reset the library's state.
//    - Operations:
//      a) Take snapshot of all clean LazyRational objects from the registry.
//      b) For each such object: decrement its refcount, destroy it via
//         placement destructor, then reconstruct it as a dirty CONST(0)
//         using placement new. This removes all objects from the registry.
//      c) Clear the global pool (assign a brand new empty NodePool).
//      d) Clear the π cache (reset_pi_cache()).
//      e) Clear the clean object registry.
//    - After reset_pool(), all LazyRational objects are in dirty state
//      containing the rational 0. No memory leaks – everything is cleaned.
//    - This function is useful for testing and for applications that need
//      to reclaim memory after heavy symbolic computations.
//
// -----------------------------------------------------------------------------
// KEY INVARIANTS AND DESIGN CHOICES
// -----------------------------------------------------------------------------
//
// - The pool is thread_local → different threads do not share nodes.
//   This simplifies locking but increases memory usage if many threads.
//
// - GC is conservative: it only runs when the pool is nearly full and GC is not
//   disabled (gc_disabled flag). During canonicalization, GC is temporarily
//   disabled to avoid interfering with the construction of a large tree.
//
// - The pool is append‑only between GC cycles. After GC, a completely new pool
//   is created, and the old one is discarded. This means:
//   * No need to clear individual node fields on zero refcount.
//   * No individual slot reuse – simpler and faster.
//   * Memory fragmentation is bounded – a full GC resets everything.
//
// - Memory grows in chunks of 4096 nodes, not pre‑allocated to max_size.
//
// - reset_pool() vs collect_garbage():
//   * collect_garbage(): preserves clean objects, collapses their trees to
//     constants, keeps the same indices, does NOT touch dirty objects.
//   * reset_pool(): destroys ALL clean objects (replaces them with dirty 0),
//     completely wipes the pool and caches, resets global state. Use for
//     full cleanup or between test cases.
//
// - The clean object registry (g_clean_rationals) enables GC to find all
//   live roots without scanning the entire pool. It also enables potential
//   future optimisations like pool compaction.
//
// -----------------------------------------------------------------------------
// TODO: FUTURE IMPROVEMENTS
// -----------------------------------------------------------------------------
//
// 1. Separate GC policies: hard vs soft
//    - hard_collect_garbage(): current implementation – evaluate all live roots,
//      collapse to constants, create new pool. Heavy but thorough.
//    - soft_collect_garbage(): lightweight version that traverses the pool,
//      finds all nodes with refcount == 0, marks their slots as free, and
//      resets next_free_index to the first free slot from the beginning.
//      Does NOT evaluate or collapse trees. Useful for quick memory recovery
//      when the pool is mostly garbage but roots are still complex.
//    - Heuristic: call soft_GC on every allocation when pool is 80% full,
//      call hard_GC only when soft_GC fails to free enough space.
//
// 2. Pool compaction during hard_GC
//    - Currently, after hard_GC the pool may be sparse (indices 0, 1000, 50000).
//    - With the clean object registry, we have direct access to all live roots.
//    - We could renumber all living CONST nodes to start from 0 sequentially,
//      then update the clean_index_ of each LazyRational in the registry.
//    - This would eliminate sparse high indices and reduce pool memory footprint
//      after GC, especially in long‑running applications with many GC cycles.
//    - Trade‑off: requires writing back to the objects in the registry
//      (which is possible – we have pointers), but adds complexity.
//    - Priority: low – sparseness is usually not extreme, and 4096‑chunk
//      allocation mitigates the issue.
//
// 3. Configurable chunk size
//    - Currently hard‑coded to 4096. Might be too large for embedded systems
//      or too small for HPC workloads. Could become a compile‑time or run‑time
//      parameter.
//
// 4. Pool statistics
//    - Add debugging counters: number of nodes allocated, number of GC cycles,
//      average pool utilisation, number of cache hits/misses.
//    - Useful for performance tuning and understanding usage patterns.
//
// -----------------------------------------------------------------------------
// USAGE NOTE
// -----------------------------------------------------------------------------
// The functions in this header are internal and used only by LazyRational.
// External code should never call these directly.
// -----------------------------------------------------------------------------


#pragma once

#include "node_types.h"
#include "global_state.h"
#include <absl/container/flat_hash_map.h>
#include <absl/hash/hash.h>
#include <memory>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <type_traits>
#include <stdexcept>

namespace delta::internal {

    // ------------------------------------------------------------------------
    // Hash and equality for Value (use absl::Hash)
    // ------------------------------------------------------------------------
    struct ValueHash {
        size_t operator()(const Value& v) const noexcept {
            return absl::Hash<Value>()(v);
        }
    };

    struct ValueEqual {
        bool operator()(const Value& a, const Value& b) const noexcept {
            return a == b;
        }
    };

    // ------------------------------------------------------------------------
    // Forward declarations
    // ------------------------------------------------------------------------
    void collect_garbage();

    // ------------------------------------------------------------------------
    // Cache keys for sum/product and unary nodes
    // ------------------------------------------------------------------------
    struct SumProductKey {
        LazyOp op;                         // must be SUM or PRODUCT
        std::vector<Value> leaf_values;    // constant factors/terms (canonical order)
        absl::InlinedVector<int32_t, 2> children;  // child node indices (canonical order)

        bool operator==(const SumProductKey& other) const {
            return op == other.op &&
                leaf_values == other.leaf_values &&
                children == other.children;
        }
    };

    struct SumProductKeyHash {
        size_t operator()(const SumProductKey& key) const {
            size_t h = absl::Hash<LazyOp>()(key.op);
            ValueHash value_hasher;
            for (const auto& v : key.leaf_values) {
                h = absl::HashOf(h, value_hasher(v));
            }
            for (int32_t c : key.children) {
                h = absl::HashOf(h, c);
            }
            return h;
        }
    };

    struct UnaryKey {
        LazyOp op;                              // any non‑SUM/PRODUCT op (NEG, RECIP, SQRT, …)
        absl::InlinedVector<int32_t, 2> children;
        int32_t eps_idx;                       // index of epsilon constant, or -1

        bool operator==(const UnaryKey& other) const = default;
    };

    struct UnaryKeyHash {
        size_t operator()(const UnaryKey& k) const {
            size_t h = absl::HashOf(k.op, k.eps_idx);
            for (int32_t c : k.children) {
                h = absl::HashOf(h, c);
            }
            return h;
        }
    };

    // ------------------------------------------------------------------------
    // NodePool – global hash‑consed pool (thread_local)
    // ------------------------------------------------------------------------
    struct NodePool {
        size_t max_size = DEFAULT_POOL_MAX_SIZE;    // soft limit (may be exceeded temporarily)
        size_t gc_threshold = 0;                    // 0.9 * max_size, triggers GC
        std::vector<Node> nodes;                    // all nodes (some may be free)
        std::vector<Value> values;                  // constant values (indexed by value_idx / eps_idx)
        std::vector<int> refcount;                  // reference count per node (0 = free/unused)
        size_t next_free_index = 0;                 // hint for next free slot

        // Caches for hash‑consing
        absl::flat_hash_map<Value, int, ValueHash, ValueEqual> value_cache;
        absl::flat_hash_map<Value, int, ValueHash, ValueEqual> constant_cache;
        absl::flat_hash_map<SumProductKey, int, SumProductKeyHash> sum_product_cache;
        absl::flat_hash_map<UnaryKey, int, UnaryKeyHash> unary_cache;

        void update_gc_threshold() {
            gc_threshold = static_cast<size_t>(0.9 * max_size);
        }

        void ensure_initialized() {
            if (nodes.empty()) {
                nodes.reserve(4096);
                refcount.reserve(4096);
                next_free_index = 0;
                update_gc_threshold();
            }
        }

        int add_value(const Value& v) {
            auto it = value_cache.find(v);
            if (it != value_cache.end()) return it->second;
            int idx = static_cast<int>(values.size());
            values.push_back(v);
            value_cache[v] = idx;
            return idx;
        }

        int allocate_node() {
            ensure_initialized();

            // Trigger GC if pool is near full and GC is not disabled
            if (!internal::gc_disabled && next_free_index >= gc_threshold) {
                collect_garbage();
                if (next_free_index >= max_size) {
                    throw std::runtime_error("NodePool exhausted: all slots occupied by roots");
                }
            }

            // Find a free slot starting from next_free_index
            size_t idx = next_free_index;
            while (idx < nodes.size() && is_occupied(nodes[idx])) {
                ++idx;
            }

            // If no free slot, expand the vector
            if (idx >= nodes.size()) {
                size_t old_size = nodes.size();
                size_t new_size = old_size + 4096;

                // Do not exceed max_size unless GC is disabled
                if (!internal::gc_disabled && new_size > max_size) {
                    new_size = max_size;
                }
                if (new_size <= old_size) {
                    throw std::runtime_error("NodePool exhausted: cannot expand beyond max_size");
                }

                nodes.resize(new_size);
                refcount.resize(new_size, 0);
                for (size_t i = old_size; i < new_size; ++i) {
                    nodes[i] = Node();
                    refcount[i] = 0;
                }
                idx = old_size;
            }

            // Claim the slot
            next_free_index = idx + 1;
            refcount[idx] = 0;
            return static_cast<int>(idx);
        }

        bool is_occupied(const Node& node) const {
            if (node.op == LazyOp::SUM || node.op == LazyOp::PRODUCT) {
                return !node.leaf_values.empty() || !node.children.empty();
            }
            if (node.op == LazyOp::CONST) {
                return node.value_idx != -1;
            }
            return !node.children.empty() || node.eps_idx != -1;
        }
    };

    inline thread_local NodePool pool;

    // ------------------------------------------------------------------------
    // Helper function implementations from node_types.h
    // ------------------------------------------------------------------------
    inline uint64_t compute_hash_const(const Value& v) {
        return ValueHash{}(v);
    }

    inline uint64_t combine_hash(LazyOp op, uint64_t h0, uint64_t h1, int64_t extra) {
        uint64_t h = static_cast<uint64_t>(op);
        auto combine = [](uint64_t& seed, uint64_t v) {
            seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            };
        combine(h, h0);
        if (h1 != 0) combine(h, h1);
        if (extra != 0) combine(h, static_cast<uint64_t>(extra));
        return h;
    }

    // ------------------------------------------------------------------------
    // Node constructors (implementation) – no depth or approx fields
    // ------------------------------------------------------------------------
    inline Node::Node(LazyOp op, int32_t val_idx, uint64_t hash)
        : op(op), hash(hash), value_idx(val_idx), eps_idx(-1) {
    }

    inline Node::Node(LazyOp op, absl::InlinedVector<int32_t, 2> children,
        int32_t eps_idx, uint64_t hash)
        : op(op), hash(hash), children(std::move(children)), value_idx(-1), eps_idx(eps_idx) {
    }

    inline Node::Node(LazyOp op, std::vector<Value> leaf_values,
        absl::InlinedVector<int32_t, 2> children,
        uint64_t hash)
        : op(op), hash(hash), value_idx(-1), eps_idx(-1),
        leaf_values(std::move(leaf_values)), children(std::move(children)) {
    }

    // ------------------------------------------------------------------------
    // Factory functions for clean nodes (hash‑consed)
    // ------------------------------------------------------------------------
    inline int add_const(const Value& v) {
        pool.ensure_initialized();
        auto it = pool.constant_cache.find(v);
        if (it != pool.constant_cache.end()) return it->second;

        int idx = pool.allocate_node();
        int val_idx = pool.add_value(v);
        Node node(LazyOp::CONST, val_idx, compute_hash_const(v));
        pool.nodes[idx] = std::move(node);
        pool.refcount[idx] = 0;
        pool.constant_cache[v] = idx;
        return idx;
    }

    inline int make_sum_node(std::vector<Value> leaf_values,
        absl::InlinedVector<int32_t, 2> children) {
        pool.ensure_initialized();
        SumProductKey key{ LazyOp::SUM, leaf_values, children };
        auto it = pool.sum_product_cache.find(key);
        if (it != pool.sum_product_cache.end()) return it->second;

        uint64_t hash = static_cast<uint64_t>(LazyOp::SUM);
        ValueHash value_hasher;
        for (int32_t child : children) {
            hash = combine_hash(LazyOp::SUM, hash, pool.nodes[child].hash);
        }
        for (const auto& v : leaf_values) {
            hash = absl::HashOf(hash, value_hasher(v));
        }

        int idx = pool.allocate_node();
        Node node(LazyOp::SUM, std::move(leaf_values), std::move(children), hash);
        pool.nodes[idx] = std::move(node);
        pool.refcount[idx] = 0;
        pool.sum_product_cache[std::move(key)] = idx;
        return idx;
    }

    inline int make_product_node(std::vector<Value> leaf_values,
        absl::InlinedVector<int32_t, 2> children) {
        pool.ensure_initialized();
        SumProductKey key{ LazyOp::PRODUCT, leaf_values, children };
        auto it = pool.sum_product_cache.find(key);
        if (it != pool.sum_product_cache.end()) return it->second;

        uint64_t hash = static_cast<uint64_t>(LazyOp::PRODUCT);
        ValueHash value_hasher;
        for (int32_t child : children) {
            hash = combine_hash(LazyOp::PRODUCT, hash, pool.nodes[child].hash);
        }
        for (const auto& v : leaf_values) {
            hash = absl::HashOf(hash, value_hasher(v));
        }

        int idx = pool.allocate_node();
        Node node(LazyOp::PRODUCT, std::move(leaf_values), std::move(children), hash);
        pool.nodes[idx] = std::move(node);
        pool.refcount[idx] = 0;
        pool.sum_product_cache[std::move(key)] = idx;
        return idx;
    }

    inline int get_unary_node(LazyOp op, absl::InlinedVector<int32_t, 2> children, int eps_idx) {
        pool.ensure_initialized();
        UnaryKey key{ op, children, eps_idx };
        auto it = pool.unary_cache.find(key);
        if (it != pool.unary_cache.end()) return it->second;

        uint64_t hash = static_cast<uint64_t>(op);
        if (children.empty()) {
            hash = combine_hash(op, 0, eps_idx);
        }
        else if (children.size() == 1) {
            int32_t child = children[0];
            hash = combine_hash(op, pool.nodes[child].hash, 0, eps_idx);
        }
        else if (children.size() == 2) {
            int32_t base = children[0];
            int32_t exp = children[1];
            hash = combine_hash(LazyOp::POW, pool.nodes[base].hash, pool.nodes[exp].hash, eps_idx);
        }
        else {
            throw std::logic_error("get_unary_node: invalid children count");
        }

        int idx = pool.allocate_node();
        Node node(op, std::move(children), eps_idx, hash);
        pool.nodes[idx] = std::move(node);
        pool.refcount[idx] = 0;
        pool.unary_cache[std::move(key)] = idx;
        return idx;
    }

    inline int get_pow_node(int base, int exponent, int eps_idx) {
        absl::InlinedVector<int32_t, 2> children = { base, exponent };
        return get_unary_node(LazyOp::POW, std::move(children), eps_idx);
    }

    // ------------------------------------------------------------------------
    // Reference counting
    // ------------------------------------------------------------------------
    inline void increment_ref(int idx) {
        if (idx < 0) return;
        pool.ensure_initialized();
        if (static_cast<size_t>(idx) >= pool.nodes.size()) return;
        ++pool.refcount[idx];
    }

    inline void decrement_ref(int idx) {
        if (idx < 0) return;
        if (static_cast<size_t>(idx) >= pool.nodes.size()) return;
        if (pool.refcount[idx] > 0) {
            --pool.refcount[idx];
            if (pool.refcount[idx] == 0) {
                const Node& node = pool.nodes[idx];
                for (int32_t child : node.children) {
                    decrement_ref(child);
                }
                // FIXME: node fields are NOT cleared! The slot will remain marked
                // as occupied by is_occupied(), causing memory leak. Should set
                // node.op = LazyOp::CONST? or clear children and leaf_values.
            }
        }
    }

} // namespace delta::internal

// ----------------------------------------------------------------------------
// Include evaluate_impl.h and define evaluate() for clean trees, plus GC
// ----------------------------------------------------------------------------
#include "evaluate_impl.h"

namespace delta::internal {

    // Evaluate a clean tree rooted at root_idx to a Value
    inline Value evaluate(int root_idx) {
        struct Accessor {
            Value const_value(const Node& node) const {
                return pool.values[node.value_idx];
            }
            Value eps_value(const Node& node) const {
                return (node.eps_idx != -1) ? pool.values[node.eps_idx] : Value{};
            }
        };
        SumStrategy_Standard sum_strategy;
        ProdStrategy_Sequential prod_strategy;
        return evaluate_tree<Node>(root_idx, pool.nodes, Accessor{}, sum_strategy, prod_strategy);
    }

    // Garbage collection: replace all live roots with their evaluated constants,
    // then rebuild a compacted pool.
    inline void collect_garbage() {
        // 1. Get snapshot of all clean LazyRational objects (roots)
        auto clean_objects = get_clean_objects_snapshot();
        if (clean_objects.empty()) {
            pool = NodePool();
            pool.update_gc_threshold();
            return;
        }

        // 2. Collect root indices and find max index
        std::unordered_set<int> root_indices;
        size_t max_root_index = 0;
        for (delta::LazyRational* obj : clean_objects) {
            int idx = obj->clean_index_;
            if (idx >= 0 && static_cast<size_t>(idx) < pool.nodes.size() && pool.refcount[idx] > 0) {
                root_indices.insert(idx);
                if (static_cast<size_t>(idx) > max_root_index) max_root_index = idx;
            }
        }

        if (root_indices.empty()) {
            pool = NodePool();
            pool.update_gc_threshold();
            return;
        }

        // 3. Create a new pool sized to (max_root_index + 1)
        NodePool new_pool;
        new_pool.max_size = pool.max_size;
        new_pool.update_gc_threshold();
        size_t new_size = max_root_index + 1;
        new_pool.nodes.resize(new_size);
        new_pool.refcount.assign(new_size, 0);
        new_pool.values = pool.values;
        new_pool.value_cache = pool.value_cache;

        // 4. For each live root, evaluate its subtree and replace with a CONST node
        for (int root_idx : root_indices) {
            Value v = evaluate(root_idx);
            int val_idx = new_pool.add_value(v);
            Node const_node(LazyOp::CONST, val_idx, compute_hash_const(v));
            new_pool.nodes[root_idx] = std::move(const_node);
            new_pool.refcount[root_idx] = 1;   // each stub has exactly one owner (the LazyRational)
        }

        // 5. Find the first free slot for next_free_index
        new_pool.next_free_index = 0;
        while (new_pool.next_free_index < new_size &&
            new_pool.is_occupied(new_pool.nodes[new_pool.next_free_index])) {
            ++new_pool.next_free_index;
        }

        // 6. Replace the old pool
        pool = std::move(new_pool);
    }

    inline void set_pool_max_size(size_t new_size) {
        if (pool.nodes.empty()) {
            pool.max_size = new_size;
            pool.update_gc_threshold();
        }
        else if (new_size > pool.max_size) {
            pool.max_size = new_size;
            pool.update_gc_threshold();
        }
    }

    inline void force_garbage_collect() {
        collect_garbage();
    }

    // -------------------------------------------------------------------------
    // reset_pool – complete pool reset, invalidating all clean objects
    // -------------------------------------------------------------------------
    inline void reset_pool() {
        // 1. Take snapshot of all clean objects (while the old pool is still alive)
        auto clean_objects = get_clean_objects_snapshot();

        // 2. Invalidate each clean object by reinitialising it as a dirty zero
        for (delta::LazyRational* obj : clean_objects) {
            decrement_ref(obj->clean_index_);
            obj->~LazyRational();
            new (obj) delta::LazyRational();   // default constructor creates dirty CONST(0)
        }

        // 3. Clear the global pool and caches
        pool = NodePool();
        pool.update_gc_threshold();
        reset_pi_cache();

        // 4. Clear the registry (all objects already removed, but for safety)
        clear_clean_registry();
    }

} // namespace delta::internal