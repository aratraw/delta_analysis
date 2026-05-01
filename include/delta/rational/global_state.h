// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// global_state.h
// -----------------------------------------------------------------------------
// SINGLE POINT OF CONTROL FOR ALL GLOBAL STATE IN THE LIBRARY
// -----------------------------------------------------------------------------
// This header does NOT include evaluation_core.h or node_pool.h.
// It is included by EVERYONE who needs access to caches and the clean object
// registry.
//
// Contents:
//   - π cache (thread‑local, used by evaluation_core.h)
//   - Registry of clean LazyRational objects (for garbage collection)
//   - GC disable flag and pool size control
//
// TODO: merge context.h with global_state.h, priority low
// -----------------------------------------------------------------------------

#pragma once

#include "storage.h"
#include <map>
#include <unordered_set>
#include <vector>

// Forward declaration for LazyRational (avoids circular dependency)
namespace delta {
    class LazyRational;
}

namespace delta::internal {

    // ------------------------------------------------------------------------
    // π cache (used by evaluation_core.h)
    // Thread‑local to avoid contention; each thread computes its own π.
    // ------------------------------------------------------------------------
    inline thread_local std::map<Value, Value> pi_cache;

    inline void reset_pi_cache() {
        pi_cache.clear();
    }

    // ------------------------------------------------------------------------
    // Registry of clean LazyRational objects (clean state only)
    // ------------------------------------------------------------------------
    // Used by garbage collection to find all live roots.
    // Each LazyRational registers itself when it becomes clean and
    // unregisters when mutated or destroyed.
    // ------------------------------------------------------------------------
    inline thread_local std::unordered_set<delta::LazyRational*> g_clean_rationals;

    inline void register_clean(delta::LazyRational* obj) {
        g_clean_rationals.insert(obj);
    }

    inline void unregister_clean(delta::LazyRational* obj) {
        g_clean_rationals.erase(obj);
    }

    inline std::vector<delta::LazyRational*> get_clean_objects_snapshot() {
        return std::vector<delta::LazyRational*>(g_clean_rationals.begin(),
            g_clean_rationals.end());
    }

    inline void clear_clean_registry() {
        g_clean_rationals.clear();
    }

    // ------------------------------------------------------------------------
    // Pool configuration
    // ------------------------------------------------------------------------
    static constexpr size_t DEFAULT_POOL_MAX_SIZE = 1000000;

    // ------------------------------------------------------------------------
    // GC disable flag and temporary pool size limit override
    // ------------------------------------------------------------------------
    // When gc_disabled == true, collect_garbage() does NOT run even if the
    // pool threshold is exceeded. Used during canonicalization to prevent
    // premature GC while building large expressions.
    // ------------------------------------------------------------------------
    inline thread_local bool gc_disabled = false;

} // namespace delta::internal