// global_state.h
// Единая точка управления всем глобальным состоянием библиотеки.
// НЕ подключает ни evaluation_core.h, ни node_pool.h.
// Подключается ВСЕМИ, кому нужен доступ к кэшам и реестру чистых объектов.

#pragma once

#include "storage.h"
#include <map>
#include <unordered_set>
#include <vector>

// Forward declaration для LazyRational (избегаем циклической зависимости)
namespace delta {
    class LazyRational;
}

namespace delta::internal {

    // ------------------------------------------------------------------------
    // Кэш числа π (используется в evaluation_core.h)
    // ------------------------------------------------------------------------
    inline thread_local std::map<Value, Value> pi_cache;

    inline void reset_pi_cache() {
        pi_cache.clear();
    }

    // ------------------------------------------------------------------------
    // Реестр чистых объектов LazyRational (только clean state)
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
    static constexpr size_t DEFAULT_POOL_MAX_SIZE = 1000000;
    // ------------------------------------------------------------------------
    // Флаг отключения сборщика мусора (GC) и временное снятие лимита размера пула
    // ------------------------------------------------------------------------
    inline thread_local bool gc_disabled = false;

} // namespace delta::internal