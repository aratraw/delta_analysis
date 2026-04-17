// gc.h
#pragma once

#include "node_pool.h"
#include "evaluate_impl.h"
#include <stdexcept>
#include <algorithm>

namespace delta::internal {

    // ----------------------------------------------------------------------------
    // collect_garbage – сборка мусора в глобальном пуле
    // Живые узлы (refcount > 0) вычисляются и превращаются в CONST.
    // Мёртвые узлы удаляются (затираются).
    // ----------------------------------------------------------------------------
    inline void collect_garbage() {
        const size_t n = pool.nodes.size();
        if (n == 0) return;

        // Определяем живые узлы
        std::vector<bool> alive(n, false);
        size_t alive_count = 0;
        for (size_t i = 0; i < n; ++i) {
            if (pool.refcount[i] > 0) {
                alive[i] = true;
                ++alive_count;
            }
        }

        // Если нет живых узлов, просто сбрасываем пул
        if (alive_count == 0) {
            pool.nodes.clear();
            pool.refcount.clear();
            pool.next_free_index = 0;
            pool.value_cache.clear();
            pool.constant_cache.clear();
            pool.sum_product_cache.clear();
            pool.unary_cache.clear();
            pool.ternary_cache.clear();
            return;
        }

        // Создаём новый пул такого же размера max_size
        NodePool new_pool;
        new_pool.max_size = pool.max_size;
        new_pool.update_gc_threshold();
        new_pool.nodes.resize(pool.max_size);
        new_pool.refcount.assign(pool.max_size, 0);

        // Копируем values (все значения остаются, так как на них могут ссылаться)
        new_pool.values = pool.values;
        new_pool.value_cache = pool.value_cache;

        // Для каждого живого узла:
        // 1. Вычисляем его значение
        // 2. Создаём новый CONST узел с этим значением
        // 3. Сохраняем его в новом пуле на том же индексе
        // 4. Копируем refcount
        for (size_t i = 0; i < n; ++i) {
            if (alive[i]) {
                // Вычисляем значение узла
                Value v = evaluate(static_cast<int>(i));
                // Нормализуем перед добавлением как константу
                if (v.tag == ValueType::Small && !v.small_reduced) {
                    v.normalize();
                }
                int val_idx = new_pool.add_value(v);
                Node const_node(LazyOp::CONST, val_idx, 0, Interval(to_double(v)), compute_hash_const(v));
                new_pool.nodes[i] = std::move(const_node);
                new_pool.refcount[i] = pool.refcount[i];
            }
        }

        // Находим первый свободный слот
        new_pool.next_free_index = 0;
        while (new_pool.next_free_index < new_pool.max_size &&
            new_pool.is_occupied(new_pool.nodes[new_pool.next_free_index])) {
            ++new_pool.next_free_index;
        }

        // Заменяем старый пул новым
        pool = std::move(new_pool);
    }

    // ----------------------------------------------------------------------------
    // set_pool_max_size – установка максимального размера пула
    // ----------------------------------------------------------------------------
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

    // ----------------------------------------------------------------------------
    // force_garbage_collect – принудительный запуск GC
    // ----------------------------------------------------------------------------
    inline void force_garbage_collect() {
        collect_garbage();
    }

    // ----------------------------------------------------------------------------
    // reset_pool – полный сброс пула (для тестов)
    // ----------------------------------------------------------------------------
    inline void reset_pool() {
        pool.~NodePool();
        new (&pool) NodePool();
        pool.update_gc_threshold();
    }

} // namespace delta::internal