// gc.h
#pragma once

#include "node_pool.h"
#include "evaluate_impl.h"
#include <stdexcept>
#include <algorithm>

namespace delta::internal {

    // Проверка, занят ли слот (используем метод из NodePool)
    inline bool is_occupied(const Node& node) {
        return !(node.value_idx == -1 && node.child0 == -1 && node.child1 == -1);
    }

    // Сборка мусора: создаём новый пул, копируем в него живые узлы (refcount > 0),
    // превращая их в константы. Новый пул имеет размер ровно столько, сколько живых узлов,
    // но не более max_size. Индексы живых узлов сохраняются (копируются на те же места),
    // поэтому все ссылки на них остаются валидными.
    inline void collect_garbage() {
        const size_t n = pool.nodes.size();
        if (n == 0) return;

        // Определяем, какие узлы живы
        std::vector<bool> alive(n, false);
        size_t alive_count = 0;
        for (size_t i = 0; i < n; ++i) {
            if (pool.refcount[i] > 0) {
                alive[i] = true;
                ++alive_count;
            }
        }

        // Создаём новый пул с размером max_size (как в исходной версии)
        NodePool new_pool;
        new_pool.max_size = pool.max_size;
        new_pool.update_gc_threshold();
        new_pool.nodes.assign(pool.max_size, Node());
        new_pool.refcount.assign(pool.max_size, 0);

        // Копируем живые узлы в новый пул (на те же индексы)
        for (size_t i = 0; i < n; ++i) {
            if (alive[i]) {
                // Вычисляем значение узла
                Value v = evaluate(static_cast<int>(i));
                int val_idx = new_pool.add_value(v);
                Node const_node(LazyOp::CONST, -1, -1, val_idx, 0,
                    Interval(to_double(v)), compute_hash_const(v));
                new_pool.nodes[i] = const_node;
                new_pool.refcount[i] = pool.refcount[i];
            }
        }

        // Находим первый свободный слот для next_free_index
        new_pool.next_free_index = 0;
        while (new_pool.next_free_index < new_pool.max_size &&
            is_occupied(new_pool.nodes[new_pool.next_free_index])) {
            ++new_pool.next_free_index;
        }

        // Заменяем старый пул новым
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

    inline void reset_pool() {
        pool.~NodePool();
        new (&pool) NodePool();
        pool.update_gc_threshold();
    }

} // namespace delta::internal