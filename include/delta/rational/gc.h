// gc.h
#pragma once

#include "node_pool.h"
#include "evaluate_impl.h"
#include <stdexcept>

namespace delta::internal {

    // Проверка, занят ли слот в заданном пуле
    inline bool is_occupied(const NodePool& p, size_t idx) {
        const Node& node = p.nodes[idx];
        return !(node.value_idx == -1 && node.child0 == -1 && node.child1 == -1);
    }

    // Выделение нового слота в глобальном пуле
    inline int allocate_node() {
        if (pool.next_free_index >= pool.gc_threshold) {
            collect_garbage();
            if (pool.next_free_index >= pool.max_size) {
                throw std::runtime_error("NodePool exhausted: all slots occupied by roots");
            }
        }
        size_t idx = pool.next_free_index;
        while (idx < pool.max_size && is_occupied(pool, idx)) {
            ++idx;
        }
        if (idx >= pool.max_size) {
            collect_garbage();
            return allocate_node();
        }
        pool.next_free_index = idx + 1;
        pool.refcount[idx] = 0;
        return static_cast<int>(idx);
    }

    // Сборка мусора: все корневые узлы заменяются на константы, остальные удаляются
    inline void collect_garbage() {
        NodePool new_pool;
        new_pool.max_size = pool.max_size;
        new_pool.update_gc_threshold();   // синхронизируем порог
        new_pool.nodes.assign(pool.max_size, Node());
        new_pool.refcount.assign(pool.max_size, 0);

        // Переносим корневые узлы как константы
        for (int idx = 0; idx < static_cast<int>(pool.nodes.size()); ++idx) {
            if (pool.refcount[idx] == 0) continue;
            Value v = evaluate(idx);
            int val_idx = new_pool.add_value(v);
            Node const_node(LazyOp::CONST, -1, -1, val_idx, 0,
                Interval(to_double(v)), compute_hash_const(v));
            new_pool.nodes[idx] = const_node;
            new_pool.refcount[idx] = pool.refcount[idx];
        }

        // Находим первый свободный слот для next_free_index
        new_pool.next_free_index = 0;
        while (new_pool.next_free_index < new_pool.max_size &&
            is_occupied(new_pool, new_pool.next_free_index)) {
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
    }

    inline void force_garbage_collect() {
        collect_garbage();
    }

    inline void reset_pool() {
        pool.~NodePool();
        new (&pool) NodePool();   // placement new
        // После сброса нужно обновить порог (конструктор NodePool инициализирует max_size = DEFAULT_MAX_SIZE)
        pool.update_gc_threshold();
    }

} // namespace delta::internal