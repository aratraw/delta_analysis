// node_pool.h
// Версия 3.2 – удалены поля approx и depth из Node, удалены вспомогательные функции интервалов.
// ----------------------------------------------------------------------------
// Модификация для реестра чистых объектов: reset_pool() инвалидирует все чистые LazyRational.

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
    // Хэшер и компаратор для Value (явные, используют absl::Hash)
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
    // Ключи кэширования
    // ------------------------------------------------------------------------
    struct SumProductKey {
        LazyOp op;
        std::vector<Value> leaf_values;
        absl::InlinedVector<int32_t, 2> children;

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
        LazyOp op;
        absl::InlinedVector<int32_t, 2> children;
        int32_t eps_idx;

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
    // NodePool
    // ------------------------------------------------------------------------
    struct NodePool {
        size_t max_size = DEFAULT_POOL_MAX_SIZE;
        size_t gc_threshold = 0;
        std::vector<Node> nodes;
        std::vector<Value> values;
        std::vector<int> refcount;
        size_t next_free_index = 0;

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

            // GC запускается только если не запрещён флагом gc_disabled
            if (!internal::gc_disabled && next_free_index >= gc_threshold) {
                collect_garbage();
                if (next_free_index >= max_size) {
                    throw std::runtime_error("NodePool exhausted: all slots occupied by roots");
                }
            }

            // Поиск свободного слота, начиная с next_free_index
            size_t idx = next_free_index;
            while (idx < nodes.size() && is_occupied(nodes[idx])) {
                ++idx;
            }

            // Если свободного слота не нашлось, расширяем вектор
            if (idx >= nodes.size()) {
                size_t old_size = nodes.size();
                size_t new_size = old_size + 4096;

                // Если GC не отключён, не превышаем max_size
                if (!internal::gc_disabled && new_size > max_size) {
                    new_size = max_size;
                }
                // Если new_size всё ещё <= old_size, значит max_size уже достигнут и расширяться некуда
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

            // Увеличиваем указатель на следующий свободный слот
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
    // Реализации вспомогательных функций из node_types.h
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
    // Конструкторы Node (реализация) – без depth и approx
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
    // Фабричные функции чистых узлов
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
    // Управление ссылками
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
            }
        }
    }

} // namespace delta::internal

// ----------------------------------------------------------------------------
// Подключаем evaluate_impl.h и реализуем evaluate для чистого дерева и GC
// ----------------------------------------------------------------------------
#include "evaluate_impl.h"

namespace delta::internal {

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

    inline void collect_garbage() {
        // 1. Получаем список всех чистых объектов (корней)
        auto clean_objects = get_clean_objects_snapshot();
        if (clean_objects.empty()) {
            pool = NodePool();
            pool.update_gc_threshold();
            return;
        }

        // 2. Собираем корневые индексы и максимальный из них
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

        // 3. Создаём новый пул с размером, достаточным для max_root_index+1
        NodePool new_pool;
        new_pool.max_size = pool.max_size;
        new_pool.update_gc_threshold();
        size_t new_size = max_root_index + 1;
        new_pool.nodes.resize(new_size);
        new_pool.refcount.assign(new_size, 0);
        new_pool.values = pool.values;
        new_pool.value_cache = pool.value_cache;

        // 4. Для каждого корня вычисляем значение и создаём константу
        for (int root_idx : root_indices) {
            Value v = evaluate(root_idx);
            int val_idx = new_pool.add_value(v);
            Node const_node(LazyOp::CONST, val_idx, compute_hash_const(v));
            new_pool.nodes[root_idx] = std::move(const_node);
            new_pool.refcount[root_idx] = 1;   // один владелец – сам LazyRational
        }

        // 5. Находим первый свободный слот для next_free_index
        new_pool.next_free_index = 0;
        while (new_pool.next_free_index < new_size &&
            new_pool.is_occupied(new_pool.nodes[new_pool.next_free_index])) {
            ++new_pool.next_free_index;
        }

        // 6. Заменяем старый пул
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
    // reset_pool – полный сброс пула с инвалидацией всех чистых объектов
    // -------------------------------------------------------------------------
    inline void reset_pool() {
        // 1. Получить снимок всех чистых объектов (пока старый пул жив)
        auto clean_objects = get_clean_objects_snapshot();

        // 2. Инвалидировать каждый чистый объект
        for (delta::LazyRational* obj : clean_objects) {
            // Уменьшить счётчик ссылок на узел в старом пуле
            decrement_ref(obj->clean_index_);
            // Пересоздать объект как грязный ноль (конструктор по умолчанию не регистрирует)
            obj->~LazyRational();
            new (obj) delta::LazyRational();
        }

        // 3. Очистить глобальный пул и кэши
        pool = NodePool();
        pool.update_gc_threshold();
        reset_pi_cache();

        // 4. Очистить реестр (все объекты уже удалены, но для гарантии)
        clear_clean_registry();
    }

} // namespace delta::internal