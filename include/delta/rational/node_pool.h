// node_pool.h (адаптирован под новый tagged union Value, флаг small_reduced в Value)
#pragma once

#include "storage.h"
#include "interval.h"
#include "utils.h"
#include "evaluation_core.h"

#include <absl/hash/hash.h>
#include <absl/numeric/int128.h>
#include <absl/container/flat_hash_map.h>

#include <memory>          // для std::unique_ptr
#include <algorithm>       // для std::sort, std::swap, std::max
#include <vector>
#include <cstdint>
#include <cmath>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

namespace delta::internal {

    // Forward declaration
    void collect_garbage();

    // ----------------------------------------------------------------------------
    // AbslHashValue для нового Value (прямой доступ к полям union)
    // ----------------------------------------------------------------------------
    template <typename H>
    H AbslHashValue(H h, const SmallStorage& s) {
        SmallStorage norm = s;
        bool red = false;
        norm.normalize(red);
        return H::combine(std::move(h), norm.num, norm.den);
    }

    template <typename H>
    H AbslHashValue(H h, const BigStorage& b) {
        // Для хэширования BigStorage используем numerator и denominator (копирование допустимо, т.к. не в горячем пути)
        return H::combine(std::move(h), b.numerator(), b.denominator());
    }

    template <typename H>
    H AbslHashValue(H h, const Value& v) {
        if (v.tag == ValueType::Small) {
            return AbslHashValue(std::move(h), v.storage.small);
        }
        else if (v.tag == ValueType::Big) {
            return AbslHashValue(std::move(h), v.storage.big);
        }
        else {
            // Ленивый индекс – хэшируем сам индекс
            return H::combine(std::move(h), v.storage.lazy);
        }
    }

    enum class LazyOp : uint8_t {
        CONST, ADD, MUL, NEG, RECIP,
        SQRT, EXP, LOG, SIN, COS, ACOS,
        PI, E, POW,
        SUM   // N-арный оператор сложения
    };

    struct Node {
        LazyOp op;
        int32_t child0;
        int32_t child1;
        int32_t value_idx;
        int32_t depth;
        Interval approx;
        uint64_t hash;
        std::unique_ptr<std::vector<int>> sum_children;   // только для SUM

        // Конструктор по умолчанию
        Node() : op(LazyOp::CONST), child0(-1), child1(-1), value_idx(-1), depth(0), approx(Interval()), hash(0), sum_children(nullptr) {}

        // Конструктор для бинарных/унарных/константных узлов
        Node(LazyOp op, int32_t c0, int32_t c1, int32_t v_idx, int32_t depth,
            Interval approx, uint64_t hash)
            : op(op), child0(c0), child1(c1), value_idx(v_idx),
            depth(depth), approx(approx), hash(hash), sum_children(nullptr) {
        }

        // Конструктор для SUM
        Node(LazyOp op, std::vector<int>&& children, int32_t depth, Interval approx, uint64_t hash)
            : op(op), child0(-1), child1(-1), value_idx(-1),
            depth(depth), approx(approx), hash(hash),
            sum_children(std::make_unique<std::vector<int>>(std::move(children))) {
        }

        // Запрет копирования (из-за unique_ptr)
        Node(const Node&) = delete;
        Node& operator=(const Node&) = delete;

        // Move-конструктор
        Node(Node&& other) noexcept
            : op(other.op), child0(other.child0), child1(other.child1),
            value_idx(other.value_idx), depth(other.depth),
            approx(other.approx), hash(other.hash),
            sum_children(std::move(other.sum_children)) {
            other.child0 = other.child1 = -1;
            other.value_idx = -1;
            other.depth = 0;
            other.hash = 0;
            other.op = LazyOp::CONST;   // необязательно, но безопасно
        }

        // Move-присваивание
        Node& operator=(Node&& other) noexcept {
            if (this != &other) {
                op = other.op;
                child0 = other.child0;
                child1 = other.child1;
                value_idx = other.value_idx;
                depth = other.depth;
                approx = other.approx;
                hash = other.hash;
                sum_children = std::move(other.sum_children);
                other.child0 = other.child1 = -1;
                other.value_idx = -1;
                other.depth = 0;
                other.hash = 0;
                other.op = LazyOp::CONST;
            }
            return *this;
        }

        ~Node() = default;
    };

    struct BinaryKey {
        LazyOp op;
        int32_t left;
        int32_t right;
        bool operator==(const BinaryKey&) const = default;
    };

    struct UnaryKey {
        LazyOp op;
        int32_t child;
        int32_t value_idx;
        bool operator==(const UnaryKey&) const = default;
    };

    struct TernaryKey {
        LazyOp op;
        int32_t left;
        int32_t right;
        int32_t value_idx;
        bool operator==(const TernaryKey&) const = default;
    };

    struct BinaryKeyHash {
        size_t operator()(const BinaryKey& k) const {
            return absl::HashOf(k.op, k.left, k.right);
        }
    };

    struct UnaryKeyHash {
        size_t operator()(const UnaryKey& k) const {
            return absl::HashOf(k.op, k.child, k.value_idx);
        }
    };

    struct TernaryKeyHash {
        size_t operator()(const TernaryKey& k) const {
            return absl::HashOf(k.op, k.left, k.right, k.value_idx);
        }
    };

    // Ключ и хэш для SUM
    struct SumKey {
        std::vector<int> children; // уже отсортированы, без нулей
        bool operator==(const SumKey& other) const {
            return children == other.children;
        }
    };

    struct SumKeyHash {
        size_t operator()(const SumKey& key) const {
            size_t h = key.children.size();
            for (int c : key.children) {
                h = absl::HashOf(h, c);
            }
            return h;
        }
    };

    struct NodePool {
        static constexpr size_t DEFAULT_MAX_SIZE = 1'000'000;
        size_t max_size = DEFAULT_MAX_SIZE;
        size_t gc_threshold = 0;
        std::vector<Node> nodes;
        std::vector<Value> values;
        std::vector<int> refcount;
        size_t next_free_index = 0;

        // Кэши
        absl::flat_hash_map<Value, int> value_cache;
        absl::flat_hash_map<Value, int> constant_cache;
        absl::flat_hash_map<BinaryKey, int, BinaryKeyHash> binary_cache;
        absl::flat_hash_map<UnaryKey, int, UnaryKeyHash> unary_cache;
        absl::flat_hash_map<TernaryKey, int, TernaryKeyHash> ternary_cache;
        absl::flat_hash_map<SumKey, int, SumKeyHash> sum_cache;   // кэш для SUM

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
            Value normalized = v;
            if (normalized.tag == ValueType::Small && !normalized.small_reduced) {
                normalized.normalize();
            }
            auto it = value_cache.find(normalized);
            if (it != value_cache.end()) return it->second;
            int idx = static_cast<int>(values.size());
            values.push_back(normalized);
            value_cache[normalized] = idx;
            return idx;
        }

        int allocate_node() {
            ensure_initialized();

            if (next_free_index >= gc_threshold) {
                collect_garbage();
                if (next_free_index >= max_size) {
                    throw std::runtime_error("NodePool exhausted: all slots occupied by roots");
                }
            }

            size_t idx = next_free_index;
            while (idx < nodes.size() && is_occupied(nodes[idx])) {
                ++idx;
            }

            if (idx >= nodes.size()) {
                size_t old_size = nodes.size();
                size_t new_size = old_size + 4096;
                if (new_size > max_size) new_size = max_size;
                if (new_size <= old_size) {
                    throw std::runtime_error("NodePool exhausted: cannot expand beyond max_size");
                }
                nodes.resize(new_size);
                refcount.resize(new_size, 0);
                for (size_t i = old_size; i < new_size; ++i) {
                    nodes[i] = Node(LazyOp::CONST, -1, -1, -1, 0, Interval(), 0);
                    refcount[i] = 0;
                }
                idx = old_size;
            }

            next_free_index = idx + 1;
            refcount[idx] = 0;
            return static_cast<int>(idx);
        }

        // Вспомогательная функция для проверки занятости узла
        bool is_occupied(const Node& node) const {
            if (node.op == LazyOp::SUM) {
                return node.sum_children && !node.sum_children->empty();
            }
            return !(node.value_idx == -1 && node.child0 == -1 && node.child1 == -1);
        }
    };

    inline thread_local NodePool pool;


    // ----------------------------------------------------------------------------
    // Объявления функций
    // ----------------------------------------------------------------------------
    int add_const(const Value& v);
    int get_binary_node(LazyOp op, int left, int right);
    int get_binary_node(LazyOp op, int left, int right, int value_idx);
    int get_pow_node(int left, int right, int value_idx);
    int get_unary_node(LazyOp op, int child, int value_idx = -1);
    int make_sum_node(std::vector<int> raw_children);
    Interval compute_interval(LazyOp op, const Interval& a, const Interval& b = Interval());
    uint64_t combine_hash(LazyOp op, uint64_t h0, uint64_t h1 = 0, int64_t extra = 0);
    uint64_t compute_hash_const(const Value& v);

    inline void increment_ref(int idx);
    inline void decrement_ref(int idx);
    inline void set_pool_max_size(size_t new_size);
    inline void force_garbage_collect();

    // ----------------------------------------------------------------------------
    // Реализации
    // ----------------------------------------------------------------------------
    inline int add_const(const Value& v) {
        pool.ensure_initialized();
        auto it = pool.constant_cache.find(v);
        if (it != pool.constant_cache.end()) return it->second;
        int idx = pool.allocate_node();
        int val_idx = pool.add_value(v);
        Node node(LazyOp::CONST, -1, -1, val_idx, 0, Interval(to_double(v)), compute_hash_const(v));
        pool.nodes[idx] = std::move(node);
        pool.refcount[idx] = 0;
        pool.constant_cache[v] = idx;
        return idx;
    }

    inline int get_binary_node(LazyOp op, int left, int right) {
        pool.ensure_initialized();
        if (op == LazyOp::ADD || op == LazyOp::MUL) {
            if (pool.nodes[left].hash > pool.nodes[right].hash)
                std::swap(left, right);
            else if (pool.nodes[left].hash == pool.nodes[right].hash && left > right)
                std::swap(left, right);
        }
        BinaryKey key{ op, left, right };
        auto it = pool.binary_cache.find(key);
        if (it != pool.binary_cache.end()) return it->second;
        int idx = pool.allocate_node();
        int depth = 1 + std::max(pool.nodes[left].depth, pool.nodes[right].depth);
        Interval approx = compute_interval(op, pool.nodes[left].approx, pool.nodes[right].approx);
        uint64_t hash = combine_hash(op, pool.nodes[left].hash, pool.nodes[right].hash);
        Node node(op, left, right, -1, depth, approx, hash);
        pool.nodes[idx] = std::move(node);
        pool.refcount[idx] = 0;
        pool.binary_cache[key] = idx;
        return idx;
    }

    inline int get_binary_node(LazyOp op, int left, int right, int value_idx) {
        pool.ensure_initialized();
        if (op == LazyOp::ADD || op == LazyOp::MUL) {
            if (pool.nodes[left].hash > pool.nodes[right].hash)
                std::swap(left, right);
            else if (pool.nodes[left].hash == pool.nodes[right].hash && left > right)
                std::swap(left, right);
        }
        BinaryKey key{ op, left, right };
        auto it = pool.binary_cache.find(key);
        if (it != pool.binary_cache.end()) return it->second;
        int idx = pool.allocate_node();
        int depth = 1 + std::max(pool.nodes[left].depth, pool.nodes[right].depth);
        Interval approx = compute_interval(op, pool.nodes[left].approx, pool.nodes[right].approx);
        uint64_t hash = combine_hash(op, pool.nodes[left].hash, pool.nodes[right].hash, value_idx);
        Node node(op, left, right, value_idx, depth, approx, hash);
        pool.nodes[idx] = std::move(node);
        pool.refcount[idx] = 0;
        pool.binary_cache[key] = idx;
        return idx;
    }

    inline int get_pow_node(int left, int right, int value_idx) {
        pool.ensure_initialized();
        TernaryKey key{ LazyOp::POW, left, right, value_idx };
        auto it = pool.ternary_cache.find(key);
        if (it != pool.ternary_cache.end()) return it->second;
        int idx = pool.allocate_node();
        int depth = 1 + std::max(pool.nodes[left].depth, pool.nodes[right].depth);
        Interval approx = compute_interval(LazyOp::POW, pool.nodes[left].approx, pool.nodes[right].approx);
        uint64_t hash = combine_hash(LazyOp::POW, pool.nodes[left].hash, pool.nodes[right].hash, value_idx);
        Node node(LazyOp::POW, left, right, value_idx, depth, approx, hash);
        pool.nodes[idx] = std::move(node);
        pool.refcount[idx] = 0;
        pool.ternary_cache[key] = idx;
        return idx;
    }

    inline int get_unary_node(LazyOp op, int child, int value_idx) {
        pool.ensure_initialized();
        UnaryKey key{ op, child, value_idx };
        auto it = pool.unary_cache.find(key);
        if (it != pool.unary_cache.end()) return it->second;
        int idx = pool.allocate_node();
        int depth;
        Interval approx;
        uint64_t hash;
        if (child == -1) {
            depth = 0;
            approx = compute_interval(op, Interval());
            hash = combine_hash(op, 0, value_idx);
        }
        else {
            depth = 1 + pool.nodes[child].depth;
            approx = compute_interval(op, pool.nodes[child].approx);
            hash = combine_hash(op, pool.nodes[child].hash, value_idx);
        }
        Node node(op, child, -1, value_idx, depth, approx, hash);
        pool.nodes[idx] = std::move(node);
        pool.refcount[idx] = 0;
        pool.unary_cache[key] = idx;
        return idx;
    }

    // ----------------------------------------------------------------------------
    // make_sum_node – создание N-арного узла SUM (каноническое создание)
    // ----------------------------------------------------------------------------
    inline int make_sum_node(std::vector<int> raw_children) {
        pool.ensure_initialized();

        // 1. Фильтрация нулевых констант
        std::vector<int> filtered;
        filtered.reserve(raw_children.size());
        for (int child : raw_children) {
            const Node& node = pool.nodes[child];
            if (node.op == LazyOp::CONST && is_zero(pool.values[node.value_idx]))
                continue;
            filtered.push_back(child);
        }

        // 2. Дегенерация
        if (filtered.empty()) {
            return add_const(Value(SmallStorage(0)));
        }
        if (filtered.size() == 1) {
            return filtered[0];
        }
        // ДЕГЕНЕРАЦИЯ SUM ОТ ДВУХ ОПЕРАНДОВ ДО БИНАРНОГО ADD - ТОЛЬКО В SIMPLIFY

        // 3. Канонизация: сортировка по (hash, индекс)
        std::sort(filtered.begin(), filtered.end(),
            [&](int a, int b) {
                uint64_t ha = pool.nodes[a].hash;
                uint64_t hb = pool.nodes[b].hash;
                if (ha != hb) return ha < hb;
                return a < b;
            });

        // 4. Проверка кэша
        SumKey key{ filtered };
        auto it = pool.sum_cache.find(key);
        if (it != pool.sum_cache.end()) {
            return it->second;
        }

        // 5. Расчёт метаданных
        int depth = 0;
        Interval approx = Interval::zero();
        uint64_t hash = static_cast<uint64_t>(LazyOp::SUM);
        for (int child : filtered) {
            depth = std::max(depth, pool.nodes[child].depth + 1);
            approx = approx + pool.nodes[child].approx;
            hash = combine_hash(LazyOp::SUM, hash, pool.nodes[child].hash);
        }

        // 6. Выделение узла
        int idx = pool.allocate_node();
        Node node(LazyOp::SUM, std::move(filtered), depth, approx, hash);
        pool.nodes[idx] = std::move(node);
        pool.refcount[idx] = 0;
        pool.sum_cache[std::move(key)] = idx;
        return idx;
    }

    inline Interval compute_interval(LazyOp op, const Interval& a, const Interval& b) {
        using std::sqrt, std::exp, std::log, std::sin, std::cos, std::acos;
        switch (op) {
        case LazyOp::ADD:  return a + b;
        case LazyOp::MUL:  return a * b;
        case LazyOp::NEG:  return -a;
        case LazyOp::RECIP: {
            if (a.lower() <= 0.0 && a.upper() >= 0.0)
                return Interval(-std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity());
            double lo = 1.0 / a.upper();
            double hi = 1.0 / a.lower();
            if (lo > hi) std::swap(lo, hi);
            return Interval(lo, hi);
        }
        case LazyOp::SQRT: {
            if (a.upper() < 0) return Interval();
            double lo = a.lower() < 0 ? 0.0 : sqrt(a.lower());
            double hi = sqrt(a.upper());
            return Interval(lo, hi);
        }
        case LazyOp::EXP: {
            double lo = exp(a.lower());
            double hi = exp(a.upper());
            return Interval(lo, hi);
        }
        case LazyOp::LOG: {
            if (a.upper() <= 0) return Interval(-std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity());
            double lo = log(a.lower());
            double hi = log(a.upper());
            return Interval(lo, hi);
        }
        case LazyOp::SIN:   return Interval(-1.0, 1.0);
        case LazyOp::COS:   return Interval(-1.0, 1.0);
        case LazyOp::ACOS: {
            if (a.lower() < -1 || a.upper() > 1)
                return Interval(-std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity());
            double lo = acos(a.upper());
            double hi = acos(a.lower());
            return Interval(lo, hi);
        }
        case LazyOp::PI:   return Interval(M_PI);
        case LazyOp::E:    return Interval(M_E);
        case LazyOp::POW: {
            double lo = std::pow(a.lower(), b.lower());
            double hi = std::pow(a.upper(), b.upper());
            return Interval(lo, hi);
        }
        case LazyOp::SUM:
            // Для SUM интервал вычисляется при создании узла; здесь он не используется.
            return Interval();
        default: return Interval();
        }
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

    inline uint64_t compute_hash_const(const Value& v) {
        return absl::Hash<Value>{}(v);
    }

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
                if (node.op == LazyOp::SUM && node.sum_children) {
                    for (int child : *node.sum_children) {
                        decrement_ref(child);
                    }
                }
            }
        }
    }

    // ----------------------------------------------------------------------------
    // Функции для мутации SUM (COW-семантика)
    // ----------------------------------------------------------------------------
    inline bool can_mutate_sum(int idx) {
        if (idx < 0 || static_cast<size_t>(idx) >= pool.nodes.size()) return false;
        const Node& node = pool.nodes[idx];
        return node.op == LazyOp::SUM && pool.refcount[idx] == 1 && node.sum_children != nullptr;
    }

    //if something breaks mysteriously - you should definitely look closely if you have tampered with this function
    // check if you invalidate cache when needed, if you even use this cache. 
    inline void append_to_sum(int idx, int child_idx) {
        assert(can_mutate_sum(idx));
        Node& node = pool.nodes[idx];
        auto& children = *node.sum_children;

        // Просто добавляем в конец (без сортировки, без кэша)
        children.push_back(child_idx);
        increment_ref(child_idx);

        // Инкрементальное обновление метаданных
        node.depth = std::max(node.depth, pool.nodes[child_idx].depth + 1);
        node.approx = node.approx + pool.nodes[child_idx].approx;

        // Узел стал неканоническим
        node.hash = 0;
    }
} // namespace delta::internal

#include "gc.h"