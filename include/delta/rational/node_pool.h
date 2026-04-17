// node_pool.h
// Версия 2.0 – "глупые" фабрики: без упрощений, только интернирование и вычисление метаданных.
// Упрощения вынесены в simplify_impl (локальный упроститель TempNode).

#pragma once

#include "storage.h"
#include "interval.h"
#include "utils.h"

#include <absl/hash/hash.h>
#include <absl/container/flat_hash_map.h>

#include <memory>
#include <vector>
#include <cstdint>
#include <algorithm>

namespace delta::internal {
    void collect_garbage();   // forward declaration

    // ----------------------------------------------------------------------------
    // LazyOp для чистых узлов
    // ----------------------------------------------------------------------------
    enum class LazyOp : uint8_t {
        CONST,
        SUM,
        PRODUCT,
        NEG,
        RECIP,
        SQRT,
        EXP,
        LOG,
        SIN,
        COS,
        ACOS,
        PI,
        E,
        POW
    };

    // ----------------------------------------------------------------------------
    // Node (чистый узел)
    // ----------------------------------------------------------------------------
    struct Node {
        LazyOp op;
        int32_t depth;
        Interval approx;
        uint64_t hash;

        // Для SUM/PRODUCT
        std::unique_ptr<std::vector<int32_t>> children;

        // Для CONST
        int32_t value_idx;

        // Для унарных (и PI/E)
        int32_t child0;   // -1 если нет

        // Для POW
        int32_t child1;   // -1 если нет

        // Для операций с eps
        int32_t eps_idx;  // -1 если нет

        // Конструкторы
        Node();
        Node(LazyOp op, int32_t val_idx, int32_t depth, Interval approx, uint64_t hash);
        Node(LazyOp op, std::vector<int32_t>&& children, int32_t depth, Interval approx, uint64_t hash);
        Node(LazyOp op, int32_t child, int32_t eps_idx, int32_t depth, Interval approx, uint64_t hash);
        Node(LazyOp op, int32_t base, int32_t exp, int32_t eps_idx, int32_t depth, Interval approx, uint64_t hash);

        Node(const Node&) = delete;
        Node& operator=(const Node&) = delete;
        Node(Node&&) noexcept = default;
        Node& operator=(Node&&) noexcept = default;
    };

    // ----------------------------------------------------------------------------
    // Ключи кэширования – используют исходные (неупрощённые) детей
    // ----------------------------------------------------------------------------
    struct SumProductKey {
        LazyOp op;
        std::vector<int32_t> children;   // исходный порядок, без изменений
        bool operator==(const SumProductKey&) const = default;
    };

    struct SumProductKeyHash {
        size_t operator()(const SumProductKey& key) const {
            size_t h = static_cast<size_t>(key.op);
            for (int32_t c : key.children) h = absl::HashOf(h, c);
            return h;
        }
    };

    struct UnaryKey {
        LazyOp op;
        int32_t child;      // -1 для PI/E
        int32_t eps_idx;    // -1 если нет
        bool operator==(const UnaryKey&) const = default;
    };

    struct UnaryKeyHash {
        size_t operator()(const UnaryKey& k) const {
            return absl::HashOf(k.op, k.child, k.eps_idx);
        }
    };

    struct TernaryKey {
        LazyOp op;
        int32_t base;
        int32_t exponent;
        int32_t eps_idx;
        bool operator==(const TernaryKey&) const = default;
    };

    struct TernaryKeyHash {
        size_t operator()(const TernaryKey& k) const {
            return absl::HashOf(k.op, k.base, k.exponent, k.eps_idx);
        }
    };

    // ----------------------------------------------------------------------------
    // NodePool
    // ----------------------------------------------------------------------------
    struct NodePool {
        static constexpr size_t DEFAULT_MAX_SIZE = 1'000'000;
        size_t max_size = DEFAULT_MAX_SIZE;
        size_t gc_threshold = 0;
        std::vector<Node> nodes;
        std::vector<Value> values;
        std::vector<int> refcount;
        size_t next_free_index = 0;

        absl::flat_hash_map<Value, int> value_cache;
        absl::flat_hash_map<Value, int> constant_cache;
        absl::flat_hash_map<SumProductKey, int, SumProductKeyHash> sum_product_cache;
        absl::flat_hash_map<UnaryKey, int, UnaryKeyHash> unary_cache;
        absl::flat_hash_map<TernaryKey, int, TernaryKeyHash> ternary_cache;

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
                    nodes[i] = Node();
                    refcount[i] = 0;
                }
                idx = old_size;
            }

            next_free_index = idx + 1;
            refcount[idx] = 0;
            return static_cast<int>(idx);
        }

        bool is_occupied(const Node& node) const {
            if (node.op == LazyOp::SUM || node.op == LazyOp::PRODUCT) {
                return node.children && !node.children->empty();
            }
            return !(node.value_idx == -1 && node.child0 == -1 && node.child1 == -1);
        }
    };

    inline thread_local NodePool pool;

    // ----------------------------------------------------------------------------
    // Реализации Node конструкторов
    // ----------------------------------------------------------------------------
    inline Node::Node() : op(LazyOp::CONST), depth(0), approx(Interval()), hash(0),
        value_idx(-1), child0(-1), child1(-1), eps_idx(-1) {
    }

    inline Node::Node(LazyOp op, int32_t val_idx, int32_t depth, Interval approx, uint64_t hash)
        : op(op), depth(depth), approx(approx), hash(hash),
        value_idx(val_idx), child0(-1), child1(-1), eps_idx(-1) {
    }

    inline Node::Node(LazyOp op, std::vector<int32_t>&& children, int32_t depth, Interval approx, uint64_t hash)
        : op(op), depth(depth), approx(approx), hash(hash),
        children(std::make_unique<std::vector<int32_t>>(std::move(children))),
        value_idx(-1), child0(-1), child1(-1), eps_idx(-1) {
    }

    inline Node::Node(LazyOp op, int32_t child, int32_t eps_idx, int32_t depth, Interval approx, uint64_t hash)
        : op(op), depth(depth), approx(approx), hash(hash),
        child0(child), child1(-1), value_idx(-1), eps_idx(eps_idx) {
    }

    inline Node::Node(LazyOp op, int32_t base, int32_t exp, int32_t eps_idx, int32_t depth, Interval approx, uint64_t hash)
        : op(op), depth(depth), approx(approx), hash(hash),
        child0(base), child1(exp), value_idx(-1), eps_idx(eps_idx) {
    }

    // ----------------------------------------------------------------------------
    // Вспомогательные функции
    // ----------------------------------------------------------------------------
    inline uint64_t compute_hash_const(const Value& v) {
        return absl::Hash<Value>{}(v);
    }

    inline uint64_t combine_hash(LazyOp op, uint64_t h0, uint64_t h1 = 0, int64_t extra = 0) {
        uint64_t h = static_cast<uint64_t>(op);
        auto combine = [](uint64_t& seed, uint64_t v) {
            seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            };
        combine(h, h0);
        if (h1 != 0) combine(h, h1);
        if (extra != 0) combine(h, static_cast<uint64_t>(extra));
        return h;
    }

    inline Interval compute_interval(LazyOp op, const Interval& a, const Interval& b = Interval()) {
        using std::sqrt, std::exp, std::log, std::sin, std::cos, std::acos;
        switch (op) {
        case LazyOp::SUM:      return a + b;
        case LazyOp::PRODUCT:  return a * b;
        case LazyOp::NEG:      return -a;
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
        case LazyOp::PI: {
            constexpr double pi = 3.14159265358979323846;
            return Interval(pi);
        }
        case LazyOp::E: {
            constexpr double e = 2.71828182845904523536;
            return Interval(e);
        }
        case LazyOp::POW: {
            double lo = std::pow(a.lower(), b.lower());
            double hi = std::pow(a.upper(), b.upper());
            return Interval(lo, hi);
        }
        default: return Interval();
        }
    }

    // ----------------------------------------------------------------------------
    // Фабричные функции чистых узлов (без упрощений – только интернирование)
    // ----------------------------------------------------------------------------

    // add_const – нормализует значение и интернирует константу
    inline int add_const(const Value& v) {
        pool.ensure_initialized();
        auto it = pool.constant_cache.find(v);
        if (it != pool.constant_cache.end()) return it->second;
        int idx = pool.allocate_node();
        int val_idx = pool.add_value(v);
        Node node(LazyOp::CONST, val_idx, 0, Interval(to_double(v)), compute_hash_const(v));
        pool.nodes[idx] = std::move(node);
        pool.refcount[idx] = 0;
        pool.constant_cache[v] = idx;
        return idx;
    }

    // make_sum_node – создаёт узел SUM с переданными детьми (без фильтрации, сортировки, свёртки)
    inline int make_sum_node(std::vector<int32_t> children) {
        pool.ensure_initialized();
        // Ключ использует исходные дети (в том же порядке)
        SumProductKey key{ LazyOp::SUM, children };
        auto it = pool.sum_product_cache.find(key);
        if (it != pool.sum_product_cache.end()) return it->second;

        // Вычисляем метаданные по переданным детям
        int32_t depth = 0;
        Interval approx = Interval::zero();
        uint64_t hash = static_cast<uint64_t>(LazyOp::SUM);
        for (int32_t child : children) {
            depth = std::max(depth, pool.nodes[child].depth + 1);
            approx = approx + pool.nodes[child].approx;
            hash = combine_hash(LazyOp::SUM, hash, pool.nodes[child].hash);
        }

        int idx = pool.allocate_node();
        Node node(LazyOp::SUM, std::move(children), depth, approx, hash);
        pool.nodes[idx] = std::move(node);
        pool.refcount[idx] = 0;
        pool.sum_product_cache[std::move(key)] = idx;
        return idx;
    }

    // make_product_node – создаёт узел PRODUCT с переданными детьми (без фильтрации, сортировки, свёртки)
    inline int make_product_node(std::vector<int32_t> children) {
        pool.ensure_initialized();
        SumProductKey key{ LazyOp::PRODUCT, children };
        auto it = pool.sum_product_cache.find(key);
        if (it != pool.sum_product_cache.end()) return it->second;

        int32_t depth = 0;
        Interval approx = Interval::one();
        uint64_t hash = static_cast<uint64_t>(LazyOp::PRODUCT);
        for (int32_t child : children) {
            depth = std::max(depth, pool.nodes[child].depth + 1);
            approx = approx * pool.nodes[child].approx;
            hash = combine_hash(LazyOp::PRODUCT, hash, pool.nodes[child].hash);
        }

        int idx = pool.allocate_node();
        Node node(LazyOp::PRODUCT, std::move(children), depth, approx, hash);
        pool.nodes[idx] = std::move(node);
        pool.refcount[idx] = 0;
        pool.sum_product_cache[std::move(key)] = idx;
        return idx;
    }

    // get_unary_node – создаёт узел для унарной операции (без алгебраических упрощений)
    inline int get_unary_node(LazyOp op, int child, int eps_idx) {
        pool.ensure_initialized();
        UnaryKey key{ op, child, eps_idx };
        auto it = pool.unary_cache.find(key);
        if (it != pool.unary_cache.end()) return it->second;

        int idx = pool.allocate_node();
        int32_t depth;
        Interval approx;
        uint64_t hash;

        if (child == -1) { // PI, E
            depth = 0;
            approx = compute_interval(op, Interval());
            hash = combine_hash(op, 0, eps_idx);
        }
        else {
            depth = 1 + pool.nodes[child].depth;
            approx = compute_interval(op, pool.nodes[child].approx);
            hash = combine_hash(op, pool.nodes[child].hash, 0, eps_idx);
        }

        Node node(op, child, eps_idx, depth, approx, hash);
        pool.nodes[idx] = std::move(node);
        pool.refcount[idx] = 0;
        pool.unary_cache[key] = idx;
        return idx;
    }

    // get_pow_node – создаёт узел POW (без упрощений)
    inline int get_pow_node(int base, int exponent, int eps_idx) {
        pool.ensure_initialized();
        TernaryKey key{ LazyOp::POW, base, exponent, eps_idx };
        auto it = pool.ternary_cache.find(key);
        if (it != pool.ternary_cache.end()) return it->second;

        int idx = pool.allocate_node();
        int32_t depth = 1 + std::max(pool.nodes[base].depth, pool.nodes[exponent].depth);
        Interval approx = compute_interval(LazyOp::POW,
            pool.nodes[base].approx,
            pool.nodes[exponent].approx);
        uint64_t hash = combine_hash(LazyOp::POW,
            pool.nodes[base].hash,
            pool.nodes[exponent].hash,
            eps_idx);

        Node node(LazyOp::POW, base, exponent, eps_idx, depth, approx, hash);
        pool.nodes[idx] = std::move(node);
        pool.refcount[idx] = 0;
        pool.ternary_cache[key] = idx;
        return idx;
    }

    // ----------------------------------------------------------------------------
    // Управление ссылками
    // ----------------------------------------------------------------------------
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
                if ((node.op == LazyOp::SUM || node.op == LazyOp::PRODUCT) && node.children) {
                    for (int32_t child : *node.children) {
                        decrement_ref(child);
                    }
                }
                else {
                    if (node.child0 != -1) decrement_ref(node.child0);
                    if (node.child1 != -1) decrement_ref(node.child1);
                }
            }
        }
    }

    //// ----------------------------------------------------------------------------
    //// Структурное сравнение (реализация в simplify.h, но здесь не используется)
    //// ----------------------------------------------------------------------------
    //inline bool structurally_equal(int root_a, int root_b) {
    //    // Заглушка – в новом пайплайне не используется, оставлена для совместимости.
    //    return root_a == root_b;
    //}

} // namespace delta::internal

// Подключаем gc.h в конце для разрешения циклической зависимости
#include "gc.h"