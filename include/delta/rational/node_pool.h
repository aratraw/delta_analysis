// node_pool.h
#pragma once

#include "storage.h"
#include "interval.h"
#include "utils.h"
#include "evaluation_core.h"

#include <absl/hash/hash.h>
#include <absl/numeric/int128.h>
#include <absl/container/flat_hash_map.h>

#include <vector>
#include <cstdint>
#include <optional>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

namespace delta::internal {

    // ----------------------------------------------------------------------------
    // AbslHashValue для SmallStorage, BigStorage и Value (в том же пространстве имён)
    // ----------------------------------------------------------------------------
    template <typename H>
    H AbslHashValue(H h, const SmallStorage& s) {
        SmallStorage norm = s;
        norm.normalize();
        return H::combine(std::move(h), norm.num, norm.den);
    }

    template <typename H>
    H AbslHashValue(H h, const BigStorage& b) {
        return H::combine(std::move(h), b.num(), b.den());
    }

    template <typename H>
    H AbslHashValue(H h, const Value& v) {
        return std::visit([&](const auto& val) { return H::combine(std::move(h), val); }, v);
    }

    // ----------------------------------------------------------------------------
    // LazyOp – виды узлов
    // ----------------------------------------------------------------------------
    enum class LazyOp : uint8_t {
        CONST, ADD, MUL, NEG, RECIP,
        SQRT, EXP, LOG, SIN, COS, ACOS,
        PI, E
    };

    // ----------------------------------------------------------------------------
    // Node – узел ленивого дерева
    // ----------------------------------------------------------------------------
    struct Node {
        LazyOp op;
        int32_t child0;
        int32_t child1;
        int32_t value_idx;
        int32_t depth;
        Interval approx;
        uint64_t hash;

        Node(LazyOp op, int32_t c0, int32_t c1, int32_t v_idx, int32_t depth,
            Interval approx, uint64_t hash)
            : op(op), child0(c0), child1(c1), value_idx(v_idx),
            depth(depth), approx(approx), hash(hash) {
        }
    };

    // ----------------------------------------------------------------------------
    // Ключи для кэшей интернирования
    // ----------------------------------------------------------------------------
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

    struct BinaryKeyHash {
        size_t operator()(const BinaryKey& k) const {
            size_t h = 0;
            h ^= static_cast<size_t>(k.op) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(k.left) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(k.right) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    struct UnaryKeyHash {
        size_t operator()(const UnaryKey& k) const {
            size_t h = 0;
            h ^= static_cast<size_t>(k.op) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(k.child) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= static_cast<size_t>(k.value_idx) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    // ----------------------------------------------------------------------------
    // NodePool – thread‑local пул узлов и значений
    // ----------------------------------------------------------------------------
    struct NodePool {
        std::vector<Node> nodes;
        std::vector<Value> values;
        absl::flat_hash_map<Value, int> value_cache;
        absl::flat_hash_map<BinaryKey, int, BinaryKeyHash> binary_cache;
        absl::flat_hash_map<UnaryKey, int, UnaryKeyHash> unary_cache;
    };

    inline thread_local NodePool pool;

    // ----------------------------------------------------------------------------
    // Вспомогательные функции для работы с пулом (объявления)
    // ----------------------------------------------------------------------------
    int add_value(const Value& v);
    int add_const(const Value& v);
    int get_binary_node(LazyOp op, int left, int right);
    int get_unary_node(LazyOp op, int child, int value_idx = -1);
    Interval compute_interval(LazyOp op, const Interval& a, const Interval& b = Interval());
    uint64_t combine_hash(LazyOp op, uint64_t h0, uint64_t h1 = 0, int64_t extra = 0);
    uint64_t compute_hash_const(const Value& v);

    // ----------------------------------------------------------------------------
    // Реализации (inline)
    // ----------------------------------------------------------------------------

    inline int add_value(const Value& v) {
        // Нормализуем копию для поиска и вставки
        Value normalized = v;
        if (auto* s = std::get_if<SmallStorage>(&normalized)) {
            s->normalize();
        }
        // BigStorage уже нормализован в конструкторе
        auto it = pool.value_cache.find(normalized);
        if (it != pool.value_cache.end()) return it->second;
        int idx = static_cast<int>(pool.values.size());
        pool.values.push_back(normalized);
        pool.value_cache[normalized] = idx;
        return idx;
    }

    inline int add_const(const Value& v) {
        int val_idx = add_value(v);
        Interval approx(to_double(v));
        uint64_t h = compute_hash_const(v);
        Node node(LazyOp::CONST, -1, -1, val_idx, 0, approx, h);
        int idx = static_cast<int>(pool.nodes.size());
        pool.nodes.push_back(node);
        return idx;
    }

    inline int get_binary_node(LazyOp op, int left, int right) {
        //if (op == LazyOp::ADD || op == LazyOp::MUL) {
        //    if (pool.nodes[left].hash > pool.nodes[right].hash)
        //        std::swap(left, right);
        //    else if (pool.nodes[left].hash == pool.nodes[right].hash && left > right)
        //        std::swap(left, right);
        //}
        BinaryKey key{ op, left, right };
        auto it = pool.binary_cache.find(key);
        if (it != pool.binary_cache.end()) return it->second;

        int depth = 1 + std::max(pool.nodes[left].depth, pool.nodes[right].depth);
        Interval approx = compute_interval(op, pool.nodes[left].approx, pool.nodes[right].approx);
        uint64_t hash = combine_hash(op, pool.nodes[left].hash, pool.nodes[right].hash);
        Node node(op, left, right, -1, depth, approx, hash);
        int idx = static_cast<int>(pool.nodes.size());
        pool.nodes.push_back(node);
        pool.binary_cache[key] = idx;
        return idx;
    }

    inline int get_unary_node(LazyOp op, int child, int value_idx) {
        UnaryKey key{ op, child, value_idx };
        auto it = pool.unary_cache.find(key);
        if (it != pool.unary_cache.end()) return it->second;

        int depth;
        Interval approx;
        uint64_t hash;

        if (child == -1) {
            // Узел без детей (например, PI, E)
            depth = 0;
            // Вычисляем интервал для константы без аргумента
            approx = compute_interval(op, Interval());
            // Хеш: только операция и value_idx (дети отсутствуют)
            hash = combine_hash(op, 0, value_idx);
        }
        else {
            depth = 1 + pool.nodes[child].depth;
            approx = compute_interval(op, pool.nodes[child].approx);
            hash = combine_hash(op, pool.nodes[child].hash, value_idx);
        }

        Node node(op, child, -1, value_idx, depth, approx, hash);
        int idx = static_cast<int>(pool.nodes.size());
        pool.nodes.push_back(node);
        pool.unary_cache[key] = idx;
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

} // namespace delta::internal