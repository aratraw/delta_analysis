// evaluate_impl.h
#pragma once

#include "node_pool.h"
#include "evaluation_core.h"
#include "utils.h"

#include <stack>
#include <vector>
#include <optional>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

namespace delta::internal {

    // ============================================================================
    // Batching configuration
    // ============================================================================
    inline constexpr size_t BATCH_SIZE = 32;   // размер батча для leapfrog

    // ============================================================================
    // Forward declarations
    // ============================================================================
    inline Value leapfrog_reduce(std::vector<Value>& values);

    // ============================================================================
    // compute_node – dispatcher for non‑SUM operations
    // ============================================================================
    inline Value compute_node(LazyOp op, const Value& left, const Value& right, const Value& eps) {
        switch (op) {
        case LazyOp::SUM:
            throw std::logic_error("compute_node: SUM must be handled by leapfrog_reduce");
        case LazyOp::PRODUCT:
            throw std::logic_error("compute_node: PRODUCT not yet batched (use sequential eager_mul)");
        case LazyOp::NEG:      return eager_neg(left);
        case LazyOp::RECIP:    return eager_div(Value(SmallStorage(1)), left);
        case LazyOp::SQRT:     return eager_sqrt(left, eps);
        case LazyOp::EXP:      return eager_exp(left, eps);
        case LazyOp::LOG:      return eager_log(left, eps);
        case LazyOp::SIN:      return eager_sin(left, eps);
        case LazyOp::COS:      return eager_cos(left, eps);
        case LazyOp::ACOS:     return eager_acos(left, eps);
        case LazyOp::PI:       return eager_pi(eps);
        case LazyOp::E:        return eager_e(eps);
        case LazyOp::POW:      return eager_pow(left, right, eps);
        default:
            throw std::logic_error("compute_node: unknown LazyOp");
        }
    }

    // ============================================================================
    // Simple sequential summation for a batch (stride > 0)
    // ============================================================================
    inline Value sum_batch_sequential(const Value* batch, size_t stride, size_t count) {
        if (count == 0) return Value(SmallStorage(0));
        Value result = batch[0];
        for (size_t i = 1; i < count; ++i) {
            add_inplace(result, batch[i * stride]);
        }
        return result;
    }

    // ============================================================================
    // Main Leapfrog Reduction – in‑place, minimal allocations
    // ============================================================================
    inline Value leapfrog_reduce(std::vector<Value>& values) {
        const size_t N = values.size();
        if (N == 0) return Value(SmallStorage(0));
        if (N == 1) return values[0];
        if (N <= BATCH_SIZE) {
            return sum_batch_sequential(values.data(), 1, N);
        }

        // ---------- First pass: compress into v2 ----------
        const size_t M = (N + BATCH_SIZE - 1) / BATCH_SIZE;
        const size_t M_padded = ((M + BATCH_SIZE - 1) / BATCH_SIZE) * BATCH_SIZE;

        std::vector<Value> v2(M_padded, Value(SmallStorage(0)));

        for (size_t i = 0; i < M; ++i) {
            size_t start = i * BATCH_SIZE;
            size_t cnt = std::min(BATCH_SIZE, N - start);
            v2[i] = sum_batch_sequential(&values[start], 1, cnt);
        }

        // ---------- Iterative leapfrog reduction ----------
        size_t stride = 1;
        size_t current_len = M_padded;

        while (stride * BATCH_SIZE <= current_len) {
            for (size_t i = 0; i + (BATCH_SIZE - 1) * stride < current_len; i += BATCH_SIZE * stride) {
                v2[i] = sum_batch_sequential(&v2[i], stride, BATCH_SIZE);
            }
            stride *= BATCH_SIZE;
        }

        // ---------- Final tail collection ----------
        std::vector<Value> tail;
        tail.reserve((current_len + stride - 1) / stride);
        for (size_t i = 0; i < current_len; i += stride) {
            tail.push_back(std::move(v2[i]));
        }

        return sum_batch_sequential(tail.data(), 1, tail.size());
    }


    // Перегрузка для работы с индексами и кэшем (без копирования Value, для грязного дерева.)
// Версия для работы с массивом индексов и кэшем (без копирования Value)
    inline Value leapfrog_reduce_from_cache(const int* indices, size_t N,
        const std::vector<std::optional<Value>>& cache) {
        if (N == 0) return Value(SmallStorage(0));
        if (N == 1) return cache[indices[0]].value();
        if (N <= BATCH_SIZE) {
            // Последовательное суммирование по индексам
            Value result = cache[indices[0]].value();
            for (size_t i = 1; i < N; ++i) {
                add_inplace(result, cache[indices[i]].value());
            }
            return result;
        }

        // Первый проход: сжатие в v2
        const size_t M = (N + BATCH_SIZE - 1) / BATCH_SIZE;
        const size_t M_padded = ((M + BATCH_SIZE - 1) / BATCH_SIZE) * BATCH_SIZE;

        std::vector<Value> v2(M_padded, Value(SmallStorage(0)));

        for (size_t i = 0; i < M; ++i) {
            size_t start = i * BATCH_SIZE;
            size_t cnt = std::min(BATCH_SIZE, N - start);
            Value batch_sum = cache[indices[start]].value();
            for (size_t j = 1; j < cnt; ++j) {
                add_inplace(batch_sum, cache[indices[start + j]].value());
            }
            v2[i] = std::move(batch_sum);
        }

        // Leapfrog редукция по v2
        size_t stride = 1;
        size_t current_len = M_padded;

        while (stride * BATCH_SIZE <= current_len) {
            for (size_t i = 0; i + (BATCH_SIZE - 1) * stride < current_len; i += BATCH_SIZE * stride) {
                v2[i] = sum_batch_sequential(&v2[i], stride, BATCH_SIZE);
            }
            stride *= BATCH_SIZE;
        }

        // Финальный хвост
        std::vector<Value> tail;
        tail.reserve((current_len + stride - 1) / stride);
        for (size_t i = 0; i < current_len; i += stride) {
            tail.push_back(std::move(v2[i]));
        }

        return sum_batch_sequential(tail.data(), 1, tail.size());
    }


    // ============================================================================
    // In‑place Leapfrog Reduction – мутирует переданный вектор values, нужен для eval_inplace(skip_simplify=true) 
    // где нам плевать на сохранение структуры дерева, а главное быстрый результат.
    // ============================================================================
    inline void leapfrog_reduce_inplace(std::vector<Value>& values) {
        const size_t N = values.size();
        if (N <= 1) return;
        if (N <= BATCH_SIZE) {
            values[0] = sum_batch_sequential(values.data(), 1, N);
            values.resize(1);
            return;
        }

        // Первый проход: сжатие в начало вектора (работаем как с v2, но без отдельного вектора)
        const size_t M = (N + BATCH_SIZE - 1) / BATCH_SIZE;
        const size_t M_padded = ((M + BATCH_SIZE - 1) / BATCH_SIZE) * BATCH_SIZE;

        // Расширяем вектор до M_padded, заполняя нулями
        values.resize(M_padded, Value(SmallStorage(0)));

        // Сворачиваем батчи из исходных N элементов и записываем в начало (индексы 0..M-1)
        for (size_t i = 0; i < M; ++i) {
            size_t start = i * BATCH_SIZE;
            size_t cnt = std::min(BATCH_SIZE, N - start);
            values[i] = sum_batch_sequential(&values[start], 1, cnt);
        }
        // Остальные элементы (M..M_padded-1) уже нули

        // Leapfrog итерация
        size_t stride = 1;
        size_t current_len = M_padded;

        while (stride * BATCH_SIZE <= current_len) {
            for (size_t i = 0; i + (BATCH_SIZE - 1) * stride < current_len; i += BATCH_SIZE * stride) {
                values[i] = sum_batch_sequential(&values[i], stride, BATCH_SIZE);
            }
            stride *= BATCH_SIZE;
        }

        // Сбор хвоста в начало
        size_t tail_idx = 0;
        for (size_t i = 0; i < current_len; i += stride) {
            if (i != tail_idx) {
                values[tail_idx] = std::move(values[i]);
            }
            ++tail_idx;
        }
        values.resize(tail_idx);
        // Финальное суммирование хвоста
        values[0] = sum_batch_sequential(values.data(), 1, tail_idx);
        values.resize(1);
    }
    // ============================================================================
    // evaluate – computes a clean tree (NodePool)
    // ============================================================================
    inline Value evaluate(int root_idx) {
        const auto& nodes = pool.nodes;
        const auto& values = pool.values;
        const size_t n = nodes.size();

        std::vector<std::optional<Value>> cache(n);
        std::stack<int> st;
        st.push(root_idx);

        while (!st.empty()) {
            int idx = st.top();
            if (cache[idx].has_value()) {
                st.pop();
                continue;
            }

            const Node& node = nodes[idx];

            if (node.op == LazyOp::CONST) {
                cache[idx] = values[node.value_idx];
                st.pop();
                continue;
            }

            bool children_ready = true;
            if (node.children) {
                for (int32_t child : *node.children) {
                    if (!cache[child].has_value()) {
                        st.push(child);
                        children_ready = false;
                    }
                }
            }
            else {
                if (node.child0 != -1 && !cache[node.child0].has_value()) {
                    st.push(node.child0);
                    children_ready = false;
                }
                if (node.child1 != -1 && !cache[node.child1].has_value()) {
                    st.push(node.child1);
                    children_ready = false;
                }
            }
            if (!children_ready) continue;

            Value result;
            if (node.op == LazyOp::SUM) {
                const auto& children = *node.children;
                if (children.empty()) {
                    result = Value(SmallStorage(0));
                }
                else {
                    std::vector<Value> child_values;
                    child_values.reserve(children.size());
                    for (int32_t child_idx : children) {
                        child_values.push_back(cache[child_idx].value());
                    }
                    result = leapfrog_reduce(child_values);
                }
            }
            else if (node.op == LazyOp::PRODUCT) {
                const auto& children = *node.children;
                if (children.empty()) {
                    result = Value(SmallStorage(1));
                }
                else {
                    result = cache[children[0]].value();
                    for (size_t i = 1; i < children.size(); ++i) {
                        result = eager_mul(result, cache[children[i]].value());
                    }
                }
            }
            else {
                Value left = (node.child0 != -1) ? cache[node.child0].value() : Value{};
                Value right = (node.child1 != -1) ? cache[node.child1].value() : Value{};
                Value eps = (node.eps_idx != -1) ? values[node.eps_idx] : Value{};
                result = compute_node(node.op, left, right, eps);
            }

            cache[idx] = result;
            st.pop();
        }

        return cache[root_idx].value();
    }

} // namespace delta::internal