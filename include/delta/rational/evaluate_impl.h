// evaluate_impl.h
#pragma once

#include "node_pool.h"
#include "evaluation_core.h"
#include <boost/multiprecision/integer.hpp>   // для lcm
#include <stack>
#include <optional>
#include <vector>
#include <utility>      // для std::move

namespace delta::internal {

    // ----------------------------------------------------------------------------
    // compute_node – вычисление узла по операции и уже готовым детям
    // (без изменений)
    // ----------------------------------------------------------------------------
    inline Value compute_node(LazyOp op, const Value& left, const Value& right, const Value& eps) {
        switch (op) {
        case LazyOp::ADD:   return eager_add(left, right);
        case LazyOp::MUL:   return eager_mul(left, right);
        case LazyOp::NEG:   return eager_neg(left);
        case LazyOp::RECIP: return eager_div(Value(SmallStorage(1)), left);
        case LazyOp::SQRT:  return eager_sqrt(left, eps);
        case LazyOp::EXP:   return eager_exp(left, eps);
        case LazyOp::LOG:   return eager_log(left, eps);
        case LazyOp::SIN:   return eager_sin(left, eps);
        case LazyOp::COS:   return eager_cos(left, eps);
        case LazyOp::ACOS:  return eager_acos(left, eps);
        case LazyOp::PI:    return eager_pi(eps);
        case LazyOp::E:     return eager_e(eps);
        case LazyOp::POW:   return eager_pow(left, right, eps);
        default:
            throw std::logic_error("compute_node: unknown LazyOp");
        }
    }

    // ----------------------------------------------------------------------------
    // batch_add_values – пакетное сложение нескольких значений из кэша
    // ----------------------------------------------------------------------------
    inline Value batch_add_values(const std::vector<int>& indices,
        const std::vector<std::optional<Value>>& cache) {
        if (indices.empty()) {
            return SmallStorage(0);
        }
        if (indices.size() == 1) {
            return cache[indices[0]].value();
        }

        using boost::multiprecision::cpp_int;
        cpp_int common_denom(1);
        std::vector<cpp_int> numerators;
        numerators.reserve(indices.size());

        // Первый проход: вычисляем общий знаменатель (LCM всех знаменателей)
        for (int idx : indices) {
            const Value& v = cache[idx].value();
            auto [num, den] = normalize_to_cpp_int(v);
            numerators.push_back(num);
            common_denom = boost::multiprecision::lcm(common_denom, den);
        }

        // Второй проход: суммируем числители, приведённые к общему знаменателю
        cpp_int sum_num(0);
        for (size_t i = 0; i < indices.size(); ++i) {
            const Value& v = cache[indices[i]].value();
            auto [num, den] = normalize_to_cpp_int(v);
            cpp_int factor = common_denom / den;
            sum_num += num * factor;
        }

        // Сокращение
        cpp_int g = boost::multiprecision::gcd(sum_num, common_denom);
        if (g != 0) {
            sum_num /= g;
            common_denom /= g;
        }

        // Преобразование обратно в Value (SmallStorage, если помещается)
        if (sum_num <= to_cpp_int((std::numeric_limits<absl::int128>::max)()) &&
            common_denom <= to_cpp_int((std::numeric_limits<absl::uint128>::max)())) {
            return SmallStorage(int128_from_string(sum_num.str()),
                uint128_from_string(common_denom.str()));
        }
        return BigStorage(sum_num, common_denom);
    }

    // ----------------------------------------------------------------------------
    // evaluate – итеративный пост-порядок с локальным кэшем и пакетной обработкой ADD
    // ----------------------------------------------------------------------------
    inline Value evaluate(int root_idx) {
        const auto& nodes = pool.nodes;
        const auto& values = pool.values;
        const size_t n = nodes.size();

        std::vector<std::optional<Value>> cache(n);
        // Кэш плоских списков операндов для каждого ADD-узла (индексы из cache)
        std::vector<std::optional<std::vector<int>>> add_cache(n);

        std::stack<int> st;
        st.push(root_idx);

        // Максимальный размер батча для пакетного сложения (предотвращает взрывной LCM)
        constexpr size_t MAX_BATCH_SIZE = 64;
        // Порог: если операндов <= 3, складываем последовательно (меньше накладных расходов)
        constexpr size_t DIRECT_ADD_LIMIT = 3;

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

            // Узел без детей (например, PI, E)
            if (node.child0 == -1 && node.child1 == -1) {
                Value left{}, right{};
                Value eps = node.value_idx != -1 ? values[node.value_idx] : Value{};
                Value result = compute_node(node.op, left, right, eps);
                cache[idx] = result;
                st.pop();
                continue;
            }

            // Проверяем готовность детей
            bool need_children = false;
            if (node.child0 != -1 && !cache[node.child0].has_value()) {
                st.push(node.child0);
                need_children = true;
            }
            if (node.child1 != -1 && !cache[node.child1].has_value()) {
                st.push(node.child1);
                need_children = true;
            }
            if (need_children) {
                continue;
            }

            // --- Специальная обработка для ADD ---
            if (node.op == LazyOp::ADD) {
                std::vector<int> operands;
                operands.reserve(16); // небольшая начальная ёмкость

                // Левый ребёнок
                if (node.child0 != -1) {
                    const Node& left_node = nodes[node.child0];
                    if (left_node.op == LazyOp::ADD && add_cache[node.child0].has_value()) {
                        auto& child_vec = add_cache[node.child0].value();
                        // Если у дочернего ADD только один родитель (этот узел), перемещаем вектор
                        if (pool.refcount[node.child0] == 1) {
                            operands.insert(operands.end(),
                                std::make_move_iterator(child_vec.begin()),
                                std::make_move_iterator(child_vec.end()));
                            child_vec.clear();
                            add_cache[node.child0].reset();
                        }
                        else {
                            // Иначе копируем индексы (они маленькие)
                            operands.insert(operands.end(), child_vec.begin(), child_vec.end());
                        }
                    }
                    else {
                        // Обычный узел – добавляем его индекс
                        operands.push_back(node.child0);
                    }
                }

                // Правый ребёнок
                if (node.child1 != -1) {
                    const Node& right_node = nodes[node.child1];
                    if (right_node.op == LazyOp::ADD && add_cache[node.child1].has_value()) {
                        auto& child_vec = add_cache[node.child1].value();
                        if (pool.refcount[node.child1] == 1) {
                            operands.insert(operands.end(),
                                std::make_move_iterator(child_vec.begin()),
                                std::make_move_iterator(child_vec.end()));
                            child_vec.clear();
                            add_cache[node.child1].reset();
                        }
                        else {
                            operands.insert(operands.end(), child_vec.begin(), child_vec.end());
                        }
                    }
                    else {
                        operands.push_back(node.child1);
                    }
                }

                // Сохраняем список операндов для возможного использования родительским ADD
                add_cache[idx] = std::move(operands);
                const auto& indices = add_cache[idx].value();

                // Мелкие суммы – складываем последовательно (меньше накладных расходов)
                if (indices.size() <= DIRECT_ADD_LIMIT) {
                    Value sum = cache[indices[0]].value();
                    for (size_t i = 1; i < indices.size(); ++i) {
                        sum = eager_add(sum, cache[indices[i]].value());
                    }
                    cache[idx] = sum;
                }
                else {
                    // Разбиваем на батчи, если операндов больше MAX_BATCH_SIZE
                    if (indices.size() <= MAX_BATCH_SIZE) {
                        cache[idx] = batch_add_values(indices, cache);
                    }
                    else {
                        std::vector<Value> batch_results;
                        batch_results.reserve((indices.size() + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE);
                        for (size_t start = 0; start < indices.size(); start += MAX_BATCH_SIZE) {
                            size_t end = std::min(start + MAX_BATCH_SIZE, indices.size());
                            std::vector<int> chunk(indices.begin() + start, indices.begin() + end);
                            batch_results.push_back(batch_add_values(chunk, cache));
                        }
                        // Суммируем результаты батчей последовательно через eager_add
                        Value sum = batch_results[0];
                        for (size_t i = 1; i < batch_results.size(); ++i) {
                            sum = eager_add(sum, batch_results[i]);
                        }
                        cache[idx] = sum;
                    }
                }
                st.pop();
                continue;
            }

            // --- Обработка всех остальных операций (без изменений) ---
            Value left = node.child0 != -1 ? cache[node.child0].value() : Value{};
            Value right = node.child1 != -1 ? cache[node.child1].value() : Value{};
            Value eps = node.value_idx != -1 ? values[node.value_idx] : Value{};
            Value result = compute_node(node.op, left, right, eps);
            cache[idx] = result;
            st.pop();
        }

        return cache[root_idx].value();
    }

} // namespace delta::internal