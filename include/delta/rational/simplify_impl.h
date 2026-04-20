// simplify_impl.h
// Версия 3.0 – упрощение дерева TempNode с поддержкой гетерогенного хранения
// ----------------------------------------------------------------------------
// Стратегия упрощения:
//   - Flattening вложенных SUM/PRODUCT
//   - Удаление нейтральных элементов (0 в SUM, 1 в PRODUCT)
//   - Сортировка операндов по хэшу (детерминизм)
//   - Сокращение противоположных пар: x + NEG(x) → 0, x * RECIP(x) → 1
//   - Алгебраические упрощения: NEG(NEG(x)) → x, RECIP(RECIP(x)) → x,
//     EXP(LOG(x)) → x, LOG(EXP(x)) → x
//   - Упрощения POW: x^0 → 1, x^1 → x, 1^x → 1, 0^positive → 0,
//     (a^b)^c → a^(b*c) для целых показателей
//   - КОНСТАНТЫ НЕ СВОРАЧИВАЮТСЯ (кроме особых случаев выше)
// ----------------------------------------------------------------------------

#pragma once

#include "lazy_nodes.h"
#include "storage.h"
#include "evaluation_core.h"
#include <vector>
#include <algorithm>
#include <stack>
#include <unordered_map>
#include <optional>

namespace delta::internal {

    // ------------------------------------------------------------------------
    // Вспомогательные функции для TempNode
    // ------------------------------------------------------------------------
    inline bool is_temp_zero(const TempNode& node, const std::vector<Value>& values) {
        if (node.op != LazyOp::CONST) return false;
        return is_zero(values[node.value_idx]);
    }

    inline bool is_temp_one(const TempNode& node, const std::vector<Value>& values) {
        if (node.op != LazyOp::CONST) return false;
        return is_one(values[node.value_idx]);
    }

    inline bool is_temp_minus_one(const TempNode& node, const std::vector<Value>& values) {
        if (node.op != LazyOp::CONST) return false;
        const Value& v = values[node.value_idx];
        return v == -1;
    }

    inline bool is_temp_positive_const(const TempNode& node, const std::vector<Value>& values) {
        if (node.op != LazyOp::CONST) return false;
        return is_positive(values[node.value_idx]);
    }

    inline int make_temp_const(std::vector<Value>& values, const Value& v) {
        int idx = static_cast<int>(values.size());
        values.push_back(v);
        return idx;
    }

    // Создание TempNode (универсальная версия)
    inline int make_temp_node(std::vector<TempNode>& nodes,
        std::vector<Value>& values,
        LazyOp op,
        std::vector<int> children,
        int value_idx = -1,
        int eps_idx = -1) {
        uint64_t hash = static_cast<uint64_t>(op);
        Interval approx;
        int32_t depth = 0;

        if (op == LazyOp::CONST) {
            const Value& v = values[value_idx];
            hash = compute_hash_const(v);
            approx = Interval(to_double(v));
            depth = 0;
        }
        else if (op == LazyOp::SUM) {
            approx = Interval::zero();
            for (int c : children) {
                depth = std::max(depth, nodes[c].depth + 1);
                approx = approx + nodes[c].approx;
                hash = combine_hash(LazyOp::SUM, hash, nodes[c].hash);
            }
        }
        else if (op == LazyOp::PRODUCT) {
            approx = Interval::one();
            for (int c : children) {
                depth = std::max(depth, nodes[c].depth + 1);
                approx = approx * nodes[c].approx;
                hash = combine_hash(LazyOp::PRODUCT, hash, nodes[c].hash);
            }
        }
        else if (op == LazyOp::NEG || op == LazyOp::RECIP || op == LazyOp::SQRT ||
            op == LazyOp::EXP || op == LazyOp::LOG || op == LazyOp::SIN ||
            op == LazyOp::COS || op == LazyOp::ACOS) {
            int c = children[0];
            depth = 1 + nodes[c].depth;
            approx = compute_interval(op, nodes[c].approx);
            hash = combine_hash(op, nodes[c].hash, 0, eps_idx);
        }
        else if (op == LazyOp::PI || op == LazyOp::E) {
            depth = 0;
            approx = compute_interval(op, Interval());
            hash = combine_hash(op, 0, eps_idx);
        }
        else if (op == LazyOp::POW) {
            int base = children[0];
            int exp = children[1];
            depth = 1 + std::max(nodes[base].depth, nodes[exp].depth);
            approx = compute_interval(LazyOp::POW, nodes[base].approx, nodes[exp].approx);
            hash = combine_hash(LazyOp::POW, nodes[base].hash, nodes[exp].hash, eps_idx);
        }
        else {
            throw std::logic_error("make_temp_node: unknown op");
        }

        int idx = static_cast<int>(nodes.size());
        nodes.emplace_back(op, std::move(children), value_idx, eps_idx, hash, approx, depth);
        return idx;
    }

    // Создание TempNode с гетерогенным хранением (для SUM/PRODUCT)
    inline int make_temp_sum_node(std::vector<TempNode>& nodes,
        std::vector<Value>& values,
        std::vector<Value> leaf_values,
        std::vector<int> complex_children) {
        uint64_t hash = static_cast<uint64_t>(LazyOp::SUM);
        Interval approx = Interval::zero();
        int32_t depth = 0;

        for (const auto& v : leaf_values) {
            approx = approx + Interval(to_double(v));
            hash = absl::HashOf(hash, v);
        }
        for (int c : complex_children) {
            depth = std::max(depth, nodes[c].depth + 1);
            approx = approx + nodes[c].approx;
            hash = combine_hash(LazyOp::SUM, hash, nodes[c].hash);
        }

        int idx = static_cast<int>(nodes.size());
        nodes.emplace_back(LazyOp::SUM, std::move(leaf_values), std::move(complex_children),
            -1, -1, hash, approx, depth);
        return idx;
    }

    inline int make_temp_product_node(std::vector<TempNode>& nodes,
        std::vector<Value>& values,
        std::vector<Value> leaf_values,
        std::vector<int> complex_children) {
        uint64_t hash = static_cast<uint64_t>(LazyOp::PRODUCT);
        Interval approx = Interval::one();
        int32_t depth = 0;

        for (const auto& v : leaf_values) {
            approx = approx * Interval(to_double(v));
            hash = absl::HashOf(hash, v);
        }
        for (int c : complex_children) {
            depth = std::max(depth, nodes[c].depth + 1);
            approx = approx * nodes[c].approx;
            hash = combine_hash(LazyOp::PRODUCT, hash, nodes[c].hash);
        }

        int idx = static_cast<int>(nodes.size());
        nodes.emplace_back(LazyOp::PRODUCT, std::move(leaf_values), std::move(complex_children),
            -1, -1, hash, approx, depth);
        return idx;
    }

    // ------------------------------------------------------------------------
    // Основная функция упрощения дерева TempNode
    // ------------------------------------------------------------------------
    inline int simplify_tree(std::vector<TempNode>& nodes,
        std::vector<Value>& values,
        int root) {
        std::vector<int> simplified(nodes.size(), -1);
        std::stack<int> st;
        st.push(root);
        std::vector<int> postorder;
        while (!st.empty()) {
            int idx = st.top(); st.pop();
            postorder.push_back(idx);
            const auto& tn = nodes[idx];
            if (tn.op == LazyOp::SUM || tn.op == LazyOp::PRODUCT) {
                for (int child : tn.complex_children) st.push(child);
            }
            else {
                for (int child : tn.children) st.push(child);
            }
        }

        for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
            int idx = *it;
            TempNode& node = nodes[idx];

            // Рекурсивно упрощаем детей
            if (node.op == LazyOp::SUM || node.op == LazyOp::PRODUCT) {
                for (int& child : node.complex_children) {
                    child = simplified[child];
                }
            }
            else {
                for (int& child : node.children) {
                    child = simplified[child];
                }
            }

            switch (node.op) {
            case LazyOp::CONST:
                simplified[idx] = idx;
                break;

            case LazyOp::SUM: {
                // Flattening: собираем leaf_values и complex_children из всех вложенных SUM
                std::vector<Value> all_leaf_values = std::move(node.leaf_values);
                std::vector<int> all_complex_children = std::move(node.complex_children);

                // Проходим по complex_children, извлекая вложенные SUM
                for (size_t i = 0; i < all_complex_children.size(); ) {
                    int child = all_complex_children[i];
                    const TempNode& child_node = nodes[child];
                    if (child_node.op == LazyOp::SUM) {
                        // Поглощаем leaf_values (перемещаем)
                        all_leaf_values.insert(all_leaf_values.end(),
                            std::make_move_iterator(child_node.leaf_values.begin()),
                            std::make_move_iterator(child_node.leaf_values.end()));
                        // Поглощаем complex_children
                        all_complex_children.insert(all_complex_children.end(),
                            child_node.complex_children.begin(),
                            child_node.complex_children.end());
                        // Удаляем поглощённый узел
                        all_complex_children.erase(all_complex_children.begin() + i);
                        // i не увеличиваем, т.к. элемент удалён
                    }
                    else {
                        ++i;
                    }
                }

                // Удаление нулей из leaf_values
                all_leaf_values.erase(
                    std::remove_if(all_leaf_values.begin(), all_leaf_values.end(),
                        [](const Value& v) { return is_zero(v); }),
                    all_leaf_values.end());

                // Извлечение CONST из complex_children в leaf_values
                for (size_t i = 0; i < all_complex_children.size(); ) {
                    int child = all_complex_children[i];
                    if (nodes[child].op == LazyOp::CONST) {
                        Value v = values[nodes[child].value_idx];
                        if (!is_zero(v)) {
                            all_leaf_values.push_back(std::move(v));
                        }
                        all_complex_children.erase(all_complex_children.begin() + i);
                    }
                    else {
                        ++i;
                    }
                }

                // Сортировка leaf_values по хэшу (для детерминизма)
                std::sort(all_leaf_values.begin(), all_leaf_values.end(),
                    [](const Value& a, const Value& b) {
                        return absl::Hash<Value>{}(a) < absl::Hash<Value>{}(b);
                    });

                // Сортировка complex_children по хэшу узлов
                std::sort(all_complex_children.begin(), all_complex_children.end(),
                    [&](int a, int b) {
                        return nodes[a].hash < nodes[b].hash;
                    });

                // Сокращение пар: x + NEG(x) → 0
                std::vector<bool> keep_complex(all_complex_children.size(), true);
                for (size_t i = 0; i < all_complex_children.size(); ++i) {
                    if (!keep_complex[i]) continue;
                    int child_i = all_complex_children[i];
                    const TempNode& node_i = nodes[child_i];
                    if (node_i.op == LazyOp::NEG && node_i.children.size() == 1) {
                        int neg_child = node_i.children[0];
                        for (size_t j = i + 1; j < all_complex_children.size(); ++j) {
                            if (!keep_complex[j]) continue;
                            if (all_complex_children[j] == neg_child) {
                                keep_complex[i] = false;
                                keep_complex[j] = false;
                                break;
                            }
                        }
                    }
                }
                std::vector<int> after_cancel;
                for (size_t i = 0; i < all_complex_children.size(); ++i) {
                    if (keep_complex[i]) after_cancel.push_back(all_complex_children[i]);
                }

                // Сборка результата
                if (all_leaf_values.empty() && after_cancel.empty()) {
                    int zero_idx = make_temp_const(values, Value(0));
                    int new_idx = make_temp_node(nodes, values, LazyOp::CONST, {}, zero_idx);
                    simplified[idx] = new_idx;
                }
                else if (all_leaf_values.size() == 1 && after_cancel.empty()) {
                    // Единственное слагаемое – просто возвращаем его как CONST
                    int const_idx = make_temp_const(values, all_leaf_values[0]);
                    int new_idx = make_temp_node(nodes, values, LazyOp::CONST, {}, const_idx);
                    simplified[idx] = new_idx;
                }
                else if (all_leaf_values.empty() && after_cancel.size() == 1) {
                    // Единственный комплексный ребёнок
                    simplified[idx] = after_cancel[0];
                }
                else {
                    int new_idx = make_temp_sum_node(nodes, values,
                        std::move(all_leaf_values),
                        std::move(after_cancel));
                    simplified[idx] = new_idx;
                }
                break;
            }

            case LazyOp::PRODUCT: {
                // Локальный блок для устранения ошибок goto (объявления переменных не будут пересечены)
                {
                    // Flattening
                    std::vector<Value> all_leaf_values = std::move(node.leaf_values);
                    std::vector<int> all_complex_children = std::move(node.complex_children);

                    for (size_t i = 0; i < all_complex_children.size(); ) {
                        int child = all_complex_children[i];
                        const TempNode& child_node = nodes[child];
                        if (child_node.op == LazyOp::PRODUCT) {
                            all_leaf_values.insert(all_leaf_values.end(),
                                std::make_move_iterator(child_node.leaf_values.begin()),
                                std::make_move_iterator(child_node.leaf_values.end()));
                            all_complex_children.insert(all_complex_children.end(),
                                child_node.complex_children.begin(),
                                child_node.complex_children.end());
                            all_complex_children.erase(all_complex_children.begin() + i);
                        }
                        else {
                            ++i;
                        }
                    }

                    // Удаление единиц из leaf_values
                    all_leaf_values.erase(
                        std::remove_if(all_leaf_values.begin(), all_leaf_values.end(),
                            [](const Value& v) { return is_one(v); }),
                        all_leaf_values.end());

                    // Если среди leaf_values есть ноль, весь PRODUCT → 0
                    bool has_zero = std::any_of(all_leaf_values.begin(), all_leaf_values.end(),
                        [](const Value& v) { return is_zero(v); });
                    if (has_zero) {
                        int zero_idx = make_temp_const(values, Value(0));
                        int new_idx = make_temp_node(nodes, values, LazyOp::CONST, {}, zero_idx);
                        simplified[idx] = new_idx;
                        break;
                    }

                    // Извлечение CONST из complex_children в leaf_values
                    for (size_t i = 0; i < all_complex_children.size(); ) {
                        int child = all_complex_children[i];
                        if (nodes[child].op == LazyOp::CONST) {
                            Value v = values[nodes[child].value_idx];
                            if (is_zero(v)) {
                                int zero_idx = make_temp_const(values, Value(0));
                                int new_idx = make_temp_node(nodes, values, LazyOp::CONST, {}, zero_idx);
                                simplified[idx] = new_idx;
                                break;  // выходим из блока (внешний break сработает после)
                            }
                            if (!is_one(v)) {
                                all_leaf_values.push_back(std::move(v));
                            }
                            all_complex_children.erase(all_complex_children.begin() + i);
                        }
                        else {
                            ++i;
                        }
                    }

                    // Если уже был обнаружен ноль и мы вышли через break, пропускаем дальнейшую обработку
                    if (simplified[idx] != -1) break;

                    // Сортировка
                    std::sort(all_leaf_values.begin(), all_leaf_values.end(),
                        [](const Value& a, const Value& b) {
                            return absl::Hash<Value>{}(a) < absl::Hash<Value>{}(b);
                        });
                    std::sort(all_complex_children.begin(), all_complex_children.end(),
                        [&](int a, int b) {
                            return nodes[a].hash < nodes[b].hash;
                        });

                    // Сокращение пар: x * RECIP(x) → 1
                    std::vector<bool> keep_complex(all_complex_children.size(), true);
                    for (size_t i = 0; i < all_complex_children.size(); ++i) {
                        if (!keep_complex[i]) continue;
                        int child_i = all_complex_children[i];
                        const TempNode& node_i = nodes[child_i];
                        if (node_i.op == LazyOp::RECIP && node_i.children.size() == 1) {
                            int recip_child = node_i.children[0];
                            for (size_t j = i + 1; j < all_complex_children.size(); ++j) {
                                if (!keep_complex[j]) continue;
                                if (all_complex_children[j] == recip_child) {
                                    keep_complex[i] = false;
                                    keep_complex[j] = false;
                                    // Добавляем 1 в leaf_values
                                    all_leaf_values.push_back(Value(1));
                                    break;
                                }
                            }
                        }
                    }
                    std::vector<int> after_cancel;
                    for (size_t i = 0; i < all_complex_children.size(); ++i) {
                        if (keep_complex[i]) after_cancel.push_back(all_complex_children[i]);
                    }

                    // Удаление единиц, которые могли появиться после сокращения
                    all_leaf_values.erase(
                        std::remove_if(all_leaf_values.begin(), all_leaf_values.end(),
                            [](const Value& v) { return is_one(v); }),
                        all_leaf_values.end());

                    if (all_leaf_values.empty() && after_cancel.empty()) {
                        int one_idx = make_temp_const(values, Value(1));
                        int new_idx = make_temp_node(nodes, values, LazyOp::CONST, {}, one_idx);
                        simplified[idx] = new_idx;
                    }
                    else if (all_leaf_values.size() == 1 && after_cancel.empty()) {
                        int const_idx = make_temp_const(values, all_leaf_values[0]);
                        int new_idx = make_temp_node(nodes, values, LazyOp::CONST, {}, const_idx);
                        simplified[idx] = new_idx;
                    }
                    else if (all_leaf_values.empty() && after_cancel.size() == 1) {
                        simplified[idx] = after_cancel[0];
                    }
                    else {
                        int new_idx = make_temp_product_node(nodes, values,
                            std::move(all_leaf_values),
                            std::move(after_cancel));
                        simplified[idx] = new_idx;
                    }
                }
                break;
            }

            case LazyOp::NEG: {
                int child = node.children[0];
                if (nodes[child].op == LazyOp::NEG) {
                    simplified[idx] = nodes[child].children[0];
                    break;
                }
                // NEG не сворачивает константы
                simplified[idx] = idx;
                break;
            }

            case LazyOp::RECIP: {
                int child = node.children[0];
                if (nodes[child].op == LazyOp::RECIP) {
                    simplified[idx] = nodes[child].children[0];
                    break;
                }
                // RECIP не сворачивает константы
                simplified[idx] = idx;
                break;
            }

            case LazyOp::EXP: {
                int child = node.children[0];
                if (nodes[child].op == LazyOp::LOG) {
                    simplified[idx] = nodes[child].children[0];
                    break;
                }
                simplified[idx] = idx;
                break;
            }

            case LazyOp::LOG: {
                int child = node.children[0];
                if (nodes[child].op == LazyOp::EXP) {
                    simplified[idx] = nodes[child].children[0];
                    break;
                }
                simplified[idx] = idx;
                break;
            }

            case LazyOp::SQRT:
            case LazyOp::SIN:
            case LazyOp::COS:
            case LazyOp::ACOS:
            case LazyOp::PI:
            case LazyOp::E:
                simplified[idx] = idx;
                break;

            case LazyOp::POW: {
                int base = node.children[0];
                int exp = node.children[1];
                const TempNode& base_node = nodes[base];
                const TempNode& exp_node = nodes[exp];

                // x^0 → 1
                if (is_temp_zero(exp_node, values)) {
                    int one_idx = make_temp_const(values, Value(1));
                    int new_idx = make_temp_node(nodes, values, LazyOp::CONST, {}, one_idx);
                    simplified[idx] = new_idx;
                    break;
                }
                // x^1 → x
                if (is_temp_one(exp_node, values)) {
                    simplified[idx] = base;
                    break;
                }
                // 1^x → 1
                if (is_temp_one(base_node, values)) {
                    int one_idx = make_temp_const(values, Value(1));
                    int new_idx = make_temp_node(nodes, values, LazyOp::CONST, {}, one_idx);
                    simplified[idx] = new_idx;
                    break;
                }
                // 0^positive → 0
                if (is_temp_zero(base_node, values) && is_temp_positive_const(exp_node, values)) {
                    int zero_idx = make_temp_const(values, Value(0));
                    int new_idx = make_temp_node(nodes, values, LazyOp::CONST, {}, zero_idx);
                    simplified[idx] = new_idx;
                    break;
                }
                // (a^b)^c → a^(b*c) для целых показателей
                if (base_node.op == LazyOp::POW &&
                    nodes[base_node.children[1]].op == LazyOp::CONST &&
                    exp_node.op == LazyOp::CONST) {
                    const Value& b_val = values[nodes[base_node.children[1]].value_idx];
                    const Value& c_val = values[exp_node.value_idx];
                    // Проверка целочисленности через denominator()
                    if (denominator(b_val) == 1 && denominator(c_val) == 1) {
                        Value prod = b_val * c_val;
                        int new_exp = make_temp_const(values, prod);
                        int new_base = base_node.children[0];
                        // Создаём новый POW
                        int new_pow = make_temp_node(nodes, values, LazyOp::POW, { new_base, new_exp }, -1, node.eps_idx);
                        simplified[idx] = new_pow;
                        break;
                    }
                }
                simplified[idx] = idx;
                break;
            }

            default:
                simplified[idx] = idx;
                break;
            }
        }

        return simplified[root];
    }

} // namespace delta::internal