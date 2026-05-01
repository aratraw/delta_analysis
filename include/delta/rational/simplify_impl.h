// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// simplify_impl.h
// Версия 5.5 – однопроходная группировка и оптимизированные эвристики
// ----------------------------------------------------------------------------
// Философия: все свёртки выполняются через построение новых узлов (PRODUCT, POW),
// а не через вычисление значений. Это гарантирует сохранение символьного
// представления и возможность дальнейших упрощений.
//
// СТРАТЕГИЯ УПРОЩЕНИЯ ДЛЯ КАЖДОГО ТИПА УЗЛА:
//
// 1. CONST – всегда остаётся без изменений (уже константа).
// 2. SUM:
//    a) Flattening: разворачиваем вложенные SUM (SUM(a, SUM(b,c)) -> SUM(a,b,c)).
//    b) Удаляем нули из leaf_values и константных детей.
//    c) ЭВРИСТИКА ПРОПУСКА:
//       - Если узел не имеет детей (all_children.empty()) и среди leaf_values
//         нет повторяющихся значений (проверяется через однопроходную карту
//         частот с ранним выходом), то группировка и дистрибутивность невозможны.
//         В этом случае после сортировки leaf_values сразу создаётся SUM узел.
//    d) Группировка скалярных констант, свёртка повторяющихся (a+a -> 2*a)
//       выполняется ЕДИНЫМ ПРОХОДОМ с использованием flat_hash_map.
//    e) Свёртка одинаковых дочерних узлов (A+A -> 2*A).
//    f) Дистрибутивность (a*b + a*c -> a*(b+c)) – только если среди детей есть
//       хотя бы один узел PRODUCT.
//    g) Каноническая сортировка leaf_values и детей.
//    h) Сокращение x + NEG(x) -> 0.
//    i) Сборка итогового узла.
//
// 3. PRODUCT:
//    a) Flattening вложенных PRODUCT.
//    b) Удаление единиц, обработка нуля (весь продукт -> 0).
//    c) ЭВРИСТИКА ПРОПУСКА аналогично SUM: если нет детей и все множители
//       уникальны, группировка не нужна.
//    d) Группировка скалярных множителей с возведением в степень (a*a -> a^2)
//       выполняется однопроходной картой частот.
//    e) Свёртка одинаковых подузлов (A*A -> A^2).
//    f) Каноническая сортировка leaf_values и детей.
//    g) Сокращение x * RECIP(x) -> 1.
//    h) Сборка итогового узла.
//
// 4. Унарные операции: упрощение цепочек (NEG(NEG(x)) -> x, RECIP(RECIP(x)) -> x,
//    EXP(LOG(x)) -> x, LOG(EXP(x)) -> x).
//
// 5. POW: частные случаи (0^положит=0, 1^любое=1, степень0=1, степень1=base,
//    (x^a)^b -> x^(a*b) для целых показателей).
//
// ----------------------------------------------------------------------------
// ИСТОРИЯ ИЗМЕНЕНИЙ:
// Версия 5.3 – радикальное ускорение сортировок Value и безопасный компаратор.
// Версия 5.4 – добавлена эвристика пропуска группировки и дистрибутивности
//              для SUM с уникальными константами.
// Версия 5.5 – заменена двухпроходная группировка (set + map) на однопроходную
//              с flat_hash_map и флагом has_duplicate; добавлена аналогичная
//              эвристика для PRODUCT.
// ----------------------------------------------------------------------------

#pragma once

#include "lazy_nodes.h"
#include "storage.h"
#include "evaluation_core.h"
#include <vector>
#include <algorithm>
#include <stack>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>

namespace delta::internal {

    // ------------------------------------------------------------------------
    // Вспомогательные предикаты для TempNode
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
        return values[node.value_idx] == -1;
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

    // Создание TempNode для не-SUM/PRODUCT
    inline int make_temp_node(std::vector<TempNode>& nodes,
        std::vector<Value>& values,
        LazyOp op,
        std::vector<int> children,
        int value_idx = -1,
        int eps_idx = -1) {
        uint64_t hash = static_cast<uint64_t>(op);
        if (op == LazyOp::CONST) {
            hash = compute_hash_const(values[value_idx]);
        }
        else if (op == LazyOp::NEG || op == LazyOp::RECIP || op == LazyOp::SQRT ||
            op == LazyOp::EXP || op == LazyOp::LOG || op == LazyOp::SIN ||
            op == LazyOp::COS || op == LazyOp::ACOS) {
            int c = children[0];
            hash = combine_hash(op, nodes[c].hash, 0, eps_idx);
        }
        else if (op == LazyOp::PI || op == LazyOp::E) {
            hash = combine_hash(op, 0, eps_idx);
        }
        else if (op == LazyOp::POW) {
            int base = children[0];
            int exp = children[1];
            hash = combine_hash(LazyOp::POW, nodes[base].hash, nodes[exp].hash, eps_idx);
        }
        else {
            throw std::logic_error("make_temp_node: unknown op");
        }
        int idx = static_cast<int>(nodes.size());
        nodes.emplace_back(op, std::move(children), value_idx, eps_idx, hash);
        return idx;
    }

    inline int make_temp_sum_node(std::vector<TempNode>& nodes,
        std::vector<Value>& values,
        std::vector<Value> leaf_values,
        std::vector<int> children) {
        uint64_t hash = static_cast<uint64_t>(LazyOp::SUM);
        for (const auto& v : leaf_values) hash = absl::HashOf(hash, v);
        for (int c : children) hash = combine_hash(LazyOp::SUM, hash, nodes[c].hash);
        int idx = static_cast<int>(nodes.size());
        nodes.emplace_back(LazyOp::SUM, std::move(leaf_values), std::move(children), -1, -1, hash);
        return idx;
    }

    inline int make_temp_product_node(std::vector<TempNode>& nodes,
        std::vector<Value>& values,
        std::vector<Value> leaf_values,
        std::vector<int> children) {
        uint64_t hash = static_cast<uint64_t>(LazyOp::PRODUCT);
        for (const auto& v : leaf_values) hash = absl::HashOf(hash, v);
        for (int c : children) hash = combine_hash(LazyOp::PRODUCT, hash, nodes[c].hash);
        int idx = static_cast<int>(nodes.size());
        nodes.emplace_back(LazyOp::PRODUCT, std::move(leaf_values), std::move(children), -1, -1, hash);
        return idx;
    }

    // ------------------------------------------------------------------------
    // Каноническая сортировка вектора Value (по хешу, затем по значению)
    // ------------------------------------------------------------------------
    inline void sort_value_vector_canonical(std::vector<Value>& vals) {
        struct Pair { size_t hash; Value val; };
        std::vector<Pair> pairs;
        pairs.reserve(vals.size());
        for (auto& v : vals) {
            pairs.push_back({ absl::Hash<Value>{}(v), std::move(v) });
        }
        std::sort(pairs.begin(), pairs.end(),
            [](const Pair& a, const Pair& b) {
                if (a.hash != b.hash) return a.hash < b.hash;
                return a.val < b.val;
            });
        vals.clear();
        for (auto& p : pairs) {
            vals.push_back(std::move(p.val));
        }
    }

    // ------------------------------------------------------------------------
    // Сравнение TempNode (использует хеши)
    // ------------------------------------------------------------------------
    inline bool temp_nodes_equal(const std::vector<TempNode>& nodes,
        const std::vector<Value>& values,
        int a, int b) {
        if (a == b) return true;
        const TempNode& na = nodes[a];
        const TempNode& nb = nodes[b];
        if (na.op != nb.op || na.hash != nb.hash) return false;

        if (na.op == LazyOp::CONST) {
            return values[na.value_idx] == values[nb.value_idx];
        }
        if (na.op == LazyOp::SUM || na.op == LazyOp::PRODUCT) {
            if (na.leaf_values.size() != nb.leaf_values.size()) return false;
            std::vector<Value> la = na.leaf_values;
            std::vector<Value> lb = nb.leaf_values;
            sort_value_vector_canonical(la);
            sort_value_vector_canonical(lb);
            if (la != lb) return false;

            if (na.children.size() != nb.children.size()) return false;
            auto sorted_children = [&](const TempNode& n) {
                std::vector<int> c = n.children;
                std::sort(c.begin(), c.end(), [&](int x, int y) {
                    if (nodes[x].hash != nodes[y].hash) return nodes[x].hash < nodes[y].hash;
                    return x < y;
                    });
                return c;
                };
            const auto ca = sorted_children(na);
            const auto cb = sorted_children(nb);
            for (size_t i = 0; i < ca.size(); ++i) {
                if (!temp_nodes_equal(nodes, values, ca[i], cb[i])) return false;
            }
            return true;
        }
        else if (na.op == LazyOp::NEG || na.op == LazyOp::RECIP ||
            na.op == LazyOp::SQRT || na.op == LazyOp::EXP ||
            na.op == LazyOp::LOG || na.op == LazyOp::SIN ||
            na.op == LazyOp::COS || na.op == LazyOp::ACOS) {
            return temp_nodes_equal(nodes, values, na.children[0], nb.children[0]);
        }
        else if (na.op == LazyOp::POW) {
            return temp_nodes_equal(nodes, values, na.children[0], nb.children[0]) &&
                temp_nodes_equal(nodes, values, na.children[1], nb.children[1]);
        }
        return true;
    }

    // ------------------------------------------------------------------------
    // Основная функция упрощения дерева
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
            for (int child : tn.children) st.push(child);
        }

        for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
            int idx = *it;
            TempNode& node = nodes[idx];
            for (int& child : node.children) child = simplified[child];

            switch (node.op) {
            case LazyOp::CONST:
                simplified[idx] = idx;
                break;

            case LazyOp::SUM: {
                std::vector<Value> all_leaf_values = std::move(node.leaf_values);
                std::vector<int> all_children = std::move(node.children);

                // 1. Flattening вложенных SUM
                for (size_t i = 0; i < all_children.size(); ) {
                    int child = all_children[i];
                    const TempNode& child_node = nodes[child];
                    if (child_node.op == LazyOp::SUM) {
                        all_leaf_values.insert(all_leaf_values.end(),
                            std::make_move_iterator(child_node.leaf_values.begin()),
                            std::make_move_iterator(child_node.leaf_values.end()));
                        all_children.insert(all_children.end(),
                            child_node.children.begin(), child_node.children.end());
                        all_children.erase(all_children.begin() + i);
                    }
                    else ++i;
                }

                // 2. Удаление нулей и константных нулевых детей
                {
                    std::vector<Value> new_leaf;
                    for (auto& v : all_leaf_values) {
                        if (!is_zero(v)) new_leaf.push_back(std::move(v));
                    }
                    all_leaf_values = std::move(new_leaf);

                    std::vector<int> new_children;
                    for (int child : all_children) {
                        if (nodes[child].op == LazyOp::CONST) {
                            Value v = values[nodes[child].value_idx];
                            if (!is_zero(v)) {
                                all_leaf_values.push_back(std::move(v));
                            }
                        }
                        else {
                            new_children.push_back(child);
                        }
                    }
                    all_children = std::move(new_children);
                }

                // 3. Быстрая проверка на возможность упрощений
                bool has_product = false;
                for (int child : all_children) {
                    if (nodes[child].op == LazyOp::PRODUCT) {
                        has_product = true;
                        break;
                    }
                }

                // Если нет детей и все leaf_values уникальны – пропускаем группировку
                if (all_children.empty()) {
                    // Однопроходная проверка уникальности через map (с ранним выходом)
                    absl::flat_hash_map<Value, dumb_int, ValueHash, ValueEqual> freq;
                    bool has_duplicate = false;
                    for (const auto& v : all_leaf_values) {
                        auto it = freq.find(v);
                        if (it == freq.end()) {
                            freq.emplace(v, 1);
                        }
                        else {
                            has_duplicate = true;
                            break;
                        }
                    }
                    if (!has_duplicate) {
                        // Все уникальны – просто сортируем и создаём SUM
                        sort_value_vector_canonical(all_leaf_values);
                        if (all_leaf_values.empty()) {
                            int zero_idx = make_temp_const(values, Value(0));
                            simplified[idx] = make_temp_node(nodes, values, LazyOp::CONST, {}, zero_idx);
                        }
                        else if (all_leaf_values.size() == 1) {
                            int const_idx = make_temp_const(values, all_leaf_values[0]);
                            simplified[idx] = make_temp_node(nodes, values, LazyOp::CONST, {}, const_idx);
                        }
                        else {
                            simplified[idx] = make_temp_sum_node(nodes, values, std::move(all_leaf_values), {});
                        }
                        break;
                    }
                }

                // 4. Группировка скалярных констант (однопроходная)
                {
                    absl::flat_hash_map<Value, dumb_int, ValueHash, ValueEqual> freq;
                    for (auto& v : all_leaf_values) {
                        ++freq[std::move(v)];
                    }
                    all_leaf_values.clear();

                    for (auto& [val, cnt] : freq) {
                        if (cnt == 1) {
                            all_leaf_values.push_back(std::move(val));
                        }
                        else {
                            // Создаём PRODUCT(CONST(cnt), CONST(val))
                            int cnt_val_idx = make_temp_const(values, Value(cnt));
                            int cnt_node = make_temp_node(nodes, values, LazyOp::CONST, {}, cnt_val_idx);
                            int v_val_idx = make_temp_const(values, val);
                            int v_node = make_temp_node(nodes, values, LazyOp::CONST, {}, v_val_idx);
                            int prod_idx = make_temp_product_node(nodes, values, {},
                                std::vector<int>{cnt_node, v_node});
                            prod_idx = simplify_tree(nodes, values, prod_idx);
                            all_children.push_back(prod_idx);
                        }
                    }
                }

                // 5. Свёртка одинаковых дочерних узлов (A+A+... → N*A)
                {
                    absl::flat_hash_map<uint64_t, std::vector<int>> hash_buckets;
                    for (int child : all_children) {
                        hash_buckets[nodes[child].hash].push_back(child);
                    }

                    std::vector<int> unique_children;
                    absl::flat_hash_map<int, dumb_int> cnt_map;

                    for (auto& [hash_val, bucket] : hash_buckets) {
                        for (int candidate : bucket) {
                            bool found = false;
                            for (int u : unique_children) {
                                if (nodes[u].hash != nodes[candidate].hash) continue;
                                if (temp_nodes_equal(nodes, values, u, candidate)) {
                                    cnt_map[u] += 1;
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                unique_children.push_back(candidate);
                                cnt_map[candidate] = 1;
                            }
                        }
                    }

                    all_children.clear();
                    for (int u : unique_children) {
                        dumb_int cnt = cnt_map[u];
                        if (cnt == 1) {
                            all_children.push_back(u);
                        }
                        else {
                            int cnt_val_idx = make_temp_const(values, Value(cnt));
                            int cnt_node = make_temp_node(nodes, values, LazyOp::CONST, {}, cnt_val_idx);
                            int prod_idx = make_temp_product_node(nodes, values, {},
                                std::vector<int>{cnt_node, u});
                            prod_idx = simplify_tree(nodes, values, prod_idx);
                            all_children.push_back(prod_idx);
                        }
                    }
                }

                // 6. Дистрибутивность (только если есть произведения)
                if (has_product && all_children.size() >= 2) {
                    struct ProdInfo {
                        int idx;
                        std::vector<int> operands;
                    };
                    std::vector<ProdInfo> prods;

                    absl::flat_hash_map<Value, int, ValueHash, ValueEqual> const_cache;

                    auto get_const_node = [&](const Value& v) -> int {
                        auto it = const_cache.find(v);
                        if (it != const_cache.end()) return it->second;
                        int vi = make_temp_const(values, v);
                        int ni = make_temp_node(nodes, values, LazyOp::CONST, {}, vi);
                        const_cache[v] = ni;
                        return ni;
                        };

                    for (int child : all_children) {
                        if (nodes[child].op == LazyOp::PRODUCT) {
                            ProdInfo info;
                            info.idx = child;
                            const auto& prod_node = nodes[child];
                            for (const auto& v : prod_node.leaf_values) {
                                info.operands.push_back(get_const_node(v));
                            }
                            for (int c : prod_node.children)
                                info.operands.push_back(c);
                            prods.push_back(std::move(info));
                        }
                    }

                    if (prods.size() >= 2) {
                        absl::flat_hash_map<int, std::vector<size_t>> op_to_prods;
                        for (size_t pi = 0; pi < prods.size(); ++pi) {
                            for (int op : prods[pi].operands)
                                op_to_prods[op].push_back(pi);
                        }

                        std::vector<bool> used(prods.size(), false);
                        std::vector<int> new_factored;

                        for (const auto& [op, prod_indices] : op_to_prods) {
                            if (prod_indices.size() < 2) continue;
                            std::vector<size_t> group;
                            for (size_t pi : prod_indices) if (!used[pi]) group.push_back(pi);
                            if (group.size() < 2) continue;

                            std::vector<int> residuals;
                            for (size_t pi : group) {
                                used[pi] = true;
                                ProdInfo& prod = prods[pi];
                                std::vector<int> new_ops;
                                for (int o : prod.operands) if (o != op) new_ops.push_back(o);

                                int new_prod;
                                if (new_ops.empty()) {
                                    int one_idx = make_temp_const(values, Value(1));
                                    new_prod = make_temp_node(nodes, values, LazyOp::CONST, {}, one_idx);
                                }
                                else if (new_ops.size() == 1) {
                                    new_prod = new_ops[0];
                                }
                                else {
                                    std::vector<Value> leaf_vals;
                                    std::vector<int> children;
                                    for (int o : new_ops) {
                                        if (nodes[o].op == LazyOp::CONST)
                                            leaf_vals.push_back(values[nodes[o].value_idx]);
                                        else
                                            children.push_back(o);
                                    }
                                    new_prod = make_temp_product_node(nodes, values,
                                        std::move(leaf_vals), std::move(children));
                                }
                                new_prod = simplify_tree(nodes, values, new_prod);
                                residuals.push_back(new_prod);
                            }

                            int sum_idx = make_temp_sum_node(nodes, values, {}, residuals);
                            sum_idx = simplify_tree(nodes, values, sum_idx);

                            int factored = make_temp_product_node(nodes, values, {},
                                std::vector<int>{op, sum_idx});
                            factored = simplify_tree(nodes, values, factored);
                            new_factored.push_back(factored);
                        }

                        std::vector<int> updated_children;
                        for (size_t i = 0; i < prods.size(); ++i)
                            if (!used[i]) updated_children.push_back(prods[i].idx);
                        updated_children.insert(updated_children.end(),
                            new_factored.begin(), new_factored.end());
                        for (int child : all_children)
                            if (nodes[child].op != LazyOp::PRODUCT)
                                updated_children.push_back(child);
                        all_children = std::move(updated_children);
                    }
                }

                // 7. Каноническая сортировка
                sort_value_vector_canonical(all_leaf_values);
                std::sort(all_children.begin(), all_children.end(),
                    [&](int a, int b) {
                        if (nodes[a].hash != nodes[b].hash)
                            return nodes[a].hash < nodes[b].hash;
                        return a < b;
                    });

                // 8. Сокращение x + NEG(x) → 0
                std::vector<bool> keep(all_children.size(), true);
                for (size_t i = 0; i < all_children.size(); ++i) {
                    if (!keep[i]) continue;
                    int child_i = all_children[i];
                    const TempNode& node_i = nodes[child_i];
                    if (node_i.op == LazyOp::NEG && node_i.children.size() == 1) {
                        int neg_child = node_i.children[0];
                        for (size_t j = i + 1; j < all_children.size(); ++j) {
                            if (!keep[j]) continue;
                            if (all_children[j] == neg_child) {
                                keep[i] = false;
                                keep[j] = false;
                                break;
                            }
                        }
                    }
                }
                std::vector<int> after_cancel;
                for (size_t i = 0; i < all_children.size(); ++i)
                    if (keep[i]) after_cancel.push_back(all_children[i]);

                // 9. Сборка итогового узла
                if (all_leaf_values.empty() && after_cancel.empty()) {
                    int zero_idx = make_temp_const(values, Value(0));
                    simplified[idx] = make_temp_node(nodes, values, LazyOp::CONST, {}, zero_idx);
                }
                else if (all_leaf_values.size() == 1 && after_cancel.empty()) {
                    int const_idx = make_temp_const(values, all_leaf_values[0]);
                    simplified[idx] = make_temp_node(nodes, values, LazyOp::CONST, {}, const_idx);
                }
                else if (all_leaf_values.empty() && after_cancel.size() == 1) {
                    simplified[idx] = after_cancel[0];
                }
                else {
                    simplified[idx] = make_temp_sum_node(nodes, values,
                        std::move(all_leaf_values), std::move(after_cancel));
                }
                break;
            }

            case LazyOp::PRODUCT: {
                std::vector<Value> all_leaf_values = std::move(node.leaf_values);
                std::vector<int> all_children = std::move(node.children);

                // Flattening вложенных PRODUCT
                for (size_t i = 0; i < all_children.size(); ) {
                    int child = all_children[i];
                    if (nodes[child].op == LazyOp::PRODUCT) {
                        all_leaf_values.insert(all_leaf_values.end(),
                            std::make_move_iterator(nodes[child].leaf_values.begin()),
                            std::make_move_iterator(nodes[child].leaf_values.end()));
                        all_children.insert(all_children.end(),
                            nodes[child].children.begin(), nodes[child].children.end());
                        all_children.erase(all_children.begin() + i);
                    }
                    else ++i;
                }

                // Обработка нуля и единиц
                {
                    std::vector<Value> new_leaf;
                    bool found_zero = false;
                    for (auto& v : all_leaf_values) {
                        if (is_zero(v)) { found_zero = true; break; }
                        if (!is_one(v)) new_leaf.push_back(std::move(v));
                    }
                    if (found_zero) {
                        int zero_idx = make_temp_const(values, Value(0));
                        simplified[idx] = make_temp_node(nodes, values, LazyOp::CONST, {}, zero_idx);
                        break;
                    }
                    all_leaf_values = std::move(new_leaf);

                    std::vector<int> new_children;
                    for (int child : all_children) {
                        if (nodes[child].op == LazyOp::CONST) {
                            Value v = values[nodes[child].value_idx];
                            if (is_zero(v)) {
                                int zero_idx = make_temp_const(values, Value(0));
                                simplified[idx] = make_temp_node(nodes, values, LazyOp::CONST, {}, zero_idx);
                                break;
                            }
                            if (!is_one(v)) all_leaf_values.push_back(std::move(v));
                        }
                        else {
                            new_children.push_back(child);
                        }
                    }
                    all_children = std::move(new_children);
                }

                // Быстрая проверка: если нет детей и все множители уникальны – пропускаем группировку
                if (all_children.empty()) {
                    absl::flat_hash_map<Value, dumb_int, ValueHash, ValueEqual> freq;
                    bool has_duplicate = false;
                    for (const auto& v : all_leaf_values) {
                        auto it = freq.find(v);
                        if (it == freq.end()) {
                            freq.emplace(v, 1);
                        }
                        else {
                            has_duplicate = true;
                            break;
                        }
                    }
                    if (!has_duplicate) {
                        // Все уникальны – сортируем и создаём PRODUCT
                        sort_value_vector_canonical(all_leaf_values);
                        if (all_leaf_values.empty()) {
                            int one_idx = make_temp_const(values, Value(1));
                            simplified[idx] = make_temp_node(nodes, values, LazyOp::CONST, {}, one_idx);
                        }
                        else if (all_leaf_values.size() == 1) {
                            int const_idx = make_temp_const(values, all_leaf_values[0]);
                            simplified[idx] = make_temp_node(nodes, values, LazyOp::CONST, {}, const_idx);
                        }
                        else {
                            simplified[idx] = make_temp_product_node(nodes, values, std::move(all_leaf_values), {});
                        }
                        break;
                    }
                }

                // Группировка скалярных множителей (однопроходная)
                {
                    absl::flat_hash_map<Value, dumb_int, ValueHash, ValueEqual> freq;
                    for (auto& v : all_leaf_values) {
                        ++freq[std::move(v)];
                    }
                    all_leaf_values.clear();
                    for (auto& [val, cnt] : freq) {
                        if (cnt == 1) {
                            all_leaf_values.push_back(std::move(val));
                        }
                        else {
                            // POW(CONST(val), CONST(cnt))
                            int base_val_idx = make_temp_const(values, val);
                            int base_node = make_temp_node(nodes, values, LazyOp::CONST, {}, base_val_idx);
                            int exp_val_idx = make_temp_const(values, Value(cnt));
                            int exp_node = make_temp_node(nodes, values, LazyOp::CONST, {}, exp_val_idx);
                            int pow_idx = make_temp_node(nodes, values, LazyOp::POW, { base_node, exp_node });
                            pow_idx = simplify_tree(nodes, values, pow_idx);
                            all_children.push_back(pow_idx);
                        }
                    }
                }

                // Свёртка одинаковых дочерних узлов (A*A*A... → A^N)
                {
                    absl::flat_hash_map<uint64_t, std::vector<int>> hash_buckets;
                    for (int child : all_children) {
                        hash_buckets[nodes[child].hash].push_back(child);
                    }

                    std::vector<int> unique;
                    absl::flat_hash_map<int, dumb_int> cnt_map;
                    for (auto& [hash_val, bucket] : hash_buckets) {
                        for (int candidate : bucket) {
                            bool found = false;
                            for (int u : unique) {
                                if (nodes[u].hash != nodes[candidate].hash) continue;
                                if (temp_nodes_equal(nodes, values, u, candidate)) {
                                    cnt_map[u] += 1;
                                    found = true;
                                    break;
                                }
                            }
                            if (!found) {
                                unique.push_back(candidate);
                                cnt_map[candidate] = 1;
                            }
                        }
                    }
                    all_children.clear();
                    for (int u : unique) {
                        dumb_int cnt = cnt_map[u];
                        if (cnt == 1) {
                            all_children.push_back(u);
                        }
                        else {
                            int exp_idx = make_temp_const(values, Value(cnt));
                            int exp_node = make_temp_node(nodes, values, LazyOp::CONST, {}, exp_idx);
                            int pow_idx = make_temp_node(nodes, values, LazyOp::POW, { u, exp_node });
                            pow_idx = simplify_tree(nodes, values, pow_idx);
                            all_children.push_back(pow_idx);
                        }
                    }
                }

                // Каноническая сортировка
                sort_value_vector_canonical(all_leaf_values);
                std::sort(all_children.begin(), all_children.end(),
                    [&](int a, int b) {
                        if (nodes[a].hash != nodes[b].hash)
                            return nodes[a].hash < nodes[b].hash;
                        return a < b;
                    });

                // Сокращение x * RECIP(x) → 1
                std::vector<bool> keep(all_children.size(), true);
                for (size_t i = 0; i < all_children.size(); ++i) {
                    if (!keep[i]) continue;
                    int child_i = all_children[i];
                    const TempNode& node_i = nodes[child_i];
                    if (node_i.op == LazyOp::RECIP && node_i.children.size() == 1) {
                        int recip_child = node_i.children[0];
                        for (size_t j = i + 1; j < all_children.size(); ++j) {
                            if (!keep[j]) continue;
                            if (all_children[j] == recip_child) {
                                keep[i] = false;
                                keep[j] = false;
                                all_leaf_values.push_back(Value(1));
                                break;
                            }
                        }
                    }
                }
                std::vector<int> after_cancel;
                for (size_t i = 0; i < all_children.size(); ++i)
                    if (keep[i]) after_cancel.push_back(all_children[i]);

                all_leaf_values.erase(
                    std::remove_if(all_leaf_values.begin(), all_leaf_values.end(),
                        [](const Value& v) { return is_one(v); }),
                    all_leaf_values.end());

                if (all_leaf_values.empty() && after_cancel.empty()) {
                    int one_idx = make_temp_const(values, Value(1));
                    simplified[idx] = make_temp_node(nodes, values, LazyOp::CONST, {}, one_idx);
                }
                else if (all_leaf_values.size() == 1 && after_cancel.empty()) {
                    int const_idx = make_temp_const(values, all_leaf_values[0]);
                    simplified[idx] = make_temp_node(nodes, values, LazyOp::CONST, {}, const_idx);
                }
                else if (all_leaf_values.empty() && after_cancel.size() == 1) {
                    simplified[idx] = after_cancel[0];
                }
                else {
                    simplified[idx] = make_temp_product_node(nodes, values,
                        std::move(all_leaf_values), std::move(after_cancel));
                }
                break;
            }

            case LazyOp::NEG: {
                int child = node.children[0];
                if (nodes[child].op == LazyOp::NEG) {
                    simplified[idx] = nodes[child].children[0];
                }
                else {
                    simplified[idx] = idx;
                }
                break;
            }
            case LazyOp::RECIP: {
                int child = node.children[0];
                if (nodes[child].op == LazyOp::RECIP) {
                    simplified[idx] = nodes[child].children[0];
                }
                else {
                    simplified[idx] = idx;
                }
                break;
            }
            case LazyOp::EXP: {
                int child = node.children[0];
                if (nodes[child].op == LazyOp::LOG) {
                    simplified[idx] = nodes[child].children[0];
                }
                else {
                    simplified[idx] = idx;
                }
                break;
            }
            case LazyOp::LOG: {
                int child = node.children[0];
                if (nodes[child].op == LazyOp::EXP) {
                    simplified[idx] = nodes[child].children[0];
                }
                else {
                    simplified[idx] = idx;
                }
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

                if (is_temp_zero(exp_node, values)) {
                    int one_idx = make_temp_const(values, Value(1));
                    simplified[idx] = make_temp_node(nodes, values, LazyOp::CONST, {}, one_idx);
                }
                else if (is_temp_one(exp_node, values)) {
                    simplified[idx] = base;
                }
                else if (is_temp_one(base_node, values)) {
                    int one_idx = make_temp_const(values, Value(1));
                    simplified[idx] = make_temp_node(nodes, values, LazyOp::CONST, {}, one_idx);
                }
                else if (is_temp_zero(base_node, values) && is_temp_positive_const(exp_node, values)) {
                    int zero_idx = make_temp_const(values, Value(0));
                    simplified[idx] = make_temp_node(nodes, values, LazyOp::CONST, {}, zero_idx);
                }
                else if (base_node.op == LazyOp::POW &&
                    nodes[base_node.children[1]].op == LazyOp::CONST &&
                    exp_node.op == LazyOp::CONST) {
                    const Value& b_val = values[nodes[base_node.children[1]].value_idx];
                    const Value& c_val = values[exp_node.value_idx];
                    if (denominator(b_val) == 1 && denominator(c_val) == 1) {
                        Value prod = b_val * c_val;
                        int new_exp = make_temp_const(values, prod);
                        int new_base = base_node.children[0];
                        simplified[idx] = make_temp_node(nodes, values, LazyOp::POW, { new_base, new_exp }, -1, node.eps_idx);
                    }
                    else {
                        simplified[idx] = idx;
                    }
                }
                else {
                    simplified[idx] = idx;
                }
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