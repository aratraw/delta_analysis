// simplify_impl.h
// ============================================================================
// СТРАТЕГИЯ УПРОЩЕНИЯ (SIMPLIFICATION) ВЕРСИИ 2.0
// ============================================================================
//
// 1.  ОСНОВНАЯ ЦЕЛЬ: ПРЕОБРАЗОВАНИЕ ДЕРЕВА В КАНОНИЧЕСКУЮ ФОРМУ
//     БЕЗ ВЫЧИСЛЕНИЯ КОНСТАНТНЫХ ПОДВЫРАЖЕНИЙ.
//
//     Упрощение НЕ ДОЛЖНО:
//       - сворачивать 1+2 в 3
//       - вычислять sqrt(2) или exp(1) до рационального значения
//       - выполнять любые нетривиальные арифметические действия над константами
//         (сложение, умножение, GCD, LCM, перевод в double и т.д.)
//
//     Упрощение ДОЛЖНО:
//       - расплющивать вложенные SUM и PRODUCT (flattening), если они есть.
//       - удалять нейтральные элементы (0 в сумме, 1 в произведении)
//       - канонизировать операции по порядку операндов: сортировать детей SUM/PRODUCT по хэшу (детерминированный порядок)
//       - сокращать противоположные пары: x + NEG(x) → 0, x * RECIP(x) → 1
//       - выполнять алгебраические упрощения: NEG(NEG(x)) → x, RECIP(RECIP(x)) → x,
//         EXP(LOG(x)) → x, LOG(EXP(x)) → x, x^0 → 1, x^1 → x, 1^x → 1, 0^positive → 0,
//         (a^b)^c → a^(b*c) для целых показателей.
//
// 2.  ПОЧЕМУ НЕЛЬЗЯ "ПРОСТО СВЕРНУТЬ ВСЁ В КОНСТАНТЫ"?
//
//     А) РАЗДЕЛЕНИЕ ОТВЕТСТВЕННОСТИ:
//        Упрощение готовит дерево к будущему вычислению, но не выполняет само
//        вычисление. Вычисление (eval) – отдельная операция, которая может быть
//        выполнена позже - ОПТИМАЛЬНЫМ АЛГОРИТМОМ КОТОРЫЙ ЗАНИМАЕТСЯ ТОЛЬКО ВЫЧИСЛЕНИЕМ.
//
//     Б) ПРОИЗВОДИТЕЛЬНОСТЬ:
//        Свёртка констант в процессе упрощения приводит к преждевременным
//        затратам на GCD/LCM и переход в BigStorage. Например, гармонический ряд
//        из 10 000 членов, если каждый раз сворачивать константы, вызовет
//        многократное сокращение дробей, что катастрофически медленно.
//
// 3.  БУДУЩЕЕ РАСШИРЕНИЕ: СИМВОЛЬНЫЕ ВЫЧИСЛЕНИЯ
//
//     Если в библиотеку будут добавлены неизвестные (symbolic variables),
//     будет введена отдельная функция упрощения – "решение дерева
//     относительно неизвестных". В этой функции можно будет выполнять полную свёртку
//     констант (например, вычислять 2+3, но не вычислять sin(pi/4)), а также
//     применять дистрибутивность, группировку подобных членов и другие
//     преобразования. Будет ли эта функция реализована как simplify_* или eval_*, пока не имеет значения.
//
// 4.  РЕАЛИЗАЦИЯ В ДАННОМ ФАЙЛЕ:
//
//     Функция simplify_tree() получает на вход временное дерево (TempNode) и
//     локальный пул значений (values). Она выполняет пост-порядный обход и
//     применяет описанные выше правила. В результате возвращается индекс
//     упрощённого корня в том же массиве TempNode. После этого канонизированное
//     дерево интернируется в глобальный пул NodePool.
//
//     ВАЖНО: Все ветки, которые ранее сворачивали константы (для NEG, RECIP,
//     EXP, LOG, SQRT, SIN, COS, ACOS, POW), удалены. Оставлены только чисто
//     алгебраические преобразования. Исключение – случаи, когда результат
//     очевидно не зависит от значения (x^0 → 1, 0^positive → 0 и т.п.).
//
// ============================================================================

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
    // ... (остальной код без изменений, как в предыдущем ответе)
}
#pragma once

#include "lazy_nodes.h"   // для TempNode
#include "storage.h"
#include "evaluation_core.h"
#include <vector>
#include <algorithm>
#include <stack>
#include <unordered_map>
#include <optional>

namespace delta::internal {

    // ----------------------------------------------------------------------------
    // Вспомогательные функции для работы с TempNode
    // ----------------------------------------------------------------------------

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
        if (v.tag == ValueType::Small) {
            const SmallStorage& s = v.storage.small;
            return s.num == -1 && s.den == 1;
        }
        return false;
    }

    inline double temp_const_value(const TempNode& node, const std::vector<Value>& values) {
        return to_double(values[node.value_idx]);
    }

    inline int make_temp_const(std::vector<Value>& values, const Value& v) {
        int idx = static_cast<int>(values.size());
        values.push_back(v);
        return idx;
    }

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

    // ----------------------------------------------------------------------------
    // Основная функция упрощения дерева TempNode
    // ----------------------------------------------------------------------------
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
            for (int child : nodes[idx].children) {
                st.push(child);
            }
        }

        for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
            int idx = *it;
            TempNode& node = nodes[idx];
            std::vector<int> simp_children;
            for (int child : node.children) {
                simp_children.push_back(simplified[child]);
            }

            if (node.op != LazyOp::CONST && !simp_children.empty()) {
                node.children = std::move(simp_children);
            }

            switch (node.op) {
            case LazyOp::CONST:
                simplified[idx] = idx;
                break;

            case LazyOp::SUM: {
                std::vector<int> flat_children;
                for (int child : node.children) {
                    if (nodes[child].op == LazyOp::SUM) {
                        for (int sub : nodes[child].children) {
                            flat_children.push_back(sub);
                        }
                    }
                    else {
                        flat_children.push_back(child);
                    }
                }

                std::vector<int> filtered;
                for (int child : flat_children) {
                    if (!(nodes[child].op == LazyOp::CONST && is_zero(values[nodes[child].value_idx]))) {
                        filtered.push_back(child);
                    }
                }

                std::sort(filtered.begin(), filtered.end(),
                    [&](int a, int b) {
                        uint64_t ha = nodes[a].hash;
                        uint64_t hb = nodes[b].hash;
                        if (ha != hb) return ha < hb;
                        return a < b;
                    });

                std::vector<bool> keep(filtered.size(), true);
                for (size_t i = 0; i < filtered.size(); ++i) {
                    if (!keep[i]) continue;
                    int child_i = filtered[i];
                    if (nodes[child_i].op == LazyOp::NEG && nodes[child_i].children.size() == 1) {
                        int neg_child = nodes[child_i].children[0];
                        for (size_t j = i + 1; j < filtered.size(); ++j) {
                            if (!keep[j]) continue;
                            int child_j = filtered[j];
                            if (child_j == neg_child) {
                                keep[i] = false;
                                keep[j] = false;
                                break;
                            }
                        }
                    }
                }
                std::vector<int> after_cancel;
                for (size_t i = 0; i < filtered.size(); ++i) {
                    if (keep[i]) after_cancel.push_back(filtered[i]);
                }

                if (after_cancel.empty()) {
                    int zero_idx = make_temp_const(values, Value(SmallStorage(0)));
                    int new_idx = make_temp_node(nodes, values, LazyOp::CONST, {}, zero_idx);
                    simplified[idx] = new_idx;
                }
                else if (after_cancel.size() == 1) {
                    simplified[idx] = after_cancel[0];
                }
                else {
                    int new_idx = make_temp_node(nodes, values, LazyOp::SUM, std::move(after_cancel));
                    simplified[idx] = new_idx;
                }
                break;
            }

            case LazyOp::PRODUCT: {
                std::vector<int> flat_children;
                for (int child : node.children) {
                    if (nodes[child].op == LazyOp::PRODUCT) {
                        for (int sub : nodes[child].children) {
                            flat_children.push_back(sub);
                        }
                    }
                    else {
                        flat_children.push_back(child);
                    }
                }

                std::vector<int> filtered;
                for (int child : flat_children) {
                    if (!(nodes[child].op == LazyOp::CONST && is_one(values[nodes[child].value_idx]))) {
                        filtered.push_back(child);
                    }
                }

                std::sort(filtered.begin(), filtered.end(),
                    [&](int a, int b) {
                        uint64_t ha = nodes[a].hash;
                        uint64_t hb = nodes[b].hash;
                        if (ha != hb) return ha < hb;
                        return a < b;
                    });

                std::vector<bool> keep(filtered.size(), true);
                for (size_t i = 0; i < filtered.size(); ++i) {
                    if (!keep[i]) continue;
                    int child_i = filtered[i];
                    if (nodes[child_i].op == LazyOp::RECIP && nodes[child_i].children.size() == 1) {
                        int recip_child = nodes[child_i].children[0];
                        for (size_t j = i + 1; j < filtered.size(); ++j) {
                            if (!keep[j]) continue;
                            int child_j = filtered[j];
                            if (child_j == recip_child) {
                                keep[i] = false;
                                keep[j] = false;
                                break;
                            }
                        }
                    }
                }
                std::vector<int> after_cancel;
                for (size_t i = 0; i < filtered.size(); ++i) {
                    if (keep[i]) after_cancel.push_back(filtered[i]);
                }

                if (after_cancel.empty()) {
                    int one_idx = make_temp_const(values, Value(SmallStorage(1)));
                    int new_idx = make_temp_node(nodes, values, LazyOp::CONST, {}, one_idx);
                    simplified[idx] = new_idx;
                }
                else if (after_cancel.size() == 1) {
                    simplified[idx] = after_cancel[0];
                }
                else {
                    int new_idx = make_temp_node(nodes, values, LazyOp::PRODUCT, std::move(after_cancel));
                    simplified[idx] = new_idx;
                }
                break;
            }

            case LazyOp::NEG: {
                int child = node.children[0];
                if (nodes[child].op == LazyOp::NEG) {
                    simplified[idx] = nodes[child].children[0];
                    break;
                }
                // Убрано сворачивание констант
                simplified[idx] = idx;
                break;
            }

            case LazyOp::RECIP: {
                int child = node.children[0];
                if (nodes[child].op == LazyOp::RECIP) {
                    simplified[idx] = nodes[child].children[0];
                    break;
                }
                // Убрано сворачивание констант
                simplified[idx] = idx;
                break;
            }

            case LazyOp::EXP: {
                int child = node.children[0];
                if (nodes[child].op == LazyOp::LOG) {
                    simplified[idx] = nodes[child].children[0];
                    break;
                }
                // Убрано сворачивание констант
                simplified[idx] = idx;
                break;
            }

            case LazyOp::LOG: {
                int child = node.children[0];
                if (nodes[child].op == LazyOp::EXP) {
                    simplified[idx] = nodes[child].children[0];
                    break;
                }
                // Убрано сворачивание констант
                simplified[idx] = idx;
                break;
            }

            case LazyOp::SQRT:
            case LazyOp::SIN:
            case LazyOp::COS:
            case LazyOp::ACOS: {
                // Убрано сворачивание констант; оставлены только алгебраические упрощения (пока нет)
                simplified[idx] = idx;
                break;
            }

            case LazyOp::PI:
            case LazyOp::E:
                simplified[idx] = idx;
                break;

            case LazyOp::POW: {
                int base = node.children[0];
                int exp = node.children[1];

                // x^0 -> 1
                if (nodes[exp].op == LazyOp::CONST && is_zero(values[nodes[exp].value_idx])) {
                    int one_idx = make_temp_const(values, Value(SmallStorage(1)));
                    int new_idx = make_temp_node(nodes, values, LazyOp::CONST, {}, one_idx);
                    simplified[idx] = new_idx;
                    break;
                }
                // x^1 -> x
                if (nodes[exp].op == LazyOp::CONST && is_one(values[nodes[exp].value_idx])) {
                    simplified[idx] = base;
                    break;
                }
                // 1^x -> 1
                if (nodes[base].op == LazyOp::CONST && is_one(values[nodes[base].value_idx])) {
                    int one_idx = make_temp_const(values, Value(SmallStorage(1)));
                    int new_idx = make_temp_node(nodes, values, LazyOp::CONST, {}, one_idx);
                    simplified[idx] = new_idx;
                    break;
                }
                // 0^positive -> 0
                if (nodes[base].op == LazyOp::CONST && is_zero(values[nodes[base].value_idx]) &&
                    nodes[exp].op == LazyOp::CONST && is_positive(values[nodes[exp].value_idx])) {
                    int zero_idx = make_temp_const(values, Value(SmallStorage(0)));
                    int new_idx = make_temp_node(nodes, values, LazyOp::CONST, {}, zero_idx);
                    simplified[idx] = new_idx;
                    break;
                }
                // (a^b)^c -> a^(b*c) для целых показателей
                if (nodes[base].op == LazyOp::POW &&
                    nodes[nodes[base].children[1]].op == LazyOp::CONST &&
                    nodes[exp].op == LazyOp::CONST) {
                    const Value& b_val = values[nodes[nodes[base].children[1]].value_idx];
                    const Value& c_val = values[nodes[exp].value_idx];
                    auto is_integer_val = [](const Value& v) {
                        if (v.tag == ValueType::Small) return v.storage.small.den == 1;
                        if (v.tag == ValueType::Big) return v.storage.big.denominator() == 1;
                        return false;
                        };
                    if (is_integer_val(b_val) && is_integer_val(c_val)) {
                        Value prod = eager_mul(b_val, c_val);
                        int new_exp = make_temp_const(values, prod);
                        int new_base = nodes[base].children[0];
                        int new_pow = make_temp_node(nodes, values, LazyOp::POW, { new_base, new_exp }, -1, node.eps_idx);
                        simplified[idx] = new_pow;
                        break;
                    }
                }
                // Убрано сворачивание констант (блок, вычисляющий eager_pow)
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