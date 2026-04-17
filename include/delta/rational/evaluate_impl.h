// evaluate_impl.h
#pragma once

#include "node_pool.h"
#include "evaluation_core.h"

#include <stack>
#include <vector>
#include <optional>
#include <algorithm>
#include <stdexcept>

namespace delta::internal {

    // ----------------------------------------------------------------------------
    // compute_node – вычисление значения узла по его операции и аргументам
    // ----------------------------------------------------------------------------
    inline Value compute_node(LazyOp op, const Value& left, const Value& right, const Value& eps) {
        switch (op) {
        case LazyOp::SUM:      // не должен вызываться для SUM (он обрабатывается отдельно в evaluate)
            throw std::logic_error("compute_node: SUM not supported (use batch_add)");
        case LazyOp::PRODUCT:  // аналогично, PRODUCT обрабатывается отдельно
            throw std::logic_error("compute_node: PRODUCT not supported (use eager_mul loop)");
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

    // ----------------------------------------------------------------------------
    // batch_add_values – эффективное суммирование группы Value с общим знаменателем
    // ----------------------------------------------------------------------------
    inline Value batch_add_values(const std::vector<int>& indices,
        const std::vector<std::optional<Value>>& cache) {
        if (indices.empty()) {
            return Value(SmallStorage(0));
        }
        if (indices.size() == 1) {
            return cache[indices[0]].value();
        }

        // Быстрый путь: все SmallStorage и количество ≤ 64
        bool all_small = true;
        for (int idx : indices) {
            const Value& v = cache[idx].value();
            if (v.tag != ValueType::Small) {
                all_small = false;
                break;
            }
        }

        if (all_small && indices.size() <= 64) {
            absl::uint128 common_denom = 1;
            std::vector<absl::int128> nums;
            std::vector<absl::uint128> dens;
            nums.reserve(indices.size());
            dens.reserve(indices.size());

            for (int idx : indices) {
                const SmallStorage& s = cache[idx].value().storage.small;
                nums.push_back(s.num);
                dens.push_back(s.den);

                absl::uint128 g = binary_gcd(common_denom, s.den);
                absl::uint128 lcm_candidate = common_denom / g;
                if (lcm_candidate > (std::numeric_limits<absl::uint128>::max)() / s.den) {
                    all_small = false;
                    break;
                }
                common_denom = lcm_candidate * s.den;
            }

            if (all_small) {
                absl::int128 sum_num = 0;
                bool overflow = false;
                for (size_t i = 0; i < indices.size(); ++i) {
                    absl::uint128 factor = common_denom / dens[i];
                    if (nums[i] != 0 && factor > 0) {
                        absl::uint128 abs_num = nums[i] < 0 ? static_cast<absl::uint128>(-nums[i]) : static_cast<absl::uint128>(nums[i]);
                        if (abs_num > (std::numeric_limits<absl::uint128>::max)() / factor) {
                            overflow = true;
                            break;
                        }
                    }
                    absl::int128 term = nums[i] * static_cast<absl::int128>(factor);
                    if (would_overflow_add(sum_num, term)) {
                        overflow = true;
                        break;
                    }
                    sum_num += term;
                }
                if (!overflow) {
                    bool negative = (sum_num < 0);
                    absl::uint128 abs_sum = negative ? static_cast<absl::uint128>(-sum_num) : static_cast<absl::uint128>(sum_num);
                    absl::uint128 g = binary_gcd(abs_sum, common_denom);
                    if (g > 1) {
                        abs_sum /= g;
                        common_denom /= g;
                    }
                    if (abs_sum <= static_cast<absl::uint128>((std::numeric_limits<absl::int128>::max)()) &&
                        common_denom <= (std::numeric_limits<absl::uint128>::max)()) {
                        absl::int128 final_num = negative ? -static_cast<absl::int128>(abs_sum) : static_cast<absl::int128>(abs_sum);
                        SmallStorage result_small(final_num, common_denom);
                        Value result_val(result_small, false);
                        result_val.normalize();
                        return result_val;
                    }
                }
            }
        }

        // Общий путь через dumb_int
        using delta::internal::dumb_int;
        std::vector<std::pair<dumb_int, dumb_int>> norms;
        norms.reserve(indices.size());
        dumb_int common_denom(1);

        for (int idx : indices) {
            auto nd = normalize_to_dumb_int(cache[idx].value());
            norms.emplace_back(nd.first, nd.second);
            common_denom = boost::multiprecision::lcm(common_denom, nd.second);
        }

        dumb_int sum_num(0);
        for (size_t i = 0; i < norms.size(); ++i) {
            dumb_int factor = common_denom / norms[i].second;
            sum_num += norms[i].first * factor;
        }

        dumb_int g = boost::multiprecision::gcd(sum_num, common_denom);
        if (g != 0) {
            sum_num /= g;
            common_denom /= g;
        }

        if (fits_in_int128(sum_num) && fits_in_uint128(common_denom)) {
            SmallStorage result_small(dumb_int_to_int128(sum_num), dumb_int_to_uint128(common_denom));
            Value result_val(result_small, false);
            result_val.normalize();
            return result_val;
        }
        return Value(BigStorage(sum_num, common_denom));
    }

    // ----------------------------------------------------------------------------
    // process_batch_list – разбивает большой список на батчи и суммирует
    // ----------------------------------------------------------------------------
    inline Value process_batch_list(const std::vector<int>& indices,
        size_t batch_size,
        const std::vector<std::optional<Value>>& cache) {
        if (indices.empty()) return Value(SmallStorage(0));
        if (indices.size() <= batch_size) {
            return batch_add_values(indices, cache);
        }
        std::vector<Value> partial_sums;
        partial_sums.reserve((indices.size() + batch_size - 1) / batch_size);
        for (size_t start = 0; start < indices.size(); start += batch_size) {
            size_t end = std::min(start + batch_size, indices.size());
            std::vector<int> chunk(indices.begin() + start, indices.begin() + end);
            partial_sums.push_back(batch_add_values(chunk, cache));
        }
        Value sum = partial_sums[0];
        for (size_t i = 1; i < partial_sums.size(); ++i) {
            sum = eager_add(sum, partial_sums[i]);
        }
        return sum;
    }

    // ----------------------------------------------------------------------------
    // evaluate – итеративное вычисление чистого дерева
    // ----------------------------------------------------------------------------
    inline Value evaluate(int root_idx) {
        const auto& nodes = pool.nodes;
        const auto& values = pool.values;
        const size_t n = nodes.size();

        std::vector<std::optional<Value>> cache(n);
        std::stack<int> st;
        st.push(root_idx);

        constexpr size_t MAX_BATCH_SIZE = 64;

        while (!st.empty()) {
            int idx = st.top();
            if (cache[idx].has_value()) {
                st.pop();
                continue;
            }

            const Node& node = nodes[idx];

            // CONST – сразу вычисляем
            if (node.op == LazyOp::CONST) {
                cache[idx] = values[node.value_idx];
                st.pop();
                continue;
            }

            // Проверяем готовность детей
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

            // Все дети готовы – вычисляем текущий узел
            Value result;
            switch (node.op) {
            // БАТЧ-ВЕРСИЯ КОТОРАЯ В ТЕОРИИ ДОЛЖНА РАБОТАТЬ БЫСТРЕЕ НО НУЖНО ПРОСТО СДЕЛАТЬ БАТЧИ РЕКУРСИВНЫМИ ПО 32/16/8
            // В ТЕКУЩЕМ ИСПОЛНЕНИИ НЕ МОЖЕТ ПОСЧИТАТЬ ДАЖЕ 100 узлов за время жизни вселенной.
            // Но теперь мы хотя бы знаем достоверно в чём была проблема...
            // 
            //case LazyOp::SUM: {
            //    // Собираем индексы детей и используем batch-суммирование
            //    std::vector<int> child_indices(node.children->begin(), node.children->end());
            //    result = process_batch_list(child_indices, MAX_BATCH_SIZE, cache);
            //    break;
            //}
            // Тупая последовательная версия которую предлагают нейронки потому что это гарантированное решение. 
            // Что в целом так и есть, но я всё ещё считаю что правильный батчинг должен быть быстрее. 
            case LazyOp::SUM: {
                // Последовательное сложение с нормализацией после каждого шага
                // Это предотвращает экспоненциальный рост знаменателя, в отличие от глобального LCM.
                const auto& children = *node.children;
                if (children.empty()) {
                    result = Value(SmallStorage(0));
                }
                else {
                    result = cache[children[0]].value();
                    for (size_t i = 1; i < children.size(); ++i) {
                        result = eager_add(result, cache[children[i]].value());
                    }
                }
                break;
            }
            case LazyOp::PRODUCT: {
                // Последовательное умножение (можно добавить batch-оптимизацию позже)
                Value prod = cache[(*node.children)[0]].value();
                for (size_t i = 1; i < node.children->size(); ++i) {
                    prod = eager_mul(prod, cache[(*node.children)[i]].value());
                }
                result = prod;
                break;
            }
            default: {
                // Унарные, бинарные, PI, E, POW
                Value left = (node.child0 != -1) ? cache[node.child0].value() : Value{};
                Value right = (node.child1 != -1) ? cache[node.child1].value() : Value{};
                Value eps = (node.eps_idx != -1) ? values[node.eps_idx] : Value{};
                result = compute_node(node.op, left, right, eps);
                break;
            }
            }

            cache[idx] = result;
            st.pop();
        }

        return cache[root_idx].value();
    }

} // namespace delta::internal