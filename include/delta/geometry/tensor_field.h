// include/delta/geometry/tensor_field.h
#pragma once

#include <map>
#include <concepts>
#include <stdexcept>
#include <Eigen/Core>
#include "delta/core/rational.h"

namespace delta::geometry {

    // -------------------------------------------------------------------------
    // Вспомогательный шаблон для выбора типа тензора по рангу и размерности
    // -------------------------------------------------------------------------
    template<typename Scalar, int Rank, int Dim>
    struct TensorTypeSelector;

    template<typename Scalar, int Dim>
    struct TensorTypeSelector<Scalar, 0, Dim> {
        using type = Scalar;
    };

    template<typename Scalar, int Dim>
    struct TensorTypeSelector<Scalar, 1, Dim> {
        using type = Eigen::Matrix<Scalar, Dim, 1>;
    };

    template<typename Scalar, int Dim>
    struct TensorTypeSelector<Scalar, 2, Dim> {
        using type = Eigen::Matrix<Scalar, Dim, Dim>;
    };

    template<typename Scalar, int Rank, int Dim>
    using TensorType = typename TensorTypeSelector<Scalar, Rank, Dim>::type;

    // -------------------------------------------------------------------------
    // Концепт TensorField (опционально)
    // -------------------------------------------------------------------------
    template<typename T, typename Addr>
    concept TensorFieldConcept = requires(T t, const T ct, Addr a) {
        typename T::value_type;
        typename T::address_type;
        { t.set(a, typename T::value_type{}) } -> std::same_as<void>;
        { ct.at(a) } -> std::convertible_to<typename T::value_type>;
        { ct.contains(a) } -> std::convertible_to<bool>;
        { ct.begin() } -> std::input_or_output_iterator;
        { ct.end() } -> std::input_or_output_iterator;
    };

    // -------------------------------------------------------------------------
    // Базовый класс тензорного поля
    // -------------------------------------------------------------------------
    template<typename Addr, typename Scalar, int Rank, int Dim = 0>
    class TensorField {
        static_assert(Rank >= 0 && Rank <= 2, "Only ranks 0,1,2 are currently supported");
    public:
        using value_type = TensorType<Scalar, Rank, Dim>;
        using address_type = Addr;
        using scalar_type = Scalar;
        static constexpr int rank = Rank;
        static constexpr int dim = Dim;

        TensorField() = default;

        void set(const Addr& addr, const value_type& val) {
            data_[addr] = val;
        }

        value_type& operator[](const Addr& addr) {
            return data_[addr];
        }

        const value_type& at(const Addr& addr) const {
            auto it = data_.find(addr);
            if (it == data_.end()) {
                throw std::out_of_range("TensorField: address not found");
            }
            return it->second;
        }

        bool contains(const Addr& addr) const {
            return data_.find(addr) != data_.end();
        }

        auto begin() const { return data_.begin(); }
        auto end() const { return data_.end(); }
        auto begin() { return data_.begin(); }
        auto end() { return data_.end(); }

        std::size_t size() const { return data_.size(); }

        template<typename Func>
        void apply(Func&& f) {
            for (auto& [addr, val] : data_) {
                f(val);
            }
        }

        template<typename Func>
        void apply(Func&& f) const {
            for (const auto& [addr, val] : data_) {
                f(val);
            }
        }

    protected:
        std::map<Addr, value_type> data_;
    };

    // -------------------------------------------------------------------------
    // Свободные функции для алгебраических операций
    // -------------------------------------------------------------------------

    template<typename Addr, typename Scalar, int Rank, int Dim>
    TensorField<Addr, Scalar, Rank, Dim>
        operator+(const TensorField<Addr, Scalar, Rank, Dim>& a,
            const TensorField<Addr, Scalar, Rank, Dim>& b) {
        TensorField<Addr, Scalar, Rank, Dim> result;
        for (const auto& [addr, val_a] : a) {
            if (b.contains(addr)) {
                result.set(addr, val_a + b.at(addr));
            }
        }
        return result;
    }

    template<typename Addr, typename Scalar, int Rank, int Dim>
    TensorField<Addr, Scalar, Rank, Dim>
        operator-(const TensorField<Addr, Scalar, Rank, Dim>& a,
            const TensorField<Addr, Scalar, Rank, Dim>& b) {
        TensorField<Addr, Scalar, Rank, Dim> result;
        for (const auto& [addr, val_a] : a) {
            if (b.contains(addr)) {
                result.set(addr, val_a - b.at(addr));
            }
        }
        return result;
    }

    template<typename Addr, typename Scalar, int Rank, int Dim>
    TensorField<Addr, Scalar, Rank, Dim>
        operator*(const TensorField<Addr, Scalar, Rank, Dim>& f, const Scalar& s) {
        TensorField<Addr, Scalar, Rank, Dim> result;
        for (const auto& [addr, val] : f) {
            result.set(addr, val * s);
        }
        return result;
    }

    template<typename Addr, typename Scalar, int Rank, int Dim>
    TensorField<Addr, Scalar, Rank, Dim>
        operator*(const Scalar& s, const TensorField<Addr, Scalar, Rank, Dim>& f) {
        return f * s;
    }

    template<typename Addr, typename Scalar, int Rank, int Dim>
    TensorField<Addr, Scalar, Rank, Dim>&
        operator*=(TensorField<Addr, Scalar, Rank, Dim>& f, const Scalar& s) {
        f.apply([s](auto& val) { val *= s; });
        return f;
    }

    // -------------------------------------------------------------------------
    // Тензорное произведение (для всех комбинаций рангов 0,1,2)
    // -------------------------------------------------------------------------
    template<typename Addr, typename Scalar, int RankA, int RankB, int Dim>
    auto tensor_product(const TensorField<Addr, Scalar, RankA, Dim>& a,
        const TensorField<Addr, Scalar, RankB, Dim>& b) {
        constexpr int RankR = RankA + RankB;
        TensorField<Addr, Scalar, RankR, Dim> result;
        for (const auto& [addr, valA] : a) {
            if (b.contains(addr)) {
                const auto& valB = b.at(addr);
                if constexpr (RankA == 0 && RankB == 0) {
                    result.set(addr, valA * valB);
                }
                else if constexpr (RankA == 0 && RankB == 1) {
                    result.set(addr, valA * valB);
                }
                else if constexpr (RankA == 0 && RankB == 2) {
                    result.set(addr, valA * valB);
                }
                else if constexpr (RankA == 1 && RankB == 0) {
                    result.set(addr, valA * valB);
                }
                else if constexpr (RankA == 1 && RankB == 1) {
                    // вектор * вектор -> матрица (внешнее произведение)
                    result.set(addr, valA * valB.transpose());
                }
                else if constexpr (RankA == 1 && RankB == 2) {
                    // вектор * матрица -> ранг 3, не поддерживаем
                    static_assert(RankA + RankB <= 2,
                        "Tensor product of vector and matrix not supported");
                }
                else if constexpr (RankA == 2 && RankB == 0) {
                    result.set(addr, valA * valB);
                }
                else if constexpr (RankA == 2 && RankB == 1) {
                    static_assert(RankA + RankB <= 2,
                        "Tensor product of matrix and vector not supported");
                }
                else if constexpr (RankA == 2 && RankB == 2) {
                    static_assert(RankA + RankB <= 2,
                        "Tensor product of matrix and matrix not supported");
                }
            }
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Свёртка (contract) – обобщение следа
    // -------------------------------------------------------------------------
    template<typename Addr, typename Scalar, int Dim>
    TensorField<Addr, Scalar, 0, Dim> contract(const TensorField<Addr, Scalar, 2, Dim>& t) {
        return trace(t);  // пока только для матриц
    }

    // -------------------------------------------------------------------------
    // Свёртка для матриц (след)
    // -------------------------------------------------------------------------
    template<typename Addr, typename Scalar, int Dim>
    TensorField<Addr, Scalar, 0, Dim> trace(const TensorField<Addr, Scalar, 2, Dim>& t) {
        TensorField<Addr, Scalar, 0, Dim> result;
        for (const auto& [addr, mat] : t) {
            result.set(addr, mat.trace());
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Операции поднятия и опускания индексов с метрикой
    // -------------------------------------------------------------------------

    /**
     * @brief Опускание индекса у контравариантного вектора: v_i = g_{ij} v^j
     * @param v  Векторное поле (ранг 1)
     * @param g  Метрическое поле (ранг 2, ковариантное)
     * @return   Ковариантное векторное поле (ранг 1, но в том же классе)
     */
    template<typename Addr, typename Scalar, int Dim>
    TensorField<Addr, Scalar, 1, Dim>
        lower_index(const TensorField<Addr, Scalar, 1, Dim>& v,
            const TensorField<Addr, Scalar, 2, Dim>& g) {
        TensorField<Addr, Scalar, 1, Dim> result;
        for (const auto& [addr, vec] : v) {
            if (g.contains(addr)) {
                const auto& metric = g.at(addr);
                result.set(addr, metric * vec);
            }
        }
        return result;
    }

    /**
     * @brief Поднятие индекса у ковариантного вектора: v^i = g^{ij} v_j
     * @param cov    Ковариантное векторное поле (ранг 1)
     * @param g_inv  Обратное метрическое поле (ранг 2, контравариантное)
     * @return       Контравариантное векторное поле (ранг 1)
     */
    template<typename Addr, typename Scalar, int Dim>
    TensorField<Addr, Scalar, 1, Dim>
        raise_index(const TensorField<Addr, Scalar, 1, Dim>& cov,
            const TensorField<Addr, Scalar, 2, Dim>& g_inv) {
        TensorField<Addr, Scalar, 1, Dim> result;
        for (const auto& [addr, vec] : cov) {
            if (g_inv.contains(addr)) {
                const auto& metric_inv = g_inv.at(addr);
                result.set(addr, metric_inv * vec);
            }
        }
        return result;
    }

    /**
     * @brief Поднятие второго индекса у смешанного тензора (матрицы):
     *        M^{i}_{j} -> M^{ik} = g^{kj} M^{i}_{j}
     * @param m       Матричное поле (ранг 2, предполагается смешанное)
     * @param g_inv   Обратное метрическое поле
     * @return        Поле с двумя верхними индексами (ранг 2)
     */
    template<typename Addr, typename Scalar, int Dim>
    TensorField<Addr, Scalar, 2, Dim>
        raise_second_index(const TensorField<Addr, Scalar, 2, Dim>& m,
            const TensorField<Addr, Scalar, 2, Dim>& g_inv) {
        TensorField<Addr, Scalar, 2, Dim> result;
        for (const auto& [addr, mat] : m) {
            if (g_inv.contains(addr)) {
                const auto& metric_inv = g_inv.at(addr);
                result.set(addr, mat * metric_inv);
            }
        }
        return result;
    }

    /**
     * @brief Опускание второго индекса у смешанного тензора:
     *        M^{i}_{j} -> M_{ij} = g_{jk} M^{i}_{k}? На самом деле нужно определить.
     *        Упростим: опускаем первый индекс или второй в зависимости от конвенции.
     *        Здесь реализуем опускание первого индекса: M_{ij} = g_{ik} M^{k}_{j}
     */
    template<typename Addr, typename Scalar, int Dim>
    TensorField<Addr, Scalar, 2, Dim>
        lower_first_index(const TensorField<Addr, Scalar, 2, Dim>& m,
            const TensorField<Addr, Scalar, 2, Dim>& g) {
        TensorField<Addr, Scalar, 2, Dim> result;
        for (const auto& [addr, mat] : m) {
            if (g.contains(addr)) {
                const auto& metric = g.at(addr);
                result.set(addr, metric * mat);  // g * m
            }
        }
        return result;
    }

} // namespace delta::geometry