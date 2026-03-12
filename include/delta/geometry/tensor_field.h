// include/delta/geometry/tensor_field.h
#pragma once

#include <map>
#include <concepts>
#include <stdexcept>
#include <Eigen/Core>
#include "delta/core/rational.h"
#include "delta/geometry/simplicial_complex.h"
#include "delta/geometry/hat_basis.h"

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

    // Для рангов >2 пока не поддерживаем – можно добавить позже через Eigen::Tensor
    template<typename Scalar, int Rank, int Dim>
        requires (Rank > 2)
    struct TensorTypeSelector {
        static_assert(Rank <= 2, "Tensor ranks > 2 are not supported in this version");
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
    // Базовый класс тензорного поля (ранги 0,1,2)
    // -------------------------------------------------------------------------
    template<typename Addr, typename Scalar, int Rank, int Dim = 0>
        requires (Rank >= 0 && Rank <= 2)
    class TensorField {
    public:
        using value_type = TensorType<Scalar, Rank, Dim>;
        using address_type = Addr;
        using scalar_type = Scalar;
        static constexpr int rank = Rank;
        static constexpr int dim = Dim;

        TensorField() = default;

        // Доступ и модификация
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

        // Применить функцию ко всем значениям
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

        // -------------------------------------------------------------------------
        // Интерполяция на новый комплекс (refine)
        // -------------------------------------------------------------------------
        template<typename Complex, typename NewComplex, typename Metric>
        TensorField<Addr, Scalar, Rank, Dim>
            refine(const Complex& old_mesh,
                const NewComplex& new_mesh,
                const SubdivisionMap& subdiv_map,
                const Metric& metric) const {
            TensorField<Addr, Scalar, Rank, Dim> result;

            if constexpr (Rank == 0) {
                // Для скалярного поля используем HatBasis для интерполяции значений в новых точках
                HatBasis<NewComplex> basis(new_mesh);
                for (const auto& addr : new_mesh) {
                    // Ищем исходный симплекс, содержащий addr
                    // Можно использовать locate_point_in_simplex из HatBasis
                    std::optional<std::vector<typename Scalar>> bary;
                    int found_dim = -1;
                    std::size_t found_idx = 0;
                    for (int dim = old_mesh.dim(); dim >= 0 && !bary; --dim) {
                        for (std::size_t i = 0; i < old_mesh.num_simplices(dim); ++i) {
                            bary = basis.locate_point_in_simplex(addr, old_mesh, dim, i);
                            if (bary) {
                                found_dim = dim;
                                found_idx = i;
                                break;
                            }
                        }
                    }
                    if (!bary) {
                        // Точка вне комплекса – пропускаем (или throw)
                        continue;
                    }
                    // Интерполяция по вершинам исходного симплекса
                    const auto& simp = old_mesh.get_simplex(found_dim, found_idx);
                    Scalar val = 0;
                    for (std::size_t j = 0; j < simp.size(); ++j) {
                        val += (*bary)[j] * this->at(old_mesh.vertex(simp[j]));
                    }
                    result.set(addr, val);
                }
            }
            else {
                // Для тензоров ранга 1 и 2 используем карту подразделения:
                // каждый новый симплекс (вершина/ребро/грань) наследует значение от исходного,
                // если он является потомком одного исходного симплекса.
                // Если новый симплекс является потомком нескольких исходных (например, вершина-барицентр),
                // нужно усреднять. Пока реализуем простое копирование от первого найденного предка.
                // Для более точной интерполяции потребуется покомпонентное взвешивание,
                // но оставим это на будущее.
                for (const auto& [old_key, new_keys] : subdiv_map) {
                    if (old_key.dim != Rank) continue; // рассматриваем только симплексы нужной размерности
                    const auto& old_val = this->at(old_mesh.vertex(old_key.idx)); // ???
                    // Внимание: old_key.idx – это индекс симплекса размерности Rank в старом комплексе.
                    // Но у нас поле привязано к адресам, а не к симплексам. Для тензора на симплексах
                    // нужна другая структура. Однако в нашем TensorField адреса – это вершины (для ранга 0)
                    // или, возможно, другие типы адресов. Здесь мы смешиваем концепции.
                    // Чтобы избежать путаницы, ограничим refine только для ранга 0 (скалярные поля на вершинах).
                    // Для тензоров на симплексах нужен отдельный класс, например, TensorFieldOnSimplex.
                    // Поэтому пока оставим static_assert.
                    static_assert(Rank == 0,
                        "refine for tensor fields of rank >0 is not implemented yet; "
                        "use scalar fields and component-wise interpolation if needed.");
                }
            }
            return result;
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
        static_assert(RankR <= 2, "Tensor product of ranks >2 is not supported");
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
                else if constexpr (RankA == 2 && RankB == 0) {
                    result.set(addr, valA * valB);
                }
                else {
                    // остальные комбинации (1x2, 2x1, 2x2) дают ранг 3 или 4 – не поддерживаем
                    static_assert((RankA == 1 && RankB == 2) || (RankA == 2 && RankB == 1) || (RankA == 2 && RankB == 2),
                        "Tensor product for these ranks not implemented");
                }
            }
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Симметризация и антисимметризация для тензоров второго ранга
    // -------------------------------------------------------------------------
    template<typename Addr, typename Scalar, int Dim>
    TensorField<Addr, Scalar, 2, Dim> symmetrize(const TensorField<Addr, Scalar, 2, Dim>& t) {
        TensorField<Addr, Scalar, 2, Dim> result;
        for (const auto& [addr, mat] : t) {
            result.set(addr, (mat + mat.transpose()) / Scalar{ 2 });
        }
        return result;
    }

    template<typename Addr, typename Scalar, int Dim>
    TensorField<Addr, Scalar, 2, Dim> antisymmetrize(const TensorField<Addr, Scalar, 2, Dim>& t) {
        TensorField<Addr, Scalar, 2, Dim> result;
        for (const auto& [addr, mat] : t) {
            result.set(addr, (mat - mat.transpose()) / Scalar{ 2 });
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Свёртка (след) для матриц
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
     * @brief Опускание первого индекса у смешанного тензора:
     *        M^{i}_{j} -> M_{ij} = g_{ik} M^{k}_{j}
     */
    template<typename Addr, typename Scalar, int Dim>
    TensorField<Addr, Scalar, 2, Dim>
        lower_first_index(const TensorField<Addr, Scalar, 2, Dim>& m,
            const TensorField<Addr, Scalar, 2, Dim>& g) {
        TensorField<Addr, Scalar, 2, Dim> result;
        for (const auto& [addr, mat] : m) {
            if (g.contains(addr)) {
                const auto& metric = g.at(addr);
                result.set(addr, metric * mat);
            }
        }
        return result;
    }
    template<typename Complex, typename NewComplex, typename Metric>
    TensorField<Addr, Scalar, Rank, Dim>
        refine(const Complex& old_mesh,
            const NewComplex& new_mesh,
            const SubdivisionMap& subdiv_map,
            const Metric& metric) const {
        TensorField<Addr, Scalar, Rank, Dim> result;

        if constexpr (Rank == 0) {
            // ... (существующая реализация)
        }
        else {
            // Для рангов 1 и 2: интерполируем каждую компоненту отдельно
            // Создадим временные скалярные поля для каждой компоненты
            constexpr int comp_count = (Rank == 1) ? Dim : Dim * Dim;
            std::array<TensorField<Addr, Scalar, 0, Dim>, comp_count> comp_fields;
            // Заполняем компоненты из текущего поля
            for (const auto& [addr, val] : data_) {
                if constexpr (Rank == 1) {
                    for (int c = 0; c < Dim; ++c) {
                        comp_fields[c].set(addr, val(c));
                    }
                }
                else { // Rank == 2
                    for (int i = 0; i < Dim; ++i) {
                        for (int j = 0; j < Dim; ++j) {
                            comp_fields[i * Dim + j].set(addr, val(i, j));
                        }
                    }
                }
            }

            // Интерполируем каждую компоненту (используем refine для ранга 0)
            std::array<TensorField<Addr, Scalar, 0, Dim>, comp_count> new_comp_fields;
            for (int c = 0; c < comp_count; ++c) {
                new_comp_fields[c] = comp_fields[c].refine(old_mesh, new_mesh, subdiv_map, metric);
            }

            // Собираем результат
            for (const auto& addr : new_mesh) {
                if constexpr (Rank == 1) {
                    Eigen::Matrix<Scalar, Dim, 1> vec;
                    for (int c = 0; c < Dim; ++c) {
                        vec(c) = new_comp_fields[c].at(addr);
                    }
                    result.set(addr, vec);
                }
                else { // Rank == 2
                    Eigen::Matrix<Scalar, Dim, Dim> mat;
                    for (int i = 0; i < Dim; ++i) {
                        for (int j = 0; j < Dim; ++j) {
                            mat(i, j) = new_comp_fields[i * Dim + j].at(addr);
                        }
                    }
                    result.set(addr, mat);
                }
            }
        }
        return result;
    }
} // namespace delta::geometry