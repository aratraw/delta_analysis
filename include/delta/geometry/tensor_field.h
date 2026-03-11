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

        // Доступ для записи (создаёт элемент при необходимости)
        void set(const Addr& addr, const value_type& val) {
            data_[addr] = val;
        }

        // Оператор индексирования для записи (создаёт элемент при необходимости)
        value_type& operator[](const Addr& addr) {
            return data_[addr];
        }

        // Доступ только для чтения (кидает исключение, если нет)
        const value_type& at(const Addr& addr) const {
            auto it = data_.find(addr);
            if (it == data_.end()) {
                throw std::out_of_range("TensorField: address not found");
            }
            return it->second;
        }

        // Проверка наличия
        bool contains(const Addr& addr) const {
            return data_.find(addr) != data_.end();
        }

        // Итераторы для обхода всех точек
        auto begin() const { return data_.begin(); }
        auto end() const { return data_.end(); }
        auto begin() { return data_.begin(); }
        auto end() { return data_.end(); }

        // Количество точек
        std::size_t size() const { return data_.size(); }

        // Операции над всем полем (применить функцию к каждому значению)
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
        std::map<Addr, value_type> data_;   // для простоты используем map
    };

    // -------------------------------------------------------------------------
    // Свободные функции для алгебраических операций (поточечные)
    // -------------------------------------------------------------------------

    // Сложение двух полей одинакового ранга и размерности
    template<typename Addr, typename Scalar, int Rank, int Dim>
    TensorField<Addr, Scalar, Rank, Dim>
        operator+(const TensorField<Addr, Scalar, Rank, Dim>& a,
            const TensorField<Addr, Scalar, Rank, Dim>& b) {
        TensorField<Addr, Scalar, Rank, Dim> result;
        // Проходим по всем точкам из a, если есть в b, складываем
        for (const auto& [addr, val_a] : a) {
            if (b.contains(addr)) {
                result.set(addr, val_a + b.at(addr));
            }
        }
        // Можно также добавить точки, которые есть только в b, но тогда сумма не определена.
        // Оставим только пересечение.
        return result;
    }

    // Вычитание
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

    // Умножение на скаляр (слева и справа)
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

    // Умножение на скаляр с присваиванием
    template<typename Addr, typename Scalar, int Rank, int Dim>
    TensorField<Addr, Scalar, Rank, Dim>&
        operator*=(TensorField<Addr, Scalar, Rank, Dim>& f, const Scalar& s) {
        f.apply([s](auto& val) { val *= s; });
        return f;
    }

    // Тензорное произведение двух полей (разных рангов)
    template<typename Addr, typename Scalar, int RankA, int RankB, int Dim>
    auto tensor_product(const TensorField<Addr, Scalar, RankA, Dim>& a,
        const TensorField<Addr, Scalar, RankB, Dim>& b) {
        constexpr int RankR = RankA + RankB;
        TensorField<Addr, Scalar, RankR, Dim> result;
        for (const auto& [addr, val_a] : a) {
            if (b.contains(addr)) {
                const auto& val_b = b.at(addr);   // <-- исправлено: получаем val_b
                if constexpr (RankA == 1 && RankB == 1) {
                    // val_a - вектор-столбец, val_b - вектор-столбец, хотим матрицу val_a * val_b^T
                    result.set(addr, val_a * val_b.transpose());
                }
                else {
                    // Для других комбинаций пока не реализовано
                    static_assert(RankA == 1 && RankB == 1, "tensor_product only implemented for vector-vector");
                }
            }
        }
        return result;
    }

    // Свёртка для поля ранга 2 (получаем скалярное поле - след)
    template<typename Addr, typename Scalar, int Dim>
    TensorField<Addr, Scalar, 0, Dim> trace(const TensorField<Addr, Scalar, 2, Dim>& t) {
        TensorField<Addr, Scalar, 0, Dim> result;
        for (const auto& [addr, mat] : t) {
            result.set(addr, mat.trace());
        }
        return result;
    }

} // namespace delta::geometry