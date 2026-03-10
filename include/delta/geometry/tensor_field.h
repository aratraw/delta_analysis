// include/delta/geometry/tensor_field.h
#pragma once

#include <map>
#include <type_traits>
#include <Eigen/Core>
#include "delta/core/regulative_idea.h" // для Address концепта (опционально)

namespace delta::geometry{

    // Вспомогательный шаблон для выбора типа тензора
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

    /**
     * @class TensorField
     * @brief A field assigning a tensor of fixed rank and dimension to each address.
     *
     * @tparam Addr    Address type (must be copyable and comparable).
     * @tparam Scalar  Scalar type (e.g., double, Rational).
     * @tparam Rank    Tensor rank (0 = scalar, 1 = vector, 2 = matrix).
     * @tparam Dim     Dimension of the space (e.g., 3 for 3D vectors/matrices).
     */
    template<typename Addr, typename Scalar, int Rank, int Dim>
    class TensorField {
        static_assert(Rank >= 0 && Rank <= 2, "Only ranks 0,1,2 are currently supported");
    public:
        using value_type = TensorType<Scalar, Rank, Dim>;
        using address_type = Addr;
        using map_type = std::map<Addr, value_type>;

        // Конструкторы
        TensorField() = default;

        // Задание значения для конкретного адреса
        void set(const Addr& addr, const value_type& value) {
            data_[addr] = value;
        }

        // Доступ на чтение (бросает исключение, если адрес отсутствует)
        const value_type& at(const Addr& addr) const {
            auto it = data_.find(addr);
            if (it == data_.end()) {
                throw std::out_of_range("TensorField: address not found");
            }
            return it->second;
        }

        // Доступ с возможностью модификации (если адрес отсутствует, создаёт значение по умолчанию)
        value_type& operator[](const Addr& addr) {
            return data_[addr];
        }

        // Проверка наличия адреса
        bool contains(const Addr& addr) const {
            return data_.find(addr) != data_.end();
        }

        // Количество адресов в поле
        std::size_t size() const { return data_.size(); }

        // Итераторы для обхода
        auto begin() const { return data_.begin(); }
        auto end() const { return data_.end(); }
        auto begin() { return data_.begin(); }
        auto end() { return data_.end(); }

        // Применить функцию к каждому значению (in-place)
        template<typename Func>
        void apply(Func&& f) {
            for (auto& [addr, val] : data_) {
                f(val);
            }
        }

        // Применить функцию к каждому значению (константная версия)
        template<typename Func>
        void apply(Func&& f) const {
            for (const auto& [addr, val] : data_) {
                f(val);
            }
        }

    private:
        map_type data_;
    };

    // Операторы поэлементного сложения и вычитания
    template<typename Addr, typename Scalar, int Rank, int Dim>
    TensorField<Addr, Scalar, Rank, Dim> operator+(
        const TensorField<Addr, Scalar, Rank, Dim>& a,
        const TensorField<Addr, Scalar, Rank, Dim>& b) {
        // Предполагаем, что множества адресов совпадают (иначе можно определить политику)
        // Для простоты копируем из a и добавляем значения из b
        TensorField<Addr, Scalar, Rank, Dim> result = a;
        for (const auto& [addr, val] : b) {
            result[addr] += val;
        }
        return result;
    }

    // Аналогично operator-, умножение на скаляр
    template<typename Addr, typename Scalar, int Rank, int Dim>
    TensorField<Addr, Scalar, Rank, Dim> operator*(
        const TensorField<Addr, Scalar, Rank, Dim>& f,
        const Scalar& s) {
        TensorField<Addr, Scalar, Rank, Dim> result = f;
        result.apply([&](auto& val) { val *= s; });
        return result;
    }

    // Скалярное произведение двух полей (сумма поэлементных свёрток) - опционально
    // Но для начала достаточно.

} // namespace delta::geometry