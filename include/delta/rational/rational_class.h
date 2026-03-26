// rational_class.h
#pragma once

#include "rational_fwd.h"
#include "storage.h"
#include "expression_root.h"   // для ExpressionRoot

#include <absl/numeric/int128.h>
#include <boost/multiprecision/cpp_int.hpp>

#include <memory>
#include <string>
#include <variant>

namespace delta {

    class Rational {
    public:
        // Конструктор из Value (для immediate)
        Rational(internal::Value val);

        // Конструкторы immediate
        Rational() noexcept;
        Rational(absl::int128 num);
        Rational(absl::int128 num, absl::uint128 den);
        explicit Rational(const boost::multiprecision::cpp_int& num);
        Rational(const boost::multiprecision::cpp_int& num, const boost::multiprecision::cpp_int& den);
        explicit Rational(const std::string& s);          // парсит "1/2" или "0.125"
        Rational(int num, int den);   // конструктор от двух int

        // Конструктор для lazy (внутренний) – сигнатура уникальна, так как нет конструктора от одного int
        explicit Rational(std::size_t root_idx);               // для индекса корня

        // Запрет на float/double
        //Rational(double) = delete;
        //Rational(float) = delete;
        //Rational(long double) = delete;

        // Копирование/перемещение по умолчанию
        Rational(const Rational&) = default;
        Rational(Rational&&) = default;
        Rational& operator=(const Rational&) = default;
        Rational& operator=(Rational&&) = default;
        ~Rational() = default;

        // Состояние
        bool is_immediate() const noexcept;
        bool is_lazy() const noexcept;

        // Доступ к immediate данным
        const internal::SmallStorage* as_small() const noexcept;
        const internal::BigStorage* as_big() const noexcept;

        // Доступ к lazy (индекс корня)
        int root_index() const;          // требует is_lazy()

        // Преобразование в lazy (создаёт константу из текущего значения)
        Rational lazy() const;

        // Упрощение и вычисление
        Rational simplify() const;
        Rational eval(bool skip_simplify = false) const;

        // Для внутреннего использования
        internal::Value to_value() const;

        // Интервальная оценка
        internal::Interval approx_interval() const noexcept;

        // Дружественные операторы
        friend Rational operator+(const Rational&, const Rational&);
        friend Rational operator-(const Rational&, const Rational&);
        friend Rational operator*(const Rational&, const Rational&);
        friend Rational operator/(const Rational&, const Rational&);
        friend Rational operator-(const Rational&);
        friend Rational batch_add(const std::vector<Rational>&);

        // Дружественные функции из transcendentals.h
        friend Rational sqrt(const Rational&, const Rational&);
        friend Rational exp(const Rational&, const Rational&);
        friend Rational log(const Rational&, const Rational&);
        friend Rational sin(const Rational&, const Rational&);
        friend Rational cos(const Rational&, const Rational&);
        friend Rational acos(const Rational&, const Rational&);
        friend Rational pi(const Rational&);
        friend Rational e(const Rational&);
        friend Rational pow(const Rational&, int);

    private:
        using Storage = std::variant<internal::SmallStorage,
            internal::BigStorage,
            int>;   // int для lazy
        Storage storage_;

        friend std::string to_string(const Rational& r);
    };

    // Включение реализаций


} // namespace delta
#include "rational_impl.h"