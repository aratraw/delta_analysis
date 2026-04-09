// rational_class.h
#pragma once

#include "rational_fwd.h"
#include "storage.h"
#include "expression_root.h"

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

        // Конструкторы immediate (неявные)
        Rational() noexcept;
        Rational(absl::int128 num);
        Rational(absl::int128 num, absl::uint128 den);  // для внутреннего использования

        // Конструкторы от целых типов (неявные)
        Rational(int num);
        Rational(long long num);
        Rational(unsigned long long num);

        // ЕДИНСТВЕННЫЙ конструктор от двух аргументов – от long long
        explicit Rational(long long num, long long den);

        // Конструкторы для больших чисел (explicit)
        explicit Rational(const boost::multiprecision::cpp_int& num);
        explicit Rational(const boost::multiprecision::cpp_int& num, const boost::multiprecision::cpp_int& den);
        explicit Rational(const std::string& s);

        // 🔥 Новые конструкторы от dumb_int (для внутреннего использования)
        explicit Rational(const internal::dumb_int& num);
        explicit Rational(const internal::dumb_int& num, const internal::dumb_int& den);

        // Фабричный метод для ленивого объекта
        static Rational from_lazy_index(std::size_t root_idx);

        // Копирование/перемещение по умолчанию
        Rational(const Rational&);
        Rational(Rational&&) noexcept;
        Rational& operator=(const Rational&);
        Rational& operator=(Rational&&) noexcept;
        ~Rational();

        double to_double() const;

        template<typename T>
        T convert_to() const;

        bool is_immediate() const noexcept;
        bool is_lazy() const noexcept;

        const internal::SmallStorage* as_small() const noexcept;
        const internal::BigStorage* as_big() const noexcept;
        Rational numerator() const;
        Rational denominator() const;

        int root_index() const;
        Rational lazy() const;
        Rational immediate() const;
        Rational simplify() const;
        Rational eval(bool skip_simplify = false) const;

        internal::Value to_value() const;
        internal::Interval approx_interval() const noexcept;

        // Дружественные операторы
        friend Rational operator+(const Rational&, const Rational&);
        friend Rational operator-(const Rational&, const Rational&);
        friend Rational operator*(const Rational&, const Rational&);
        friend Rational operator/(const Rational&, const Rational&);
        friend Rational operator-(const Rational&);
        friend Rational batch_add(const std::vector<Rational>&);

        friend Rational sqrt(const Rational&, const Rational&);
        friend Rational exp(const Rational&, const Rational&);
        friend Rational log(const Rational&, const Rational&);
        friend Rational sin(const Rational&, const Rational&);
        friend Rational cos(const Rational&, const Rational&);
        friend Rational acos(const Rational&, const Rational&);
        friend Rational pi(const Rational&);
        friend Rational e(const Rational&);
        friend Rational pow(const Rational&, int);

        // 🔥 Друзья для inplace-операций (доступ к storage_)
        friend void inplace_add(Rational& a, const Rational& b);
        friend void inplace_mul(Rational& a, const Rational& b);

    private:
        using Storage = std::variant<internal::SmallStorage, internal::BigStorage, int>;
        Storage storage_;

        friend std::string to_string(const Rational& r);
    };

} // namespace delta
#include "rational_impl.h"