// rational_class.h
#pragma once

#include "rational_fwd.h"
#include "storage.h"

#include <boost/multiprecision/cpp_int.hpp>
#include <string>
#include <vector>

namespace delta {

    class Rational {
    public:
        // ------------------------------------------------------------------------
        // Конструкторы
        // ------------------------------------------------------------------------
        Rational() noexcept;
        Rational(int num);
        Rational(long long num);
        Rational(unsigned long long num);
        explicit Rational(long long num, long long den);
        explicit Rational(const boost::multiprecision::cpp_int& num);
        explicit Rational(const boost::multiprecision::cpp_int& num, const boost::multiprecision::cpp_int& den);
        explicit Rational(const std::string& s);
        explicit Rational(const internal::dumb_int& num);
        explicit Rational(const internal::dumb_int& num, const internal::dumb_int& den);

        // Внутренний конструктор из Value
        explicit Rational(internal::Value val);

        // Копирование и перемещение
        Rational(const Rational&);
        Rational(Rational&&) noexcept;
        Rational& operator=(const Rational&);
        Rational& operator=(Rational&&) noexcept;
        ~Rational();

        // ------------------------------------------------------------------------
        // Преобразование в LazyRational
        // ------------------------------------------------------------------------
        LazyRational as_lazy() const;

        // ------------------------------------------------------------------------
        // Доступ к числителю и знаменателю (нормализованным)
        // ------------------------------------------------------------------------
        Rational numerator() const;
        Rational denominator() const;

        // ------------------------------------------------------------------------
        // Преобразование в double и строку
        // ------------------------------------------------------------------------
        double to_double() const;
        std::string to_string() const;

        internal::Interval approx_interval() const;

        // ------------------------------------------------------------------------
        // Доступ к внутреннему Value (для внутреннего использования)
        // ------------------------------------------------------------------------
        const internal::Value& value() const noexcept { return storage_; }

        // ------------------------------------------------------------------------
        // Арифметические операторы (всегда eager, возвращают новый Rational)
        // ------------------------------------------------------------------------
        friend Rational operator+(const Rational& a, const Rational& b);
        friend Rational operator-(const Rational& a, const Rational& b);
        friend Rational operator*(const Rational& a, const Rational& b);
        friend Rational operator/(const Rational& a, const Rational& b);
        friend Rational operator-(const Rational& a);   // унарный минус

        // Составные операторы
        friend Rational& operator+=(Rational& a, const Rational& b);
        friend Rational& operator-=(Rational& a, const Rational& b);
        friend Rational& operator*=(Rational& a, const Rational& b);
        friend Rational& operator/=(Rational& a, const Rational& b);

        // ------------------------------------------------------------------------
        // Сравнения
        // ------------------------------------------------------------------------
        friend bool operator==(const Rational& a, const Rational& b);
        friend bool operator!=(const Rational& a, const Rational& b);
        friend bool operator<(const Rational& a, const Rational& b);
        friend bool operator<=(const Rational& a, const Rational& b);
        friend bool operator>(const Rational& a, const Rational& b);
        friend bool operator>=(const Rational& a, const Rational& b);

        // ------------------------------------------------------------------------
        // Batch addition (eager)
        // ------------------------------------------------------------------------
        friend Rational batch_add(const std::vector<Rational>& terms);

        // ------------------------------------------------------------------------
        // Абсолютное значение
        // ------------------------------------------------------------------------
        friend Rational abs(const Rational& x);

        // ------------------------------------------------------------------------
        // Шаблонное преобразование
        // ------------------------------------------------------------------------
        template<typename T>
        T convert_to() const;

    private:
        internal::Value storage_;

        // Дружественные eager-функции (определены в evaluation_core.h)
        friend Rational eager_sqrt(const Rational& x, const Rational& eps);
        friend Rational eager_exp(const Rational& x, const Rational& eps);
        friend Rational eager_log(const Rational& x, const Rational& eps);
        friend Rational eager_sin(const Rational& x, const Rational& eps);
        friend Rational eager_cos(const Rational& x, const Rational& eps);
        friend Rational eager_acos(const Rational& x, const Rational& eps);
        friend Rational eager_pi(const Rational& eps);
        friend Rational eager_e(const Rational& eps);
        friend Rational eager_pow(const Rational& base, const Rational& exp, const Rational& eps);

        // In-place операции (из evaluation_core.h)
        friend void inplace_add(Rational& a, const Rational& b);
        friend void inplace_mul(Rational& a, const Rational& b);
    };

    std::ostream& operator<<(std::ostream& os, const Rational& r);

} // namespace delta

#include "rational_impl.h"