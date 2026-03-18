#pragma once

#include <optional>
#include <set>
#include <cmath>
#include <Eigen/Dense>
#include <boost/multiprecision/number.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include "delta/core/rational.h"

namespace delta::geometry {

    // -------------------------------------------------------------------------
    // Вспомогательные функции для разложения на простые множители
    // -------------------------------------------------------------------------

    namespace detail {

        /**
         * @brief Разложить целое число на простые множители.
         * @tparam Integer Целочисленный тип (boost::multiprecision::cpp_int или int)
         * @param n Число для разложения (положительное)
         * @return std::set<int> Множество простых множителей (уникальные)
         */
        template<typename Integer>
        std::set<int> prime_factors(Integer n) {
            std::set<int> factors;

            // Работаем с положительным числом
            if (n < 0) n = -n;
            if (n <= 1) return factors;  // 1 не имеет простых множителей

            // Обрабатываем множитель 2
            if (n % 2 == 0) {
                factors.insert(2);
                while (n % 2 == 0) {
                    n /= 2;
                }
            }

            // Обрабатываем нечётные множители
            int p = 3;
            while (static_cast<double>(p) * static_cast<double>(p) <= static_cast<double>(n)) {
                if (n % p == 0) {
                    factors.insert(p);
                    while (n % p == 0) {
                        n /= p;
                    }
                }
                p += 2;
            }

            // Если остаток больше 1, это тоже простой множитель
            if (n > 1) {
                factors.insert(static_cast<int>(n));
            }

            return factors;
        }

        /**
         * @brief Специализация для int (оптимизированная).
         */
        inline std::set<int> prime_factors(int n) {
            std::set<int> factors;
            if (n < 0) n = -n;
            if (n <= 1) return factors;

            // Factor 2
            if (n % 2 == 0) {
                factors.insert(2);
                while (n % 2 == 0) n /= 2;
            }

            // Factor odd numbers
            for (int p = 3; p * p <= n; p += 2) {
                if (n % p == 0) {
                    factors.insert(p);
                    while (n % p == 0) n /= p;
                }
            }

            if (n > 1) factors.insert(n);
            return factors;
        }

        /**
         * @brief Получить числитель и знаменатель Rational в несократимом виде.
         * @return std::pair<boost::multiprecision::cpp_int, boost::multiprecision::cpp_int>
         */
        inline std::pair<boost::multiprecision::cpp_int, boost::multiprecision::cpp_int>
            get_numerator_denominator(const Rational& x) {
            using boost::multiprecision::numerator;
            using boost::multiprecision::denominator;

            auto num = numerator(x);
            auto den = denominator(x);

            return { num, den };
        }

    } // namespace detail

    // -------------------------------------------------------------------------
    // Finite base numbers - проверка представимости в заданном основании
    // -------------------------------------------------------------------------

    /**
     * @brief Проверка представимости числа в системе счисления с основанием Base.
     *
     * Число представимо в основании Base, если в его несократимой дроби p/q
     * все простые множители q являются простыми множителями Base.
     *
     * @tparam Base Основание системы счисления (2, 3, 10 и т.д.)
     */
    template<int Base>
    struct FiniteBaseNumbers {
        /**
         * @brief Проверить, представимо ли рациональное число x в основании Base.
         * @param x Рациональное число (ненулевое)
         * @return true если представимо, false если нет или x == 0
         */
        static bool is_representable(const Rational& x) {
            if (x == 0) return false;  // ноль исключаем согласно определению K*

            auto [num, den] = detail::get_numerator_denominator(x);
            auto den_factors = detail::prime_factors(den);

            // Если знаменатель == 1, число целое - представимо в любом основании
            if (den_factors.empty()) return true;

            auto base_factors = detail::prime_factors(Base);

            // Проверяем, что все множители знаменателя содержатся в множителях основания
            for (int p : den_factors) {
                if (base_factors.find(p) == base_factors.end()) {
                    return false;
                }
            }

            return true;
        }
    };

    // -------------------------------------------------------------------------
    // Универсальное ядро K*
    // -------------------------------------------------------------------------

    /**
     * @brief Проверить, принадлежит ли число универсальному конструктивному ядру K*.
     *
     * Универсальное ядро K* = Q \ {0} - все ненулевые рациональные числа.
     *
     * @param x Проверяемое число
     * @return true если x != 0, false если x == 0
     */
    inline bool is_in_universal_core(const Rational& x) {
        return x != 0;
    }

    // -------------------------------------------------------------------------
    // Точка - просто алиас на Eigen::Matrix (ВАРИАНТ А)
    // -------------------------------------------------------------------------

    /**
     * @brief Точка в пространстве размерности Dim.
     *
     * Просто алиас на Eigen::Matrix для удобства. Все координаты хранятся как Rational.
     *
     * @tparam Scalar Тип скаляра (обычно Rational)
     * @tparam Dim Размерность пространства
     */
    template<typename Scalar, int Dim>
    using Point = Eigen::Matrix<Scalar, Dim, 1>;

    // -------------------------------------------------------------------------
    // Vector - отдельный класс для векторов
    // -------------------------------------------------------------------------

    /**
     * @brief Вектор в пространстве размерности Dim.
     *
     * Векторы не имеют ограничений на координаты (могут быть нулевыми).
     * Хранит Eigen::Matrix внутри.
     *
     * @tparam Scalar Тип скаляра (обычно Rational)
     * @tparam Dim Размерность пространства
     */
    template<typename Scalar, int Dim>
    class Vector {
        static_assert(Dim > 0, "Dimension must be positive");

    public:
        using vector_type = Eigen::Matrix<Scalar, Dim, 1>;

        Vector() : data_(vector_type::Zero()) {}

        // 1. Конструктор от Eigen::Matrix (не-explicit)
        Vector(const vector_type& data) : data_(data) {}

        // 2. Универсальный конструктор от любого Eigen-выражения
        template<typename OtherDerived>
        Vector(const Eigen::MatrixBase<OtherDerived>& other)
            : data_(other) {
        }

        // 3. Конструктор от списка координат (только если аргументы конвертируются в Scalar)
        template<typename... Args, typename = std::enable_if_t<(std::is_convertible_v<Args, Scalar> && ...)>>
        explicit Vector(Args... args) {
            static_assert(sizeof...(Args) == Dim, "Wrong number of components");
            vector_type tmp;
            int i = 0;
            ((tmp[i++] = args), ...);
            data_ = tmp;
        }

        const vector_type& data() const { return data_; }
        const Scalar& operator[](int i) const { return data_[i]; }
        bool operator==(const Vector& other) const { return data_ == other.data_; }

    private:
        vector_type data_;
    };
    // -------------------------------------------------------------------------
    // Операции с точками и векторами
    // -------------------------------------------------------------------------

    /**
     * @brief Разность двух точек даёт вектор.
     */
    template<typename Scalar, int Dim>
    Vector<Scalar, Dim> operator-(const Point<Scalar, Dim>& a, const Point<Scalar, Dim>& b) {
        return Vector<Scalar, Dim>(a.array() - b.array());
    }

    // -------------------------------------------------------------------------
    // Проверка принадлежности точки конструктивному ядру K
    // -------------------------------------------------------------------------

    /**
     * @brief Проверить, принадлежит ли точка конструктивному ядру K.
     *
     * В упрощённой реализации (для Этапа 0) точка принадлежит K,
     * если все её координаты ненулевые.
     *
     * @tparam Dim Размерность
     * @param p Проверяемая точка (Eigen::Matrix)
     * @return true если все координаты ненулевые
     */
    template<int Dim>
    bool is_in_K(const Eigen::Matrix<Rational, Dim, 1>& p) {
        for (int i = 0; i < Dim; ++i) {
            if (p[i] == 0) {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Сумма точки и вектора даёт новую точку.
     *
     * Возвращает std::nullopt, если результат содержит нулевую координату
     * (т.е. не принадлежит K).
     */
    template<typename Scalar, int Dim>
    std::optional<Point<Scalar, Dim>> operator+(const Point<Scalar, Dim>& p,
        const Vector<Scalar, Dim>& v) {
        Point<Scalar, Dim> new_coords = p + v.data();

        // Проверяем принадлежность K (все координаты ненулевые)
        if (is_in_K(new_coords)) {
            return new_coords;
        }
        return std::nullopt;
    }

    /**
     * @brief Сумма векторов всегда разрешена.
     */
    template<typename Scalar, int Dim>
    Vector<Scalar, Dim> operator+(const Vector<Scalar, Dim>& u, const Vector<Scalar, Dim>& v) {
        return Vector<Scalar, Dim>(u.data() + v.data());  // ЭТО МЕСТО ТЕПЕРЬ РАБОТАЕТ БЛАГОДАРЯ УНИВЕРСАЛЬНОМУ КОНСТРУКТОРУ
    }

    /**
     * @brief Умножение вектора на скаляр всегда разрешено.
     */
    template<typename Scalar, int Dim>
    Vector<Scalar, Dim> operator*(const Scalar& s, const Vector<Scalar, Dim>& v) {
        return Vector<Scalar, Dim>(s * v.data());
    }

    /**
     * @brief Умножение вектора на скаляр (коммутативный вариант).
     */
    template<typename Scalar, int Dim>
    Vector<Scalar, Dim> operator*(const Vector<Scalar, Dim>& v, const Scalar& s) {
        return s * v;
    }

    // -------------------------------------------------------------------------
    // Вспомогательные функции для тестов (не входят в спецификацию, но удобны)
    // -------------------------------------------------------------------------

    /**
     * @brief Создать точку из списка Rational (для удобства тестов).
     */
    template<int Dim, typename... Args>
    Point<Rational, Dim> make_point(Args... args) {
        Point<Rational, Dim> p;
        int i = 0;
        ((p[i++] = static_cast<Rational>(args)), ...);
        return p;
    }

    /**
     * @brief Создать вектор из списка Rational (для удобства тестов).
     */
    template<int Dim, typename... Args>
    Vector<Rational, Dim> make_vector(Args... args) {
        return Vector<Rational, Dim>(static_cast<Rational>(args)...);
    }

} // namespace delta::geometry