// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/geometry/constructive_core.h
// ============================================================================
// constructive_core.h
// Конструктивное ядро Δ‑анализа: точки, векторы и операции над ними
// ============================================================================
//
// В этом файле определены фундаментальные типы для конструктивного описания
// пространства: Point (точка) и Vector (вектор), а также допустимые операции.
// Все вычисления ведутся с использованием типа delta::Rational – точной
// дроби p/q (числитель/знаменатель), которая не зависит от системы счисления.
//
// ----------------------------------------------------------------------------
// 1. Почему Rational и почему не нужна система счисления?
// ----------------------------------------------------------------------------
// Классическая проблема "конечного представления чисел" возникает, когда
// мы пытаемся записать число в фиксированной системе счисления (например,
// десятичной): 1/3 = 0.333... – бесконечная строка. Однако Rational хранит
// число как пару целых (числитель, знаменатель). Это конечное и точное
// представление, не зависящее от основания. Поэтому для нас любое ненулевое
// рациональное число уже является конструктивным адресом.
//
// ----------------------------------------------------------------------------
// 2. Универсальное конструктивное ядро K*
// ----------------------------------------------------------------------------
// Согласно блоку A4e, универсальное конструктивное ядро K* = Q \ {0} –
// все ненулевые рациональные числа. Поскольку мы работаем с Rational как
// с дробью, мы автоматически опираемся на K* и не нуждаемся в дополнительных
// ограничениях на знаменатель (например, не требуется, чтобы он был степенью 2 и 5).
//
// ----------------------------------------------------------------------------
// 3. Точка (Point) – конструктивный адрес
// ----------------------------------------------------------------------------
// Точка представляет физическое положение. Чтобы быть адресом, точка должна:
//   • иметь ненулевые координаты (иначе это "ничто", не может быть указано);
//   • координаты должны быть ненулевыми рациональными числами.
// Проверка принадлежности ядру K реализована функцией is_in_K(p).
//
// ----------------------------------------------------------------------------
// 4. Вектор (Vector) – свободное перемещение
// ----------------------------------------------------------------------------
// Вектор – это перемещение, скорость, сила. Он не обязан быть ненулевым
// (нулевой вектор – допустимое "ничего не делать"). Координаты вектора могут
// быть любыми рациональными числами, включая нули. Векторы образуют полное
// векторное пространство с операциями сложения и умножения на скаляр.
//
// ----------------------------------------------------------------------------
// 5. Отношения между точками и векторами
// ----------------------------------------------------------------------------
//   • Разность двух точек даёт вектор: p - q = v. Всегда допустимо.
//   • Сумма точки и вектора даёт новую точку ТОЛЬКО если результат принадлежит
//     ядру K (т.е. все координаты ненулевые). В противном случае операция
//     возвращает std::nullopt.
//
// Это фундаментальное отличие от стандартной геометрии:
//   • В стандартной геометрии точка + вектор всегда точка.
//   • В Δ‑анализе мы не можем гарантировать, что в результате сложения
//     не возникнет нулевая координата (которая не является адресом) или
//     иррациональное число (которое не есть Rational). Поэтому результат
//     опционален – операция успешна только тогда, когда новый адрес
//     остаётся конструктивным.
//
// ----------------------------------------------------------------------------
// 6. Почему нельзя «просто сделать как в обычной геометрии»?
// ----------------------------------------------------------------------------
// Обычный подход предполагает, что ℝⁿ дано со всеми своими точками, включая
// начало координат и иррациональные точки. Это удобно для математического
// анализа, но противоречит конструктивной природе физической реальности:
//   • Никакое реальное измерение не может точно указать точку с нулевой
//     координатой (это отсутствие места).
//   • Иррациональные координаты невозможно записать конечной строкой.
//
// Δ‑анализ устраняет эти проблемы, принимая за основу только конструктивные
// адреса (K*) и делая операции над ними явно проверяющими допустимость
// результата. Использование optional – прямое выражение этого принципа.
//
// ----------------------------------------------------------------------------
// 7. Применение в библиотеке
// ----------------------------------------------------------------------------
// Типы Point и Vector используются во всех модулях, работающих с 
// дискретными операторами (градиент, дивергенция, curl)
// ,вариационными принципами итд. Операции над ними
// строго соответствуют аксиомам Δ‑анализа.
//
// ============================================================================

#pragma once

#include <optional>
#include <set>
#include <cmath>
#include <Eigen/Dense>
#include <boost/multiprecision/cpp_int.hpp>
#include "delta/core/rational.h"

namespace delta::geometry {

    // -------------------------------------------------------------------------
    // Вспомогательные функции для разложения на простые множители
    // -------------------------------------------------------------------------

    namespace detail {

        inline std::set<int> prime_factors(const delta::internal::dumb_int& n) {
            std::set<int> factors;
            if (n <= 1) return factors;
            delta::internal::dumb_int m = n;
            if (m % 2 == 0) {
                factors.insert(2);
                while (m % 2 == 0) m /= 2;
            }
            delta::internal::dumb_int p = 3;
            while (p * p <= m) {
                if (m % p == 0) {
                    factors.insert(p.convert_to<int>());
                    while (m % p == 0) m /= p;
                }
                p += 2;
            }
            if (m > 1) factors.insert(m.convert_to<int>());
            return factors;
        }

        inline std::set<int> prime_factors(const Rational& x) {
            if (x.denominator() != 1) {
                throw std::domain_error("prime_factors: argument must be an integer");
            }
            auto num = x.numerator();
            if (num < 0) num = -num;
            delta::internal::dumb_int n = num.convert_to<delta::internal::dumb_int>();
            return prime_factors(n);
        }

        inline std::pair<Rational, Rational> get_numerator_denominator(const Rational& x) {
            return { x.numerator(), x.denominator() };
        }

    } // namespace detail

    // -------------------------------------------------------------------------
    // Finite base numbers
    // -------------------------------------------------------------------------

    template<int Base>
    struct FiniteBaseNumbers {
        static bool is_representable(const Rational& x) {
            if (x == 0) return false;
            auto [num, den] = detail::get_numerator_denominator(x);
            auto den_factors = detail::prime_factors(den);
            if (den_factors.empty()) return true;
            auto base_factors = detail::prime_factors(static_cast<delta::internal::dumb_int>(Base));
            for (int p : den_factors) {
                if (base_factors.find(p) == base_factors.end()) return false;
            }
            return true;
        }
    };

    // -------------------------------------------------------------------------
    // Универсальное ядро K*
    // -------------------------------------------------------------------------

    inline bool is_in_universal_core(const Rational& x) {
        return x != 0;
    }

    // -------------------------------------------------------------------------
    // Point - алиас на Eigen::Matrix
    // -------------------------------------------------------------------------

    template<typename Scalar, int Dim>
    using Point = Eigen::Matrix<Scalar, Dim, 1>;

    // -------------------------------------------------------------------------
    // Vector - отдельный класс для векторов с полной арифметикой и геометрией
    // -------------------------------------------------------------------------

    template<typename Scalar, int Dim>
    class Vector {
        static_assert(Dim > 0, "Dimension must be positive");

    public:
        using vector_type = Eigen::Matrix<Scalar, Dim, 1>;

        Vector() : data_(vector_type::Zero()) {}

        Vector(const vector_type& data) : data_(data) {}

        template<typename OtherDerived>
        Vector(const Eigen::MatrixBase<OtherDerived>& other) : data_(other) {}

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
        Scalar& operator[](int i) { return data_[i]; }

        bool operator==(const Vector& other) const { return data_ == other.data_; }

        const Scalar& x() const { static_assert(Dim >= 1, "Dimension too low"); return data_[0]; }
        const Scalar& y() const { static_assert(Dim >= 2, "Dimension too low"); return data_[1]; }
        const Scalar& z() const { static_assert(Dim >= 3, "Dimension too low"); return data_[2]; }
        Scalar& x() { static_assert(Dim >= 1, "Dimension too low"); return data_[0]; }
        Scalar& y() { static_assert(Dim >= 2, "Dimension too low"); return data_[1]; }
        Scalar& z() { static_assert(Dim >= 3, "Dimension too low"); return data_[2]; }

        // -----------------------------------------------------------------
        // Арифметические операторы (покомпонентные)
        // -----------------------------------------------------------------
        Vector operator+(const Vector& other) const { return Vector(data_ + other.data_); }
        Vector operator-(const Vector& other) const { return Vector(data_ - other.data_); }
        Vector operator*(Scalar s) const { return Vector(data_ * s); }
        Vector operator/(Scalar s) const { return Vector(data_ / s); }
        Vector operator-() const { return Vector(-data_); }

        Vector& operator+=(const Vector& other) { data_ += other.data_; return *this; }
        Vector& operator-=(const Vector& other) { data_ -= other.data_; return *this; }
        Vector& operator*=(Scalar s) { data_ *= s; return *this; }
        Vector& operator/=(Scalar s) { data_ /= s; return *this; }

        // -----------------------------------------------------------------
        // Геометрические методы
        // -----------------------------------------------------------------
        Scalar dot(const Vector& other) const { return data_.dot(other.data_); }

        Vector cross(const Vector& other) const {
            static_assert(Dim == 3, "cross only for 3D vectors");
            return Vector(data_.cross(other.data_));
        }

        Scalar squaredNorm() const { return data_.squaredNorm(); }
        Scalar norm() const { return data_.norm(); }

        Vector normalized() const {
            Scalar n = norm();
            if (n == 0) return *this;
            return *this / n;
        }

    private:
        vector_type data_;
    };

    // -------------------------------------------------------------------------
    // Свободные операторы для Vector (коммутативное умножение на скаляр)
    // -------------------------------------------------------------------------
    template<typename Scalar, int Dim>
    Vector<Scalar, Dim> operator*(Scalar s, const Vector<Scalar, Dim>& v) {
        return v * s;
    }

    // -------------------------------------------------------------------------
    // Операции с точками и векторами
    // -------------------------------------------------------------------------

    template<typename Scalar, int Dim>
    Vector<Scalar, Dim> operator-(const Point<Scalar, Dim>& a, const Point<Scalar, Dim>& b) {
        return Vector<Scalar, Dim>(a.array() - b.array());
    }

    template<int Dim>
    bool is_in_K(const Eigen::Matrix<Rational, Dim, 1>& p) {
        for (int i = 0; i < Dim; ++i) {
            if (p[i] == 0) return false;
        }
        return true;
    }

    template<typename Scalar, int Dim>
    std::optional<Point<Scalar, Dim>> operator+(const Point<Scalar, Dim>& p,
        const Vector<Scalar, Dim>& v) {
        Point<Scalar, Dim> new_coords = p + v.data();
        if (is_in_K(new_coords)) {
            return new_coords;
        }
        return std::nullopt;
    }

    template<typename Scalar, int Dim>
    Vector<Scalar, Dim> operator+(const Vector<Scalar, Dim>& u, const Vector<Scalar, Dim>& v) {
        return Vector<Scalar, Dim>(u.data() + v.data());
    }

    // -------------------------------------------------------------------------
    // Вспомогательные функции для тестов
    // -------------------------------------------------------------------------

    template<int Dim, typename... Args>
    Point<Rational, Dim> make_point(Args... args) {
        Point<Rational, Dim> p;
        int i = 0;
        ((p[i++] = static_cast<Rational>(args)), ...);
        return p;
    }

    template<int Dim, typename... Args>
    Vector<Rational, Dim> make_vector(Args... args) {
        return Vector<Rational, Dim>(static_cast<Rational>(args)...);
    }

} // namespace delta::geometry