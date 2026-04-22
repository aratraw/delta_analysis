// lazy_rational.h
#pragma once

#include "rational_class.h"
#include "lazy_nodes.h"
#include "absl/container/inlined_vector.h"
#include <vector>
#include <optional>          // для std::optional

#ifdef DELTA_TESTING
namespace delta::testing {
    class LazyRationalTestFixture;   // forward declaration
}
#endif

namespace delta {

    // forward declaration для дружественной функции
    namespace internal {
        class Interval;
    }
    internal::Interval compute_interval_dirty(const LazyRational& lr);

    class LazyRational {
    public:
        // ------------------------------------------------------------------------
        // Конструкторы и деструктор
        // ------------------------------------------------------------------------
        LazyRational();                                    // грязный CONST(0)
        explicit LazyRational(const Rational& r);          // грязный CONST(r)
        explicit LazyRational(Rational&& r);               // грязный CONST(std::move(r))

        // Запрет копирования (move-only)
        LazyRational(const LazyRational&) = delete;
        LazyRational& operator=(const LazyRational&) = delete;

        // Перемещение
        LazyRational(LazyRational&& other) noexcept;
        LazyRational& operator=(LazyRational&& other) noexcept;

        ~LazyRational();

        // ------------------------------------------------------------------------
        // Глубокое копирование
        // ------------------------------------------------------------------------
        LazyRational clone() const;

        // ------------------------------------------------------------------------
        // Преобразование в Rational (вычисление)
        // ------------------------------------------------------------------------
        Rational eval(bool skip_simplify = false) const;

        // ------------------------------------------------------------------------
        // Вычисление на месте – превращает объект в чистое дерево из одного узла CONST
        // ------------------------------------------------------------------------
        void eval_inplace(bool skip_simplify = false);

        // ------------------------------------------------------------------------
        // Упрощение (канонизация in-place)
        // ------------------------------------------------------------------------
        void simplify_inplace();          // Dirty -> Clean
        LazyRational simplify() const;    // возвращает новый Clean LazyRational (копируя)

        // ------------------------------------------------------------------------
        // Приблизительный интервал (не требует канонизации, кэшируется)
        // ------------------------------------------------------------------------
        internal::Interval approx_interval() const;

        // ------------------------------------------------------------------------
        // Принудительное приведение к грязному состоянию
        // ------------------------------------------------------------------------
        void ensure_dirty();

        // ------------------------------------------------------------------------
        // Состояние (для отладки)
        // ------------------------------------------------------------------------
        bool is_dirty() const { return state_ == State::Dirty; }
        bool is_clean() const { return state_ == State::Clean; }

        // ------------------------------------------------------------------------
        // Сброс кэшированного интервала (вызывается при мутациях)
        // ------------------------------------------------------------------------
        void invalidate_interval() const { cached_interval_.reset(); }

        // ------------------------------------------------------------------------
        // Методы массовой вставки (bulk append)
        // ------------------------------------------------------------------------
        void append_values(std::vector<internal::Value>&& values);
        void append_nodes(std::vector<int>&& node_indices);

        // ------------------------------------------------------------------------
        // Мутирующие операторы (всегда изменяют левый операнд)
        // ------------------------------------------------------------------------
        // Сложение
        friend LazyRational& operator+(LazyRational& a, const LazyRational& b);
        friend LazyRational&& operator+(LazyRational&& a, const LazyRational& b);
        friend LazyRational& operator+(LazyRational& a, const Rational& b);
        friend LazyRational&& operator+(LazyRational&& a, const Rational& b);

        // Вычитание (a - b = a + NEG(b))
        friend LazyRational& operator-(LazyRational& a, const LazyRational& b);
        friend LazyRational&& operator-(LazyRational&& a, const LazyRational& b);
        friend LazyRational& operator-(LazyRational& a, const Rational& b);
        friend LazyRational&& operator-(LazyRational&& a, const Rational& b);

        // Умножение
        friend LazyRational& operator*(LazyRational& a, const LazyRational& b);
        friend LazyRational&& operator*(LazyRational&& a, const LazyRational& b);
        friend LazyRational& operator*(LazyRational& a, const Rational& b);
        friend LazyRational&& operator*(LazyRational&& a, const Rational& b);

        // Деление (a / b = a * RECIP(b))
        friend LazyRational& operator/(LazyRational& a, const LazyRational& b);
        friend LazyRational&& operator/(LazyRational&& a, const LazyRational& b);
        friend LazyRational& operator/(LazyRational& a, const Rational& b);
        friend LazyRational&& operator/(LazyRational&& a, const Rational& b);

        // Унарный минус (создаёт новый LazyRational)
        // Чтобы НЕ мутировать правый операнд для использования в operator-(LazyRational a,LazyRational b);
        // Таким образом: МУТАЦИЯ НА УНАРНОМ МИНУСЕ В ОБЩЕМ СЛУЧАЕ ВРЕДНА ПРИ ТЕКУЩЕЙ РЕАЛИЗАЦИИ БИНАРНОГО МИНУСА.
        // А ВООБЩЕ ПОЧЕМУ МЫ ВЫЗЫВАЕМ УНАРНЫЙ МИНУС ПРИ operator-(LazyRational a,LazyRational b)? Лишнее копирование объекта!
        // ПРОСТО ОБРАБОТАТЬ ВСЁ В ТОМ ЖЕ operator-: элементы правого операнда обернуть в NEG и пихнуть в правый операнд.
        friend LazyRational operator-(const LazyRational& a);

        // ------------------------------------------------------------------------
        // Дружественные операторы с Rational слева (Rational + LazyRational и т.д.)
        // ------------------------------------------------------------------------
        friend LazyRational& operator+(const Rational& a, LazyRational& b);
        friend LazyRational&& operator+(const Rational& a, LazyRational&& b);
        friend LazyRational operator-(const Rational& a, const LazyRational& b);
        friend LazyRational operator-(const Rational& a, LazyRational&& b);
        friend LazyRational& operator*(const Rational& a, LazyRational& b);
        friend LazyRational&& operator*(const Rational& a, LazyRational&& b);
        friend LazyRational operator/(const Rational& a, const LazyRational& b);
        friend LazyRational operator/(const Rational& a, LazyRational&& b);
        friend LazyRational mutating_unary_minus(LazyRational&& a);

        // Составные операторы
        friend LazyRational& operator+=(LazyRational& a, const LazyRational& b);
        friend LazyRational& operator+=(LazyRational& a, const Rational& b);
        friend LazyRational& operator-=(LazyRational& a, const LazyRational& b);
        friend LazyRational& operator-=(LazyRational& a, const Rational& b);
        friend LazyRational& operator*=(LazyRational& a, const LazyRational& b);
        friend LazyRational& operator*=(LazyRational& a, const Rational& b);
        friend LazyRational& operator/=(LazyRational& a, const LazyRational& b);
        friend LazyRational& operator/=(LazyRational& a, const Rational& b);

        // ------------------------------------------------------------------------
        // Сравнения (неявно вызывают канонизацию, изменяют объекты)
        // ------------------------------------------------------------------------
        friend bool operator==(const LazyRational& a, const LazyRational& b);
        friend bool operator!=(const LazyRational& a, const LazyRational& b);
        friend bool operator<(const LazyRational& a, const LazyRational& b);
        friend bool operator<=(const LazyRational& a, const LazyRational& b);
        friend bool operator>(const LazyRational& a, const LazyRational& b);
        friend bool operator>=(const LazyRational& a, const LazyRational& b);

        // ------------------------------------------------------------------------
         // Дружественные функции для ленивых трансцендентных (все перегрузки)
         // ------------------------------------------------------------------------
        friend LazyRational lazy_sqrt(const LazyRational& x, const Rational& eps);
        friend LazyRational lazy_sqrt(const Rational& x, const Rational& eps);
        friend LazyRational lazy_exp(const LazyRational& x, const Rational& eps);
        friend LazyRational lazy_exp(const Rational& x, const Rational& eps);
        friend LazyRational lazy_log(const LazyRational& x, const Rational& eps);
        friend LazyRational lazy_log(const Rational& x, const Rational& eps);
        friend LazyRational lazy_sin(const LazyRational& x, const Rational& eps);
        friend LazyRational lazy_sin(const Rational& x, const Rational& eps);
        friend LazyRational lazy_cos(const LazyRational& x, const Rational& eps);
        friend LazyRational lazy_cos(const Rational& x, const Rational& eps);
        friend LazyRational lazy_acos(const LazyRational& x, const Rational& eps);
        friend LazyRational lazy_acos(const Rational& x, const Rational& eps);
        friend LazyRational lazy_pi(const Rational& eps);
        friend LazyRational lazy_e(const Rational& eps);
        friend LazyRational lazy_pow(const LazyRational& base, const LazyRational& exponent, const Rational& eps);
        friend LazyRational lazy_pow(const Rational& base, const LazyRational& exponent, const Rational& eps);
        friend LazyRational lazy_pow(const Rational& base, const Rational& exponent, const Rational& eps);
        friend LazyRational lazy_pow(const LazyRational& base, const Rational& exponent, const Rational& eps);
        friend LazyRational lazy_pow(const LazyRational& base, int exponent);

        // Друг для доступа к грязному дереву при вычислении интервала
        friend internal::Interval compute_interval_dirty(const LazyRational& lr);

#ifdef DELTA_TESTING
        friend class delta::testing::LazyRationalTestFixture;
#endif

    private:
        enum class State { Dirty, Clean };
        mutable State state_ = State::Dirty;

        // Для Dirty (mutable для ленивой канонизации в const-методах)
        mutable std::vector<internal::DirtyNode> nodes_;
        mutable std::vector<internal::Value> constants_;
        mutable int root_ = -1;

        // Для Clean:
        mutable int clean_index_ = -1;      // mutable для канонизации в const-методах

        // Кэш интервальной оценки (nullopt, если не вычислен или дерево изменилось)
        mutable std::optional<internal::Interval> cached_interval_;

        // ------------------------------------------------------------------------
        // Приватные методы
        // ------------------------------------------------------------------------
        void canonicalize() const;          // Dirty -> Clean, меняет state_ и clean_index_
        int import_tree(const LazyRational& other);
        int add_constant(const internal::Value& v);
        int new_dirty_node(internal::LazyOp op, absl::InlinedVector<int32_t, 4> children,
            int value_idx = -1, int eps_idx = -1);
        void append_sum_children(int sum_node, const LazyRational& other);
        void append_product_children(int prod_node, const LazyRational& other);
    };

    std::ostream& operator<<(std::ostream& os, const LazyRational& lr);

} // namespace delta

#include "lazy_rational_impl.h"