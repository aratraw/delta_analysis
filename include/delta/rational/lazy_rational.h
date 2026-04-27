// lazy_rational.h

// ---------------------------------------------------------------------------
// LAZYRATIONAL: МУТАБЕЛЬНОЕ ОТЛОЖЕННОЕ ВЫРАЖЕНИЕ
// ---------------------------------------------------------------------------
//
// LazyRational — move-only тип, представляющий отложенное (lazy) вычисление
// арифметико-трансцендентного выражения над рациональными числами.
// Копирование запрещено, перемещение разрешено. Для явного глубокого
// копирования используйте метод .clone().
//
// ---------------------------------------------------------------------------
// ФИЛОСОФИЯ МУТАЦИИ: ПОЧЕМУ acc + term, А НЕ acc = acc + term
// ---------------------------------------------------------------------------
//
// LazyRational спроектирован под основной сценарий вычислительной математики:
// накопление суммы (или произведения) в цикле с последующим ОДНИМ вычислением.
//
//   LazyRational acc;                  // создаёт CONST(0)
//   for (...) {
//       acc + term;                    // мутирует acc, добавляя term в дерево
//   }
//   Rational result = acc.eval();      // одно вычисление всего дерева
//
// Обратите внимание: НЕ «acc = acc + term», а просто «acc + term».
// Присваивание acc = acc + ... ЗАБЛОКИРОВАНО на уровне компиляции,
// потому что LazyRational запрещает копирование (оператор = удалён).
//
// ПОЧЕМУ ТАК:
//   - Если бы acc = acc + term работало, то каждое сложение создавало бы
//     ПОЛНУЮ КОПИЮ дерева acc, что дало бы O(N²) по памяти и времени.
//   - При неизменяемом (immutable) дизайне эти копии происходят НЕЯВНО,
//     и пользователь не может их избежать.
//   - Здесь же acc + term мутирует acc на месте за O(1), а полное
//     вычисление acc.eval() делается ОДИН раз в конце за O(N).
//
// В 99.99999% случаев у вас РОВНО ОДИН объект LazyRational на выражение.
// Все операнды — это eager-значения (Rational, int, литералы), которые
// поглощаются деревом без создания дополнительных LazyRational:
//
//   acc + 10_r;     // Rational поглощается как leaf_value
//   acc / 3_r;      // Rational поглощается как leaf_value
//   acc * term_r;   // Rational поглощается как leaf_value
//
// Именно под этот сценарий оптимизирована вся архитектура.
//
// The Underline: acc = acc + term; — ЭТО ГОВНОКОД.
// Мы заблокировали его на уровне компилятора.
//
// ---------------------------------------------------------------------------
// КАК РАБОТАЮТ ОПЕРАЦИИ: ЦЕПОЧКИ МУТАЦИЙ
// ---------------------------------------------------------------------------
//
// Ключевой принцип: каждый мутирующий оператор возвращает ССЫЛКУ на свой
// левый операнд. Это позволяет строить цепочки операций, где ВСЕ изменения
// применяются к ОДНОМУ объекту:
//
//   LazyRational acc;                  // CONST(0)
//   acc + 1_r + 2_r + 3_r + 4_r;      // мутирует acc четыре раза подряд!
//
// Что происходит под капотом:
//
//   1. acc + 1_r:
//      operator+(LazyRational& acc, const Rational& 1_r) вызывается.
//      Внутри: acc.ensure_dirty(), затем проверяется корень.
//      Если корень — SUM, 1_r добавляется в leaf_values.
//      Если корень — не SUM, создаётся новый узел SUM, и старый корень
//      становится его дочерним узлом, а 1_r попадает в leaf_values.
//      Возвращается acc& (ссылка на тот же acc!).
//      Теперь acc = SUM(CONST(0), leaf_values=[1_r]).
//
//   2. acc + 2_r:
//      acc — это тот же объект, его корень уже SUM.
//      operator+(acc, 2_r) снова добавляет 2_r в leaf_values того же узла SUM.
//      Возвращается acc&.
//      Теперь acc = SUM(CONST(0), leaf_values=[1_r, 2_r]).
//
//   3. acc + 3_r:
//      В leaf_values добавляется 3_r.
//      Возвращается acc&.
//      Теперь acc = SUM(CONST(0), leaf_values=[1_r, 2_r, 3_r]).
//
//   4. acc + 4_r:
//      В leaf_values добавляется 4_r.
//      Возвращается acc&.
//      Теперь acc = SUM(CONST(0), leaf_values=[1_r, 2_r, 3_r, 4_r]).
//
// Итог: ОДИН объект acc, ОДИН узел SUM, в leaf_values — все слагаемые.
// Все четыре операции выполнились за O(1) каждая, без единой аллокации
// нового LazyRational и без копирования дерева.
//
// Трансцендентные функции работают аналогично:
//
//   LazyRational x = LazyRational(0.5_r);
//   Sin(x) + Cos(x.eval() * 2_r) + 1_r;
//
//   1. Sin(x) — создаёт копию x, мутирует её в SIN(CONST(0.5)), возвращает её.
//   2. Полученный SIN складывается с Cos(...) — мутирует SIN в SUM(SIN, COS).
//   3. К SUM добавляется 1_r — попадает в leaf_values того же SUM.
//
// Везде, где оператор возвращает ссылку на левый операнд, цепочка
// продолжает мутировать ОДИН И ТОТ ЖЕ объект, не создавая новых.
//
// ---
// ЦЕНА МУТАЦИИ: КОГДА НУЖЕН .clone()
// ---
//
// Мутация означает, что операторы изменяют свой ЛЕВЫЙ операнд.
// Если вам по какой-то причине нужно использовать один и тот же
// LazyRational в нескольких подвыражениях, вы должны ЯВНО создать
// копию через .clone():
//
//   LazyRational x = ...;
//   LazyRational expr = Sin(x.clone() * 2_r) + Cos(x.clone() + 1_r);
//
// Без .clone() компилятор может вычислить подвыражения в любом порядке,
// и x будет испорчен первым же мутирующим оператором.
//
// В неизменяемых библиотеках те же копии происходят НЕЯВНО внутри
// каждого оператора. Здесь же пользователь ПОЛНОСТЬЮ КОНТРОЛИРУЕТ,
// когда копировать, и в 99.99999% случаев копии просто не нужны.
//
// ---------------------------------------------------------------------------
// О МУТАЦИИ И АРИФМЕТИКЕ: ЧТО МУТИРУЕТ, А ЧТО НЕТ
// ---------------------------------------------------------------------------
//
// Операторы с LazyRational устроены так, что всегда мутируют ЛЕВЫЙ
// операнд, даже если правый операнд — LazyRational. Это может быть
// неочевидно в редких случаях, когда у вас ДВА LazyRational:
//
//   LazyRational a = ...;
//   LazyRational b = ...;
//   a + b;   // a МУТИРУЕТСЯ (становится SUM(a,b)), b не изменяется
//            // (его дерево импортируется в a через import_tree)
//
// После этой операции a содержит сумму, а b остаётся в исходном
// состоянии (его дерево скопировано внутрь a).
//
// Если вы хотите сохранить a и получить новый LazyRational:
//
//   LazyRational c = a.clone() + b;   // a не затронут, c содержит сумму
//
// ---
// БЕЗОПАСНЫЕ ОПЕРАТОРЫ (НЕ мутируют аргумент)
// ---
//   - Sin(x), Cos(x), Exp(x), Log(x), Sqrt(x), Acos(x) — принимают const&,
//     внутри клонируют x, мутируют клон и возвращают его.
//   - Унарный минус: -x создаёт новый LazyRational (копирует x).
//   - x.clone() — создаёт явную глубокую копию.
//   - x.eval()  — вычисляет x в Rational (для CONST-узлов за O(1)).
//
// МУТИРУЮЩИЕ ОПЕРАТОРЫ (изменяют левый операнд)
// ---
//   - a + b, a - b, a * b, a / b  (все бинарные арифметические)
//   - a += b, a -= b, a *= b, a /= b
//
// ---------------------------------------------------------------------------

#pragma once

#include "rational_class.h"
#include "lazy_nodes.h"
#include "absl/container/inlined_vector.h"
#include "global_state.h"   // <-- ДОБАВЛЕНО для register_clean/unregister_clean
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
        void reset_pool();
        void collect_garbage();
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
        // Доступ к константам (для тестирования через друзей)
        // ------------------------------------------------------------------------
        int add_constant(const internal::Value& v);

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

        friend void internal::reset_pool();
        friend void internal::collect_garbage();

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
        int new_dirty_node(internal::LazyOp op, absl::InlinedVector<int32_t, 2> children,
            int value_idx = -1, int eps_idx = -1);
        void append_sum_children(int sum_node, const LazyRational& other);
        void append_product_children(int prod_node, const LazyRational& other);

        // ------------------------------------------------------------------------
        // Регистрация/дерегистрация в глобальном реестре чистых объектов
        // ------------------------------------------------------------------------
        void register_clean() {
            internal::register_clean(this);
        }
        void unregister_clean() {
            internal::unregister_clean(this);
        }
    };

    std::ostream& operator<<(std::ostream& os, const LazyRational& lr);

} // namespace delta

#include "lazy_rational_impl.h"