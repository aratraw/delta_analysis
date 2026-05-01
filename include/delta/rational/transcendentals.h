// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// transcendentals.h

// ---------------------------------------------------------------------------
// ТРАНСЦЕНДЕНТНЫЕ ФУНКЦИИ: EAGER И LAZY ВЕРСИИ
// ---------------------------------------------------------------------------
// Этот файл предоставляет два семейства функций для работы с трансцендентными
// выражениями: eager (возвращают Rational) и lazy (возвращают LazyRational).
//
// ---------------------------------------------------------------------------
// ДВА СТИЛЯ ПОСТРОЕНИЯ ВЫРАЖЕНИЙ: ПОЛНОСТЬЮ ЛЕНИВЫЙ vs СМЕШАННЫЙ
// ---------------------------------------------------------------------------
//
// Библиотека поддерживает два принципиально разных подхода к построению
// выражений, каждый из которых имеет свои преимущества.
//
// ---
// Стиль 1: Полностью ленивое построение (LazyRational на всех уровнях)
// ---
//
//   LazyRational x = LazyRational("1.5"_r);
//   LazyRational expr = Sin(x.clone() * 2_r + 1_r);   // <-- .clone() !
//
// Что происходит:
//   1. x.clone()        → создаёт глубокую копию x (новый независимый объект)
//   2. clone * 2_r      → мутирует клон, строит узел PRODUCT
//   3. clone + 1_r      → мутирует клон дальше, строит узел SUM
//   4. Sin( SUM(...) )  → создаёт узел SIN над поддеревом
//
// Результат: дерево из трёх узлов (PRODUCT, SUM, SIN). Ни одно вычисление
// не выполнено — построен только план вычислений (граф).
//
// ВАЖНО: никогда не пишите Sin(x * 2_r + 1_r) без .clone()! Операторы
// арифметики МУТИРУЮТ левый lvalue-операнд, и x будет безвозвратно
// испорчен. Всегда используйте .clone() при построении аргументных
// подвыражений, если планируете использовать x далее.
//
// Плюсы:
//   - Аргументы могут быть сколь угодно сложными, всё откладывается.
//   - Канонизация видит всё дерево целиком и может выполнить алгебраические
//     упрощения: Acos(Cos(x)) → x, Exp(Log(x)) → x, сокращение NEG, RECIP.
//   - Один проход канонизации устраняет избыточность во всём выражении.
//
// Минусы:
//   - Требуется явный .clone() при использовании x в нескольких местах
//     (в неизменяемых библиотеках те же копии делаются неявно).
//   - Дерево растёт с каждым оператором, что увеличивает время канонизации.
//
// ---
// Стиль 2: Смешанное построение (eval() для аргументов)
// ---
//
//   LazyRational x = LazyRational("1.5"_r);
//   LazyRational expr = Cos(x.eval() * 2_r + 1_r);
//
// Что происходит:
//   1. x.eval()                   → немедленно вычисляет x → Rational(1.5)
//   2. Rational * 2_r             → eager-умножение → Rational(3.0)
//   3. Rational + 1_r             → eager-сложение  → Rational(4.0)
//   4. Cos( Rational(4.0) )       → создаёт LazyRational с узлом COS(CONST(4.0))
//
// Результат: дерево из ДВУХ узлов (CONST внутри COS). Аргумент косинуса
// уже вычислен до вызова Cos.
//
// Плюсы:
//   - НЕ мутирует исходный x — можно использовать многократно без .clone().
//   - Дерево минимального размера → быстрее канонизация и вычисление.
//   - Арифметика аргументов выполняется eager-способом (быстро, без
//     построения промежуточных узлов графа).
//   - eval() на CONST-узле выполняется за O(1) — это просто доступ к полю,
//     без обхода дерева и без аллокаций.
//
// Минусы:
//   - Теряется возможность алгебраического упрощения подвыражения
//     (x*2+1 уже «впечатано» в константу).
//   - Если снаружи находится Acos(Cos(...)), канонизация НЕ сократит
//     их, потому что аргумент косинуса — готовая константа, а не
//     исходное подвыражение.
//
// ---
// КОГДА ЧТО ИСПОЛЬЗОВАТЬ: ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ
// ---
//
// ┌──────────────────────────────────────┬─────────────────────────────────┐
// │ Сценарий                             │ Рекомендация                    │
// ├──────────────────────────────────────┼─────────────────────────────────┤
// │ Аргумент — простая константа         │ eval() — нечего упрощать,       │
// │                                      │ зато быстро и без .clone()      │
// ├──────────────────────────────────────┼─────────────────────────────────┤
// │ Аргумент сложный, используется       │ .clone() + ленивый — строим     │
// │ ОДИН раз                             │ дерево, канонизируем потом      │
// ├──────────────────────────────────────┼─────────────────────────────────┤
// │ Аргумент сложный, используется       │ .clone() для каждой мутирующей  │
// │ МНОГО раз                            │ позиции или eval()              │
// ├──────────────────────────────────────┼─────────────────────────────────┤
// │ Ожидается алгебраическое упрощение   │ Только ленивый — канонизация    │
// │ (сокращение)                         │ видит всё дерево целиком        │
// ├──────────────────────────────────────┼─────────────────────────────────┤
// │ Критична производительность          │ eval() для константных частей,  │
// │ построения выражения                 │ ленивый для остального          │
// ├──────────────────────────────────────┼─────────────────────────────────┤
// │ Накопление суммы/произведения        │ Ленивый с мутацией — O(N)       │
// │ в цикле (90% вычислительных задач)   │ вместо O(N²) у неизменяемых    │
// └──────────────────────────────────────┴─────────────────────────────────┘
//
// ---
// ПРИМЕР ОПТИМАЛЬНОГО СМЕШАННОГО ИСПОЛЬЗОВАНИЯ
// ---
//
//   LazyRational x = LazyRational("1.23456789"_r);
//   LazyRational expr = Sin(x)                       // x не мутируется
//                     + Cos(x.eval() * 2_r)           // eager-аргумент
//                     + Exp(Log(x.eval() + 1_r));     // eager-аргумент
//
// Здесь Sin(x) остаётся ленивым (может сократиться с внешним Asin),
// а простые константные аргументы для Cos и Log вычисляются сразу,
// не создавая лишних узлов и не мутируя x.
// ВНИМАНИЕ: eval в данном случае логичен только потому что в дереве x один узел CONST:
// Именно на этот случай в eval есть короткий путь, которым мы здесь и ползуемся.
// Если бы x содержало дерево выражений - eval был бы неоптимален, но всё ещё корректен. Ориентируйтесь на ситуацию.

#pragma once

#include "rational_class.h"
#include "lazy_rational.h"
#include "context.h"

namespace delta {

    // ----------------------------------------------------------------------------
    // Eager версии (возвращают Rational)
    // ----------------------------------------------------------------------------
    inline Rational sqrt(const Rational& x, const Rational& eps = default_eps()) {
        return eager_sqrt(x, eps);
    }
    inline Rational exp(const Rational& x, const Rational& eps = default_eps()) {
        return eager_exp(x, eps);
    }
    inline Rational log(const Rational& x, const Rational& eps = default_eps()) {
        return eager_log(x, eps);
    }
    inline Rational sin(const Rational& x, const Rational& eps = default_eps()) {
        return eager_sin(x, eps);
    }
    inline Rational cos(const Rational& x, const Rational& eps = default_eps()) {
        return eager_cos(x, eps);
    }
    inline Rational acos(const Rational& x, const Rational& eps = default_eps()) {
        return eager_acos(x, eps);
    }
    inline Rational pi(const Rational& eps = default_eps()) {
        return eager_pi(eps);
    }
    inline Rational e(const Rational& eps = default_eps()) {
        return eager_e(eps);
    }
    inline Rational pow(const Rational& base, const Rational& exponent, const Rational& eps = default_eps()) {
        return eager_pow(base, exponent, eps);
    }
    inline Rational pow(const Rational& base, int exponent) {
        if (exponent == 0) return Rational(1);
        if (exponent < 0) {
            Rational pos = pow(base, -exponent);
            return 1 / pos;
        }
        Rational result = Rational(1);
        Rational b = base;
        int e = exponent;
        while (e > 0) {
            if (e & 1) result *= b;
            e >>= 1;
            if (e != 0) b = b * b;
        }
        return result;
    }

    // ----------------------------------------------------------------------------
    // НОВЫЕ EAGER ФУНКЦИИ: asin, atan, tan
    // ----------------------------------------------------------------------------
    inline Rational asin(const Rational& x, const Rational& eps = default_eps()) {
        return eager_asin(x, eps);
    }
    inline Rational atan(const Rational& x, const Rational& eps = default_eps()) {
        return eager_atan(x, eps);
    }
    inline Rational tan(const Rational& x, const Rational& eps = default_eps()) {
        return eager_tan(x, eps);
    }

    // ----------------------------------------------------------------------------
    // Lazy версии (возвращают LazyRational) с LazyRational аргументом
    // ----------------------------------------------------------------------------
    inline LazyRational lazy_sqrt(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::SQRT, { child }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_exp(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::EXP, { child }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_log(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::LOG, { child }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_sin(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::SIN, { child }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_cos(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::COS, { child }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_acos(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::ACOS, { child }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_pow(const LazyRational& base, const LazyRational& exponent, const Rational& eps = default_eps()) {
        LazyRational result = base.clone();
        result.ensure_dirty();
        int exp_root = result.import_tree(exponent);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::POW, { result.root_, exp_root }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    // ----------------------------------------------------------------------------
    // НОВЫЕ LAZY ФУНКЦИИ (ЗАКОММЕНТИРОВАНЫ)
    // ----------------------------------------------------------------------------
    // Для полноценной поддержки LazyRational необходимо:
    // 1. Добавить ASIN, ATAN, TAN в enum LazyOp (node_types.h)
    // 2. Обработать их в evaluate_tree (evaluate_impl.h)
    // 3. Добавить в compute_interval (node_pool.h)
    // 4. Поддержать в simplify_tree (simplify_impl.h)
    // 5. Обновить конструкторы DirtyNode/TempNode (lazy_nodes.h)
    // 6. Раскомментировать код ниже
    /*
    inline LazyRational lazy_asin(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::ASIN, { child }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_atan(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::ATAN, { child }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_tan(const LazyRational& x, const Rational& eps = default_eps()) {
        LazyRational result = x.clone();
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int child = result.root_;
        int node = result.new_dirty_node(internal::LazyOp::TAN, { child }, -1, eps_idx);
        result.root_ = node;
        return result;
    }
    */

    // ----------------------------------------------------------------------------
    // Lazy версии с аргументом Rational (без лишнего создания LazyRational)
    // ----------------------------------------------------------------------------
    inline LazyRational lazy_sqrt(const Rational& x, const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int child_const = result.add_constant(x.value());
        int child_node = result.new_dirty_node(internal::LazyOp::CONST, {}, child_const, -1);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::SQRT, { child_node }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_exp(const Rational& x, const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int child_const = result.add_constant(x.value());
        int child_node = result.new_dirty_node(internal::LazyOp::CONST, {}, child_const, -1);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::EXP, { child_node }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_log(const Rational& x, const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int child_const = result.add_constant(x.value());
        int child_node = result.new_dirty_node(internal::LazyOp::CONST, {}, child_const, -1);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::LOG, { child_node }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_sin(const Rational& x, const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int child_const = result.add_constant(x.value());
        int child_node = result.new_dirty_node(internal::LazyOp::CONST, {}, child_const, -1);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::SIN, { child_node }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_cos(const Rational& x, const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int child_const = result.add_constant(x.value());
        int child_node = result.new_dirty_node(internal::LazyOp::CONST, {}, child_const, -1);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::COS, { child_node }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_acos(const Rational& x, const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int child_const = result.add_constant(x.value());
        int child_node = result.new_dirty_node(internal::LazyOp::CONST, {}, child_const, -1);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::ACOS, { child_node }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_pow(const Rational& base, const LazyRational& exponent, const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int base_const = result.add_constant(base.value());
        int base_node = result.new_dirty_node(internal::LazyOp::CONST, {}, base_const, -1);
        int exp_root = result.import_tree(exponent);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::POW, { base_node, exp_root }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_pow(const Rational& base, const Rational& exponent, const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int base_const = result.add_constant(base.value());
        int base_node = result.new_dirty_node(internal::LazyOp::CONST, {}, base_const, -1);
        int exp_const = result.add_constant(exponent.value());
        int exp_node = result.new_dirty_node(internal::LazyOp::CONST, {}, exp_const, -1);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::POW, { base_node, exp_node }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_pow(const LazyRational& base, const Rational& exponent, const Rational& eps = default_eps()) {
        LazyRational result = base.clone();
        result.ensure_dirty();
        int exp_const = result.add_constant(exponent.value());
        int exp_node = result.new_dirty_node(internal::LazyOp::CONST, {}, exp_const, -1);
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::POW, { result.root_, exp_node }, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_pow(const LazyRational& base, int exponent) {
        return lazy_pow(base, Rational(exponent), default_eps());
    }

    // Статические фабрики для констант
    inline LazyRational lazy_pi(const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::PI, {}, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    inline LazyRational lazy_e(const Rational& eps = default_eps()) {
        LazyRational result;
        result.ensure_dirty();
        int eps_idx = result.add_constant(eps.value());
        int node = result.new_dirty_node(internal::LazyOp::E, {}, -1, eps_idx);
        result.root_ = node;
        return result;
    }

    // ----------------------------------------------------------------------------
    // Удобные короткие имена для lazy-вычислений (заглавные буквы)
    // ----------------------------------------------------------------------------
    inline LazyRational Sqrt(const LazyRational& x, const Rational& eps = default_eps()) {
        return lazy_sqrt(x, eps);
    }
    inline LazyRational Sqrt(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_sqrt(x, eps);
    }

    inline LazyRational Exp(const LazyRational& x, const Rational& eps = default_eps()) {
        return lazy_exp(x, eps);
    }
    inline LazyRational Exp(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_exp(x, eps);
    }

    inline LazyRational Log(const LazyRational& x, const Rational& eps = default_eps()) {
        return lazy_log(x, eps);
    }
    inline LazyRational Log(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_log(x, eps);
    }

    inline LazyRational Sin(const LazyRational& x, const Rational& eps = default_eps()) {
        return lazy_sin(x, eps);
    }
    inline LazyRational Sin(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_sin(x, eps);
    }

    inline LazyRational Cos(const LazyRational& x, const Rational& eps = default_eps()) {
        return lazy_cos(x, eps);
    }
    inline LazyRational Cos(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_cos(x, eps);
    }

    inline LazyRational Acos(const LazyRational& x, const Rational& eps = default_eps()) {
        return lazy_acos(x, eps);
    }
    inline LazyRational Acos(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_acos(x, eps);
    }

    inline LazyRational Pi(const Rational& eps = default_eps()) {
        return lazy_pi(eps);
    }

    inline LazyRational E(const Rational& eps = default_eps()) {
        return lazy_e(eps);
    }

    inline LazyRational Pow(const LazyRational& base, const LazyRational& exponent, const Rational& eps = default_eps()) {
        return lazy_pow(base, exponent, eps);
    }
    inline LazyRational Pow(const Rational& base, const LazyRational& exponent, const Rational& eps = default_eps()) {
        return lazy_pow(base, exponent, eps);
    }
    inline LazyRational Pow(const LazyRational& base, const Rational& exponent, const Rational& eps = default_eps()) {
        return lazy_pow(base, exponent, eps);
    }
    inline LazyRational Pow(const Rational& base, const Rational& exponent, const Rational& eps = default_eps()) {
        return lazy_pow(base, exponent, eps);
    }
    inline LazyRational Pow(const LazyRational& base, int exponent) {
        return lazy_pow(base, exponent);
    }

    // ----------------------------------------------------------------------------
    // НОВЫЕ КОРОТКИЕ ИМЕНА (ЗАКОММЕНТИРОВАНЫ, зависят от lazy_* выше)
    // ----------------------------------------------------------------------------
    /*
    inline LazyRational Asin(const LazyRational& x, const Rational& eps = default_eps()) {
        return lazy_asin(x, eps);
    }
    inline LazyRational Asin(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_asin(x, eps);
    }

    inline LazyRational Atan(const LazyRational& x, const Rational& eps = default_eps()) {
        return lazy_atan(x, eps);
    }
    inline LazyRational Atan(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_atan(x, eps);
    }

    inline LazyRational Tan(const LazyRational& x, const Rational& eps = default_eps()) {
        return lazy_tan(x, eps);
    }
    inline LazyRational Tan(const Rational& x, const Rational& eps = default_eps()) {
        return lazy_tan(x, eps);
    }
    */

} // namespace delta