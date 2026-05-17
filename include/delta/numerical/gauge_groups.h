// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/numerical/gauge_groups.h
// ============================================================================
// CALIBRATIONAL GROUPS – U(1), SU(2), SU(3)
// ============================================================================
//
// Provides template classes for the gauge groups U(1) (as SO(2)), SU(2) and
// SU(3).  For Scalar = Rational the SU(2) and SU(3) groups use GaussQi as the
// complex number type, and the exponential / logarithm are implemented via the
// general matrix functions delta::exp / delta::log (which employ trace
// normalisation for the complex case).
//
// ============================================================================

#pragma once

#include <Eigen/Dense>
#include <complex>
#include "delta/core/rational.h"
#include "delta/rational/gauss_qi.h"             // GaussQi
#include "delta/rational/transcendentals.h"      // delta::cos, sin, acos, exp, log
#include "delta/core/eigen_integration.h"                         // matrix exp/log for GaussQi

   // =========================================================================
   // СТРАТЕГИЧЕСКИЙ ОБЗОР: ВАРИАНТЫ РЕФАКТОРИНГА ПОД ПОЛИМОРФИЗМ (Rational vs P-adic)
   // =========================================================================
   // ТЕКУЩИЙ КОД завязан на архимедову геометрию (cos/sin). Чтобы в будущем 
   // внедрить сюда p-адические числа (где вместо тригонометрии нужны ряды Тейлора),
   // у нас есть два принципиальных архитектурных пути на шаблонах C++. 
   // Мы детально распишем оба, но выбирать прямо сейчас не будем — голова пухнет.
   //
   // ПОДХОД А: Полная специализация всего класса (Incomplete Generic Template)
   // -------------------------------------------------------------------------
   // Базовый шаблон оставляется пустым декларативным интерфейсом:
   // `template<typename Scalar> struct U1;`
   // А для каждого типа скаляра пишется своя изолированная с нуля копия класса:
   // `template<> struct U1<Rational> { ... };` (аналитическая тригонометрия)
   // `template<> struct U1<PadicNumber> { ... };` (матричные ряды Тейлора)
   // * Плюсы: Полная свобода. В P-adic версии можно объявить другие связанные 
   //          типы (например, метрический скаляр) или убрать лишние параметры.
   // * Минусы: Приходится дублировать общий код (matrix_type, identity, N = 1).
   //
   // ПОДХОД Б: Делегирование стратегий вычисления (Generic Каркас + Внешний Движок)
   // -------------------------------------------------------------------------
   // Шаблон `U1` остается ЕДИНЫМ и generic (как сейчас). Все общие типы и методы 
   // (identity, N) пишутся ОДИН РАЗ. А специфичная математика выносится во внешние 
   // перегруженные функции (стратегии), которые компилятор выбирает по типу Scalar:
   // `static matrix_type exp(...) { return delta::exp_u1_impl(A, eps); }`
   // Мы просто пишем две внешние реализации `exp_u1_impl` для Rational и PadicNumber.
   // * Плюсы: Ноль дублирования кода. Класс U1 становится абсолютно всеядным.
   // * Минусы: Все типы скаляров обязаны строго укладываться в единую структуру класса.
   //
   // ВЫВОД:
   // Оба подхода рабочие. Оставляем этот комментарий как ментальную карту, 
   // закрываем файл и идём пить кофе, пока не взорвался мозг.
   // =========================================================================

namespace delta::numerical {

    // =========================================================================
    // U(1) = SO(2)
    // =========================================================================
    template<typename Scalar>
    struct U1 {
        using matrix_type = Eigen::Matrix<Scalar, 2, 2>;
        using algebra_type = Eigen::Matrix<Scalar, 2, 2>;   // skew‑symmetric
        static constexpr int N = 1;
        static matrix_type identity() {
            return matrix_type::Identity();
        }

        //used to compute Wilson Action, needs to be Rational - metric scalar type.
        static Rational /*metric scalar type*/ real_trace(const matrix_type& U) {
            return U.trace();
        }
        // Exponential of a skew‑symmetric matrix [[0, -θ],[θ, 0]]
        static matrix_type exp(const algebra_type& A,
            const Scalar& eps = delta::default_eps()) {
            Scalar theta = A(1, 0);   // the (1,0) entry is θ
            Scalar c = delta::cos(theta, eps);
            Scalar s = delta::sin(theta, eps);
            matrix_type R;
            R << c, -s,
                s, c;
            return R;
        }

        // Principal logarithm of a rotation matrix
        static algebra_type log(const matrix_type& U,
            const Scalar& eps = delta::default_eps()) {
            // U = [[c, -s], [s, c]];  c = cos θ, s = sin θ
            Scalar c = U(0, 0);
            Scalar s = U(1, 0);
            Scalar theta = delta::acos(c, eps);
            // Adjust sign according to s
            if (s < 0) theta = -theta;
            algebra_type A;
            A << 0, -theta,
                theta, 0;
            return A;
        }
    };

    // =========================================================================
    // SU(2)
    // =========================================================================
    template<typename Scalar>
    struct SU2;

    template<>
    struct SU2<Rational> {
        using matrix_type = Eigen::Matrix<GaussQi, 2, 2>;
        using algebra_type = Eigen::Matrix<GaussQi, 2, 2>;   // i * Hermitian traceless
        static constexpr int N = 2;
        static matrix_type identity() {
            return matrix_type::Identity();
        }
        //used to compute Wilson Action, needs to be Rational - metric scalar type.
        static Rational /*metric scalar type*/ real_trace(const matrix_type& U) {
            return U.trace().real();
        }
        static matrix_type exp(const algebra_type& A,
            const Rational& eps = delta::default_eps()) {
            return delta::exp(A, eps);
        }

        static algebra_type log(const matrix_type& U,
            const Rational& eps = delta::default_eps()) {
            return delta::log(U, eps);
        }
    };

    // For double (if ever needed)
    template<>
    struct SU2<double> {
        using matrix_type = Eigen::Matrix<std::complex<double>, 2, 2>;
        using algebra_type = Eigen::Matrix<std::complex<double>, 2, 2>;
        static constexpr int N = 2;
        static matrix_type identity() { return matrix_type::Identity(); }
        // ... trivial wrappers if required
    };

    // =========================================================================
    // SU(3)
    // =========================================================================
    template<typename Scalar>
    struct SU3;

    template<>
    struct SU3<Rational> {
        using matrix_type = Eigen::Matrix<GaussQi, 3, 3>;
        using algebra_type = Eigen::Matrix<GaussQi, 3, 3>;   // i * Hermitian traceless
        static constexpr int N = 3;
        static matrix_type identity() {
            return matrix_type::Identity();
        }
        //used to compute Wilson Action, needs to be Rational - metric scalar type.
        static Rational /*metric scalar type*/ real_trace(const matrix_type& U) {
            return U.trace().real();
        }
        static matrix_type exp(const algebra_type& A,
            const Rational& eps = delta::default_eps()) {
            return delta::exp(A, eps);
        }

        static algebra_type log(const matrix_type& U,
            const Rational& eps = delta::default_eps()) {
            return delta::log(U, eps);
        }
    };

} // namespace delta::numerical