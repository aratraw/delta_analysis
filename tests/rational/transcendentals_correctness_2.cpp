// tests/rational/transcendentals_correctness_test_2.cpp
// ============================================================================
// ДОПОЛНИТЕЛЬНЫЕ ТЕСТЫ ТРАНСЦЕНДЕНТНЫХ ФУНКЦИЙ
// ============================================================================
//
// Покрываемые функции:
//   - sinpi, cospi, tanpi, asinpi, acospi, atanpi (eager)
//   - asin, acos, atan, tan для GaussQi (eager)
//   - atan2 для Rational
//   - floor для Rational
//   - pow с отрицательным основанием и рациональным показателем
//   - sin, cos с очень большими аргументами
//
// Принципы проверки:
//   - Точные рациональные значения -> EXPECT_EQ
//   - Значения с радикалами (√2/2 и т.п.) -> EXPECT_RATIONAL_NEAR с eps * K
//   - Тождества -> допуск, масштабированный от eps
// ============================================================================

#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <limits>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include "delta/core/rational.h"
#include "test_utils.h"

namespace delta::testing {

    // ----------------------------------------------------------------------------
    // Вспомогательные функции для генерации случайных рациональных чисел
    // ----------------------------------------------------------------------------
    static Rational random_rational(int64_t min_num, int64_t max_num, int64_t min_den, int64_t max_den, std::mt19937& rng) {
        std::uniform_int_distribution<int64_t> num_dist(min_num, max_num);
        std::uniform_int_distribution<int64_t> den_dist(min_den, max_den);
        int64_t num = num_dist(rng);
        int64_t den = den_dist(rng);
        if (den == 0) den = 1;
        return Rational(num, den);
    }

    static GaussQi random_gaussqi(int64_t min, int64_t max, std::mt19937& rng) {
        std::uniform_int_distribution<int64_t> dist(min, max);
        return GaussQi(Rational(dist(rng)), Rational(dist(rng)));
    }

    // ----------------------------------------------------------------------------
    // Высокоточный эталон для π (200 десятичных знаков, достаточно для 1e-80)
    // ----------------------------------------------------------------------------
    static Rational high_precision_pi() {
        // π до 100 десятичных знаков (достаточно для тестов на 1e-80)
        static const Rational pi_high(
            "31415926535897932384626433832795028841971693993751"
            "05820974944592307816406286208998628034825342117068"
            "/10000000000000000000000000000000000000000000000000"
            "00000000000000000000000000000000000000000000000000"
        );
        return pi_high;
    }

    // ----------------------------------------------------------------------------
    // Тестовый фикстур для всех скалярных трансцендентных функций
    // ----------------------------------------------------------------------------
    class TranscendentalExtendedTest : public RationalTest {
    protected:
        void SetUp() override {
            RationalTest::SetUp();
            // Сбрасываем глобальную эпсилон, чтобы тесты были предсказуемы
            delta::reset_default_eps();
        }

        void TearDown() override {
            delta::reset_default_eps();
        }

        // Набор эпсилон для тестирования разных режимов точности
        const Rational eps_low = Rational("1/1000000");                   // 1e-6
        const Rational eps_std = Rational("1/1000000000000000000000000000000"); // 1e-30
        const Rational eps_high = Rational("1/10000000000000000000000000000000000000000"); // 1e-40
        const Rational eps_ultra = Rational("1/10000000000000000000000000000000000000000000000000000000000000000000000000000000"); // 1e-80
    };

    // ============================================================================
    // 1. СЕМЕЙСТВО *pi
    // ============================================================================

    TEST_F(TranscendentalExtendedTest, SinPiTableExact) {
        // Точные значения (0, ±1, ±1/2)
        EXPECT_EQ(sinpi(0_r), 0_r);
        EXPECT_EQ(sinpi(1_r), 0_r);
        EXPECT_EQ(sinpi(2_r), 0_r);
        EXPECT_EQ(sinpi(-1_r), 0_r);
        EXPECT_EQ(sinpi(Rational(1, 2)), 1_r);
        EXPECT_EQ(sinpi(Rational(3, 2)), -1_r);
        EXPECT_EQ(sinpi(Rational(-1, 2)), -1_r);
        EXPECT_EQ(sinpi(Rational(5, 2)), 1_r);
    }

    TEST_F(TranscendentalExtendedTest, SinPiTableRadical) {
        Rational sqrt2 = delta::sqrt(2_r, eps_std);
        Rational sqrt3 = delta::sqrt(3_r, eps_std);

        // sinpi(1/4) = √2/2
        EXPECT_RATIONAL_NEAR(sinpi(Rational(1, 4)), sqrt2 / 2_r, eps_std * 10);
        // sinpi(3/4) = √2/2 (после редукции)
        EXPECT_RATIONAL_NEAR(sinpi(Rational(3, 4)), sqrt2 / 2_r, eps_std * 10);
        // sinpi(1/3) = √3/2
        EXPECT_RATIONAL_NEAR(sinpi(Rational(1, 3)), sqrt3 / 2_r, eps_std * 10);
        // sinpi(2/3) = √3/2
        EXPECT_RATIONAL_NEAR(sinpi(Rational(2, 3)), sqrt3 / 2_r, eps_std * 10);
        // sinpi(1/6) = 1/2
        EXPECT_EQ(sinpi(Rational(1, 6)), 1_r / 2);
        // sinpi(5/6) = 1/2
        EXPECT_EQ(sinpi(Rational(5, 6)), 1_r / 2);
    }

    TEST_F(TranscendentalExtendedTest, SinPiSymmetryPeriodicity) {
        for (int k = -5; k <= 5; ++k) {
            Rational x = Rational(k) / 3_r; // произвольная дробь
            // sinpi(x + 1) = -sinpi(x)
            EXPECT_RATIONAL_NEAR(sinpi(x + 1_r), -sinpi(x), eps_std * 10);
            // sinpi(-x) = -sinpi(x)
            EXPECT_RATIONAL_NEAR(sinpi(-x), -sinpi(x), eps_std * 10);
        }
    }

    TEST_F(TranscendentalExtendedTest, CosPiTableExact) {
        EXPECT_EQ(cospi(0_r), 1_r);
        EXPECT_EQ(cospi(1_r), -1_r);
        EXPECT_EQ(cospi(2_r), 1_r);
        EXPECT_EQ(cospi(Rational(1, 2)), 0_r);
        EXPECT_EQ(cospi(Rational(3, 2)), 0_r);
        EXPECT_EQ(cospi(Rational(-1, 2)), 0_r);
    }

    TEST_F(TranscendentalExtendedTest, CosPiTableRadical) {
        Rational sqrt2 = delta::sqrt(2_r, eps_std);
        Rational sqrt3 = delta::sqrt(3_r, eps_std);

        // cospi(1/4) = √2/2
        EXPECT_RATIONAL_NEAR(cospi(Rational(1, 4)), sqrt2 / 2_r, eps_std * 10);
        // cospi(3/4) = -√2/2
        EXPECT_RATIONAL_NEAR(cospi(Rational(3, 4)), -sqrt2 / 2_r, eps_std * 10);
        // cospi(1/3) = 1/2
        EXPECT_EQ(cospi(Rational(1, 3)), 1_r / 2);
        // cospi(2/3) = -1/2
        EXPECT_EQ(cospi(Rational(2, 3)), -1_r / 2);
        // cospi(1/6) = √3/2
        EXPECT_RATIONAL_NEAR(cospi(Rational(1, 6)), sqrt3 / 2_r, eps_std * 10);
        // cospi(5/6) = -√3/2
        EXPECT_RATIONAL_NEAR(cospi(Rational(5, 6)), -sqrt3 / 2_r, eps_std * 10);
    }

    TEST_F(TranscendentalExtendedTest, CosPiSymmetryPeriodicity) {
        for (int k = -5; k <= 5; ++k) {
            Rational x = Rational(k) / 3_r;
            // cospi(x + 1) = -cospi(x)
            EXPECT_RATIONAL_NEAR(cospi(x + 1_r), -cospi(x), eps_std * 10);
            // cospi(-x) = cospi(x)
            EXPECT_RATIONAL_NEAR(cospi(-x), cospi(x), eps_std * 10);
        }
    }

    TEST_F(TranscendentalExtendedTest, TanPiTableExact) {
        EXPECT_EQ(tanpi(0_r), 0_r);
        EXPECT_EQ(tanpi(1_r), 0_r);
        EXPECT_EQ(tanpi(Rational(1, 4)), 1_r);
        EXPECT_EQ(tanpi(Rational(5, 4)), 1_r);
        EXPECT_EQ(tanpi(Rational(-1, 4)), -1_r);
    }

    TEST_F(TranscendentalExtendedTest, TanPiTableRadical) {
        Rational sqrt3 = delta::sqrt(3_r, eps_std);
        // tanpi(1/6) = 1/√3 = √3/3
        EXPECT_RATIONAL_NEAR(tanpi(Rational(1, 6)), sqrt3 / 3_r, eps_std * 10);
        // tanpi(5/6) = -1/√3 = -√3/3
        EXPECT_RATIONAL_NEAR(tanpi(Rational(5, 6)), -sqrt3 / 3_r, eps_std * 10);
        // tanpi(1/3) = √3
        EXPECT_RATIONAL_NEAR(tanpi(Rational(1, 3)), sqrt3, eps_std * 10);
        // tanpi(2/3) = -√3
        EXPECT_RATIONAL_NEAR(tanpi(Rational(2, 3)), -sqrt3, eps_std * 10);
    }

    TEST_F(TranscendentalExtendedTest, TanPiSingularity) {
        // Нечётные полуцелые: tanpi(1/2 + k) должно выбрасывать std::domain_error
        for (int k = -3; k <= 3; ++k) {
            Rational x = Rational(1, 2) + Rational(k);
            EXPECT_THROW(tanpi(x), std::domain_error);
            EXPECT_THROW(tanpi(-x), std::domain_error);
        }
    }

    TEST_F(TranscendentalExtendedTest, TanPiSymmetry) {
        for (int k = -5; k <= 5; ++k) {
            Rational x = Rational(k) / 3_r;
            if (x == Rational(1, 2) || x == -Rational(1, 2)) continue;
            // tanpi(x + 1) = tanpi(x)
            EXPECT_RATIONAL_NEAR(tanpi(x + 1_r), tanpi(x), eps_std * 10);
            // tanpi(-x) = -tanpi(x)
            EXPECT_RATIONAL_NEAR(tanpi(-x), -tanpi(x), eps_std * 10);
        }
    }

    TEST_F(TranscendentalExtendedTest, AsinPiTableExact) {
        // Точные рациональные значения asinpi(y) для y ∈ {0, ±1/2, ±1}
        EXPECT_EQ(asinpi(0_r), 0_r);
        EXPECT_EQ(asinpi(1_r / 2), Rational(1, 6));
        EXPECT_EQ(asinpi(-1_r / 2), -Rational(1, 6));
        EXPECT_EQ(asinpi(1_r), Rational(1, 2));
        EXPECT_EQ(asinpi(-1_r), -Rational(1, 2));
    }

    TEST_F(TranscendentalExtendedTest, AsinPiTableRadical) {
        Rational sqrt2 = delta::sqrt(2_r, eps_std);
        Rational sqrt3 = delta::sqrt(3_r, eps_std);
        // asinpi(√2/2) = 1/4
        EXPECT_RATIONAL_NEAR(asinpi(sqrt2 / 2_r), Rational(1, 4), eps_std * 10);
        // asinpi(√3/2) = 1/3
        EXPECT_RATIONAL_NEAR(asinpi(sqrt3 / 2_r), Rational(1, 3), eps_std * 10);
        // asinpi(-√2/2) = -1/4
        EXPECT_RATIONAL_NEAR(asinpi(-sqrt2 / 2_r), -Rational(1, 4), eps_std * 10);
        // asinpi(-√3/2) = -1/3
        EXPECT_RATIONAL_NEAR(asinpi(-sqrt3 / 2_r), -Rational(1, 3), eps_std * 10);
    }

    TEST_F(TranscendentalExtendedTest, AcoSPiTableExact) {
        EXPECT_EQ(acospi(0_r), Rational(1, 2));
        EXPECT_EQ(acospi(1_r / 2), Rational(1, 3));
        EXPECT_EQ(acospi(1_r), 0_r);
        EXPECT_EQ(acospi(-1_r), 1_r);
    }

    TEST_F(TranscendentalExtendedTest, AcoSPiTableRadical) {
        Rational sqrt2 = delta::sqrt(2_r, eps_std);
        Rational sqrt3 = delta::sqrt(3_r, eps_std);
        // acospi(√2/2) = 1/4
        EXPECT_RATIONAL_NEAR(acospi(sqrt2 / 2_r), Rational(1, 4), eps_std * 10);
        // acospi(√3/2) = 1/6
        EXPECT_RATIONAL_NEAR(acospi(sqrt3 / 2_r), Rational(1, 6), eps_std * 10);
        // acospi(-√2/2) = 3/4
        EXPECT_RATIONAL_NEAR(acospi(-sqrt2 / 2_r), Rational(3, 4), eps_std * 10);
        // acospi(-√3/2) = 5/6
        EXPECT_RATIONAL_NEAR(acospi(-sqrt3 / 2_r), Rational(5, 6), eps_std * 10);
    }

    TEST_F(TranscendentalExtendedTest, AtanPiTableExact) {
        EXPECT_EQ(atanpi(0_r), 0_r);
        EXPECT_EQ(atanpi(1_r), Rational(1, 4));
        EXPECT_EQ(atanpi(-1_r), -Rational(1, 4));
        // atanpi(√3/3) = 1/6
        Rational sqrt3 = delta::sqrt(3_r, eps_std);
        EXPECT_RATIONAL_NEAR(atanpi(sqrt3 / 3_r), Rational(1, 6), eps_std * 10);
        // atanpi(√3) = 1/3
        EXPECT_RATIONAL_NEAR(atanpi(sqrt3), Rational(1, 3), eps_std * 10);
        // atanpi(-√3) = -1/3
        EXPECT_RATIONAL_NEAR(atanpi(-sqrt3), -Rational(1, 3), eps_std * 10);
    }

    TEST_F(TranscendentalExtendedTest, InversePiIdentities) {
        // asinpi(y) + acospi(y) = 1/2
        for (int i = -10; i <= 10; ++i) {
            Rational y = Rational(i) / 10_r;
            if (y < -1 || y > 1) continue;
            EXPECT_RATIONAL_NEAR(asinpi(y) + acospi(y), Rational(1, 2), eps_std * 10);
        }
        // asinpi(-y) = -asinpi(y)
        EXPECT_RATIONAL_NEAR(asinpi(-Rational(1, 3)), -asinpi(Rational(1, 3)), eps_std * 10);
        // atanpi(-y) = -atanpi(y)
        EXPECT_RATIONAL_NEAR(atanpi(-Rational(1, 3)), -atanpi(Rational(1, 3)), eps_std * 10);
    }

    TEST_F(TranscendentalExtendedTest, SinPiViaSin) {
        const Rational x = Rational(1234567, 1000000);
        for (const Rational& eps : { eps_low, eps_std, eps_high, eps_ultra }) {
            Rational direct = sinpi(x, eps);
            Rational via_sin = delta::sin(high_precision_pi() * x, eps);
            EXPECT_RATIONAL_NEAR(direct, via_sin, eps * 100);
        }
    }

    TEST_F(TranscendentalExtendedTest, CosPiViaCos) {
        const Rational x = Rational(7654321, 1000000);
        for (const Rational& eps : { eps_low, eps_std, eps_high, eps_ultra }) {
            Rational direct = cospi(x, eps);
            Rational via_cos = delta::cos(high_precision_pi() * x, eps);
            EXPECT_RATIONAL_NEAR(direct, via_cos, eps * 100);
        }
    }

    TEST_F(TranscendentalExtendedTest, TanPiViaTan) {
        const Rational x = Rational(123456, 100000);
        for (const Rational& eps : { eps_low, eps_std, eps_high }) { // tanpi(123456/100000) not singular
            Rational direct = tanpi(x, eps);
            Rational via_tan = delta::tan(high_precision_pi() * x, eps);
            EXPECT_RATIONAL_NEAR(direct, via_tan, eps * 100);
        }
    }

    TEST_F(TranscendentalExtendedTest, AsinPiViaAsin) {
        const Rational y = Rational(3, 7);
        for (const Rational& eps : { eps_low, eps_std, eps_high }) {
            Rational direct = asinpi(y, eps);
            Rational via_asin = delta::asin(y, eps) / delta::pi(eps);
            EXPECT_RATIONAL_NEAR(direct, via_asin, eps * 100);
        }
    }

    TEST_F(TranscendentalExtendedTest, AcoSPiViaAcos) {
        const Rational y = Rational(2, 5);
        for (const Rational& eps : { eps_low, eps_std, eps_high }) {
            Rational direct = acospi(y, eps);
            Rational via_acos = delta::acos(y, eps) / delta::pi(eps);
            EXPECT_RATIONAL_NEAR(direct, via_acos, eps * 100);
        }
    }

    TEST_F(TranscendentalExtendedTest, AtanPiViaAtan) {
        const Rational y = Rational(4, 3);
        for (const Rational& eps : { eps_low, eps_std, eps_high }) {
            Rational direct = atanpi(y, eps);
            Rational via_atan = delta::atan(y, eps) / delta::pi(eps);
            EXPECT_RATIONAL_NEAR(direct, via_atan, eps * 100);
        }
    }

    TEST_F(TranscendentalExtendedTest, SinPiRandom) {
        std::mt19937 rng(12345);
        for (int i = 0; i < 100; ++i) {
            Rational x = random_rational(-10, 10, 1, 100, rng);
            Rational expected = delta::sin(high_precision_pi() * x, eps_std);
            EXPECT_RATIONAL_NEAR(sinpi(x, eps_std), expected, eps_std * 100);
        }
    }

    TEST_F(TranscendentalExtendedTest, CosPiRandom) {
        std::mt19937 rng(12345);
        for (int i = 0; i < 100; ++i) {
            Rational x = random_rational(-10, 10, 1, 100, rng);
            Rational expected = delta::cos(high_precision_pi() * x, eps_std);
            EXPECT_RATIONAL_NEAR(cospi(x, eps_std), expected, eps_std * 100);
        }
    }

    TEST_F(TranscendentalExtendedTest, TanPiRandom) {
        std::mt19937 rng(12345);
        for (int i = 0; i < 100; ++i) {
            Rational x = random_rational(-10, 10, 1, 100, rng);
            // Пропускаем точки, где tanpi не определён
            bool skip = false;
            Rational x_red = delta::abs(x);
            Rational n = delta::floor(x_red);
            Rational f = x_red - n;
            if (f == Rational(1, 2)) skip = true;
            if (skip) continue;
            Rational expected = delta::tan(high_precision_pi() * x, eps_std);
            EXPECT_RATIONAL_NEAR(tanpi(x, eps_std), expected, eps_std * 100);
        }
    }

    // ============================================================================
    // 2. ТРАНСЦЕНДЕНТНЫЕ ФУНКЦИИ ДЛЯ GaussQi: asin, acos, atan, tan
    // ============================================================================

    TEST_F(TranscendentalExtendedTest, GaussQiAsinReal) {
        for (const Rational& eps : { eps_low, eps_std, eps_high }) {
            for (Rational x : {0_r, Rational(1, 2), 1_r, Rational(-1, 2), -1_r}) {
                GaussQi z(x, 0_r);
                GaussQi res = delta::asin(z, eps);
                Rational expected = delta::asin(x, eps);
                EXPECT_RATIONAL_NEAR(res.real(), expected, eps * 10);
                EXPECT_RATIONAL_NEAR(res.imag(), 0_r, eps * 10);
            }
        }
    }

    TEST_F(TranscendentalExtendedTest, GaussQiAcosReal) {
        for (const Rational& eps : { eps_low, eps_std, eps_high }) {
            for (Rational x : {0_r, Rational(1, 2), 1_r, Rational(-1, 2), -1_r}) {
                GaussQi z(x, 0_r);
                GaussQi res = delta::acos(z, eps);
                Rational expected = delta::acos(x, eps);
                EXPECT_RATIONAL_NEAR(res.real(), expected, eps * 10);
                EXPECT_RATIONAL_NEAR(res.imag(), 0_r, eps * 10);
            }
        }
    }

    TEST_F(TranscendentalExtendedTest, GaussQiAtanReal) {
        for (const Rational& eps : { eps_low, eps_std, eps_high }) {
            for (Rational x : {0_r, 1_r, -1_r, Rational(1, 2), Rational(-1, 2)}) {
                GaussQi z(x, 0_r);
                GaussQi res = delta::atan(z, eps);
                Rational expected = delta::atan(x, eps);
                EXPECT_RATIONAL_NEAR(res.real(), expected, eps * 10);
                EXPECT_RATIONAL_NEAR(res.imag(), 0_r, eps * 10);
            }
        }
    }

    TEST_F(TranscendentalExtendedTest, GaussQiTanReal) {
        for (const Rational& eps : { eps_low, eps_std, eps_high }) {
            for (Rational x : {0_r, Rational(1, 4), 1_r, -1_r}) {
                GaussQi z(x, 0_r);
                GaussQi res = delta::tan(z, eps);
                Rational expected = delta::tan(x, eps);
                EXPECT_RATIONAL_NEAR(res.real(), expected, eps * 10);
                EXPECT_RATIONAL_NEAR(res.imag(), 0_r, eps * 10);
            }
        }
    }

    TEST_F(TranscendentalExtendedTest, GaussQiAsinPureImag) {
        for (const Rational& eps : { eps_low, eps_std, eps_high }) {
            for (Rational y : {Rational(1, 2), 1_r, 2_r}) {
                GaussQi z(0_r, y);
                GaussQi res = delta::asin(z, eps);
                // asin(i*y) = i * asinh(y)
                Rational asinh_y = delta::log(y + delta::sqrt(y * y + 1_r, eps), eps);
                Rational expected_real = 0_r;
                Rational expected_imag = asinh_y;
                EXPECT_RATIONAL_NEAR(res.real(), expected_real, eps * 10);
                EXPECT_RATIONAL_NEAR(res.imag(), expected_imag, eps * 100);
            }
        }
    }

    TEST_F(TranscendentalExtendedTest, GaussQiAcosPureImag) {
        for (const Rational& eps : { eps_low, eps_std, eps_high }) {
            for (Rational y : {Rational(1, 2), 1_r, 2_r}) {
                GaussQi z(0_r, y);
                GaussQi res = delta::acos(z, eps);
                // acos(i*y) = π/2 - i*asinh(y)
                Rational asinh_y = delta::log(y + delta::sqrt(y * y + 1_r, eps), eps);
                Rational expected_real = delta::pi(eps) / 2_r;
                Rational expected_imag = -asinh_y;
                EXPECT_RATIONAL_NEAR(res.real(), expected_real, eps * 10);
                EXPECT_RATIONAL_NEAR(res.imag(), expected_imag, eps * 100);
            }
        }
    }
    TEST_F(TranscendentalExtendedTest, GaussQiAtanPureImag) {
        for (const Rational& eps : { eps_low, eps_std, eps_high }) {
            // Случай 1: |y| < 1  ->  atan(i*y) = i * atanh(y)
            {
                Rational y = Rational(1, 2);
                GaussQi z(0_r, y);
                GaussQi res = delta::atan(z, eps);
                Rational imag_part = "0.5"_r * delta::log((1_r + y) / (1_r - y), eps);
                EXPECT_RATIONAL_NEAR(res.real(), 0_r, eps * 10);
                EXPECT_RATIONAL_NEAR(res.imag(), imag_part, eps * 100);
            }

            // Случай 2: y > 1  ->  atan(i*y) = i * atanh(y) (вещественная часть остается 0)
            {
                Rational y = 2_r;
                GaussQi z(0_r, y);
                GaussQi res = delta::atan(z, eps);
                Rational imag_part = "0.5"_r * delta::log((y + 1_r) / (y - 1_r), eps);
                EXPECT_RATIONAL_NEAR(res.real(), 0_r, eps * 10);
                EXPECT_RATIONAL_NEAR(res.imag(), imag_part, eps * 100);
            }

            // Случай 3: y < -1  ->  atan(i*y) = i * atanh(y) (знак определяется свойством нечетности)
            {
                Rational y = -2_r;
                GaussQi z(0_r, y);
                GaussQi res = delta::atan(z, eps);
                // Используем модуль для вычисления базового значения логарифма
                Rational abs_y = 2_r;
                Rational imag_part = "0.5"_r * delta::log((abs_y + 1_r) / (abs_y - 1_r), eps);
                EXPECT_RATIONAL_NEAR(res.real(), 0_r, eps * 10);
                EXPECT_RATIONAL_NEAR(res.imag(), -imag_part, eps * 100);
            }

            // Случай 4: y = 1  ->  особая точка (точка ветвления)
            {
                GaussQi z(0_r, 1_r);
                EXPECT_THROW(delta::atan(z, eps), std::domain_error);
            }
        }
    }

    TEST_F(TranscendentalExtendedTest, GaussQiTanPureImag) {
        for (const Rational& eps : { eps_low, eps_std, eps_high }) {
            for (Rational y : {Rational(1, 2), 1_r, 2_r}) {
                GaussQi z(0_r, y);
                GaussQi res = delta::tan(z, eps);
                // tan(i*y) = i * tanh(y)
                Rational tanh_y = (delta::exp(y, eps) - delta::exp(-y, eps)) /
                    (delta::exp(y, eps) + delta::exp(-y, eps));
                Rational expected_real = 0_r;
                Rational expected_imag = tanh_y;
                EXPECT_RATIONAL_NEAR(res.real(), expected_real, eps * 10);
                EXPECT_RATIONAL_NEAR(res.imag(), expected_imag, eps * 100);
            }
        }
    }

    TEST_F(TranscendentalExtendedTest, GaussQiAsinAcosIdentities) {
        std::mt19937 rng(12345);
        for (int i = 0; i < 20; ++i) {
            GaussQi z = random_gaussqi(-2, 2, rng);
            for (const Rational& eps : { eps_low, eps_std }) {
                GaussQi asin_z = delta::asin(z, eps);
                GaussQi acos_z = delta::acos(z, eps);
                // asin(z) + acos(z) = π/2
                GaussQi sum = asin_z + acos_z;
                EXPECT_RATIONAL_NEAR(sum.real(), delta::pi(eps) / 2_r, eps * 100);
                EXPECT_RATIONAL_NEAR(sum.imag(), 0_r, eps * 100);
            }
        }
    }

    TEST_F(TranscendentalExtendedTest, GaussQiAsinSin) {
        std::mt19937 rng(12345);
        const int iterations = 8;
        for (int i = 0; i < iterations; ++i) {
            Rational re(rand() % 200 - 100, 100);
            Rational im(rand() % 200 - 100, 100);
            GaussQi z(re, im);
            GaussQi sin_z = delta::sin(z, eps_low);
            GaussQi asin_sin_z = delta::asin(sin_z, eps_low);
            EXPECT_RATIONAL_NEAR(asin_sin_z.real(), z.real(), eps_low * 100);
            EXPECT_RATIONAL_NEAR(asin_sin_z.imag(), z.imag(), eps_low * 100);
        }
    }

    TEST_F(TranscendentalExtendedTest, GaussQiCosAcos) {
        std::mt19937 rng(12345);
        const int iterations = 8;
        for (int i = 0; i < iterations; ++i) {
            Rational re(rand() % 200 - 100, 100);
            Rational im(rand() % 200 - 100, 100);
            GaussQi z(re, im);
            GaussQi cos_z = delta::cos(z, eps_low);
            GaussQi acos_cos_z = delta::acos(cos_z, eps_low);
            // Главная ветвь acos возвращает значение с Re ∈ [0, π]
            Rational expected_real = delta::abs(z.real()) <= delta::pi(eps_low) ? z.real() : delta::pi(eps_low) - z.real();
            EXPECT_RATIONAL_NEAR(acos_cos_z.real(), expected_real, eps_low * 100);
            EXPECT_RATIONAL_NEAR(acos_cos_z.imag(), z.imag(), eps_low * 100);
        }
    }

    TEST_F(TranscendentalExtendedTest, GaussQiAtanTan) {
        std::mt19937 rng(12345);
        const int iterations = 8;
        for (int i = 0; i < iterations; ++i) {
            Rational re(rand() % 200 - 100, 100);
            Rational im(rand() % 200 - 100, 100);
            GaussQi z(re, im);
            // Пропускаем точки, где tan не определён (cos ~ 0) – для простоты не проверяем
            GaussQi tan_z = delta::tan(z, eps_low);
            GaussQi atan_tan_z = delta::atan(tan_z, eps_low);
            EXPECT_RATIONAL_NEAR(atan_tan_z.real(), z.real(), eps_low * 100);
            EXPECT_RATIONAL_NEAR(atan_tan_z.imag(), z.imag(), eps_low * 100);
        }
    }
    TEST_F(TranscendentalExtendedTest, GaussQiAtanSingularity) {
        // atan(±i) должно бросать исключение
        GaussQi i(0, 1);
        GaussQi minus_i(0, -1);
        EXPECT_THROW(delta::atan(i, eps_std), std::domain_error);
        EXPECT_THROW(delta::atan(minus_i, eps_std), std::domain_error);
    }

    // ============================================================================
    // 3. atan2 ДЛЯ Rational
    // ============================================================================

    TEST_F(TranscendentalExtendedTest, Atan2Table) {
        Rational pi_val = delta::pi(eps_std);
        // (y,x) -> expected (в радианах)
        struct TestCase { Rational y, x; Rational expected_rad; };
        std::vector<TestCase> cases = {
            {0_r, 1_r, 0_r},
            {0_r, -1_r, pi_val},
            {1_r, 0_r, pi_val / 2_r},
            {-1_r, 0_r, -pi_val / 2_r},
            {1_r, 1_r, pi_val / 4_r},
            {1_r, -1_r, 3_r * pi_val / 4_r},
            {-1_r, -1_r, -3_r * pi_val / 4_r},
            {-1_r, 1_r, -pi_val / 4_r},
        };
        for (const auto& tc : cases) {
            Rational res = atan2(tc.y, tc.x, eps_std);
            EXPECT_RATIONAL_NEAR(res, tc.expected_rad, eps_std * 10);
        }
    }

    TEST_F(TranscendentalExtendedTest, Atan2Quadrants) {
        std::vector<std::pair<Rational, Rational>> points = {
            {1,1}, {1,2}, {2,1}, {-1,1}, {-2,1}, {-1,-1}, {-2,-3}, {3,-2}
        };
        for (const auto& p : points) {
            Rational y = p.first, x = p.second;
            Rational res = atan2(y, x, eps_std);
            // Сравнение с arg(GaussQi(x,y))
            GaussQi z(x, y);
            Rational arg_z = delta::arg(z, eps_std);
            EXPECT_RATIONAL_NEAR(res, arg_z, eps_std * 10);
        }
    }

    TEST_F(TranscendentalExtendedTest, Atan2Identities) {
        for (int i = -5; i <= 5; ++i) {
            for (int j = -5; j <= 5; ++j) {
                if (i == 0 && j == 0) continue;
                Rational y = Rational(i), x = Rational(j);
                if (x > 0) {
                    Rational res = atan2(y, x, eps_std);
                    Rational via_atan = delta::atan(y / x, eps_std);
                    EXPECT_RATIONAL_NEAR(res, via_atan, eps_std * 10);
                }
                else if (x < 0 && y >= 0) {
                    Rational res = atan2(y, x, eps_std);
                    Rational expected = delta::pi(eps_std) + delta::atan(y / x, eps_std);
                    EXPECT_RATIONAL_NEAR(res, expected, eps_std * 10);
                }
                else if (x < 0 && y < 0) {
                    Rational res = atan2(y, x, eps_std);
                    Rational expected = -delta::pi(eps_std) + delta::atan(y / x, eps_std);
                    EXPECT_RATIONAL_NEAR(res, expected, eps_std * 10);
                }
            }
        }
    }

    TEST_F(TranscendentalExtendedTest, Atan2ZeroZero) {
        EXPECT_THROW(atan2(0_r, 0_r, eps_std), std::domain_error);
    }

    TEST_F(TranscendentalExtendedTest, Atan2HighPrecision) {
        Rational y = Rational("1/10000000000000000000000000000000000000000"); // 1e-40
        Rational x = 1_r;
        Rational res = atan2(y, x, eps_ultra);
        // Для малых y, atan2(y,1) ≈ y
        Rational diff = delta::abs(res - y);
        EXPECT_LT(diff, eps_ultra * 10);
    }

    // ============================================================================
    // 4. floor ДЛЯ Rational
    // ============================================================================

    TEST_F(TranscendentalExtendedTest, FloorTable) {
        EXPECT_EQ(floor(0_r), 0_r);
        EXPECT_EQ(floor(1_r), 1_r);
        EXPECT_EQ(floor(-1_r), -1_r);
        EXPECT_EQ(floor(Rational(3, 2)), 1_r);
        EXPECT_EQ(floor(Rational(-3, 2)), -2_r);
        EXPECT_EQ(floor(Rational(5, 3)), 1_r);
        EXPECT_EQ(floor(Rational(-5, 3)), -2_r);
        // Большое число
        Rational big("1000000000000000000000000000000/3");
        Rational expected_big("333333333333333333333333333333");
        EXPECT_EQ(floor(big), expected_big);
    }

    TEST_F(TranscendentalExtendedTest, FloorProperties) {
        for (int num = -20; num <= 20; ++num) {
            for (int den = 1; den <= 10; ++den) {
                Rational x(num, den);
                Rational fl = floor(x);
                // целое
                EXPECT_EQ(fl.denominator(), 1_r);
                // fl <= x < fl+1
                EXPECT_LE(fl, x);
                EXPECT_LT(x, fl + 1_r);
                // floor(x + n) = floor(x) + n
                for (int n = -3; n <= 3; ++n) {
                    Rational xn = x + Rational(n);
                    EXPECT_EQ(floor(xn), fl + Rational(n));
                }
            }
        }
    }

    TEST_F(TranscendentalExtendedTest, FloorNegativeSymmetry) {
        for (int num = -20; num <= 20; ++num) {
            for (int den = 1; den <= 10; ++den) {
                Rational x(num, den);
                Rational fl = floor(x);
                Rational fl_neg = floor(-x);
                if (x.denominator() == 1_r) {   // целое?
                    EXPECT_EQ(fl + fl_neg, 0_r);
                }
                else {
                    EXPECT_EQ(fl + fl_neg, -1_r);
                }
            }
        }
    }
    // ============================================================================
    // 5. pow С ОТРИЦАТЕЛЬНЫМ ОСНОВАНИЕМ И РАЦИОНАЛЬНЫМ ПОКАЗАТЕЛЕМ
    // ============================================================================

    TEST_F(TranscendentalExtendedTest, PowNegativeBaseExact) {
        // (-8)^(1/3) = -2
        EXPECT_EQ(delta::pow(-8_r, Rational(1, 3), eps_std), -2_r);
        // (-27)^(2/3) = 9
        EXPECT_EQ(delta::pow(-27_r, Rational(2, 3), eps_std), 9_r);
        // (-8)^(2/3) = 4
        EXPECT_EQ(delta::pow(-8_r, Rational(2, 3), eps_std), 4_r);
        // (-1)^(2/3) = 1
        Rational res = delta::pow(-1_r, Rational(2, 3), eps_std);
        EXPECT_RATIONAL_NEAR(res, 1_r, eps_std * 10);
    }

    TEST_F(TranscendentalExtendedTest, PowNegativeBaseOddDenominator) {
        std::mt19937 rng(12345);
        for (int i = 0; i < 20; ++i) {
            Rational a = random_rational(1, 10, 1, 5, rng);
            int p = rand() % 5 + 1;   // 1..5
            int q = 2 * (rand() % 4) + 1; // нечётное 1,3,5,7
            Rational base = -a;
            Rational exp(p, q);
            Rational res = delta::pow(base, exp, eps_std);
            // Ожидаем (-1)^p * (a^(p/q))
            Rational a_pow_q = delta::pow(a, Rational(p, q), eps_std);
            Rational expected = (p % 2 == 0) ? a_pow_q : -a_pow_q;
            EXPECT_RATIONAL_NEAR(res, expected, eps_std * 100);
        }
    }

    TEST_F(TranscendentalExtendedTest, PowNegativeBaseEvenDenominator) {
        // Чётный знаменатель → исключение
        EXPECT_THROW(delta::pow(-2_r, Rational(1, 2), eps_std), std::domain_error);
        EXPECT_THROW(delta::pow(-3_r, Rational(3, 4), eps_std), std::domain_error);
        EXPECT_THROW(delta::pow(-1_r, Rational(1, 2), eps_std), std::domain_error);
    }

    TEST_F(TranscendentalExtendedTest, PowNegativeBaseZeroExponent) {
        EXPECT_EQ(delta::pow(-5_r, 0_r, eps_std), 1_r);
    }

    TEST_F(TranscendentalExtendedTest, PowZeroBase) {
        EXPECT_THROW(delta::pow(0_r, 0_r, eps_std), std::domain_error);
        EXPECT_THROW(delta::pow(0_r, -1_r, eps_std), std::domain_error);
        EXPECT_EQ(delta::pow(0_r, 2_r, eps_std), 0_r);
    }

    // ============================================================================
    // 6. sin / cos С ОЧЕНЬ БОЛЬШИМИ АРГУМЕНТАМИ
    // ============================================================================

    TEST_F(TranscendentalExtendedTest, SinLargeArgPeriodicity) {
        // x = 10^100, вычислим sin(x) и сравним с sin(x_red)
        std::string x_str = "1" + std::string(100, '0'); // 10^100
        Rational x(x_str);
        Rational two_pi = 2_r * delta::pi(eps_std);
        Rational k = floor(x / two_pi);
        Rational x_red = x - k * two_pi;

        Rational sin_big = delta::sin(x, eps_std);
        Rational sin_red = delta::sin(x_red, eps_std);
        EXPECT_RATIONAL_NEAR(sin_big, sin_red, eps_std * 1000);
    }

    TEST_F(TranscendentalExtendedTest, CosLargeArgPeriodicity) {
        std::string x_str = "1" + std::string(100, '0');
        Rational x(x_str);
        Rational two_pi = 2_r * delta::pi(eps_std);
        Rational k = floor(x / two_pi);
        Rational x_red = x - k * two_pi;

        Rational cos_big = delta::cos(x, eps_std);
        Rational cos_red = delta::cos(x_red, eps_std);
        EXPECT_RATIONAL_NEAR(cos_big, cos_red, eps_std * 1000);
    }

    TEST_F(TranscendentalExtendedTest, SinLargeMultipleOfPi) {
        // x = 10^50 * π
        Rational pi_val = delta::pi(eps_std);
        std::string factor_str = "1" + std::string(50, '0');
        Rational factor(factor_str);
        Rational x = factor * pi_val;
        Rational sin_x = delta::sin(x, eps_std);
        // sin(kπ) = 0, допустимая ошибка
        EXPECT_RATIONAL_NEAR(sin_x, 0_r, eps_std * 1000);
        // cos(kπ) = ±1
        Rational cos_x = delta::cos(x, eps_std);
        Rational expected_cos = (factor.numerator_raw() % 2 == 0) ? 1_r : -1_r;
        EXPECT_RATIONAL_NEAR(cos_x, expected_cos, eps_std * 1000);
    }

    TEST_F(TranscendentalExtendedTest, SinLargeHalfPi) {
        // x = 10^50 * π/2
        Rational pi_val = delta::pi(eps_std);
        std::string factor_str = "1" + std::string(50, '0');
        Rational factor(factor_str);
        Rational x = factor * pi_val / 2_r;
        Rational sin_x = delta::sin(x, eps_std);
        Rational cos_x = delta::cos(x, eps_std);
        // sin((2k+1)π/2) = (-1)^k
        int k = factor.numerator_raw().convert_to<int>();
        Rational expected_sin = (k % 2 == 0) ? 1_r : -1_r;
        EXPECT_RATIONAL_NEAR(sin_x, expected_sin, eps_std * 1000);
        // cos((2k+1)π/2) = 0
        EXPECT_RATIONAL_NEAR(cos_x, 0_r, eps_std * 1000);
    }

    TEST_F(TranscendentalExtendedTest, SinExtremePrecisionLargeArg) {
        // x = 10^100, eps = 1e-80
        std::string x_str = "1" + std::string(100, '0');
        Rational x(x_str);
        Rational pi_high = high_precision_pi();
        Rational two_pi_high = 2_r * pi_high;
        Rational k = floor(x / two_pi_high);
        Rational x_red_high = x - k * two_pi_high;
        Rational sin_high = delta::sin(x_red_high, eps_ultra);

        Rational sin_direct = delta::sin(x, eps_ultra);
        EXPECT_RATIONAL_NEAR(sin_direct, sin_high, eps_ultra * 10000);
    }

    TEST_F(TranscendentalExtendedTest, CosExtremePrecisionLargeArg) {
        std::string x_str = "1" + std::string(100, '0');
        Rational x(x_str);
        Rational pi_high = high_precision_pi();
        Rational two_pi_high = 2_r * pi_high;
        Rational k = floor(x / two_pi_high);
        Rational x_red_high = x - k * two_pi_high;
        Rational cos_high = delta::cos(x_red_high, eps_ultra);

        Rational cos_direct = delta::cos(x, eps_ultra);
        EXPECT_RATIONAL_NEAR(cos_direct, cos_high, eps_ultra * 10000);
    }

    TEST_F(TranscendentalExtendedTest, SinHugeArgNoInfiniteLoop) {
        // Проверяем, что sin(10^1000) не зацикливается и выполняется быстро
        std::string huge = "1" + std::string(1000, '0');
        Rational x(huge);
        auto start = std::chrono::steady_clock::now();
        Rational res = delta::sin(x, eps_std);
        auto end = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        EXPECT_LT(elapsed_ms, 5000) << "sin(10^1000) took too long, possible loop in argument reduction";
        EXPECT_TRUE(res >= -1_r && res <= 1_r);
    }

} // namespace delta::testing