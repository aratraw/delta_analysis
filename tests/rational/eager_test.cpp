// tests/rational/eager_test.cpp
#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "test_utils.h"

namespace delta::testing {

    class RationalEagerTest : public RationalTest {};

    // Вспомогательная функция для получения double из Rational
    inline double to_double(const Rational& r) {
        return internal::to_double(r.to_value());
    }

    // -------------------------------------------------------------------------
    // 1. Eager mode flag (default false)
    // -------------------------------------------------------------------------
    TEST_F(RationalEagerTest, EagerModeFlag) {
        EXPECT_FALSE(eager_mode());            // default
        set_eager_mode(true);
        EXPECT_TRUE(eager_mode());
        set_eager_mode(false);
        EXPECT_FALSE(eager_mode());
    }

    // -------------------------------------------------------------------------
    // 2. Eager evaluation – operations produce concrete numbers, not lazy nodes
    // -------------------------------------------------------------------------
    TEST_F(RationalEagerTest, EagerEvaluation) {
        set_eager_mode(true);
        Rational sum = "1/2"_r + "1/3"_r;
        EXPECT_FALSE(sum.is_lazy());
        EXPECT_EQ(sum.eval(), "5/6"_r);

        Rational diff = "1/2"_r - "1/3"_r;
        EXPECT_FALSE(diff.is_lazy());
        EXPECT_EQ(diff.eval(), "1/6"_r);

        Rational prod = "2/3"_r * "3/4"_r;
        EXPECT_FALSE(prod.is_lazy());
        EXPECT_EQ(prod.eval(), "1/2"_r);

        Rational quot = "2/3"_r / "4/5"_r;
        EXPECT_FALSE(quot.is_lazy());
        EXPECT_EQ(quot.eval(), "5/6"_r);
    }

    // -------------------------------------------------------------------------
    // 3. ScopedEagerEval – complex flag behavior. read source code, whatever
    // -------------------------------------------------------------------------
    TEST_F(RationalEagerTest, ScopedEagerEval) {
        // Initially lazy mode
        set_eager_mode(false);

        // Блок 1: оба операнда immediate
        Rational sum1;
        {
            ScopedEagerEval guard;  // внутри блока eager mode
            sum1 = "1/2"_r + "1/3"_r;
            EXPECT_FALSE(sum1.is_lazy());   // eager -> immediate
            EXPECT_EQ(sum1.eval(), "5/6"_r);
        }
        // После блока, режим восстановлен (global_eager_mode == false)
        Rational sum2 = "1/2"_r + "1/3"_r;  // оба immediate -> immediate
        EXPECT_FALSE(sum2.is_lazy());       // ожидаем immediate
        EXPECT_EQ(sum2.eval(), "5/6"_r);

        // Блок 2: один операнд ленивый
        Rational lazy_one = "1"_r.lazy();
        Rational eager_sum;
        {
            ScopedEagerEval guard;  // внутри блока eager mode
            eager_sum = lazy_one + "2"_r;   // lazy + immediate
            EXPECT_FALSE(eager_sum.is_lazy()); // eager -> immediate
            EXPECT_EQ(eager_sum.eval(), 3_r);
        }
        // После блока режим восстановлен, lazy_one остаётся ленивым
        Rational lazy_sum = lazy_one + "2"_r;
        EXPECT_TRUE(lazy_sum.is_lazy());   // lazy + immediate -> lazy
        EXPECT_EQ(lazy_sum.eval(), 3_r);
    }

    // -------------------------------------------------------------------------
    // 4. Eager transcendentals – sqrt, exp, etc. return evaluated values
    // -------------------------------------------------------------------------
    TEST_F(RationalEagerTest, EagerTranscendentals) {
        set_eager_mode(true);

        // sqrt(4) = 2 exactly
        Rational s = delta::sqrt(4_r);
        EXPECT_FALSE(s.is_lazy());
        EXPECT_EQ(s.eval(), 2_r);

        // sqrt(2) approximate
        Rational s2 = delta::sqrt(2_r);
        EXPECT_FALSE(s2.is_lazy());
        Rational s2_val = s2.eval();
        Rational expected_sqrt2_rat = Rational("14142135623730950488/10000000000000000000");
        EXPECT_RATIONAL_NEAR(s2_val, expected_sqrt2_rat, "1/1000000000000"_r);

        // exp(0) = 1
        Rational e0 = delta::exp(0_r);
        EXPECT_FALSE(e0.is_lazy());
        EXPECT_EQ(e0.eval(), 1_r);

        // exp(1) approximate
        Rational e1 = delta::exp(1_r);
        EXPECT_FALSE(e1.is_lazy());
        Rational e1_val = e1.eval();
        Rational expected_e_rat = Rational("27182818284590452354/10000000000000000000");
        EXPECT_RATIONAL_NEAR(e1_val, expected_e_rat, "1/1000000000000"_r);

        // log(1) = 0
        Rational l1 = delta::log(1_r);
        EXPECT_FALSE(l1.is_lazy());
        EXPECT_EQ(l1.eval(), 0_r);

        // sin(0) = 0
        Rational sin0 = delta::sin(0_r);
        EXPECT_FALSE(sin0.is_lazy());
        EXPECT_EQ(sin0.eval(), 0_r);

        // cos(0) = 1
        Rational cos0 = delta::cos(0_r);
        EXPECT_FALSE(cos0.is_lazy());
        EXPECT_EQ(cos0.eval(), 1_r);

        // acos(1) = 0
        Rational acos1 = delta::acos(1_r);
        EXPECT_FALSE(acos1.is_lazy());
        EXPECT_EQ(acos1.eval(), 0_r);
    }

        TEST_F(RationalEagerTest, EagerTranscendentalsSlow) {
        set_eager_mode(true);

        // Очень маленький epsilon (1e-60) заставляет библиотеку использовать slow_* функции
        Rational eps = "1"_r / "1000000000000000000000000000000000000000000000000000000000000"_r;

        // sqrt(2) с высокой точностью
        Rational s2 = delta::sqrt(2_r, eps);
        EXPECT_FALSE(s2.is_lazy());
        Rational s2_val = s2.eval();

        // Ожидаемое значение sqrt(2) с 60 знаками после запятой (округлено)
        Rational expected_sqrt2 = "1414213562373095048801688724209698078569671875376948073176679"_r /
            "1000000000000000000000000000000000000000000000000000000000000"_r;
        EXPECT_RATIONAL_NEAR(s2_val, expected_sqrt2, eps * 10_r);

        // exp(1) с высокой точностью
        Rational e1 = delta::exp(1_r, eps);
        EXPECT_FALSE(e1.is_lazy());
        Rational e1_val = e1.eval();

        Rational expected_e = "2718281828459045235360287471352662497757247093699959574966967"_r /
            "1000000000000000000000000000000000000000000000000000000000000"_r;
        EXPECT_RATIONAL_NEAR(e1_val, expected_e, eps * 10_r);

        // sin(1) с высокой точностью – вычисляем эталон через boost::multiprecision
        using boost::multiprecision::cpp_dec_float_100;
        cpp_dec_float_100 one(1);
        cpp_dec_float_100 sin1_mp = sin(one);
        std::string sin1_str = sin1_mp.str(60, std::ios_base::fixed);
        // Убираем десятичную точку и разбиваем на целую и дробную части
        size_t dot = sin1_str.find('.');
        std::string int_part = sin1_str.substr(0, dot);
        std::string frac_part = sin1_str.substr(dot + 1);
        // Обрезаем до 60 знаков
        if (frac_part.size() > 60) frac_part = frac_part.substr(0, 60);
        // Формируем числитель: целая часть + дробная часть (без ведущих нулей)
        std::string num_str = int_part + frac_part;
        // Удаляем ведущие нули
        size_t first_nonzero = num_str.find_first_not_of('0');
        if (first_nonzero != std::string::npos && first_nonzero > 0) {
            num_str = num_str.substr(first_nonzero);
        }
        if (num_str.empty()) num_str = "0";
        // Знаменатель = 10^len(frac_part)
        std::string den_str = "1";
        for (size_t i = 0; i < frac_part.size(); ++i) den_str += "0";
        Rational sin1_expected = Rational(num_str) / Rational(den_str);
        Rational sin1_val = delta::sin(1_r, eps).eval();

        EXPECT_RATIONAL_NEAR(sin1_val, sin1_expected, eps * 100_r);
    }
} // namespace delta::testing