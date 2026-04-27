// reset_pool_edge_cases_tests.h
#pragma once

#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "delta/rational/node_pool.h"
#include "test_utils.h"
#include "lazy_rational_test_fixture.h"

namespace delta::testing {

    // Фикстура с принудительным reset_pool перед каждым тестом.
    class ResetPoolEdgeCasesTest : public LazyRationalTestFixture {
    protected:
        void SetUp() override {
            internal::reset_pool();
        }
        void TearDown() override {
            internal::reset_pool();
        }
    };

    // -----------------------------------------------------------------------
    // 1. Простейшее трансцендентное выражение после reset_pool
    // -----------------------------------------------------------------------
    TEST_F(ResetPoolEdgeCasesTest, SimpleTranscendentalAfterReset) {
        for (int cycle = 0; cycle < 5; ++cycle) {
            internal::reset_pool();
            LazyRational x = LazyRational("0.5"_r);
            LazyRational expr = Sin(x.clone()) * Cos(x.clone());
            expr.simplify_inplace();
            ASSERT_TRUE(is_clean(expr));
            Rational val = expr.eval();
            Rational expected = sin("0.5"_r) * cos("0.5"_r);
            EXPECT_EQ(val, expected) << "Cycle " << cycle;
        }
    }

    // -----------------------------------------------------------------------
    // 2. Прямой минимальный повтор проблемного RepeatingTerm_Simplify_10
    // -----------------------------------------------------------------------
    TEST_F(ResetPoolEdgeCasesTest, RepeatingTerm10AfterReset) {
        for (int attempt = 0; attempt < 3; ++attempt) {
            internal::reset_pool();
            LazyRational term_val = Sin("0.5"_r) * Cos("0.5"_r);
            LazyRational acc;
            for (int i = 0; i < 10; ++i) acc + term_val;
            acc.simplify_inplace();   // ← если зависнет, то здесь
            ASSERT_TRUE(is_clean(acc));
            Rational val = acc.eval();
            Rational expected = (sin("0.5"_r) * cos("0.5"_r)) * 10;
            EXPECT_EQ(val, expected) << "Attempt " << attempt;
        }
    }

    // -----------------------------------------------------------------------
    // 3. Множественные сбросы с разными выражениями
    // -----------------------------------------------------------------------
    TEST_F(ResetPoolEdgeCasesTest, MultipleResetPoolCycles) {
        for (int i = 0; i < 10; ++i) {
            internal::reset_pool();
            LazyRational a = Sin(LazyRational("0.2"_r));
            LazyRational b = Cos(LazyRational("0.3"_r));
            LazyRational c = a.clone() * b.clone() + a.clone();
            c.simplify_inplace();
            EXPECT_TRUE(is_clean(c));
            Rational r = c.eval();
            EXPECT_TRUE(r > 0_r);
        }
    }

    // -----------------------------------------------------------------------
    // 4. Проверка целостности кэша π после сброса пула
    // -----------------------------------------------------------------------
    TEST_F(ResetPoolEdgeCasesTest, PiCacheIntegrity) {
        // Явно задаём eps, чтобы π вычислилось
        set_default_eps(Rational("1/1000000000000000000000000000000"));

        Rational pi_before = pi(default_eps());

        internal::reset_pool();
        internal::reset_pi_cache(); // очищаем кэш явно

        Rational pi_after = pi(default_eps());
        EXPECT_EQ(pi_before, pi_after) << "Pi must be recomputed correctly after pool reset";
    }

    // -----------------------------------------------------------------------
    // 5. Убедиться, что default_eps() не портится при сбросе пула
    // -----------------------------------------------------------------------
    TEST_F(ResetPoolEdgeCasesTest, DefaultEpsAfterReset) {
        Rational eps_before = default_eps();
        internal::reset_pool();
        Rational eps_after = default_eps();
        EXPECT_EQ(eps_before, eps_after) << "Default epsilon must survive pool reset";
    }

    // -----------------------------------------------------------------------
    // 6. Взаимодействие GC и reset_pool с упрощением
    // -----------------------------------------------------------------------
    TEST_F(ResetPoolEdgeCasesTest, GCAndResetInteraction) {
        //на самом деле команда reset_pool здесь избыточна, потому что он вызывается в SetUp и TearDown, но по ожиданию это не должно оказывать влияния на корректность выполнения.
        internal::reset_pool();//ожидаемый результат: девственно чистый пул, все кэши пусты, никаких висячих ссылок на объекты от предыдущих тестов
        internal::set_pool_max_size(120);
        LazyRational acc;
        for (int i = 0; i < 80; ++i) {
            LazyRational term = Sin(Rational(i).as_lazy()) * Cos(Rational(i + 1).as_lazy());//термы ленивые, НЕ ВЫЧИСЛЯЮТСЯ.
            acc + term;
        }
        //ожидаемый результат: acc лениво накопил термы как и должен, а term уничтожился потому что вышел из области видимости.
        acc.simplify_inplace();//ожидаемый результат: объект канонизировался и стал чистым деревом в пуле.
        internal::force_garbage_collect();//ожидаемый результат: В ТЕКУЩЕМ ПУЛЕ лежит канонизированное упрощённое дерево acc. Мы ВЫСЧИТЫВАЕМ ЕГО (канонизация не должна вызываться потому что State уже Clean). ДЕЛАЕМ НОВЫЙ ПУЛ, В НОВОМ ПУЛЕ остался ОДИН УЗЕЛ типа CONST содержащий
        //результат ленивого выражения, корнем которого был acc ранее. Переменная acc хранит индекс этого единственного узла в новом пуле. Старый пул корректно уничтожен, владения памятью освобождены.
        EXPECT_TRUE(is_clean(acc));

        internal::reset_pool();//ожидаемый результат: девственно чистый пул, пустые кэши итд. Что делать с переменной acc?
        //А вот что: В текущей области видимости есть старая переменная LazyRational acc которая к этому моменту должна была иметь один узел CONST в пуле после сборки мусора.
        //Ожидаемое поведение: освободить всю память связанную с Acc, очистить пул, очистить кэши, 
        // освободить всё владение памятью в пуле, связанное с acc, сделать acc state Dirty и инициализировать ленивым узлом 0 - который хранится в самом объекте (по умолчанию).
        //ИТОГО: Всё владение памятью должно быть очищено, переменная acc должна быть в корректном состоянии ленивого грязного нуля хранящегося локально в переменной.
        
        //По этой логике, прямо в этом месте нужно поставить проверку на то что acc равняется грязному ленивому объекту с одним узлом 0.
        internal::set_pool_max_size(200);
        LazyRational expr = Sin("0.7"_r) + Cos("0.8"_r);
        expr.simplify_inplace();//Ожидание: Канонизируется, заносится в пул.
        Rational val = expr.eval();
        Rational expected = sin("0.7"_r) + cos("0.8"_r);
        EXPECT_EQ(val, expected);
    }

    // -----------------------------------------------------------------------
    // 7. Интернирование после нескольких сбросов
    // -----------------------------------------------------------------------
    TEST_F(ResetPoolEdgeCasesTest, InterningAfterMultipleResets) {
        int idx1 = -1, idx2 = -1;
        {
            internal::reset_pool();
            LazyRational a = LazyRational(3_r);
            a + 3_r + 3_r;
            a.simplify_inplace();
            idx1 = clean_index(a);
        }
        {
            internal::reset_pool();
            LazyRational b = LazyRational(3_r);
            b + 3_r + 3_r;
            b.simplify_inplace();
            idx2 = clean_index(b);
        }
        EXPECT_EQ(idx1, idx2) << "Interning should produce identical indices after separate resets";
    }

    // -----------------------------------------------------------------------
    // 8. Стресс-тест (отключён, запускать вручную при необходимости)
    // -----------------------------------------------------------------------
    TEST_F(ResetPoolEdgeCasesTest, StressLargeAfterReset) {
        internal::reset_pool();
        LazyRational term = Sin("0.123"_r) * Cos("0.456"_r);
        LazyRational sum;
        for (int i = 0; i < 200; ++i) sum + term;
        sum.simplify_inplace();
        Rational expected = (sin("0.123"_r) * cos("0.456"_r)) * 200;
        EXPECT_EQ(sum.eval(), expected);
    }

    // -----------------------------------------------------------------------
    // 9. RepeatingTerm_Simplify_10 в цикле
    // -----------------------------------------------------------------------
    TEST_F(ResetPoolEdgeCasesTest, RepeatingTerm10ManyTimes) {
        for (int iteration = 0; iteration < 5; ++iteration) {
            internal::reset_pool();
            LazyRational term_val = Sin("0.5"_r) * Cos("0.5"_r);
            LazyRational acc;
            for (int j = 0; j < 10; ++j) acc + term_val;
            acc.simplify_inplace();
            ASSERT_TRUE(is_clean(acc));
            Rational val = acc.eval();
            Rational expected = (sin("0.5"_r) * cos("0.5"_r)) * 10;
            EXPECT_EQ(val, expected) << "Iteration " << iteration;
        }
    }

    // -----------------------------------------------------------------------
    // 10. Дистрибутивность с трансцендентными множителями после сброса
    // -----------------------------------------------------------------------
    TEST_F(ResetPoolEdgeCasesTest, DistributivityWithTranscendentalReset) {
        internal::reset_pool();
        LazyRational a = Sin("0.5"_r);
        LazyRational b = Cos("0.5"_r);
        LazyRational expr = (a.clone() * b.clone()) + (a.clone() * LazyRational(2_r));
        expr.simplify_inplace();
        EXPECT_TRUE(is_clean(expr));
        Rational val = expr.eval();
        Rational expected = sin("0.5"_r) * (cos("0.5"_r) + 2_r);
        EXPECT_EQ(val, expected);
    }

} // namespace delta::testing