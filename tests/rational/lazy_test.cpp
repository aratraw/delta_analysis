// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/rational/lazy_test.cpp
// Дополнительные тесты для LazyRational (мутации, COW, сложные сценарии)
// Адаптированы под новую архитектуру (гетерогенное хранение SUM/PRODUCT)
#include "lazy_rational_test_fixture.h"
#include "delta/core/rational.h"
#include "test_utils.h"
#include <random>
#include <sstream>

namespace delta::testing {

    class LazyRationalExtraTest : public LazyRationalTestFixture {};

    // -------------------------------------------------------------------------
    // Базовые тесты на мутации и COW
    // -------------------------------------------------------------------------
    TEST_F(LazyRationalExtraTest, SumCowOnMultipleReferences) {
        LazyRational x = LazyRational(Rational(1, 2));
        LazyRational y = LazyRational(Rational(1, 3));
        LazyRational sum = x.clone();
        sum += y;                     // sum = 1/2 + 1/3 = 5/6
        LazyRational copy = sum.clone();
        sum += Rational(1);           // sum = 5/6 + 1 = 11/6
        EXPECT_EQ(copy.eval(), Rational(5, 6));
        EXPECT_EQ(sum.eval(), Rational(11, 6));
    }

    TEST_F(LazyRationalExtraTest, PlusEqualOnImmediate) {
        LazyRational a = LazyRational(1_r); // грязный CONST(1)
        LazyRational b = LazyRational(Rational(1, 2));
        a += b;
        EXPECT_TRUE(is_dirty(a));
        EXPECT_EQ(dirty_root_op(a), internal::LazyOp::SUM);
        EXPECT_EQ(total_operands(a), 2);
        EXPECT_EQ(a.eval(), Rational(3, 2));
    }

    TEST_F(LazyRationalExtraTest, SumOfTwoSums) {
        LazyRational a = LazyRational(Rational(1, 2));
        LazyRational b = LazyRational(Rational(1, 3));
        LazyRational c = LazyRational(Rational(1, 6));
        LazyRational d = LazyRational(Rational(1, 4));

        LazyRational sum1 = a.clone();
        sum1 += b;
        LazyRational sum2 = c.clone();
        sum2 += d;
        LazyRational total = sum1.clone();
        total += sum2;   // теперь total = sum1 + sum2 (плоский SUM)
        EXPECT_EQ(dirty_root_op(total), internal::LazyOp::SUM);
        EXPECT_EQ(total_operands(total), 4);
    }

    TEST_F(LazyRationalExtraTest, ChainedMultiplication) {
        LazyRational a = LazyRational(Rational(2));
        a* Rational(3)* Rational(4);   // мутируем a, без присваивания
        EXPECT_EQ(dirty_root_op(a), internal::LazyOp::PRODUCT);
        EXPECT_EQ(total_operands(a), 3);
        EXPECT_EQ(a.eval(), Rational(24));
    }

    TEST_F(LazyRationalExtraTest, MixedOperations) {
        LazyRational a = LazyRational(Rational(2));
        a * 3_r + 5_r;   // мутируем a: сначала умножение, затем сложение
        EXPECT_EQ(dirty_root_op(a), internal::LazyOp::SUM);
        EXPECT_EQ(total_operands(a), 2);
        int root = dirty_root_index(a);
        EXPECT_EQ(dirty_node_complex_count(a, root), 1);
        EXPECT_EQ(dirty_node_leaf_count(a, root), 1);
        int prod_node = dirty_node_complex_child(a, root, 0);
        EXPECT_EQ(dirty_node_op(a, prod_node), internal::LazyOp::PRODUCT);
        EXPECT_EQ(Rational(dirty_node_leaf_value(a, root, 0)), 5_r);
        EXPECT_EQ(a.eval(), Rational(11));
    }

    // -------------------------------------------------------------------------
    // Вспомогательные функции для генерации данных и суммирования
    // -------------------------------------------------------------------------
    static std::vector<Rational> generate_powers_of_two_terms(int N, int seed = 12345, int max_exp = 20) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> num_dist(-1000, 1000);
        std::uniform_int_distribution<int> exp_dist(0, max_exp);

        std::vector<Rational> terms;
        terms.reserve(N);
        for (int i = 0; i < N; ++i) {
            int num = num_dist(rng);
            int den = 1 << exp_dist(rng);
            terms.emplace_back(num, den);
        }
        return terms;
    }

    static Rational eager_sum(const std::vector<Rational>& terms) {
        Rational s = 0_r;
        for (const auto& t : terms) s += t;
        return s;
    }

    static Rational lazy_sum(const std::vector<Rational>& terms, bool skip_simplify = true) {
        internal::reset_pool();
        LazyRational s;
        for (const auto& t : terms) s += t;
        s.eval_inplace(skip_simplify);
        return s.eval();
    }

    static internal::Value manual_pyramidal_reduce(const std::vector<internal::Value>& input,
        bool verbose = false) {
        std::vector<internal::Value> vals = input;
        constexpr size_t BATCH_SIZE = 32;
        int level = 0;

        while (vals.size() > 1) {
            if (verbose) {
                std::cout << "  Level " << level << ": " << vals.size() << " elements\n";
            }
            std::vector<internal::Value> next;
            for (size_t i = 0; i < vals.size(); i += BATCH_SIZE) {
                size_t end = std::min(i + BATCH_SIZE, vals.size());
                internal::Value sum = vals[i];
                for (size_t j = i + 1; j < end; ++j) {
                    sum += vals[j];
                }
                next.push_back(std::move(sum));
            }
            vals = std::move(next);
            ++level;
        }
        return vals.empty() ? internal::Value(0) : vals[0];
    }

    // -------------------------------------------------------------------------
    // Тесты корректности для небольших N
    // -------------------------------------------------------------------------
    TEST_F(LazyRationalExtraTest, SumMixedDenominators) {
        std::vector<Rational> terms = {
            Rational(1, 2), Rational(1, 3), Rational(1, 4),
            Rational(1, 5), Rational(1, 8), Rational(1, 6)
        };
        Rational eager = eager_sum(terms);
        Rational lazy = lazy_sum(terms);
        EXPECT_EQ(eager, lazy);
    }

    TEST_F(LazyRationalExtraTest, SumManyPowersOfTwo) {
        const int N = 1000;
        std::vector<Rational> terms = generate_powers_of_two_terms(N);
        Rational eager = eager_sum(terms);
        Rational lazy = lazy_sum(terms);
        EXPECT_EQ(eager, lazy);
    }

    TEST_F(LazyRationalExtraTest, SumManyPowersOfTwoNormalized) {
        const int N = 1000;
        std::vector<Rational> terms = generate_powers_of_two_terms(N);
        // Принудительно нормализуем каждый элемент
        for (auto& r : terms) {
            r = Rational(r.to_string());
        }
        Rational eager = eager_sum(terms);
        Rational lazy = lazy_sum(terms);
        EXPECT_EQ(eager, lazy);
    }

    TEST_F(LazyRationalExtraTest, RationalConstructorNormalizes) {
        Rational r(2, 4);
        EXPECT_EQ(r, Rational(1, 2));

        Rational r2(6, 8);
        EXPECT_EQ(r2, Rational(3, 4));
    }

    TEST_F(LazyRationalExtraTest, CloneAndMutatePreservesValues) {
        LazyRational a = LazyRational(Rational(1, 2));
        LazyRational b = a.clone();
        a += Rational(1, 4);
        EXPECT_EQ(b.eval(), Rational(1, 2));
        EXPECT_EQ(a.eval(), Rational(3, 4));
    }

    TEST_F(LazyRationalExtraTest, RepeatedAdditionOfPowerOfTwo) {
        const int N = 10000;
        Rational term(1, 8);
        std::vector<Rational> terms(N, term);
        Rational eager = eager_sum(terms);
        Rational lazy = lazy_sum(terms);
        EXPECT_EQ(eager, lazy);
        EXPECT_EQ(eager, Rational(N, 8));
    }

    TEST_F(LazyRationalExtraTest, MixedEagerLazyMutation) {
        LazyRational a = LazyRational(Rational(1, 3));
        Rational b(1, 6);
        a += b;                     // lazy + immediate
        EXPECT_EQ(a.eval(), Rational(1, 2));
        LazyRational c = a.clone();
        c += Rational(1, 4);        // 1/2 + 1/4 = 3/4
        EXPECT_EQ(c.eval(), Rational(3, 4));
        EXPECT_EQ(a.eval(), Rational(1, 2));
    }

    // -------------------------------------------------------------------------
    // Тесты для больших N с диагностикой
    // -------------------------------------------------------------------------
    TEST_F(LazyRationalExtraTest, ImmediateVsBoostLargeScale) {
        using BoostRational = boost::multiprecision::number<
            boost::multiprecision::rational_adaptor<
            boost::multiprecision::cpp_int_backend<>
            >,
            boost::multiprecision::et_off
        >;

        const std::vector<int> sizes = { 10000, 20000, 30000, 40000, 50000 };
        for (int N : sizes) {
            std::vector<Rational> terms = generate_powers_of_two_terms(N);

            // Immediate сумма Delta
            Rational delta_sum = 0_r;
            for (const auto& t : terms) delta_sum += t;

            // Boost сумма
            BoostRational boost_sum = 0;
            for (const auto& t : terms) {
                internal::dumb_int num = t.numerator().convert_to<internal::dumb_int>();
                internal::dumb_int den = t.denominator().convert_to<internal::dumb_int>();
                boost_sum += BoostRational(num, den);
            }

            // Сравнение через строки
            std::ostringstream oss_delta, oss_boost;
            oss_delta << delta_sum;
            oss_boost << boost_sum;
            EXPECT_EQ(oss_delta.str(), oss_boost.str()) << "Immediate vs Boost mismatch at N=" << N;
        }
    }

    TEST_F(LazyRationalExtraTest, SumManyPowersOfTwoLargeScale) {
        const std::vector<int> sizes = { 10000, 20000, 30000, 40000, 50000 };

        for (int N : sizes) {
            std::vector<Rational> terms = generate_powers_of_two_terms(N);
            Rational eager = eager_sum(terms);

            // Строим ленивую сумму
            internal::reset_pool();
            LazyRational lr;
            for (const auto& t : terms) lr += t;

            int root = dirty_root_index(lr);
            size_t leaf_cnt = dirty_node_leaf_count(lr, root);
            size_t complex_cnt = dirty_node_complex_count(lr, root);

            // Проверка суммы leaf_values вручную
            std::vector<internal::Value> leaf_copy;
            leaf_copy.reserve(leaf_cnt);
            for (size_t i = 0; i < leaf_cnt; ++i) {
                leaf_copy.push_back(dirty_node_leaf_value(lr, root, i));
            }
            internal::Value manual_leaf_sum = manual_pyramidal_reduce(leaf_copy);
            Rational manual_from_leaf(manual_leaf_sum);

            bool leaf_corrupted = (manual_from_leaf != eager);
            if (leaf_corrupted) {
                std::cerr << "\n=== CORRUPTION DETECTED IN LEAF_VALUES ===\n";
                std::cerr << "N = " << N << "\n";
                std::cerr << "leaf_values count = " << leaf_cnt
                    << ", children count = " << complex_cnt << "\n";
                std::cerr << "Eager sum:           " << eager << "\n";
                std::cerr << "Manual leaf sum:     " << manual_from_leaf << "\n";
                std::cerr << "Difference:          " << eager - manual_from_leaf << "\n";
            }

            // Вычисляем ленивую сумму
            Rational lazy = lazy_sum(terms, /*skip_simplify=*/true);

            if (eager != lazy) {
                std::cerr << "\n=== MISMATCH DETECTED ===\n";
                std::cerr << "N = " << N << "\n";
                std::cerr << "leaf_values count = " << leaf_cnt
                    << ", children count = " << complex_cnt << "\n";
                std::cerr << "Eager:              " << eager << "\n";
                std::cerr << "Lazy:               " << lazy << "\n";
                std::cerr << "Manual leaf sum:    " << manual_from_leaf << "\n";
                std::cerr << "Difference:         " << eager - lazy << "\n";

                // Ручное PCR для диагностики
                std::vector<internal::Value> raw_vals;
                raw_vals.reserve(terms.size());
                for (const auto& t : terms) raw_vals.push_back(t.value());

                internal::Value manual_val = manual_pyramidal_reduce(raw_vals);
                Rational manual(manual_val);
                std::cerr << "Manual PCR on raw terms: " << manual << "\n";

                if (eager != manual) {
                    std::cerr << "Eager != Manual PCR! Difference: "
                        << eager - manual << "\n";
                }

                FAIL() << "Sums differ at N=" << N;
            }
            else if (leaf_corrupted) {
                // leaf_values повреждены, но lazy каким-то образом дал правильный результат
                FAIL() << "Leaf values corrupted but lazy sum correct at N=" << N;
            }
            // Если всё хорошо — ничего не выводим
        }
    }
} // namespace delta::testing