// tests/rational/batch_test.cpp
#include <gtest/gtest.h>
#include <vector>
#include "delta/core/rational.h"
#include "test_utils.h"

namespace delta::testing {

    class RationalBatchTest : public RationalTest {};

    // -------------------------------------------------------------------------
    // Batch addition tests
    // -------------------------------------------------------------------------
    TEST_F(RationalBatchTest, BatchAddSimple) {
        std::vector<Rational> terms = { "1/2"_r, "1/3"_r, "1/6"_r };
        Rational sum = delta::batch_add(terms);
        EXPECT_EQ(sum.eval(), 1_r);
    }

    TEST_F(RationalBatchTest, BatchAddLarge) {
        std::vector<Rational> terms(100, "1/1000"_r);
        Rational sum = delta::batch_add(terms);
        EXPECT_EQ(sum.eval(), "1/10"_r);
    }

    TEST_F(RationalBatchTest, BatchAddMixed) {
        std::string denom = "1000000000000000000000000000000";
        Rational term1 = "1/2"_r;
        Rational term2 = Rational("1/" + denom);
        std::vector<Rational> terms = { term1, term2 };
        Rational sum = delta::batch_add(terms);
        Rational expected = term1 + term2;
        EXPECT_EQ(sum.eval(), expected);
    }

    TEST_F(RationalBatchTest, BatchAddOverflow) {
        std::string denom1 = "1000000000000000000000000000000";
        std::string denom2 = "1000000000000000000000000000001";
        std::vector<Rational> terms = {
            Rational("1/" + denom1),
            Rational("1/" + denom2)
        };
        Rational sum = delta::batch_add(terms);
        Rational bound = Rational(2) / Rational(denom1);
        Rational diff = bound - sum.eval();
        EXPECT_GT(diff, 0_r);
        EXPECT_GT(sum.eval(), 0_r);
    }

    TEST_F(RationalBatchTest, BatchAddLazy) {
        set_eager_mode(false);
        auto a = delta::sqrt(2_r);
        auto b = delta::exp(1_r);
        std::vector<Rational> terms = { a, b };
        Rational sum = delta::batch_add(terms);
        Rational a_val = a.eval();
        Rational b_val = b.eval();
        Rational sum_individual = a_val + b_val;
        Rational sum_batch = sum.eval();
        EXPECT_EQ(sum_batch, sum_individual);
    }

    TEST_F(RationalBatchTest, BatchAddEmpty) {
        std::vector<Rational> terms;
        Rational sum = delta::batch_add(terms);
        EXPECT_EQ(sum.eval(), 0_r);
    }

    // -------------------------------------------------------------------------
    // Harmonic series: sum_{n=1}^{1000} 1/n
    // -------------------------------------------------------------------------
    TEST_F(RationalBatchTest, HarmonicSeries) {
        std::vector<Rational> terms;
        for (int i = 1; i <= 1000; ++i) {
            terms.push_back(Rational(1, i));
        }
        Rational sum_batch = delta::batch_add(terms);
        // Compute sequentially in immediate mode for comparison
        set_eager_mode(true);
        Rational sum_seq = 0_r;
        for (int i = 1; i <= 1000; ++i) {
            sum_seq = sum_seq + Rational(1, i);
        }
        set_eager_mode(false); // restore
        EXPECT_EQ(sum_batch.eval(), sum_seq);
    }

} // namespace delta::testing