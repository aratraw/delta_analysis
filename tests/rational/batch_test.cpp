// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/rational/batch_test.cpp
#pragma once
#include <gtest/gtest.h>
#include <vector>
#include "delta/core/rational.h"
#include "test_utils.h"

namespace delta::testing {

    class RationalBatchTest : public RationalTest {};

    // -------------------------------------------------------------------------
    // Batch addition tests (всегда immediate)
    // -------------------------------------------------------------------------
    TEST_F(RationalBatchTest, BatchAddSimple) {
        std::vector<Rational> terms = { "1/2"_r, "1/3"_r, "1/6"_r };
        Rational sum = delta::batch_add(terms);
        EXPECT_EQ(sum, 1_r);
    }

    TEST_F(RationalBatchTest, BatchAddLarge) {
        std::vector<Rational> terms(100, "1/1000"_r);
        Rational sum = delta::batch_add(terms);
        EXPECT_EQ(sum, "1/10"_r);
    }

    TEST_F(RationalBatchTest, BatchAddMixed) {
        std::string denom = "1000000000000000000000000000000";
        Rational term1 = "1/2"_r;
        Rational term2 = Rational("1/" + denom);
        std::vector<Rational> terms = { term1, term2 };
        Rational sum = delta::batch_add(terms);
        Rational expected = term1 + term2;
        EXPECT_EQ(sum, expected);
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
        Rational diff = bound - sum;
        EXPECT_GT(diff, 0_r);
        EXPECT_GT(sum, 0_r);
    }

    TEST_F(RationalBatchTest, BatchAddEmpty) {
        std::vector<Rational> terms;
        Rational sum = delta::batch_add(terms);
        EXPECT_EQ(sum, 0_r);
    }

    TEST_F(RationalBatchTest, HarmonicSeriesBatch) {
        std::vector<Rational> terms;
        for (int i = 1; i <= 1000; ++i) {
            terms.push_back(Rational(1, i));
        }
        Rational sum_batch = delta::batch_add(terms);
        Rational sum_seq = 0_r;
        for (int i = 1; i <= 1000; ++i) {
            sum_seq = sum_seq + Rational(1, i);
        }
        EXPECT_EQ(sum_batch, sum_seq);
    }

} // namespace delta::testing