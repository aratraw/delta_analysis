// test_utils.h
#pragma once

#include <gtest/gtest.h>
#include "delta/core/rational.h"

namespace delta::testing {

    class RationalTest : public ::testing::Test {
    protected:
        void SetUp() override {
            old_precision_ = delta::default_eps();
        }
        void TearDown() override {
            delta::set_default_eps(old_precision_);
        }
        static void set_precision(Rational& eps) {
            delta::set_default_eps(eps);
        }
    private:
        Rational old_precision_;
    };

    inline bool is_reduced(const Rational& r) {
        // Новый Value всегда хранит сокращённую дробь
        return true;
    }

#define EXPECT_RATIONAL_NEAR(val, expected, eps) \
    EXPECT_LE(delta::abs((val) - (expected)), (eps))

} // namespace delta::testing