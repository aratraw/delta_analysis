// tests/rational/test_utils.h
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

        static void set_precision(const Rational& eps) {
            delta::set_default_eps(eps);
        }

    private:
        Rational old_precision_;
    };

    inline bool is_reduced(const Rational& r) {
        Rational imm = r;
        if (imm.is_lazy()) imm = imm.simplify();
        if (imm.is_lazy()) return false;

        internal::Value v = imm.to_value();
        internal::dumb_int num, den;
        if (v.tag == internal::ValueType::Small) {
            internal::SmallStorage norm = v.storage.small;
            bool red = false;
            norm.normalize(red);
            num = internal::to_dumb_int(norm.num);
            den = internal::to_dumb_int(norm.den);
        }
        else if (v.tag == internal::ValueType::Big) {
            const auto& b = v.storage.big;
            num = b.numerator();
            den = b.denominator();
        }
        else {
            return false;
        }
        if (num == 0) return true;
        if (den < 0) den = -den;
        if (num < 0) num = -num;
        return boost::multiprecision::gcd(num, den) == 1;
    }

#define EXPECT_RATIONAL_NEAR(val, expected, eps) \
        EXPECT_LE(delta::abs((val) - (expected)), (eps))

} // namespace delta::testing