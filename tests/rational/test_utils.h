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

        internal::Value v = imm.eval();
        boost::multiprecision::cpp_int num, den;
        if (const auto* s = std::get_if<internal::SmallStorage>(&v)) {
            internal::SmallStorage norm = *s;
            norm.normalize();
            num = internal::to_cpp_int(norm.num);
            den = internal::to_cpp_int(norm.den);
        }
        else {
            const auto& b = std::get<internal::BigStorage>(v);
            num = b.num();
            den = b.den();
        }
        if (num == 0) return true;
        if (den < 0) den = -den;
        if (num < 0) num = -num;
        return boost::multiprecision::gcd(num, den) == 1;
    }

#define EXPECT_RATIONAL_NEAR(val, expected, eps) \
        EXPECT_LE(delta::abs((val) - (expected)), (eps))

} // namespace delta::testing