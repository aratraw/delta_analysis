// tests/rational/test_utils.h
#pragma once

#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include <boost/multiprecision/cpp_int.hpp>

namespace delta::testing {

    /**
     * @brief Base test fixture for rational arithmetic tests.
     *
     * Provides:
     * - Precision management (save/restore default_eps)
     * - Helper functions for rational verification
     */
    class RationalTest : public ::testing::Test {
    protected:
        // ---------------------------------------------------------------------
        // Precision management
        // ---------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Helper: check if a rational is in canonical form (den > 0, gcd(num,den)=1)
    // -------------------------------------------------------------------------
    inline bool is_reduced(const Rational& r) {
        Rational ev = r.evaluate(); // ensure non-lazy
        if (ev == 0_r) return true;
        std::string s = ev.to_string();
        size_t slash = s.find('/');
        if (slash == std::string::npos) return true; // integer
        std::string num_str = s.substr(0, slash);
        std::string den_str = s.substr(slash + 1);
        boost::multiprecision::cpp_int num(num_str);
        boost::multiprecision::cpp_int den(den_str);
        if (num < 0) num = -num;
        return boost::multiprecision::gcd(num, den) == 1;
    }

    // -------------------------------------------------------------------------
    // Macro for comparing two rationals with tolerance (using delta::abs)
    // -------------------------------------------------------------------------
#define EXPECT_RATIONAL_NEAR(val, expected, eps) \
        EXPECT_LE(delta::abs((val) - (expected)), (eps))

} // namespace delta::testing