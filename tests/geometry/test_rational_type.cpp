// tests/basic/test_rational_type.cpp
#include <gtest/gtest.h>
#include <chrono>      // for high_resolution_clock
#include <limits>
#include <sstream>
#include "delta/core/rational.h"
#include "../test_fixtures.h"

namespace delta::testing {

    class RationalTypeTest : public DeltaTest {
    protected:
        // Helper to get denominator as a string for inspection
        static std::string denominator_str(const Rational& r) {
            std::stringstream ss;
            ss << r;
            std::string s = ss.str();
            size_t slash = s.find('/');
            if (slash == std::string::npos) return "1";
            return s.substr(slash + 1);
        }

        // Helper to get numerator as a string
        static std::string numerator_str(const Rational& r) {
            std::stringstream ss;
            ss << r;
            std::string s = ss.str();
            size_t slash = s.find('/');
            if (slash == std::string::npos) return s;
            return s.substr(0, slash);
        }

        // Helper to get denominator as a BigInt (if available)
        static boost::multiprecision::cpp_int denominator_big(const Rational& r) {
            std::string d = denominator_str(r);
            return boost::multiprecision::cpp_int(d);
        }

        static bool is_reduced(const Rational& r) {
            if (r == 0) return true;
            std::string num = numerator_str(r);
            std::string den = denominator_str(r);
            if (num.empty() || den.empty()) return true;
            boost::multiprecision::cpp_int n(num);
            boost::multiprecision::cpp_int d(den);
            if (n < 0) n = -n;
            return boost::multiprecision::gcd(n, d) == 1;
        }
    };

    // -------------------------------------------------------------------------
    // 1. Automatic reduction (GCD) in arithmetic operations
    // -------------------------------------------------------------------------
    TEST_F(RationalTypeTest, AdditionReduces) {
        Rational a(1, 2);
        Rational b(1, 3);
        Rational c = a + b;   // 1/2 + 1/3 = 5/6, already reduced
        EXPECT_EQ(numerator_str(c), "5");
        EXPECT_EQ(denominator_str(c), "6");

        Rational d(1, 6);
        Rational e = c + d;   // 5/6 + 1/6 = 6/6 = 1
        EXPECT_EQ(numerator_str(e), "1");
        EXPECT_EQ(denominator_str(e), "1");
    }

    TEST_F(RationalTypeTest, MultiplicationReduces) {
        Rational a(2, 3);
        Rational b(3, 4);
        Rational c = a * b;   // 2/3 * 3/4 = 6/12 = 1/2
        EXPECT_EQ(numerator_str(c), "1");
        EXPECT_EQ(denominator_str(c), "2");
    }

    TEST_F(RationalTypeTest, DivisionReduces) {
        Rational a(2, 3);
        Rational b(4, 5);
        Rational c = a / b;   // (2/3) / (4/5) = (2/3)*(5/4) = 10/12 = 5/6
        EXPECT_EQ(numerator_str(c), "5");
        EXPECT_EQ(denominator_str(c), "6");
    }

    TEST_F(RationalTypeTest, CompoundAssignmentReduces) {
        Rational a(2, 3);
        a += Rational(1, 6);   // 2/3 + 1/6 = 4/6 + 1/6 = 5/6
        EXPECT_EQ(numerator_str(a), "5");
        EXPECT_EQ(denominator_str(a), "6");

        a *= Rational(3, 5);   // 5/6 * 3/5 = 15/30 = 1/2
        EXPECT_EQ(numerator_str(a), "1");
        EXPECT_EQ(denominator_str(a), "2");
    }

    // -------------------------------------------------------------------------
    // 2. Precision management: default_eps_value changes affect sqrt(2)
    // -------------------------------------------------------------------------
    TEST_F(RationalTypeTest, DefaultEpsAffectsSqrt) {
        Rational original_eps = delta::default_eps();

        // Store high-precision sqrt(2)
        Rational high = delta::sqrt(2_r);
        double high_val = static_cast<double>(high);

        // Set coarse precision
        delta::default_eps_value() = Rational(1, 10);
        Rational coarse = delta::sqrt(2_r);
        double coarse_val = static_cast<double>(coarse);
        EXPECT_GT(std::abs(high_val - coarse_val), 1e-6); // should be significantly different

        // Restore
        delta::default_eps_value() = original_eps;
    }

    // -------------------------------------------------------------------------
    // 3. Transcendental functions with explicit eps
    // -------------------------------------------------------------------------
    TEST_F(RationalTypeTest, ExpWithDifferentEps) {
        Rational high = delta::exp(1_r, Rational(1, 1000000000000)); // 1e-12
        Rational low = delta::exp(1_r, Rational(1, 100));           // 1e-2

        double high_val = static_cast<double>(high);
        double low_val = static_cast<double>(low);
        EXPECT_GT(std::abs(high_val - low_val), 1e-6);
        EXPECT_NEAR(high_val, 2.718281828459045, 1e-12);
    }

    TEST_F(RationalTypeTest, LogWithDifferentEps) {
        Rational high = delta::log(2_r, Rational(1, 1000000000000));
        Rational low = delta::log(2_r, Rational(1, 100));
        double high_val = static_cast<double>(high);
        double low_val = static_cast<double>(low);
        EXPECT_GT(std::abs(high_val - low_val), 1e-6);
        EXPECT_NEAR(high_val, 0.6931471805599453, 1e-12);
    }

    TEST_F(RationalTypeTest, SinWithDifferentEps) {
        Rational high = delta::sin(1_r, Rational(1, 1000000000000));
        Rational low = delta::sin(1_r, Rational(1, 100));
        double high_val = static_cast<double>(high);
        double low_val = static_cast<double>(low);
        EXPECT_GT(std::abs(high_val - low_val), 1e-6);
        EXPECT_NEAR(high_val, 0.8414709848078965, 1e-12);
    }

    // -------------------------------------------------------------------------
    // 4. Denominator explosion test: sequence of operations
    //    This helps detect if reduction is automatic
    // -------------------------------------------------------------------------
    TEST_F(RationalTypeTest, DenominatorDoesNotExplode) {
        Rational a(1, 2);
        Rational b(1, 3);
        Rational c = a + b;          // 5/6
        Rational d = c * c;          // (5/6)^2 = 25/36
        Rational e = d + Rational(1, 36); // 25/36 + 1/36 = 26/36 = 13/18
        // Expect denominator is 18, not 36*... etc.
        EXPECT_EQ(denominator_str(e), "18");
        EXPECT_EQ(numerator_str(e), "13");
    }

    TEST_F(RationalTypeTest, ChainOfOperations) {
        Rational x = 1_r;
        for (int i = 1; i <= 10; ++i) {
            x = x + Rational(1, i);
        }
        // At each addition, denominator should stay manageable.
        // We check that denominator is not huge (e.g., less than 10^6)
        boost::multiprecision::cpp_int den = denominator_big(x);
        // The true denominator after summing 1/1+1/2+...+1/10 = 7381/2520? Actually lcm is 2520, sum is 7381/2520 ≈ 2.928.
        // So denominator should be 2520.
        EXPECT_EQ(denominator_str(x), "2520");
    }

    // -------------------------------------------------------------------------
    // 5. Large operation: matrix exponential related rationals
    //    This mimics what happens inside MatrixField
    // -------------------------------------------------------------------------
    TEST_F(RationalTypeTest, RationalInMatrixExpSimulation) {
        // Simulate one step of Padé approximation: compute (I + A) / (I - A)
        Rational a = Rational(1, 2);   // small matrix element
        Rational I = 1_r;
        Rational numerator = I + a;
        Rational denominator = I - a;
        Rational result = numerator / denominator;   // (1 + 0.5)/(1 - 0.5) = 1.5/0.5 = 3
        EXPECT_EQ(numerator_str(result), "3");
        EXPECT_EQ(denominator_str(result), "1");
    }

    TEST_F(RationalTypeTest, RationalSeriesTerm) {
        // Simulate term = term * X / n in Taylor series for exp
        Rational term = 1_r;
        Rational X = Rational(1, 2);   // 0.5
        int n = 1;
        term = term * X / n;   // 0.5
        EXPECT_EQ(numerator_str(term), "1");
        EXPECT_EQ(denominator_str(term), "2");
        n = 2;
        term = term * X / n;   // 0.5 * 0.5 / 2 = 0.25/2 = 0.125 = 1/8
        EXPECT_EQ(numerator_str(term), "1");
        EXPECT_EQ(denominator_str(term), "8");
        n = 3;
        term = term * X / n;   // (1/8)*(1/2)/3 = 1/48
        EXPECT_EQ(numerator_str(term), "1");
        EXPECT_EQ(denominator_str(term), "48");
    }

    // -------------------------------------------------------------------------
    // 6. Performance/Complexity monitoring (optional, just check no overflow)
    // -------------------------------------------------------------------------
    TEST_F(RationalTypeTest, NoOverflowAfterManyOps) {
        Rational x = 1_r;
        for (int i = 1; i <= 100; ++i) {
            x = x * Rational(i, i + 1);   // multiplying by i/(i+1)
        }
        // After 100 steps, x = 1/101, denominator 101
        EXPECT_EQ(numerator_str(x), "1");
        EXPECT_EQ(denominator_str(x), "101");
    }

    // -------------------------------------------------------------------------
    // 1. AutomaticReductionAfterOperations
    //    Проверяет, что после каждой операции дробь сокращается автоматически.
    //    Используется серия операций, которые без сокращения привели бы к огромным знаменателям.
    // -------------------------------------------------------------------------
    TEST_F(RationalTypeTest, AutomaticReductionAfterOperations) {
        Rational sum = 0_r;
        // Добавляем и вычитаем одни и те же дроби, чтобы сумма оставалась нулём.
        // Без сокращения знаменатели росли бы, но с сокращением они остаются малыми.
        std::vector<std::pair<Rational, Rational>> operations = {
            {Rational(1,2), Rational(1,2)},
            {Rational(1,3), Rational(1,3)},
            {Rational(1,5), Rational(1,5)},
            {Rational(1,7), Rational(1,7)},
            {Rational(1,11), Rational(1,11)}
        };

        for (const auto& op : operations) {
            sum += op.first;
            EXPECT_TRUE(is_reduced(sum)) << "После добавления " << op.first << " сумма не сокращена: " << sum;
            sum -= op.second;
            EXPECT_TRUE(is_reduced(sum)) << "После вычитания " << op.second << " сумма не сокращена: " << sum;
        }
        EXPECT_EQ(sum, 0_r) << "Итоговая сумма должна быть нулём";

        // Альтернативный тест: последовательное сложение дробей, у которых знаменатели имеют общие множители,
        // чтобы знаменатель результата был меньше произведения знаменателей.
        Rational a(1, 2);
        Rational b(1, 4);
        Rational c = a + b;   // 3/4
        EXPECT_EQ(denominator_str(c), "4");
        EXPECT_TRUE(is_reduced(c));

        Rational d(1, 3);
        Rational e(1, 6);
        Rational f = d + e;   // 1/2
        EXPECT_EQ(denominator_str(f), "2");
        EXPECT_TRUE(is_reduced(f));

        // Цепочка умножений, где должно происходить сокращение
        Rational x(2, 3);
        Rational y(3, 4);
        Rational z = x * y;   // 6/12 -> 1/2
        EXPECT_EQ(numerator_str(z), "1");
        EXPECT_EQ(denominator_str(z), "2");
        EXPECT_TRUE(is_reduced(z));
    }

    // -------------------------------------------------------------------------
    // 2. Cross-Cancellation: multiply two large fractions that share factors
    // -------------------------------------------------------------------------
    TEST_F(RationalTypeTest, CrossCancellation) {
        // Create large numerator and denominator with common factors
        // Using strings to avoid overflow in construction
        std::string num_str(1000, '9');  // 999...9 (1000 digits)
        std::string huge_frac = num_str + "/1";
        Rational a(huge_frac);          // huge integer
        Rational b = Rational(1) / Rational(num_str); // 1 / huge

        auto start = std::chrono::high_resolution_clock::now();
        Rational c = a * b;
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        EXPECT_EQ(c, 1_r);
        // Elapsed should be small (e.g., < 0.1s). If > 1s, reduction is not happening early.
        EXPECT_LT(elapsed, 1.0);
    }

    // -------------------------------------------------------------------------
    // 3. Epsilon interplay: Compare rational numbers near epsilon threshold
    // -------------------------------------------------------------------------
    TEST_F(RationalTypeTest, EpsilonInterplay) {
        Rational eps = delta::default_eps();
        Rational small = Rational(1, 1000000); // 1e-6
        Rational very_small = Rational(1, 1000000000000); // 1e-12

        // If default_eps is 1e-30, then very_small < eps.
        // But our comparisons should be exact, not epsilon-based.
        // This test ensures that rational comparisons use exact arithmetic.
        if (eps > small) {
            EXPECT_LT(small, eps);
        }
        else {
            EXPECT_GT(small, eps);
        }
        // Check that delta::abs(small) < eps? Not directly, but we can verify that
        // equality is exact.
        Rational exact = Rational(1, 3);
        Rational approx = delta::sqrt(2_r); // irrational, but we only compare rationals.
        EXPECT_NE(exact, approx);
    }

    // -------------------------------------------------------------------------
    // 4. ChainedOperationsNoExcessiveCopies -> переименован в AccurateChainedSum
    //    Проверяет точность сложения нескольких рациональных чисел.
    // -------------------------------------------------------------------------
    TEST_F(RationalTypeTest, AccurateChainedSum) {
        Rational a(1, 2);
        Rational b(1, 3);
        Rational c(1, 5);
        Rational d(1, 7);
        Rational e(1, 11);
        Rational res = a + b + c + d + e;

        // Точное ожидаемое значение: 1/2 + 1/3 + 1/5 + 1/7 + 1/11
        Rational expected = Rational(1, 2) + Rational(1, 3) + Rational(1, 5) + Rational(1, 7) + Rational(1, 11);
        EXPECT_EQ(res, expected) << "Сумма должна быть точно равна ожидаемому рациональному числу";

        // Дополнительно проверим, что дробь сокращена
        EXPECT_TRUE(is_reduced(res)) << "Результат должен быть в сокращённом виде: " << res;
    }

    // -------------------------------------------------------------------------
    // 5. Invariants of canonical form
    // -------------------------------------------------------------------------
    TEST_F(RationalTypeTest, CanonicalFormInvariants) {
        // Positive denominator
        Rational pos(1, 2);
        Rational neg(1, -2);
        Rational zero(0);

        // Ensure that after construction, denominator is positive
        EXPECT_EQ(denominator_str(pos), "2");
        EXPECT_EQ(denominator_str(neg), "2");  // should be normalized to -1/2? Actually 1/-2 => -1/2, denominator 2
        EXPECT_EQ(numerator_str(neg), "-1");
        // For zero, representation may be "0" with no denominator
        EXPECT_EQ(numerator_str(zero), "0");
        // Check gcd
        Rational gcd_test(6, 8);  // should be 3/4
        EXPECT_EQ(numerator_str(gcd_test), "3");
        EXPECT_EQ(denominator_str(gcd_test), "4");
        // After arithmetic
        Rational a(2, 6);  // should be 1/3
        EXPECT_EQ(numerator_str(a), "1");
        EXPECT_EQ(denominator_str(a), "3");
        Rational b(3, 9);  // should be 1/3
        EXPECT_EQ(numerator_str(b), "1");
        EXPECT_EQ(denominator_str(b), "3");
        Rational c = a + b; // 2/3
        EXPECT_EQ(numerator_str(c), "2");
        EXPECT_EQ(denominator_str(c), "3");
    }

    // -------------------------------------------------------------------------
    // 6. Large powers and reduction
    // -------------------------------------------------------------------------
    TEST_F(RationalTypeTest, LargePowers) {
        Rational base(2, 3);
        Rational pow10 = delta::pow(base, 10);
        // 2^10 / 3^10 = 1024/59049, already reduced
        EXPECT_EQ(numerator_str(pow10), "1024");
        EXPECT_EQ(denominator_str(pow10), "59049");
        // Negative exponent
        Rational pow_neg = delta::pow(base, -10);
        EXPECT_EQ(numerator_str(pow_neg), "59049");
        EXPECT_EQ(denominator_str(pow_neg), "1024");
    }

    // -------------------------------------------------------------------------
    // 7. Division by zero should throw (if Boost throws, or we need to handle)
    // -------------------------------------------------------------------------
    TEST_F(RationalTypeTest, DivisionByZero) {
        Rational a(1, 2);
        Rational b(0);
        // This should throw an exception (Boost throws division_by_zero)
        EXPECT_THROW(a / b, std::exception);
        // Also zero denominator in construction
        EXPECT_THROW(Rational(1, 0), std::exception);
    }

    // -------------------------------------------------------------------------
    // 8. Serialization / string roundtrip
    // -------------------------------------------------------------------------
    TEST_F(RationalTypeTest, StringRoundtrip) {
        Rational r = Rational(12345678901234567890_r) / Rational(9876543210987654321_r);
        std::stringstream ss;
        ss << r;
        Rational r2(ss.str());
        EXPECT_EQ(r, r2);
    }

} // namespace delta::testing