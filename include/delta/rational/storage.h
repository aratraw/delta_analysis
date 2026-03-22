// include/delta/rational/storage.h
#pragma once

#include "delta/rational/rational_fwd.h"
#include "delta/rational/interval.h"
#include <absl/numeric/int128.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace delta::internal {

    // =========================================================================
    // SmallStorage – fixed‑size representation with lazy normalisation
    // =========================================================================
    struct SmallStorage {
        absl::int128 num;
        absl::uint128 den;   // > 0
        bool reduced;        // false => need to call normalize()

        SmallStorage() noexcept : num(0), den(1), reduced(true) {}
        explicit SmallStorage(absl::int128 n) : num(n), den(1), reduced(true) {}
        SmallStorage(absl::int128 n, absl::uint128 d) : num(n), den(d), reduced(false) {}

        void normalize() {
            if (reduced) return;
            if (den == 0) throw std::domain_error("denominator cannot be zero");
            if (num == 0) {
                den = 1;
                reduced = true;
                return;
            }
            absl::uint128 a = (num < 0) ? -static_cast<absl::uint128>(num) : static_cast<absl::uint128>(num);
            absl::uint128 b = den;
            while (b != 0) {
                absl::uint128 t = b;
                b = a % b;
                a = t;
            }
            absl::uint128 g = a;
            if (g > 1) {
                num /= static_cast<absl::int128>(g);
                den /= g;
            }
            // Ensure denominator positive
            if (num < 0 && den < 0) {
                num = -num;
                den = -den;
            }
            else if (den < 0) {
                num = -num;
                den = -den;
            }
            reduced = true;
        }
    };

    // =========================================================================
    // BigStorage – arbitrary‑precision with immediate normalisation
    // =========================================================================
    struct BigStorage {
        boost::multiprecision::cpp_int num;
        boost::multiprecision::cpp_int den;   // > 0

        BigStorage() : num(0), den(1) {}
        explicit BigStorage(const boost::multiprecision::cpp_int& n) : num(n), den(1) {}
        BigStorage(const boost::multiprecision::cpp_int& n, const boost::multiprecision::cpp_int& d) : num(n), den(d) {
            normalize();
        }

        void normalize() {
            if (den == 0) throw std::domain_error("denominator cannot be zero");
            if (num == 0) {
                den = 1;
                return;
            }
            boost::multiprecision::cpp_int g = boost::multiprecision::gcd(num, den);
            if (g > 1) {
                num /= g;
                den /= g;
            }
            if (den < 0) {
                num = -num;
                den = -den;
            }
        }
    };

    // =========================================================================
    // LazyOp – operation types for deferred evaluation
    // =========================================================================
    enum class LazyOp {
        ADD,
        SUB,
        MUL,
        DIV,
        NEG,
        SQRT,
        EXP,
        LOG,
        SIN,
        COS,
        ACOS,
        PI,
        E
    };

    // Forward declaration of compute_approx (defined in simplify.h)
    Interval compute_approx(LazyOp op, const std::vector<std::shared_ptr<const Rational>>& args);

    // =========================================================================
    // LazyNode – deferred operation tree with shared ownership and immediate interval
    // =========================================================================
    struct LazyNode {
        LazyOp op;
        std::vector<std::shared_ptr<const Rational>> args;   // shared ownership to avoid recursion
        Rational precision;                                  // required accuracy for transcendentals
        mutable std::optional<Rational> cached_value;        // after eager evaluation
        Interval approx;                                     // always up‑to‑date interval for fast comparisons

        // Constructors – immediately compute approximate interval
        LazyNode(LazyOp o, std::vector<std::shared_ptr<const Rational>>&& a, const Rational& eps = Rational())
            : op(o), args(std::move(a)), precision(eps), cached_value(std::nullopt),
            approx(compute_approx(op, this->args)) {
        }

        LazyNode(LazyOp o, std::shared_ptr<const Rational> a, const Rational& eps = Rational())
            : LazyNode(o, std::vector<std::shared_ptr<const Rational>>{std::move(a)}, eps) {
        }

        LazyNode(LazyOp o, std::shared_ptr<const Rational> a, std::shared_ptr<const Rational> b, const Rational& eps = Rational())
            : LazyNode(o, std::vector<std::shared_ptr<const Rational>>{std::move(a), std::move(b)}, eps) {
        }
    };

    // =========================================================================
    // Storage variant – LazyNode is stored via shared_ptr to keep variant size small.
    // =========================================================================
    using Storage = std::variant<SmallStorage, BigStorage, std::shared_ptr<LazyNode>>;

} // namespace delta::internal