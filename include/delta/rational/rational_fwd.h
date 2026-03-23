// include/delta/rational/rational_fwd.h
#pragma once

#include <cstddef>
#include <memory>
#include <variant>
#include <vector>

namespace delta {
    class Rational;

    // Operator declarations (definitions in rational_class.h)
    Rational operator+(const Rational& a, const Rational& b);
    Rational operator-(const Rational& a, const Rational& b);
    Rational operator*(const Rational& a, const Rational& b);
    Rational operator/(const Rational& a, const Rational& b);
    Rational operator-(const Rational& a);
    bool operator==(const Rational& a, const Rational& b);
    bool operator!=(const Rational& a, const Rational& b);
    bool operator<(const Rational& a, const Rational& b);
    bool operator<=(const Rational& a, const Rational& b);
    bool operator>(const Rational& a, const Rational& b);
    bool operator>=(const Rational& a, const Rational& b);

    // Epsilon control
    const Rational& default_eps();
    void set_default_eps(const Rational& eps);
}

namespace delta::internal {
    struct SmallStorage;
    struct BigStorage;
    struct LazyNode;
    using Storage = std::variant<SmallStorage, BigStorage, std::shared_ptr<LazyNode>>;
    class Interval;

    extern thread_local bool global_eager_mode;

    Rational evaluate(const Rational& r);
    Rational simplify(const Rational& r);

    bool structurally_equal(const LazyNode& a, const LazyNode& b);
    bool structurally_equal(const Rational& a, const Rational& b);

    Rational eager_add(const Rational& a, const Rational& b);
    Rational eager_sub(const Rational& a, const Rational& b);
    Rational eager_mul(const Rational& a, const Rational& b);
    Rational eager_div(const Rational& a, const Rational& b);
    Rational eager_neg(const Rational& a);
}