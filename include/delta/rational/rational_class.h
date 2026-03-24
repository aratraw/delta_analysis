#pragma once

#include "rational_fwd.h"
#include "storage.h"

#include <absl/numeric/int128.h>
#include <boost/multiprecision/cpp_int.hpp>

#include <memory>
#include <string>
#include <variant>

namespace delta {

    class Rational {
    public:
        explicit Rational(internal::Value val);
        // Constructors (all create immediate values)
        Rational() noexcept;
        Rational(absl::int128 num);
        Rational(absl::int128 num, absl::uint128 den);
        explicit Rational(const boost::multiprecision::cpp_int& num);
        Rational(const boost::multiprecision::cpp_int& num, const boost::multiprecision::cpp_int& den);
        explicit Rational(const std::string& s);          // parses "1/2" or "0.125"
        Rational(int num) : Rational(static_cast<absl::int128>(num)) {}
        Rational(int num, int den);

        // Deleted constructors from floating-point types
        Rational(double) = delete;
        Rational(float) = delete;
        Rational(long double) = delete;

        // Copy/move defaults
        Rational(const Rational&) = default;
        Rational(Rational&&) = default;
        Rational& operator=(const Rational&) = default;
        Rational& operator=(Rational&&) = default;
        ~Rational() = default;

        // State queries
        bool is_immediate() const noexcept;
        bool is_lazy() const noexcept;

        // Raw accessors (returns nullptr if type does not match)
        const internal::SmallStorage* as_small() const noexcept;
        const internal::BigStorage* as_big() const noexcept;
        const std::shared_ptr<const internal::ExpressionRoot>& as_lazy() const;

        // Conversions
        Rational lazy() const;

        // Simplification and evaluation
        Rational simplify() const;
        internal::Value eval() const;

        // Interval estimation
        internal::Interval approx_interval() const noexcept;
        // Friend Methods
        friend Rational operator+(const Rational&, const Rational&);
        friend Rational operator-(const Rational&, const Rational&);
        friend Rational operator*(const Rational&, const Rational&);
        friend Rational operator/(const Rational&, const Rational&);
        friend Rational operator-(const Rational&);
        friend Rational batch_add(const std::vector<Rational>&);
        // Дружественные функции из transcendentals.h
        friend Rational sqrt(const Rational&, const Rational&);
        friend Rational exp(const Rational&, const Rational&);
        friend Rational log(const Rational&, const Rational&);
        friend Rational sin(const Rational&, const Rational&);
        friend Rational cos(const Rational&, const Rational&);
        friend Rational acos(const Rational&, const Rational&);
        friend Rational pi(const Rational&);
        friend Rational e(const Rational&);
        friend Rational pow(const Rational&, int);
    private:
        using Storage = std::variant<internal::SmallStorage,
            internal::BigStorage,
            std::shared_ptr<const internal::ExpressionRoot>>;
        Storage storage_;

        // Private constructors (used by eager wrappers and factories)
        explicit Rational(std::shared_ptr<const internal::ExpressionRoot> root);
        friend std::string to_string(const Rational& r);
    };


} // namespace delta