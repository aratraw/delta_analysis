// rational_impl.h
#pragma once

#include "storage.h"    
#include "expression_root.h"
#include "node_pool.h"
#include "evaluate_impl.h"
#include "simplify_impl.h"
#include "context.h"
#include "utils.h"
#include "comparisons.h"

#include <absl/numeric/int128.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <cctype>
#include <stdexcept>
#include <string>
#include <variant>
#include <functional>
#include <type_traits>

namespace delta {
    // ----------------------------------------------------------------------------
    // eager wrapper functions (formerly in eager.h)
    // ----------------------------------------------------------------------------

    inline Rational eager_add(const Rational& a, const Rational& b) {
        internal::Value va = a.to_value();
        internal::Value vb = b.to_value();
        internal::Value res = internal::eager_add(va, vb);
        return Rational(res);
    }

    inline Rational eager_sub(const Rational& a, const Rational& b) {
        internal::Value va = a.to_value();
        internal::Value vb = b.to_value();
        internal::Value res = internal::eager_sub(va, vb);
        return Rational(res);
    }

    inline Rational eager_mul(const Rational& a, const Rational& b) {
        internal::Value va = a.to_value();
        internal::Value vb = b.to_value();
        internal::Value res = internal::eager_mul(va, vb);
        return Rational(res);
    }

    inline Rational eager_div(const Rational& a, const Rational& b) {
        internal::Value va = a.to_value();
        internal::Value vb = b.to_value();
        internal::Value res = internal::eager_div(va, vb);
        return Rational(res);
    }

    inline Rational eager_neg(const Rational& a) {
        internal::Value va = a.to_value();
        internal::Value res = internal::eager_neg(va);
        return Rational(res);
    }

    inline Rational eager_sqrt(const Rational& x, const Rational& eps) {
        internal::Value vx = x.to_value();
        internal::Value veps = eps.to_value();
        internal::Value res = internal::eager_sqrt(vx, veps);
        return Rational(res);
    }

    inline Rational eager_exp(const Rational& x, const Rational& eps) {
        internal::Value vx = x.to_value();
        internal::Value veps = eps.to_value();
        internal::Value res = internal::eager_exp(vx, veps);
        return Rational(res);
    }

    inline Rational eager_log(const Rational& x, const Rational& eps) {
        internal::Value vx = x.to_value();
        internal::Value veps = eps.to_value();
        internal::Value res = internal::eager_log(vx, veps);
        return Rational(res);
    }

    inline Rational eager_sin(const Rational& x, const Rational& eps) {
        internal::Value vx = x.to_value();
        internal::Value veps = eps.to_value();
        internal::Value res = internal::eager_sin(vx, veps);
        return Rational(res);
    }

    inline Rational eager_cos(const Rational& x, const Rational& eps) {
        internal::Value vx = x.to_value();
        internal::Value veps = eps.to_value();
        internal::Value res = internal::eager_cos(vx, veps);
        return Rational(res);
    }

    inline Rational eager_acos(const Rational& x, const Rational& eps) {
        internal::Value vx = x.to_value();
        internal::Value veps = eps.to_value();
        internal::Value res = internal::eager_acos(vx, veps);
        return Rational(res);
    }

    inline Rational eager_pi(const Rational& eps) {
        internal::Value veps = eps.to_value();
        internal::Value res = internal::eager_pi(veps);
        return Rational(res);
    }

    inline Rational eager_e(const Rational& eps) {
        internal::Value veps = eps.to_value();
        internal::Value res = internal::eager_e(veps);
        return Rational(res);
    }

    inline Rational eager_pow(const Rational& base, const Rational& exponent, const Rational& eps) {
        internal::Value vbase = base.to_value();
        internal::Value vexp = exponent.to_value();
        internal::Value veps = eps.to_value();
        internal::Value res = internal::eager_pow(vbase, vexp, veps);
        return Rational(res);
    }
    // ----------------------------------------------------------------------------
    // Constructors
    // ----------------------------------------------------------------------------

    inline Rational::Rational() noexcept : storage_(internal::SmallStorage{}) {}

    inline Rational::Rational(absl::int128 num) : storage_(internal::SmallStorage(num)) {}

    // Неявные конструкторы для целых типов
    inline Rational::Rational(int num) : Rational(static_cast<absl::int128>(num)) {}
    inline Rational::Rational(long long num) : Rational(static_cast<absl::int128>(num)) {}
    inline Rational::Rational(unsigned long long num) : Rational(static_cast<absl::int128>(num)) {}

    // Конструктор от двух 128-битных чисел (внутренний, неявный)
    inline Rational::Rational(absl::int128 num, absl::uint128 den) {
        if (den == 0) {
            throw std::domain_error("Denominator cannot be zero");
        }
        internal::SmallStorage s(num, den);
        s.normalize();
        storage_ = s;
    }

    // ЕДИНСТВЕННЫЙ конструктор от двух аргументов для пользовательского кода
    inline Rational::Rational(long long num, long long den) {
        if (den == 0) {
            throw std::domain_error("Denominator cannot be zero");
        }
        if (den < 0) {
            num = -num;
            den = -den;
        }

        // Проверяем, помещается ли в 128-битные типы
        constexpr absl::int128 max_int128 = (std::numeric_limits<absl::int128>::max)();
        constexpr absl::int128 min_int128 = (std::numeric_limits<absl::int128>::min)();
        constexpr absl::uint128 max_uint128 = (std::numeric_limits<absl::uint128>::max)();

        if (num <= max_int128 && num >= min_int128 &&
            static_cast<absl::uint128>(den) <= max_uint128) {
            *this = Rational(static_cast<absl::int128>(num),
                static_cast<absl::uint128>(den));
        }
        else {
            *this = Rational(boost::multiprecision::cpp_int(num),
                boost::multiprecision::cpp_int(den));
        }
    }

    // Конструкторы для больших чисел (explicit)
    inline Rational::Rational(const boost::multiprecision::cpp_int& num)
        : storage_(internal::BigStorage(num)) {
    }

    inline Rational::Rational(const boost::multiprecision::cpp_int& num,
        const boost::multiprecision::cpp_int& den) {
        if (den == 0) {
            throw std::domain_error("Denominator cannot be zero");
        }
        storage_ = internal::BigStorage(num, den);
    }

    // Конструктор от строки
    inline Rational::Rational(const std::string& s) {
        size_t slash = s.find('/');
        if (slash != std::string::npos) {
            std::string num_str = s.substr(0, slash);
            std::string den_str = s.substr(slash + 1);
            boost::multiprecision::cpp_int num(num_str);
            boost::multiprecision::cpp_int den(den_str);
            if (den == 0) throw std::domain_error("Denominator cannot be zero");
            if (den < 0) { den = -den; num = -num; }
            boost::multiprecision::cpp_int g = boost::multiprecision::gcd(num, den);
            num /= g; den /= g;
            if (num <= internal::to_cpp_int((std::numeric_limits<absl::int128>::max)()) &&
                den <= internal::to_cpp_int((std::numeric_limits<absl::uint128>::max)())) {
                storage_ = internal::SmallStorage(
                    internal::int128_from_string(num.str()),
                    internal::uint128_from_string(den.str())
                );
            }
            else {
                storage_ = internal::BigStorage(num, den);
            }
        }
        else {
            size_t dot = s.find('.');
            if (dot == std::string::npos) {
                storage_ = internal::BigStorage(boost::multiprecision::cpp_int(s));
            }
            else {
                std::string int_part = s.substr(0, dot);
                std::string frac_part = s.substr(dot + 1);
                if (frac_part.empty()) frac_part = "0";
                size_t decimal_places = frac_part.length();

                bool negative = false;
                if (!int_part.empty() && int_part[0] == '-') {
                    negative = true;
                    int_part = int_part.substr(1);
                }

                if (int_part.empty()) int_part = "0";
                size_t int_start = int_part.find_first_not_of('0');
                if (int_start != std::string::npos) int_part = int_part.substr(int_start);
                else int_part = "0";

                std::string numerator_str = int_part + frac_part;

                size_t num_start = numerator_str.find_first_not_of('0');
                if (num_start != std::string::npos) {
                    numerator_str = numerator_str.substr(num_start);
                }
                else {
                    numerator_str = "0";
                }

                if (negative && numerator_str != "0") {
                    numerator_str = "-" + numerator_str;
                }

                boost::multiprecision::cpp_int denominator = 1;
                for (size_t i = 0; i < decimal_places; ++i) denominator *= 10;

                boost::multiprecision::cpp_int numerator(numerator_str);
                boost::multiprecision::cpp_int g = boost::multiprecision::gcd(numerator, denominator);
                numerator /= g;
                denominator /= g;

                if (numerator <= internal::to_cpp_int((std::numeric_limits<absl::int128>::max)()) &&
                    denominator <= internal::to_cpp_int((std::numeric_limits<absl::uint128>::max)())) {
                    storage_ = internal::SmallStorage(
                        internal::int128_from_string(numerator.str()),
                        internal::uint128_from_string(denominator.str())
                    );
                }
                else {
                    storage_ = internal::BigStorage(numerator, denominator);
                }
            }
        }
    }

    inline Rational::Rational(internal::Value val) {
        if (std::holds_alternative<internal::SmallStorage>(val)) {
            auto s = std::get<internal::SmallStorage>(std::move(val));
            s.normalize();
            storage_ = s;
        }
        else if (std::holds_alternative<internal::BigStorage>(val)) {
            storage_ = std::get<internal::BigStorage>(std::move(val));
        }
        else {
            throw std::logic_error("Rational::Rational(Value): invalid variant");
        }
    }

    inline Rational Rational::from_lazy_index(std::size_t root_idx) {
        Rational r;
        r.storage_ = static_cast<int>(root_idx);
        return r;
    }

    // ----------------------------------------------------------------------------
    // State queries
    // ----------------------------------------------------------------------------

    inline bool Rational::is_immediate() const noexcept {
        return std::holds_alternative<internal::SmallStorage>(storage_) ||
            std::holds_alternative<internal::BigStorage>(storage_);
    }

    inline bool Rational::is_lazy() const noexcept {
        return std::holds_alternative<int>(storage_);
    }

    // ----------------------------------------------------------------------------
    // Numerator/Denominator queries
    // ----------------------------------------------------------------------------

    inline Rational Rational::numerator() const {
        if (is_immediate()) {
            if (auto* s = std::get_if<internal::SmallStorage>(&storage_)) {
                internal::SmallStorage norm = *s;
                norm.normalize();
                return Rational(internal::to_cpp_int(norm.num));
            }
            else if (auto* b = std::get_if<internal::BigStorage>(&storage_)) {
                return Rational(b->num());
            }
        }
        else {
            return eval().numerator();
        }
        throw std::logic_error("Rational::numerator: invalid state");
    }

    inline Rational Rational::denominator() const {
        if (is_immediate()) {
            if (auto* s = std::get_if<internal::SmallStorage>(&storage_)) {
                internal::SmallStorage norm = *s;
                norm.normalize();
                return Rational(internal::to_cpp_int(norm.den));
            }
            else if (auto* b = std::get_if<internal::BigStorage>(&storage_)) {
                return Rational(b->den());
            }
        }
        else {
            return eval().denominator();
        }
        throw std::logic_error("Rational::denominator: invalid state");
    }

    // ----------------------------------------------------------------------------
    // Raw accessors
    // ----------------------------------------------------------------------------

    inline const internal::SmallStorage* Rational::as_small() const noexcept {
        return std::get_if<internal::SmallStorage>(&storage_);
    }

    inline const internal::BigStorage* Rational::as_big() const noexcept {
        return std::get_if<internal::BigStorage>(&storage_);
    }

    inline int Rational::root_index() const {
        return std::get<int>(storage_);
    }

    // ----------------------------------------------------------------------------
    // Conversions
    // ----------------------------------------------------------------------------

    inline Rational Rational::lazy() const {
        if (is_lazy()) return *this;
        internal::Value val = to_value();
        ExpressionRoot root = ExpressionRoot::make_const(val);
        return Rational::from_lazy_index(root.root_index());
    }

    // ----------------------------------------------------------------------------
    // Simplification and evaluation
    // ----------------------------------------------------------------------------

    inline Rational Rational::simplify() const {
        if (is_immediate()) return *this;
        ExpressionRoot root(root_index());
        ExpressionRoot simplified = root.simplify();
        int new_root = simplified.root_index();
        const internal::Node& node = internal::pool.nodes[new_root];
        if (node.op == internal::LazyOp::CONST) {
            return Rational(internal::pool.values[node.value_idx]);
        }
        return Rational::from_lazy_index(new_root);
    }

    inline Rational Rational::eval(bool skip_simplify) const {
        if (is_immediate()) {
            if (auto* s = std::get_if<internal::SmallStorage>(&storage_))
                return Rational(*s);
            if (auto* b = std::get_if<internal::BigStorage>(&storage_))
                return Rational(*b);
            throw std::logic_error("Rational::eval: invalid immediate state");
        }
        if (skip_simplify) {
            internal::Value v = internal::evaluate(root_index());
            return Rational(v);
        }
        Rational simp = simplify();
        if (simp.is_immediate()) return simp;
        return simp.eval(true);
    }

    // ----------------------------------------------------------------------------
    // Internal value conversion
    // ----------------------------------------------------------------------------

    inline internal::Value Rational::to_value() const {
        if (is_immediate()) {
            if (auto* s = std::get_if<internal::SmallStorage>(&storage_))
                return *s;
            if (auto* b = std::get_if<internal::BigStorage>(&storage_))
                return *b;
            throw std::logic_error("Rational::to_value: invalid immediate state");
        }
        return internal::evaluate(root_index());
    }

    // ----------------------------------------------------------------------------
    // Interval estimation
    // ----------------------------------------------------------------------------

    inline internal::Interval Rational::approx_interval() const noexcept {
        if (is_immediate()) {
            internal::Value val;
            if (auto* s = std::get_if<internal::SmallStorage>(&storage_))
                val = *s;
            else if (auto* b = std::get_if<internal::BigStorage>(&storage_))
                val = *b;
            else
                return internal::Interval();
            double d = internal::to_double(val);
            return internal::Interval(d);
        }
        return internal::pool.nodes[root_index()].approx;
    }

    // ----------------------------------------------------------------------------
    // Arithmetic operators
    // ----------------------------------------------------------------------------

    inline Rational operator+(const Rational& a, const Rational& b) {
        if (internal::global_eager_mode || (a.is_immediate() && b.is_immediate())) {
            return eager_add(a, b);
        }
        ExpressionRoot left = a.is_lazy() ? ExpressionRoot(a.root_index())
            : ExpressionRoot::make_const(a.to_value());
        ExpressionRoot right = b.is_lazy() ? ExpressionRoot(b.root_index())
            : ExpressionRoot::make_const(b.to_value());
        return Rational::from_lazy_index(left.add(right).root_index());
    }

    inline Rational operator-(const Rational& a, const Rational& b) {
        if (internal::global_eager_mode || (a.is_immediate() && b.is_immediate())) {
            return eager_sub(a, b);
        }
        ExpressionRoot left = a.is_lazy() ? ExpressionRoot(a.root_index())
            : ExpressionRoot::make_const(a.to_value());
        ExpressionRoot right = b.is_lazy() ? ExpressionRoot(b.root_index())
            : ExpressionRoot::make_const(b.to_value());
        return Rational::from_lazy_index(left.sub(right).root_index());
    }

    inline Rational operator*(const Rational& a, const Rational& b) {
        if (internal::global_eager_mode || (a.is_immediate() && b.is_immediate())) {
            return eager_mul(a, b);
        }
        ExpressionRoot left = a.is_lazy() ? ExpressionRoot(a.root_index())
            : ExpressionRoot::make_const(a.to_value());
        ExpressionRoot right = b.is_lazy() ? ExpressionRoot(b.root_index())
            : ExpressionRoot::make_const(b.to_value());
        return Rational::from_lazy_index(left.mul(right).root_index());
    }

    inline Rational operator/(const Rational& a, const Rational& b) {
        if (internal::global_eager_mode || (a.is_immediate() && b.is_immediate())) {
            return eager_div(a, b);
        }
        ExpressionRoot left = a.is_lazy() ? ExpressionRoot(a.root_index())
            : ExpressionRoot::make_const(a.to_value());
        ExpressionRoot right = b.is_lazy() ? ExpressionRoot(b.root_index())
            : ExpressionRoot::make_const(b.to_value());
        return Rational::from_lazy_index(left.div(right).root_index());
    }

    inline Rational operator-(const Rational& a) {
        if (internal::global_eager_mode || a.is_immediate()) {
            return eager_neg(a);
        }
        ExpressionRoot root(a.root_index());
        return Rational::from_lazy_index(root.neg().root_index());
    }

    // ----------------------------------------------------------------------------
    // Compound assignment
    // ----------------------------------------------------------------------------

    inline Rational& operator+=(Rational& a, const Rational& b) {
        a = a + b;
        return a;
    }

    inline Rational& operator-=(Rational& a, const Rational& b) {
        a = a - b;
        return a;
    }

    inline Rational& operator*=(Rational& a, const Rational& b) {
        a = a * b;
        return a;
    }

    inline Rational& operator/=(Rational& a, const Rational& b) {
        a = a / b;
        return a;
    }

    // ----------------------------------------------------------------------------
    // String conversion
    // ----------------------------------------------------------------------------

    inline std::string to_string(const Rational& r) {
        if (r.is_immediate()) {
            internal::Value v;
            if (const auto* s = std::get_if<internal::SmallStorage>(&r.storage_)) {
                v = *s;
            }
            else if (const auto* b = std::get_if<internal::BigStorage>(&r.storage_)) {
                v = *b;
            }
            else {
                throw std::logic_error("to_string: invalid immediate variant");
            }
            return internal::to_string(v);
        }
        return internal::to_string(r.to_value());
    }

    // ----------------------------------------------------------------------------
    // Batch addition
    // ----------------------------------------------------------------------------
    inline Rational batch_add(const std::vector<Rational>& terms) {
        if (terms.empty()) return Rational(0);

        bool all_immediate = std::all_of(terms.begin(), terms.end(),
            [](const Rational& r) { return r.is_immediate(); });

        if (all_immediate) {
            using boost::multiprecision::cpp_int;
            cpp_int common_denom(1);
            std::vector<cpp_int> nums;
            nums.reserve(terms.size());

            for (const Rational& term : terms) {
                if (term.as_small()) {
                    internal::SmallStorage s = *term.as_small();
                    s.normalize();
                    nums.push_back(internal::to_cpp_int(s.num));
                    cpp_int den = internal::to_cpp_int(s.den);
                    common_denom = boost::multiprecision::lcm(common_denom, den);
                }
                else if (term.as_big()) {
                    const auto& b = *term.as_big();
                    nums.push_back(b.num());
                    common_denom = boost::multiprecision::lcm(common_denom, b.den());
                }
                else {
                    all_immediate = false;
                    break;
                }
            }

            if (all_immediate) {
                cpp_int sum_num(0);
                for (size_t i = 0; i < terms.size(); ++i) {
                    cpp_int factor = common_denom;
                    if (terms[i].as_small()) {
                        internal::SmallStorage s = *terms[i].as_small();
                        s.normalize();
                        factor /= internal::to_cpp_int(s.den);
                        sum_num += nums[i] * factor;
                    }
                    else if (terms[i].as_big()) {
                        const auto& b = *terms[i].as_big();
                        factor /= b.den();
                        sum_num += b.num() * factor;
                    }
                }
                cpp_int g = boost::multiprecision::gcd(sum_num, common_denom);
                sum_num /= g;
                common_denom /= g;

                if (sum_num <= internal::to_cpp_int((std::numeric_limits<absl::int128>::max)()) &&
                    common_denom <= internal::to_cpp_int((std::numeric_limits<absl::uint128>::max)())) {
                    return Rational(
                        internal::int128_from_string(sum_num.str()),
                        internal::uint128_from_string(common_denom.str())
                    );
                }
                else {
                    return Rational(sum_num, common_denom);
                }
            }
        }

        std::function<ExpressionRoot(int, int)> build_tree =
            [&](int l, int r) -> ExpressionRoot {
            if (l == r) {
                const Rational& term = terms[l];
                if (term.is_lazy())
                    return ExpressionRoot(term.root_index());
                else
                    return ExpressionRoot::make_const(term.to_value());
            }
            int mid = l + (r - l) / 2;
            ExpressionRoot left = build_tree(l, mid);
            ExpressionRoot right = build_tree(mid + 1, r);
            return left.add(right);
            };

        ExpressionRoot root = build_tree(0, static_cast<int>(terms.size()) - 1);
        return Rational::from_lazy_index(root.root_index());
    }

    // ----------------------------------------------------------------------------
    // Absolute value
    // ----------------------------------------------------------------------------
    inline Rational abs(const Rational& x) {
        if (x.is_immediate()) {
            internal::Value v = x.to_value();
            if (internal::is_negative(v)) {
                return Rational(internal::eager_neg(v));
            }
            return x;
        }
        return x < Rational(0) ? -x : x;
    }

    inline ExpressionRoot ExpressionRoot::sqrt(const Rational& eps) const {
        internal::Value eps_val = eps.to_value();
        return make_unary(internal::LazyOp::SQRT, *this, eps_val);
    }

    inline ExpressionRoot ExpressionRoot::exp(const Rational& eps) const {
        internal::Value eps_val = eps.to_value();
        return make_unary(internal::LazyOp::EXP, *this, eps_val);
    }

    inline ExpressionRoot ExpressionRoot::log(const Rational& eps) const {
        internal::Value eps_val = eps.to_value();
        return make_unary(internal::LazyOp::LOG, *this, eps_val);
    }

    inline ExpressionRoot ExpressionRoot::sin(const Rational& eps) const {
        internal::Value eps_val = eps.to_value();
        return make_unary(internal::LazyOp::SIN, *this, eps_val);
    }

    inline ExpressionRoot ExpressionRoot::cos(const Rational& eps) const {
        internal::Value eps_val = eps.to_value();
        return make_unary(internal::LazyOp::COS, *this, eps_val);
    }

    inline ExpressionRoot ExpressionRoot::acos(const Rational& eps) const {
        internal::Value eps_val = eps.to_value();
        return make_unary(internal::LazyOp::ACOS, *this, eps_val);
    }

    inline ExpressionRoot ExpressionRoot::pi(const Rational& eps) {
        internal::Value eps_val = eps.to_value();
        int val_idx = internal::add_value(eps_val);
        int node_idx = internal::get_unary_node(internal::LazyOp::PI, -1, val_idx);
        return ExpressionRoot(node_idx);
    }

    inline ExpressionRoot ExpressionRoot::e(const Rational& eps) {
        internal::Value eps_val = eps.to_value();
        int val_idx = internal::add_value(eps_val);
        int node_idx = internal::get_unary_node(internal::LazyOp::E, -1, val_idx);
        return ExpressionRoot(node_idx);
    }

    inline Rational default_eps() {
        return Rational(internal::default_eps_value);
    }

    inline void set_default_eps(const Rational& eps) {
        internal::default_eps_value = eps.to_value();
    }

    inline double Rational::to_double() const {
        return internal::to_double(to_value());
    }

    inline ExpressionRoot ExpressionRoot::pow(const ExpressionRoot& exponent, const Rational& eps) const {
        internal::Value eps_val = eps.to_value();
        return make_binary_with_eps(internal::LazyOp::POW, *this, exponent, eps_val);
    }

    // ----------------------------------------------------------------------------
    // convert_to<T> functionality
    // ----------------------------------------------------------------------------
    template<typename T>
    inline T Rational::convert_to() const {
        if constexpr (std::is_same_v<T, double>) {
            return to_double();
        }
        else if constexpr (std::is_same_v<T, int>) {
            Rational v = eval();
            if (v.denominator() != 1) {
                throw std::domain_error("Rational::convert_to<int>: not an integer");
            }
            if (auto* s = v.as_small()) {
                internal::SmallStorage norm = *s;
                norm.normalize();
                absl::int128 num = norm.num;
                if (num < std::numeric_limits<int>::min() || num > std::numeric_limits<int>::max()) {
                    throw std::overflow_error("Rational::convert_to<int>: value out of int range");
                }
                return static_cast<int>(num);
            }
            else if (auto* b = v.as_big()) {
                const auto& num = b->num();
                if (num > std::numeric_limits<int>::max()) {
                    throw std::overflow_error("Rational::convert_to<int>: value out of int range (positive)");
                }
                if (num < std::numeric_limits<int>::min()) {
                    throw std::overflow_error("Rational::convert_to<int>: value out of int range (negative)");
                }
                return static_cast<int>(num.convert_to<long long>());
            }
            throw std::logic_error("Rational::convert_to<int>: invalid state");
        }
        else if constexpr (std::is_same_v<T, long long>) {
            Rational v = eval();
            if (v.denominator() != 1) {
                throw std::domain_error("Rational::convert_to<long long>: not an integer");
            }
            if (auto* s = v.as_small()) {
                internal::SmallStorage norm = *s;
                norm.normalize();
                absl::int128 num = norm.num;
                if (num < std::numeric_limits<long long>::min() || num > std::numeric_limits<long long>::max()) {
                    throw std::overflow_error("Rational::convert_to<long long>: value out of long long range");
                }
                return static_cast<long long>(num);
            }
            else if (auto* b = v.as_big()) {
                const auto& num = b->num();
                if (num > std::numeric_limits<long long>::max()) {
                    throw std::overflow_error("Rational::convert_to<long long>: value out of range (positive)");
                }
                if (num < std::numeric_limits<long long>::min()) {
                    throw std::overflow_error("Rational::convert_to<long long>: value out of range (negative)");
                }
                return static_cast<long long>(num.convert_to<long long>());
            }
            throw std::logic_error("Rational::convert_to<long long>: invalid state");
        }
        else if constexpr (std::is_same_v<T, boost::multiprecision::cpp_int>) {
            Rational v = eval();
            if (v.denominator() != 1) {
                throw std::domain_error("Rational::convert_to<cpp_int>: not an integer");
            }
            if (auto* s = v.as_small()) {
                internal::SmallStorage norm = *s;
                norm.normalize();
                // Приводим absl::int128 к cpp_int
                if (norm.num < 0) {
                    return -internal::to_cpp_int(static_cast<absl::uint128>(-norm.num));
                }
                else {
                    return internal::to_cpp_int(static_cast<absl::uint128>(norm.num));
                }
            }
            else if (auto* b = v.as_big()) {
                return b->num();  // b->num() уже cpp_int
            }
            throw std::logic_error("Rational::convert_to<cpp_int>: invalid state");
        }
        else {
            static_assert(sizeof(T) == 0, "convert_to not supported for this type");
        }
    }
} // namespace delta