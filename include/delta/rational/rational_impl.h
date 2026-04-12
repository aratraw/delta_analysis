// rational_impl.h — адаптирован под новый tagged union Value (флаг small_reduced в Value)
// Реализует оптимизации SmallStorage согласно спецификации: никакого GCD после арифметики,
// только при переполнении (try_reduce_to_small). In-place операции модифицируют поля напрямую.
#pragma once

#include "storage.h"    
#include "expression_root.h"
#include "node_pool.h"
#include "evaluate_impl.h"
#include "evaluation_core.h" 
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
#include <functional>
#include <type_traits>

namespace delta {

    // ----------------------------------------------------------------------------
    // eager wrapper functions (без изменений)
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
    // In‑place addition for immediate Rationals – согласно спецификации
    // (прямая модификация полей, сброс флага, при переполнении – пересоздание)
    // ----------------------------------------------------------------------------
    inline void inplace_add(Rational& a, const Rational& b) {
        if (!a.is_immediate() || !b.is_immediate()) {
            a = a + b;
            return;
        }

        // Small + Small
        if (a.storage_.tag == internal::ValueType::Small && b.storage_.tag == internal::ValueType::Small) {
            auto& sa = a.storage_.storage.small;
            const auto& sb = b.storage_.storage.small;

            // Быстрый путь для нуля (используем is_zero)
            if (sb.is_zero()) return;
            if (sa.is_zero()) {
                sa = sb;
                a.storage_.small_reduced = false;  // копия может быть несокращённой
                return;
            }

            // Равные знаменатели – проверяем переполнение до сложения
            if (sa.den == sb.den) {
                if (internal::would_overflow_add(sa.num, sb.num)) {
                    a = a + b;  // переполнение – пересоздаём
                    return;
                }
                sa.num += sb.num;
                if (sa.is_zero()) {
                    sa.den = 1;
                    a.storage_.small_reduced = true;
                }
                else {
                    a.storage_.small_reduced = false;
                }
                return;
            }

            // Разные знаменатели – in-place вычисление нового числителя и знаменателя
            bool denoms_small = (sa.den < (absl::uint128(1) << 62)) && (sb.den < (absl::uint128(1) << 62));
            if (denoms_small) {
                absl::int128 left = sa.num * static_cast<absl::int128>(sb.den);
                absl::int128 right = sb.num * static_cast<absl::int128>(sa.den);
                if (internal::would_overflow_add(left, right)) {
                    a = a + b;
                    return;
                }
                absl::int128 new_num = left + right;
                absl::uint128 new_den = sa.den * sb.den;
                // Произведение знаменателей при denoms_small всегда помещается в uint128
                sa.num = new_num;
                sa.den = new_den;
                a.storage_.small_reduced = false;
                return;
            }

            // Большие знаменатели – используем проверки переполнения
            absl::int128 left, right;
            if (internal::would_overflow_mul(sa.num, static_cast<absl::int128>(sb.den)) ||
                internal::would_overflow_mul(sb.num, static_cast<absl::int128>(sa.den))) {
                a = a + b;
                return;
            }
            left = sa.num * static_cast<absl::int128>(sb.den);
            right = sb.num * static_cast<absl::int128>(sa.den);
            if (internal::would_overflow_add(left, right)) {
                a = a + b;
                return;
            }
            absl::int128 new_num = left + right;
            absl::uint128 new_den;
            if (internal::would_overflow_mul(sa.den, sb.den)) {
                a = a + b;
                return;
            }
            new_den = sa.den * sb.den;
            sa.num = new_num;
            sa.den = new_den;
            a.storage_.small_reduced = false;
            return;
        }

        // Big + Big
        if (a.storage_.tag == internal::ValueType::Big && b.storage_.tag == internal::ValueType::Big) {
            *a.storage_.storage.big.ptr += *b.storage_.storage.big.ptr;
            return;
        }

        // Big + Small
        if (a.storage_.tag == internal::ValueType::Big && b.storage_.tag == internal::ValueType::Small) {
            const auto& sb = b.storage_.storage.small;
            *a.storage_.storage.big.ptr += internal::BigRationalType(internal::to_dumb_int(sb.num)) / internal::to_dumb_int(sb.den);
            return;
        }

        // Small + Big
        if (a.storage_.tag == internal::ValueType::Small && b.storage_.tag == internal::ValueType::Big) {
            auto& sa = a.storage_.storage.small;
            internal::BigRationalType res =
                (internal::BigRationalType(internal::to_dumb_int(sa.num)) / internal::to_dumb_int(sa.den)) +
                *b.storage_.storage.big.ptr;
            a.storage_ = internal::Value(internal::BigStorage(std::move(res)));
            return;
        }

        // Fallback
        a = a + b;
    }

    // ----------------------------------------------------------------------------
    // In‑place multiplication for immediate Rationals – согласно спецификации
    // ----------------------------------------------------------------------------
    inline void inplace_mul(Rational& a, const Rational& b) {
        if (!a.is_immediate() || !b.is_immediate()) {
            a = a * b;
            return;
        }

        // Small * Small
        if (a.storage_.tag == internal::ValueType::Small && b.storage_.tag == internal::ValueType::Small) {
            auto& sa = a.storage_.storage.small;
            const auto& sb = b.storage_.storage.small;

            if (sb.is_zero()) {
                sa.num = 0;
                sa.den = 1;
                a.storage_.small_reduced = true;
                return;
            }
            if (sa.is_zero()) return;

            bool denoms_small = (sa.den < (absl::uint128(1) << 62)) && (sb.den < (absl::uint128(1) << 62));
            if (denoms_small) {
                sa.num *= sb.num;
                sa.den *= sb.den;
                a.storage_.small_reduced = false;
                return;
            }

            // Большие знаменатели – проверяем переполнение
            if (internal::would_overflow_mul(sa.num, sb.num) ||
                internal::would_overflow_mul(sa.den, sb.den)) {
                a = a * b;
                return;
            }
            sa.num *= sb.num;
            sa.den *= sb.den;
            a.storage_.small_reduced = false;
            return;
        }

        // Big * Big
        if (a.storage_.tag == internal::ValueType::Big && b.storage_.tag == internal::ValueType::Big) {
            *a.storage_.storage.big.ptr *= *b.storage_.storage.big.ptr;
            return;
        }

        // В остальных случаях – через обычный оператор (он вызовет eager_mul или ленивый путь)
        a = a * b;
    }

    // ----------------------------------------------------------------------------
    // Constructors (адаптированы под новый флаг)
    // ----------------------------------------------------------------------------
    inline Rational::Rational() noexcept : storage_(internal::SmallStorage{}) {}

    inline Rational::Rational(absl::int128 num) : storage_(internal::SmallStorage(num)) {}

    inline Rational::Rational(int num) : Rational(static_cast<absl::int128>(num)) {}
    inline Rational::Rational(long long num) : Rational(static_cast<absl::int128>(num)) {}
    inline Rational::Rational(unsigned long long num) : Rational(static_cast<absl::int128>(num)) {}

    inline Rational::Rational(absl::int128 num, absl::uint128 den) {
        if (den == 0) throw std::domain_error("Denominator cannot be zero");
        internal::SmallStorage s(num, den);
        internal::Value v(s, false);
        v.normalize();   // нормализация при создании из пары (num,den)
        storage_ = v;
    }

    inline Rational::Rational(long long num, long long den) {
        if (den == 0) throw std::domain_error("Denominator cannot be zero");
        if (den < 0) { num = -num; den = -den; }

        constexpr absl::int128 max_i128 = (std::numeric_limits<absl::int128>::max)();
        constexpr absl::int128 min_i128 = (std::numeric_limits<absl::int128>::min)();
        constexpr absl::uint128 max_u128 = (std::numeric_limits<absl::uint128>::max)();

        if (num <= max_i128 && num >= min_i128 &&
            static_cast<absl::uint128>(den) <= max_u128) {
            *this = Rational(static_cast<absl::int128>(num),
                static_cast<absl::uint128>(den));
        }
        else {
            *this = Rational(internal::dumb_int(num), internal::dumb_int(den));
        }
    }

    inline Rational::Rational(const internal::dumb_int& num)
        : storage_(internal::BigStorage(num)) {
    }

    inline Rational::Rational(const internal::dumb_int& num, const internal::dumb_int& den) {
        if (den == 0) throw std::domain_error("Denominator cannot be zero");
        storage_ = internal::BigStorage(num, den);
    }

    inline Rational::Rational(const boost::multiprecision::cpp_int& num)
        : Rational(internal::dumb_int(num)) {
    }

    inline Rational::Rational(const boost::multiprecision::cpp_int& num, const boost::multiprecision::cpp_int& den)
        : Rational(internal::dumb_int(num), internal::dumb_int(den)) {
    }

    inline Rational::Rational(const std::string& s) {
        size_t slash = s.find('/');
        if (slash != std::string::npos) {
            std::string num_str = s.substr(0, slash);
            std::string den_str = s.substr(slash + 1);
            internal::dumb_int num(num_str);
            internal::dumb_int den(den_str);
            if (den == 0) throw std::domain_error("Denominator cannot be zero");
            if (den < 0) { den = -den; num = -num; }
            internal::dumb_int g = boost::multiprecision::gcd(num, den);
            num /= g; den /= g;
            if (internal::fits_in_int128(num) && internal::fits_in_uint128(den)) {
                internal::SmallStorage s_small(internal::dumb_int_to_int128(num), internal::dumb_int_to_uint128(den));
                internal::Value v(s_small, false);
                v.normalize();
                storage_ = v;
            }
            else {
                storage_ = internal::Value(internal::BigStorage(num, den));
            }
        }
        else {
            size_t dot = s.find('.');
            if (dot == std::string::npos) {
                storage_ = internal::Value(internal::BigStorage(internal::dumb_int(s)));
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

                internal::dumb_int denominator = 1;
                for (size_t i = 0; i < decimal_places; ++i) denominator *= 10;

                internal::dumb_int numerator(numerator_str);
                internal::dumb_int g = boost::multiprecision::gcd(numerator, denominator);
                numerator /= g;
                denominator /= g;

                if (internal::fits_in_int128(numerator) && internal::fits_in_uint128(denominator)) {
                    internal::SmallStorage s_small(internal::dumb_int_to_int128(numerator), internal::dumb_int_to_uint128(denominator));
                    internal::Value v(s_small, false);
                    v.normalize();
                    storage_ = v;
                }
                else {
                    storage_ = internal::Value(internal::BigStorage(numerator, denominator));
                }
            }
        }
    }

    inline Rational::Rational(internal::Value val) : storage_(std::move(val)) {}

    inline Rational Rational::from_lazy_index(std::size_t root_idx) {
        internal::increment_ref(static_cast<int>(root_idx));
        Rational r;
        r.storage_ = internal::Value(static_cast<int>(root_idx));
        return r;
    }

    // ----------------------------------------------------------------------------
    // Copy, move, destructor, assignment (без изменений)
    // ----------------------------------------------------------------------------
    inline Rational::Rational(const Rational& other) : storage_(other.storage_) {
        if (is_lazy()) internal::increment_ref(root_index());
    }

    inline Rational::Rational(Rational&& other) noexcept : storage_(std::move(other.storage_)) {
        other.storage_ = internal::SmallStorage{};
    }

    inline Rational::~Rational() {
        if (is_lazy()) internal::decrement_ref(root_index());
    }

    inline Rational& Rational::operator=(const Rational& other) {
        if (this == &other) return *this;
        if (is_lazy()) internal::decrement_ref(root_index());
        storage_ = other.storage_;
        if (is_lazy()) internal::increment_ref(root_index());
        return *this;
    }

    inline Rational& Rational::operator=(Rational&& other) noexcept {
        if (this == &other) return *this;
        if (is_lazy()) internal::decrement_ref(root_index());
        storage_ = std::move(other.storage_);
        other.storage_ = internal::SmallStorage{};
        return *this;
    }

    // ----------------------------------------------------------------------------
    // State queries
    // ----------------------------------------------------------------------------
    inline bool Rational::is_immediate() const noexcept {
        return storage_.tag == internal::ValueType::Small || storage_.tag == internal::ValueType::Big;
    }

    inline bool Rational::is_lazy() const noexcept {
        return storage_.tag == internal::ValueType::Lazy;
    }

    // ----------------------------------------------------------------------------
    // Numerator/Denominator
    // ----------------------------------------------------------------------------
    inline Rational Rational::numerator() const {
        if (is_immediate()) {
            if (storage_.tag == internal::ValueType::Small) {
                internal::SmallStorage norm = storage_.storage.small;
                bool red = false;
                norm.normalize(red);
                return Rational(internal::to_dumb_int(norm.num));
            }
            else { // Big
                return Rational(storage_.storage.big.numerator());
            }
        }
        else {
            return eval().numerator();
        }
        throw std::logic_error("Rational::numerator: invalid state");
    }

    inline Rational Rational::denominator() const {
        if (is_immediate()) {
            if (storage_.tag == internal::ValueType::Small) {
                internal::SmallStorage norm = storage_.storage.small;
                bool red = false;
                norm.normalize(red);
                return Rational(internal::to_dumb_int(norm.den));
            }
            else { // Big
                return Rational(storage_.storage.big.denominator());
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
        return (storage_.tag == internal::ValueType::Small) ? &storage_.storage.small : nullptr;
    }

    inline const internal::BigStorage* Rational::as_big() const noexcept {
        return (storage_.tag == internal::ValueType::Big) ? &storage_.storage.big : nullptr;
    }

    inline int Rational::root_index() const {
        if (storage_.tag != internal::ValueType::Lazy) throw std::logic_error("root_index on immediate");
        return storage_.storage.lazy;
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

    inline Rational Rational::immediate() const {
        if (is_immediate()) return *this;
        const internal::Node& node = internal::pool.nodes[root_index()];
        if (node.op == internal::LazyOp::CONST) {
            internal::Value v = internal::pool.values[node.value_idx];
            return Rational(v);
        }
        return eval();
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
            return *this;
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
            return storage_;
        }
        return internal::evaluate(root_index());
    }

    // ----------------------------------------------------------------------------
    // Interval estimation
    // ----------------------------------------------------------------------------
    inline internal::Interval Rational::approx_interval() const noexcept {
        if (is_immediate()) {
            double d = internal::to_double(storage_);
            return internal::Interval(d);
        }
        return internal::pool.nodes[root_index()].approx;
    }

    // ----------------------------------------------------------------------------
    // Arithmetic operators – используют eager_add/eager_mul для immediate,
    // lazy – строят выражения.
    // ----------------------------------------------------------------------------
    inline Rational operator+(const Rational& a, const Rational& b) {
        if (internal::global_eager_mode || (a.is_immediate() && b.is_immediate())) {
            return eager_add(a, b);
        }
        // Собираем детей, расплющивая вложенные SUM
        std::vector<int> children;
        auto collect = [&](const Rational& r) {
            if (r.is_lazy()) {
                int idx = r.root_index();
                const auto& node = internal::pool.nodes[idx];
                if (node.op == internal::LazyOp::SUM && node.sum_children) {
                    for (int c : *node.sum_children) children.push_back(c);
                }
                else {
                    children.push_back(idx);
                }
            }
            else {
                int const_idx = internal::add_const(r.to_value());
                children.push_back(const_idx);
            }
            };
        collect(a);
        collect(b);
        int sum_idx = internal::make_sum_node(std::move(children));
        return Rational::from_lazy_index(sum_idx);
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
    // Compound assignment – используют inplace_add/inplace_mul для immediate
    // ----------------------------------------------------------------------------
// Новая версия operator+= (заменяет старую)
    inline Rational& operator+=(Rational& a, const Rational& b) {
        if (internal::global_eager_mode || (a.is_immediate() && b.is_immediate())) {
            inplace_add(a, b);
            return a;
        }
        // Ленивый случай
        if (!a.is_lazy()) {
            a = a.lazy();   // делаем ленивым
        }
        int a_idx = a.root_index();
        // Пытаемся мутировать a на месте (COW)
        if (internal::can_mutate_sum(a_idx)) {
            int b_idx = b.is_lazy() ? b.root_index() : internal::add_const(b.to_value());
            internal::append_to_sum(a_idx, b_idx);
            return a;
        }
        // COW: создаём новый узел SUM, объединяя детей a и b
        std::vector<int> children;
        const auto& a_node = internal::pool.nodes[a_idx];
        if (a_node.op == internal::LazyOp::SUM && a_node.sum_children) {
            for (int c : *a_node.sum_children) children.push_back(c);
        }
        else {
            children.push_back(a_idx);
        }
        int b_idx = b.is_lazy() ? b.root_index() : internal::add_const(b.to_value());
        children.push_back(b_idx);
        int new_sum_idx = internal::make_sum_node(std::move(children));
        // Заменяем a
        internal::decrement_ref(a_idx);
        a.storage_ = internal::Value(new_sum_idx);
        internal::increment_ref(new_sum_idx);
        return a;
    }

    inline Rational& operator-=(Rational& a, const Rational& b) {
        a = a - b;
        return a;
    }

    inline Rational& operator*=(Rational& a, const Rational& b) {
        if (internal::global_eager_mode || (a.is_immediate() && b.is_immediate())) {
            inplace_mul(a, b);
        }
        else {
            a = a * b;
        }
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
            return internal::to_string(r.storage_);
        }
        return internal::to_string(r.to_value());
    }

    // ----------------------------------------------------------------------------
    // Batch addition (оптимизировано: без лишней нормализации)
    // ----------------------------------------------------------------------------
    inline Rational batch_add(const std::vector<Rational>& terms) {
        if (terms.empty()) return Rational(0);

        bool all_immediate = std::all_of(terms.begin(), terms.end(),
            [](const Rational& r) { return r.is_immediate(); });

        if (all_immediate) {
            using internal::dumb_int;
            dumb_int common_denom(1);
            std::vector<dumb_int> nums;
            nums.reserve(terms.size());

            for (const Rational& term : terms) {
                if (term.as_small()) {
                    internal::SmallStorage s = *term.as_small();
                    // НЕ нормализуем – работаем с сырыми значениями
                    nums.push_back(internal::to_dumb_int(s.num));
                    dumb_int den = internal::to_dumb_int(s.den);
                    common_denom = boost::multiprecision::lcm(common_denom, den);
                }
                else if (term.as_big()) {
                    const auto& b = *term.as_big();
                    nums.push_back(b.numerator());
                    common_denom = boost::multiprecision::lcm(common_denom, b.denominator());
                }
                else {
                    all_immediate = false;
                    break;
                }
            }

            if (all_immediate) {
                dumb_int sum_num(0);
                for (size_t i = 0; i < terms.size(); ++i) {
                    dumb_int factor = common_denom;
                    if (terms[i].as_small()) {
                        internal::SmallStorage s = *terms[i].as_small();
                        // знаменатель уже сырой, но для factor/den нужно деление – корректно
                        factor /= internal::to_dumb_int(s.den);
                        sum_num += nums[i] * factor;
                    }
                    else if (terms[i].as_big()) {
                        const auto& b = *terms[i].as_big();
                        factor /= b.denominator();
                        sum_num += b.numerator() * factor;
                    }
                }
                dumb_int g = boost::multiprecision::gcd(sum_num, common_denom);
                sum_num /= g;
                common_denom /= g;

                if (internal::fits_in_int128(sum_num) && internal::fits_in_uint128(common_denom)) {
                    return Rational(
                        internal::dumb_int_to_int128(sum_num),
                        internal::dumb_int_to_uint128(common_denom)
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

    // ----------------------------------------------------------------------------
    // ExpressionRoot method implementations
    // ----------------------------------------------------------------------------
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
        int val_idx = internal::pool.add_value(eps_val);
        int node_idx = internal::get_unary_node(internal::LazyOp::PI, -1, val_idx);
        return ExpressionRoot(node_idx);
    }

    inline ExpressionRoot ExpressionRoot::e(const Rational& eps) {
        internal::Value eps_val = eps.to_value();
        int val_idx = internal::pool.add_value(eps_val);
        int node_idx = internal::get_unary_node(internal::LazyOp::E, -1, val_idx);
        return ExpressionRoot(node_idx);
    }

    inline ExpressionRoot ExpressionRoot::pow(const ExpressionRoot& exponent, const Rational& eps) const {
        internal::Value eps_val = eps.to_value();
        return make_binary_with_eps(internal::LazyOp::POW, *this, exponent, eps_val);
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

    // ----------------------------------------------------------------------------
    // convert_to<T>
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
                bool red = false;
                norm.normalize(red);
                absl::int128 num = norm.num;
                if (num < std::numeric_limits<int>::min() || num > std::numeric_limits<int>::max()) {
                    throw std::overflow_error("Rational::convert_to<int>: value out of int range");
                }
                return static_cast<int>(num);
            }
            else if (auto* b = v.as_big()) {
                const auto& num = b->numerator();
                if (num > std::numeric_limits<int>::max()) {
                    throw std::overflow_error("Rational::convert_to<int>: value out of range (positive)");
                }
                if (num < std::numeric_limits<int>::min()) {
                    throw std::overflow_error("Rational::convert_to<int>: value out of range (negative)");
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
                bool red = false;
                norm.normalize(red);
                absl::int128 num = norm.num;
                if (num < std::numeric_limits<long long>::min() || num > std::numeric_limits<long long>::max()) {
                    throw std::overflow_error("Rational::convert_to<long long>: value out of long long range");
                }
                return static_cast<long long>(num);
            }
            else if (auto* b = v.as_big()) {
                const auto& num = b->numerator();
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
        else if constexpr (std::is_same_v<T, internal::dumb_int>) {
            Rational v = eval();
            if (v.denominator() != 1) {
                throw std::domain_error("Rational::convert_to<dumb_int>: not an integer");
            }
            if (auto* s = v.as_small()) {
                internal::SmallStorage norm = *s;
                bool red = false;
                norm.normalize(red);
                return internal::to_dumb_int(norm.num);
            }
            else if (auto* b = v.as_big()) {
                return b->numerator();
            }
            throw std::logic_error("Rational::convert_to<dumb_int>: invalid state");
        }
        else {
            static_assert(sizeof(T) == 0, "convert_to not supported for this type");
        }
    }

} // namespace delta