// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

#ifndef DELTA_COMPLEX_GAUSS_QI_IMPL_H
#define DELTA_COMPLEX_GAUSS_QI_IMPL_H

#include "gauss_qi.h"
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace delta {

    // ----------------------------------------------------------------------------
    // Constructors
    // ----------------------------------------------------------------------------
    inline GaussQi::GaussQi(const Rational& re) : re_(re), im_(0) {}

    inline GaussQi::GaussQi(const Rational& re, const Rational& im)
        : re_(re), im_(im) {
    }

    inline GaussQi::GaussQi(Rational&& re, Rational&& im)
        : re_(std::move(re)), im_(std::move(im)) {
    }

    inline GaussQi::GaussQi(long long re) : re_(re), im_(0) {}

    inline GaussQi::GaussQi(long long re, long long im)
        : re_(re), im_(im) {
    }

    inline GaussQi::GaussQi(const std::string& str) {
        // Упрощённый парсер: (re,im) или re+imi
        std::string s = str;
        s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());

        if (s.front() == '(' && s.back() == ')') {
            s = s.substr(1, s.size() - 2);
            size_t comma = s.find(',');
            if (comma != std::string::npos) {
                re_ = Rational(s.substr(0, comma));
                im_ = Rational(s.substr(comma + 1));
                return;
            }
        }

        // Попробуем найти 'i'
        size_t i_pos = s.find('i');
        if (i_pos != std::string::npos) {
            // отделяем действительную и мнимую части (упрощённо)
            // Для реального кода нужен полноценный парсер, здесь оставим заглушку
            throw std::runtime_error("GaussQi string parser: format with 'i' not fully implemented, use (re,im)");
        }
        else {
            re_ = Rational(s);
            im_ = 0;
        }
    }

    inline GaussQi::GaussQi(const std::string& re_str, const std::string& im_str)
        : re_(re_str), im_(im_str) {
    }

    // ----------------------------------------------------------------------------
    // Accessors
    // ----------------------------------------------------------------------------
    inline const Rational& GaussQi::real() const noexcept { return re_; }
    inline const Rational& GaussQi::imag() const noexcept { return im_; }
    inline void GaussQi::real(const Rational& r) { re_ = r; }
    inline void GaussQi::imag(const Rational& i) { im_ = i; }

    // ----------------------------------------------------------------------------
    // Norm and conjugate
    // ----------------------------------------------------------------------------
    inline Rational GaussQi::norm() const {
        return re_ * re_ + im_ * im_;
    }

    inline GaussQi GaussQi::conj() const {
        return GaussQi(re_, -im_);
    }

    // ----------------------------------------------------------------------------
    // String representation
    // ----------------------------------------------------------------------------
    inline std::string GaussQi::to_string() const {
        if (im_ == Rational(0)) {
            return re_.to_string();
        }
        std::ostringstream oss;
        oss << "(" << re_.to_string() << "," << im_.to_string() << ")";
        return oss.str();
    }

    inline std::pair<double, double> GaussQi::to_double() const {
        return { re_.to_double(), im_.to_double() };
    }

    // ----------------------------------------------------------------------------
    // Compound assignments
    // ----------------------------------------------------------------------------
    inline GaussQi& GaussQi::operator+=(const GaussQi& rhs) {
        re_ += rhs.re_;
        im_ += rhs.im_;
        return *this;
    }

    inline GaussQi& GaussQi::operator-=(const GaussQi& rhs) {
        re_ -= rhs.re_;
        im_ -= rhs.im_;
        return *this;
    }

    inline GaussQi& GaussQi::operator*=(const GaussQi& rhs) {
        Rational new_re = re_ * rhs.re_ - im_ * rhs.im_;
        Rational new_im = re_ * rhs.im_ + im_ * rhs.re_;
        re_ = std::move(new_re);
        im_ = std::move(new_im);
        return *this;
    }

    inline GaussQi& GaussQi::operator/=(const GaussQi& rhs) {
        Rational denom = rhs.norm();
        if (denom == 0)
            throw std::domain_error("GaussQi division by zero");
        Rational new_re = (re_ * rhs.re_ + im_ * rhs.im_) / denom;
        Rational new_im = (im_ * rhs.re_ - re_ * rhs.im_) / denom;
        re_ = std::move(new_re);
        im_ = std::move(new_im);
        return *this;
    }

    inline GaussQi& GaussQi::operator+=(const Rational& rhs) {
        re_ += rhs;
        return *this;
    }

    inline GaussQi& GaussQi::operator-=(const Rational& rhs) {
        re_ -= rhs;
        return *this;
    }

    inline GaussQi& GaussQi::operator*=(const Rational& rhs) {
        re_ *= rhs;
        im_ *= rhs;
        return *this;
    }

    inline GaussQi& GaussQi::operator/=(const Rational& rhs) {
        if (rhs == 0)
            throw std::domain_error("GaussQi division by scalar zero");
        re_ /= rhs;
        im_ /= rhs;
        return *this;
    }

    // ----------------------------------------------------------------------------
    // Binary arithmetic operators
    // ----------------------------------------------------------------------------
    inline GaussQi operator+(const GaussQi& lhs, const GaussQi& rhs) {
        GaussQi result = lhs;
        result += rhs;
        return result;
    }

    inline GaussQi operator-(const GaussQi& lhs, const GaussQi& rhs) {
        GaussQi result = lhs;
        result -= rhs;
        return result;
    }

    inline GaussQi operator*(const GaussQi& lhs, const GaussQi& rhs) {
        GaussQi result = lhs;
        result *= rhs;
        return result;
    }

    inline GaussQi operator/(const GaussQi& lhs, const GaussQi& rhs) {
        GaussQi result = lhs;
        result /= rhs;
        return result;
    }

    inline GaussQi operator+(const GaussQi& lhs, const Rational& rhs) {
        GaussQi result = lhs;
        result += rhs;
        return result;
    }

    inline GaussQi operator+(const Rational& lhs, const GaussQi& rhs) {
        return rhs + lhs;
    }

    inline GaussQi operator-(const GaussQi& lhs, const Rational& rhs) {
        GaussQi result = lhs;
        result -= rhs;
        return result;
    }

    inline GaussQi operator-(const Rational& lhs, const GaussQi& rhs) {
        return GaussQi(lhs) - rhs;
    }

    inline GaussQi operator*(const GaussQi& lhs, const Rational& rhs) {
        GaussQi result = lhs;
        result *= rhs;
        return result;
    }

    inline GaussQi operator*(const Rational& lhs, const GaussQi& rhs) {
        return rhs * lhs;
    }

    inline GaussQi operator/(const GaussQi& lhs, const Rational& rhs) {
        GaussQi result = lhs;
        result /= rhs;
        return result;
    }

    inline GaussQi operator/(const Rational& lhs, const GaussQi& rhs) {
        return GaussQi(lhs) / rhs;
    }

    inline GaussQi operator-(const GaussQi& z) {
        return GaussQi(-z.re_, -z.im_);
    }
    // ----------------------------------------------------------------------------
// Операторы с целыми типами (int, long long) – явное использование explicit конструктора
// ----------------------------------------------------------------------------
    inline GaussQi operator+(long long lhs, const GaussQi& rhs) {
        return GaussQi(lhs) + rhs;
    }
    inline GaussQi operator-(long long lhs, const GaussQi& rhs) {
        return GaussQi(lhs) - rhs;
    }
    inline GaussQi operator*(long long lhs, const GaussQi& rhs) {
        return GaussQi(lhs) * rhs;
    }
    inline GaussQi operator/(long long lhs, const GaussQi& rhs) {
        return GaussQi(lhs) / rhs;
    }

    inline GaussQi operator+(const GaussQi& lhs, long long rhs) {
        return lhs + GaussQi(rhs);
    }
    inline GaussQi operator-(const GaussQi& lhs, long long rhs) {
        return lhs - GaussQi(rhs);
    }
    inline GaussQi operator*(const GaussQi& lhs, long long rhs) {
        return lhs * GaussQi(rhs);
    }
    inline GaussQi operator/(const GaussQi& lhs, long long rhs) {
        return lhs / GaussQi(rhs);
    }

    // Для int – через long long
    inline GaussQi operator+(int lhs, const GaussQi& rhs) {
        return GaussQi(static_cast<long long>(lhs)) + rhs;
    }
    inline GaussQi operator-(int lhs, const GaussQi& rhs) {
        return GaussQi(static_cast<long long>(lhs)) - rhs;
    }
    inline GaussQi operator*(int lhs, const GaussQi& rhs) {
        return GaussQi(static_cast<long long>(lhs)) * rhs;
    }
    inline GaussQi operator/(int lhs, const GaussQi& rhs) {
        return GaussQi(static_cast<long long>(lhs)) / rhs;
    }

    inline GaussQi operator+(const GaussQi& lhs, int rhs) {
        return lhs + GaussQi(static_cast<long long>(rhs));
    }
    inline GaussQi operator-(const GaussQi& lhs, int rhs) {
        return lhs - GaussQi(static_cast<long long>(rhs));
    }
    inline GaussQi operator*(const GaussQi& lhs, int rhs) {
        return lhs * GaussQi(static_cast<long long>(rhs));
    }
    inline GaussQi operator/(const GaussQi& lhs, int rhs) {
        return lhs / GaussQi(static_cast<long long>(rhs));
    }
    // ----------------------------------------------------------------------------
    // Comparisons
    // ----------------------------------------------------------------------------
    inline bool operator==(const GaussQi& lhs, const GaussQi& rhs) {
        return lhs.re_ == rhs.re_ && lhs.im_ == rhs.im_;
    }

    inline bool operator!=(const GaussQi& lhs, const GaussQi& rhs) {
        return !(lhs == rhs);
    }

    // ----------------------------------------------------------------------------
    // Output stream
    // ----------------------------------------------------------------------------
    inline std::ostream& operator<<(std::ostream& os, const GaussQi& z) {
        os << z.to_string();
        return os;
    }

} // namespace delta

#endif // DELTA_COMPLEX_GAUSS_QI_IMPL_H