// include/delta/rational/interval.h
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace delta::internal {

    /**
     * @brief Интервал с направленным округлением.
     *
     * Представляет число как отрезок [lo, hi], где lo <= hi.
     * Арифметические операции гарантированно расширяют интервал так,
     * чтобы он содержал точный результат.
     */
    class Interval {
        double lo;
        double hi;

    public:
        // -------------------------------------------------------------------------
        // Конструкторы
        // -------------------------------------------------------------------------
        Interval() : lo(0.0), hi(0.0) {}
        explicit Interval(double x) : lo(x), hi(x) {}
        Interval(double l, double h) : lo(l), hi(h) {
            if (lo > hi) std::swap(lo, hi);
        }

        // -------------------------------------------------------------------------
        // Доступ к границам
        // -------------------------------------------------------------------------
        double lower() const { return lo; }
        double upper() const { return hi; }
        double width() const { return hi - lo; }

        // -------------------------------------------------------------------------
        // Арифметические операции с направленным округлением
        // -------------------------------------------------------------------------
        Interval operator+(const Interval& other) const {
            double new_lo = lo + other.lo;
            double new_hi = hi + other.hi;
            // Расширяем наружу, чтобы гарантировать покрытие точного результата
            new_lo = std::nextafter(new_lo, -std::numeric_limits<double>::infinity());
            new_hi = std::nextafter(new_hi, std::numeric_limits<double>::infinity());
            return Interval(new_lo, new_hi);
        }

        Interval operator-(const Interval& other) const {
            double new_lo = lo - other.hi;
            double new_hi = hi - other.lo;
            new_lo = std::nextafter(new_lo, -std::numeric_limits<double>::infinity());
            new_hi = std::nextafter(new_hi, std::numeric_limits<double>::infinity());
            return Interval(new_lo, new_hi);
        }

        Interval operator*(const Interval& other) const {
            // Для умножения рассматриваем все 4 комбинации границ
            double a = lo * other.lo;
            double b = lo * other.hi;
            double c = hi * other.lo;
            double d = hi * other.hi;
            double new_lo = std::min({ a, b, c, d });
            double new_hi = std::max({ a, b, c, d });
            new_lo = std::nextafter(new_lo, -std::numeric_limits<double>::infinity());
            new_hi = std::nextafter(new_hi, std::numeric_limits<double>::infinity());
            return Interval(new_lo, new_hi);
        }

        Interval operator/(const Interval& other) const {
            // Деление на ноль недопустимо – вызывающий должен убедиться, что интервал не содержит нуля
            // Для простоты расширяем бесконечно, если other содержит 0.
            if (other.lo <= 0.0 && other.hi >= 0.0) {
                return Interval(-std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity());
            }
            double a = lo / other.lo;
            double b = lo / other.hi;
            double c = hi / other.lo;
            double d = hi / other.hi;
            double new_lo = std::min({ a, b, c, d });
            double new_hi = std::max({ a, b, c, d });
            new_lo = std::nextafter(new_lo, -std::numeric_limits<double>::infinity());
            new_hi = std::nextafter(new_hi, std::numeric_limits<double>::infinity());
            return Interval(new_lo, new_hi);
        }

        Interval operator-() const {
            double new_lo = -hi;
            double new_hi = -lo;
            new_lo = std::nextafter(new_lo, -std::numeric_limits<double>::infinity());
            new_hi = std::nextafter(new_hi, std::numeric_limits<double>::infinity());
            return Interval(new_lo, new_hi);
        }

        // -------------------------------------------------------------------------
        // Логические предикаты (без расширения интервалов)
        // -------------------------------------------------------------------------
        bool operator<(const Interval& other) const {
            return hi < other.lo;
        }
        bool operator>(const Interval& other) const {
            return lo > other.hi;
        }
        bool operator<=(const Interval& other) const {
            return hi <= other.lo;
        }
        bool operator>=(const Interval& other) const {
            return lo >= other.hi;
        }
        bool operator==(const Interval& other) const {
            return lo == other.lo && hi == other.hi;
        }

        /**
         * @brief Проверяет, пересекаются ли два интервала.
         */
        bool overlaps(const Interval& other) const {
            return !(hi < other.lo || lo > other.hi);
        }

        // -------------------------------------------------------------------------
        // Статические константы
        // -------------------------------------------------------------------------
        static Interval zero() { return Interval(0.0); }
        static Interval one() { return Interval(1.0); }
    };

} // namespace delta::internal