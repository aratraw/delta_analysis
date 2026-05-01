// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// interval.h
// -----------------------------------------------------------------------------
// INTERVAL ARITHMETIC FOR FAST APPROXIMATE COMPARISONS
// -----------------------------------------------------------------------------
// This lightweight interval arithmetic is used on demand – only when logical
// comparisons (==, <, >, etc.) involve lazy expressions that cannot be resolved
// by hash equality alone.
//
// No overhead is incurred unless a comparison actually evaluates intervals.
// The intervals are based on double precision, which is sufficient for
// reliably separating non‑overlapping values. When intervals overlap, we fall
// back to exact rational evaluation.
//
// Room for Improvement: can be made more accurate by implementing:
//   - Better rounding control (currently uses std::nextafter for outward rounding)
//   - Affine arithmetic for tighter bounds
//   - Higher‑precision interval endpoints (e.g., using cpp_dec_float_100)
// However, the current implementation is deliberately simple and fast,
// and has proven sufficient for all practical use cases.
// -----------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace delta::internal {

    class Interval {
    public:
        constexpr Interval() noexcept : lo(0.0), hi(0.0) {}
        constexpr explicit Interval(double x) noexcept : lo(x), hi(x) {}
        constexpr Interval(double l, double h) noexcept
            : lo(l), hi(h)
        {
            if (lo > hi) std::swap(lo, hi);
        }

        constexpr double lower() const noexcept { return lo; }
        constexpr double upper() const noexcept { return hi; }
        constexpr double width() const noexcept { return hi - lo; }

        // Arithmetic operations with outward rounding using nextafter.
        // This guarantees that the true result is contained within the interval.
        // However, due to double precision limits, the bounds may be slightly
        // wider than mathematically necessary.

        Interval operator+(const Interval& other) const noexcept
        {
            double raw_lo = lo + other.lo;
            double raw_hi = hi + other.hi;
            return Interval(
                std::nextafter(raw_lo, -std::numeric_limits<double>::infinity()),
                std::nextafter(raw_hi, std::numeric_limits<double>::infinity())
            );
        }

        Interval operator-(const Interval& other) const noexcept
        {
            double raw_lo = lo - other.hi;
            double raw_hi = hi - other.lo;
            return Interval(
                std::nextafter(raw_lo, -std::numeric_limits<double>::infinity()),
                std::nextafter(raw_hi, std::numeric_limits<double>::infinity())
            );
        }

        Interval operator*(const Interval& other) const noexcept
        {
            double a = lo * other.lo;
            double b = lo * other.hi;
            double c = hi * other.lo;
            double d = hi * other.hi;
            double raw_lo = std::min({ a, b, c, d });
            double raw_hi = std::max({ a, b, c, d });
            return Interval(
                std::nextafter(raw_lo, -std::numeric_limits<double>::infinity()),
                std::nextafter(raw_hi, std::numeric_limits<double>::infinity())
            );
        }

        Interval operator/(const Interval& other) const
        {
            // Division by an interval containing zero -> [-∞, +∞]
            if (other.lo <= 0.0 && other.hi >= 0.0) {
                return Interval(
                    -std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity()
                );
            }
            // Compute the four possible quotients
            double a = lo / other.lo;
            double b = lo / other.hi;
            double c = hi / other.lo;
            double d = hi / other.hi;
            double raw_lo = std::min({ a, b, c, d });
            double raw_hi = std::max({ a, b, c, d });
            return Interval(
                std::nextafter(raw_lo, -std::numeric_limits<double>::infinity()),
                std::nextafter(raw_hi, std::numeric_limits<double>::infinity())
            );
        }

        Interval operator-() const noexcept
        {
            double raw_lo = -hi;
            double raw_hi = -lo;
            return Interval(
                std::nextafter(raw_lo, -std::numeric_limits<double>::infinity()),
                std::nextafter(raw_hi, std::numeric_limits<double>::infinity())
            );
        }

        // Comparisons (without rounding) – these are exact for the interval bounds.
        // If intervals do not overlap, the comparison is definitive.
        constexpr bool operator<(const Interval& other) const noexcept
        {
            return hi < other.lo;
        }
        constexpr bool operator>(const Interval& other) const noexcept
        {
            return lo > other.hi;
        }
        constexpr bool operator<=(const Interval& other) const noexcept
        {
            return hi <= other.lo;
        }
        constexpr bool operator>=(const Interval& other) const noexcept
        {
            return lo >= other.hi;
        }
        constexpr bool operator==(const Interval& other) const noexcept
        {
            return lo == other.lo && hi == other.hi;
        }

        // Returns true if the two intervals have any common point.
        // If false is returned, the values are guaranteed to be separate.
        constexpr bool overlaps(const Interval& other) const noexcept
        {
            return !(hi < other.lo || other.hi < lo);
        }

        static constexpr Interval zero() noexcept { return Interval(0.0); }
        static constexpr Interval one() noexcept { return Interval(1.0); }

    private:
        double lo, hi;
    };

} // namespace delta::internal