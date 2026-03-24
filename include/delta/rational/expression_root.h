#pragma once

#include "rational_fwd.h"
#include "interval.h"
#include "storage.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <cmath>       // for sin, cos, sqrt, exp, log, acos
#include <limits>      // for numeric_limits

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif
#ifndef M_E
#define M_E 2.71828182845904523536
#endif

namespace delta::internal {

    inline constexpr int MAX_LAZY_DEPTH = 1000;

    enum class LazyOp : uint8_t {
        CONST, ADD, SUB, MUL, DIV, NEG,
        SQRT, EXP, LOG, SIN, COS, ACOS,
        PI, E
    };

    struct Node {
        LazyOp op;
        int child0;
        int child1;
        int value_idx;
        Interval approx;
        int depth;

        Node(LazyOp op, int c0 = -1, int c1 = -1, int val_idx = -1,
            Interval approx = Interval(), int depth = 0)
            : op(op), child0(c0), child1(c1), value_idx(val_idx),
            approx(approx), depth(depth) {
        }
    };

    class ExpressionRoot {
    public:
        // Public constructors (as per spec)
        explicit ExpressionRoot(Value val);
        ExpressionRoot(LazyOp op, int c0, int c1 = -1,
            Interval approx = Interval(), int depth = 0);

        // Read-only access
        const std::vector<Node>& nodes() const noexcept { return nodes_; }
        const std::vector<Value>& values() const noexcept { return values_; }
        int root_index() const noexcept { return root_index_; }
        std::size_t hash() const noexcept { return hash_; }

        // ------------------------------------------------------------------------
        // Methods for building new lazy trees (immutable)
        // ------------------------------------------------------------------------
        ExpressionRoot add(const ExpressionRoot& other) const;
        ExpressionRoot sub(const ExpressionRoot& other) const;
        ExpressionRoot mul(const ExpressionRoot& other) const;
        ExpressionRoot div(const ExpressionRoot& other) const;
        ExpressionRoot neg() const;

        // Transcendental functions with eps (saved as Value in the node)
        ExpressionRoot sqrt(const Rational& eps) const;
        ExpressionRoot exp(const Rational& eps) const;
        ExpressionRoot log(const Rational& eps) const;
        ExpressionRoot sin(const Rational& eps) const;
        ExpressionRoot cos(const Rational& eps) const;
        ExpressionRoot acos(const Rational& eps) const;

        // Constants
        static ExpressionRoot pi(const Rational& eps);
        static ExpressionRoot e(const Rational& eps);

        // Simplification and evaluation
        ExpressionRoot simplify() const;
        Value eval() const;

        // Structural equality
        friend bool structurally_equal(const ExpressionRoot& a, const ExpressionRoot& b);

    private:
        // Private constructor for building from existing buffers
        ExpressionRoot(std::vector<Node> nodes, std::vector<Value> values, int root_index);

        // Internal helpers
        int add_node(Node node);
        int add_value(Value val);
        static Interval compute_interval(LazyOp op, const Interval& a, const Interval& b = Interval());
        static std::size_t compute_hash(const std::vector<Node>& nodes,
            const std::vector<Value>& values,
            int root);

        // Helper to automatically simplify if depth exceeds MAX_LAZY_DEPTH
        static ExpressionRoot auto_simplify(const ExpressionRoot& a, const ExpressionRoot& b, LazyOp op);
        static ExpressionRoot auto_simplify(const ExpressionRoot& a, LazyOp op);

        std::vector<Node> nodes_;
        std::vector<Value> values_;
        int root_index_;
        std::size_t hash_;

        // Allow simplify function to use private constructor
        friend ExpressionRoot simplify(const ExpressionRoot& root);
    };

    bool structurally_equal(const ExpressionRoot& a, const ExpressionRoot& b);

} // namespace delta::internal

// ----------------------------------------------------------------------------
// Implementation of compute_interval (inline, so simplify.h can see it)
// ----------------------------------------------------------------------------

namespace delta::internal {

    inline Interval ExpressionRoot::compute_interval(LazyOp op, const Interval& a, const Interval& b) {
        using std::cos;
        using std::sin;
        using std::sqrt;
        using std::exp;
        using std::log;
        using std::acos;

        switch (op) {
        case LazyOp::CONST:
            return a; // not used for CONST directly
        case LazyOp::ADD:
            return a + b;
        case LazyOp::SUB:
            return a - b;
        case LazyOp::MUL:
            return a * b;
        case LazyOp::DIV:
            return a / b;
        case LazyOp::NEG:
            return -a;
        case LazyOp::SQRT: {
            if (a.lower() < 0) {
                // Could be negative, but sqrt domain error will be caught later.
                // For interval, we use [0, upper bound] if upper bound >= 0
                double lo = a.lower() < 0 ? 0.0 : sqrt(a.lower());
                double hi = sqrt(a.upper());
                return Interval(lo, hi);
            }
            return Interval(sqrt(a.lower()), sqrt(a.upper()));
        }
        case LazyOp::EXP: {
            double lo = exp(a.lower());
            double hi = exp(a.upper());
            return Interval(lo, hi);
        }
        case LazyOp::LOG: {
            if (a.upper() <= 0) {
                // domain error, but interval might be used for comparisons; we can use [-inf, +inf]?
                // Actually we should handle only when a.lower() > 0, else return [-inf, +inf] to be safe.
                if (a.upper() <= 0) return Interval(-std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity());
            }
            double lo = log(a.lower());
            double hi = log(a.upper());
            return Interval(lo, hi);
        }
        case LazyOp::SIN: {
            // For sin, we need to handle periodicity; simple bounds: [-1,1]
            // But we can get tighter if interval is small.
            if (a.width() >= 2 * M_PI) return Interval(-1.0, 1.0);
            // Evaluate sin at endpoints and at any critical points within the interval.
            // Critical points are multiples of pi/2.
            double lo = a.lower();
            double hi = a.upper();
            double min_val = std::min(sin(lo), sin(hi));
            double max_val = std::max(sin(lo), sin(hi));
            // Find nearest critical points: floor(lo / (pi/2)) etc.
            double step = M_PI_2;
            int start = static_cast<int>(std::floor(lo / step));
            int end = static_cast<int>(std::ceil(hi / step));
            for (int k = start; k <= end; ++k) {
                double x = k * step;
                if (x >= lo && x <= hi) {
                    double y = sin(x);
                    min_val = std::min(min_val, y);
                    max_val = std::max(max_val, y);
                }
            }
            // Expand outward with nextafter to ensure inclusion
            return Interval(std::nextafter(min_val, -std::numeric_limits<double>::infinity()),
                std::nextafter(max_val, std::numeric_limits<double>::infinity()));
        }
        case LazyOp::COS: {
            if (a.width() >= 2 * M_PI) return Interval(-1.0, 1.0);
            double lo = a.lower();
            double hi = a.upper();
            double min_val = std::min(cos(lo), cos(hi));
            double max_val = std::max(cos(lo), cos(hi));
            double step = M_PI_2;
            int start = static_cast<int>(std::floor(lo / step));
            int end = static_cast<int>(std::ceil(hi / step));
            for (int k = start; k <= end; ++k) {
                double x = k * step;
                if (x >= lo && x <= hi) {
                    double y = cos(x);
                    min_val = std::min(min_val, y);
                    max_val = std::max(max_val, y);
                }
            }
            return Interval(std::nextafter(min_val, -std::numeric_limits<double>::infinity()),
                std::nextafter(max_val, std::numeric_limits<double>::infinity()));
        }
        case LazyOp::ACOS: {
            if (a.lower() < -1 || a.upper() > 1) return Interval(-std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity());
            double lo = acos(a.upper()); // acos is decreasing
            double hi = acos(a.lower());
            return Interval(lo, hi);
        }
        case LazyOp::PI:
            return Interval(M_PI);
        case LazyOp::E:
            return Interval(M_E);
        default:
            return Interval(); // fallback
        }
    }

} // namespace delta::internal