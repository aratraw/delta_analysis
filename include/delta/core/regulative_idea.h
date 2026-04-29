// include/delta/core/regulative_idea.h
#pragma once

#include <concepts>
#include <functional>
#include <array>
#include <cmath>
#include <Eigen/Dense>
#include "rational.h"

namespace delta {

    // -----------------------------------------------------------------------------
    // Address concepts
    // -----------------------------------------------------------------------------

    /**
     * @concept Address
     * @brief The most basic requirement for an address type.
     *
     * An address must be copyable and equality comparable.
     * This is the foundation for all regulative ideas.
     */
    template<typename T>
    concept Address = std::copyable<T> && std::equality_comparable<T>;

    /**
     * @concept SubtractableAddress
     * @brief Extends Address with subtraction.
     *
     * Addresses that support subtraction (difference) are needed for grids
     * where gaps are computed as `a - b`. The result must be convertible to the address type.
     */
    template<typename T>
    concept SubtractableAddress = Address<T> && requires(T a, T b) {
        { a - b } -> std::convertible_to<T>;
    };

    /**
     * @concept AddableAddress
     * @brief Extends Address with addition.
     *
     * Required for linear combinations and scaling.
     */
    template<typename T>
    concept AddableAddress = Address<T> && requires(T a, T b) {
        { a + b } -> std::convertible_to<T>;
    };

    /**
     * @concept ScalableAddress
     * @brief Extends AddableAddress with scalar multiplication.
     *
     * @tparam T The address type.
     * @tparam Scalar The scalar type (e.g., Rational, double).
     *
     * The expression `s * a` must produce a value convertible to `T`.
     */
    template<typename T, typename Scalar>
    concept ScalableAddress = AddableAddress<T> && requires(T a, Scalar s) {
        { s* a } -> std::convertible_to<T>;
    };

    /**
     * @concept LinearAddress
     * @brief Combines addition and scalar multiplication.
     *
     * This is the typical concept for addresses that behave like vectors over a scalar field.
     * The default scalar type is `Rational`.
     */
    template<typename T, typename Scalar = Rational>
    concept LinearAddress = AddableAddress<T> && ScalableAddress<T, Scalar>;

    // -----------------------------------------------------------------------------
    // Betweenness and metric concepts
    // -----------------------------------------------------------------------------

    /**
     * @concept Betweenness
     * @brief A ternary relation indicating that one address lies between two others.
     *
     * @tparam B The betweenness functor type.
     * @tparam Addr The address type.
     *
     * The expression `b(x, y, z)` must return a value convertible to `bool`.
     * Typical semantics: returns `true` iff `y` is between `x` and `z` according to the regulative idea.
     */
    template<typename B, typename Addr>
    concept Betweenness = requires(B b, const Addr & x, const Addr & y, const Addr & z) {
        { b(x, y, z) } -> std::convertible_to<bool>;
    };

    /**
     * @concept Metric
     * @brief A metric (distance function) on addresses.
     *
     * @tparam M The metric functor type.
     * @tparam Addr The address type.
     *
     * The expression `m(a, b)` must return a value that models `std::regular` (i.e., copyable, default constructible, equality comparable).
     * Typically the return type is a scalar like `Rational` or `double`.
     */
    template<typename M, typename Addr>
    concept Metric = requires(M m, const Addr & a, const Addr & b) {
        { m(a, b) } -> std::regular;
    };

    // -----------------------------------------------------------------------------
    // Core regulative idea structure
    // -----------------------------------------------------------------------------

    /**
     * @struct RegulativeIdea
     * @brief Bundles a betweenness relation and a metric into a regulative idea.
     *
     * A regulative idea defines the geometric structure on the address space:
     * how points are ordered (betweenness) and how distances are measured (metric).
     *
     * @tparam Addr The address type.
     * @tparam B    The betweenness functor type (must satisfy Betweenness<B, Addr>).
     * @tparam M    The metric functor type (must satisfy Metric<M, Addr>).
     */
    template<typename Addr, typename B, typename M>
        requires Betweenness<B, Addr>&& Metric<M, Addr>
    struct RegulativeIdea {
        using address_type = Addr;
        using betweenness_type = B;
        using metric_type = M;

        B betweenness;   ///< The betweenness relation.
        M metric;        ///< The metric.

        RegulativeIdea() = default;
        RegulativeIdea(const B& b, const M& m) : betweenness(b), metric(m) {}
    };

    // -----------------------------------------------------------------------------
    // Classical instances for linear order and Euclidean metric
    // -----------------------------------------------------------------------------

    /**
     * @struct LessBetweenness
     * @brief Betweenness for a strict total order: `x < y && y < z`.
     *
     * Works for any type that supports `operator<`.
     */
    struct LessBetweenness {
        template<typename T>
        bool operator()(const T& x, const T& y, const T& z) const {
            return x < y && y < z;
        }
    };
    static_assert(Betweenness<LessBetweenness, int>);

    /**
     * @struct EuclideanMetric
     * @brief Euclidean (absolute) distance: |a - b| for scalars, norm for vectors/matrices.
     */
    struct EuclideanMetric {
        // Для арифметических типов (int, size_t, double и т.д.)
        template<typename T>
        std::enable_if_t<std::is_arithmetic_v<T>, T>
            operator()(const T& a, const T& b) const {
            using std::abs;
            return abs(a - b);
        }

        // Для Rational
        Rational operator()(const Rational& a, const Rational& b) const {
            using delta::abs;
            return abs(a - b);
        }

        // Для Eigen-векторов и матриц
        template<typename Derived>
        typename Derived::Scalar operator()(const Eigen::MatrixBase<Derived>& a,
            const Eigen::MatrixBase<Derived>& b) const {
            return (a - b).norm();
        }

        // Для std::array<T, N>, где T — арифметический тип
        template<typename T, std::size_t N>
        std::enable_if_t<std::is_arithmetic_v<T>, T>
            operator()(const std::array<T, N>& a, const std::array<T, N>& b) const {
            T sum_sq{ 0 };
            for (std::size_t i = 0; i < N; ++i) {
                T diff = a[i] - b[i];
                sum_sq = sum_sq + diff * diff;
            }
            using std::sqrt;
            return sqrt(sum_sq);
        }

        // Для std::array<Rational, N> (специализация для Rational)
        template<std::size_t N>
        Rational operator()(const std::array<Rational, N>& a, const std::array<Rational, N>& b) const {
            Rational sum_sq{ 0 };
            for (std::size_t i = 0; i < N; ++i) {
                Rational diff = a[i] - b[i];
                sum_sq = sum_sq + diff * diff;
            }
            using delta::sqrt;
            return sqrt(sum_sq);
        }
    };
    static_assert(Metric<EuclideanMetric, int>);
    static_assert(Metric<EuclideanMetric, Rational>);
    static_assert(Metric<EuclideanMetric, Eigen::Vector2d>);

    /**
     * @struct LinearBetweenness
     * @brief Betweenness for a linear order that is not necessarily directed.
     *
     * Returns `true` if `y` lies between `x` and `z` in the usual sense,
     * i.e., either `x < y < z` or `z < y < x`. Requires `std::totally_ordered<T>`.
     */
    template<typename T>
    struct LinearBetweenness {
        bool operator()(const T& x, const T& y, const T& z) const {
            if constexpr (std::totally_ordered<T>) {
                return (x < y && y < z) || (z < y && y < x);
            }
            else {
                static_assert(std::totally_ordered<T>, "LinearBetweenness requires totally ordered types");
            }
        }
    };

    /**
     * @struct TreeBetweenness
     * @brief Betweenness for binary tree addresses (strings of '0' and '1').
     *
     * `y` is between `x` and `z` if it is a common ancestor (lowest common ancestor)
     * or if it lies on the path from one node to the other.
     */
    struct TreeBetweenness {
        bool operator()(const std::string& x, const std::string& y, const std::string& z) const {
            size_t lcp_xz = 0;
            while (lcp_xz < x.size() && lcp_xz < z.size() && x[lcp_xz] == z[lcp_xz]) ++lcp_xz;
            std::string lca = x.substr(0, lcp_xz);
            if (y == lca) return true;
            if (y.size() > lcp_xz && y.substr(0, lcp_xz) == lca) {
                if (x.size() >= y.size() && x.substr(0, y.size()) == y) return true;
                if (z.size() >= y.size() && z.substr(0, y.size()) == y) return true;
            }
            return false;
        }
    };

    // -----------------------------------------------------------------------------
    // Additional metrics
    // -----------------------------------------------------------------------------

    /**
     * @struct FrobeniusMetric
     * @brief Frobenius norm of the difference of two matrices.
     *
     * Works with `Eigen::MatrixXd`.
     */
    struct FrobeniusMetric {
        double operator()(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) const {
            return (a - b).norm();
        }
    };

    /**
     * @struct StringUltrametric
     * @brief Ultrametric on binary strings: distance = 2^{-length of common prefix}.
     *
     * If the strings are equal, distance is 0. Otherwise it is 2^{-k} where k is
     * the length of the longest common prefix.
     */
    struct StringUltrametric {
        Rational operator()(const std::string& a, const std::string& b) const {
            if (a == b) return Rational(0);
            size_t common = 0;
            while (common < a.size() && common < b.size() && a[common] == b[common]) ++common;
            return Rational(1) / delta::pow(Rational(2), static_cast<int>(common));
        }
    };

    /**
     * @struct PAdicMetric
     * @brief p‑adic metric on rational numbers.
     *
     * The distance is defined as |a - b|_p = p^{-v}, where v is the exponent of p
     * in the factorisation of the difference (v can be negative if the denominator contains p).
     *
     * @tparam p A prime number (must be >= 2).
     *
     * @note The current implementation uses `%` which may not compile for boost::multiprecision types.
     *       A future version should use valuations of numerator and denominator.
     */
    template<int p>
    struct PAdicMetric {
        static_assert(p >= 2, "p must be a prime");
        Rational operator()(const Rational& a, const Rational& b) const {
            Rational diff = a - b;
            if (diff == 0) return Rational(0);
            int v = 0;
            Rational r = diff;
            while (r % p == 0) {
                ++v;
                r /= p;
            }
            return Rational(1) / delta::pow(Rational(p), v);
        }
    };

    /**
     * @struct DiscreteMetric
     * @brief Discrete metric: 0 if equal, 1 otherwise.
     */
    struct DiscreteMetric {
        template<typename T>
        Rational operator()(const T& a, const T& b) const {
            return (a == b) ? Rational(0) : Rational(1);
        }
    };

    // -----------------------------------------------------------------------------
    // Convenience functions
    // -----------------------------------------------------------------------------

    template<typename RI, typename Addr>
    auto distance(const RI& idea, const Addr& a, const Addr& b) {
        return idea.metric(a, b);
    }

    template<typename RI, typename Addr>
    bool between(const RI& idea, const Addr& x, const Addr& y, const Addr& z) {
        return idea.betweenness(x, y, z);
    }

    // -----------------------------------------------------------------------------
    // Compile-time checks
    // -----------------------------------------------------------------------------
    static_assert(Betweenness<TreeBetweenness, std::string>);
    static_assert(Metric<FrobeniusMetric, Eigen::MatrixXd>);
    static_assert(Metric<StringUltrametric, std::string>);
    static_assert(Metric<PAdicMetric<2>, Rational>);
    static_assert(Metric<DiscreteMetric, int>);

} // namespace delta