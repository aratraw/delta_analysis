// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/geometry/tensor_field.h
#ifndef DELTA_GEOMETRY_TENSOR_FIELD_H
#define DELTA_GEOMETRY_TENSOR_FIELD_H

#include <map>
#include <stdexcept>
#include <type_traits>
#include <Eigen/Dense>
#include "delta/core/rational.h"   // for compatibility with Rational (optional)

namespace delta::geometry {

    // Helper: determine tensor value type by rank
    template<typename Scalar, int Rank, int Dim>
    struct TensorType;

    template<typename Scalar, int Dim>
    struct TensorType<Scalar, 0, Dim> {
        using type = Scalar;
    };

    template<typename Scalar, int Dim>
    struct TensorType<Scalar, 1, Dim> {
        using type = Eigen::Matrix<Scalar, Dim, 1>;
    };

    template<typename Scalar, int Dim>
    struct TensorType<Scalar, 2, Dim> {
        using type = Eigen::Matrix<Scalar, Dim, Dim>;
    };

    // Main tensor field class
    template<typename Addr, typename Scalar, int Rank, int Dim,
        typename Compare = std::less<Addr>>
        class TensorField {
        static_assert(Rank >= 0, "Rank must be non-negative");
        static_assert(Dim > 0, "Dimension must be positive");

        public:
            using value_type = typename TensorType<Scalar, Rank, Dim>::type;
            using address_type = Addr;
            using comparator_type = Compare;

            // Default constructor: empty field
            TensorField() = default;

            // Construct from a grid, initialising all points with init_val (default zero)
            template<typename Grid>
            explicit TensorField(const Grid& grid, const value_type& init_val = value_type{}) {
                for (const auto& addr : grid) {
                    set(addr, init_val);
                }
            }

            // Access value at address (throws if not present)
            const value_type& at(const Addr& addr) const {
                auto it = values_.find(addr);
                if (it == values_.end()) {
                    throw std::out_of_range("TensorField::at: address not found");
                }
                return it->second;
            }

            // Set value at address (inserts if new)
            void set(const Addr& addr, const value_type& val) {
                values_[addr] = val;
            }

            // Check if address exists
            bool contains(const Addr& addr) const {
                return values_.find(addr) != values_.end();
            }

            // Number of stored points
            std::size_t size() const { return values_.size(); }

            // Iterators for range‑based loops
            auto begin() const { return values_.begin(); }
            auto end() const { return values_.end(); }
            auto begin() { return values_.begin(); }
            auto end() { return values_.end(); }
            // Access the comparator
            const Compare& comparator() const { return values_.key_comp(); }

        private:
            std::map<Addr, value_type, Compare> values_;
    };

    // -------------------------------------------------------------------------
    // Operators
    // -------------------------------------------------------------------------

    // Addition of two fields of same rank (requires same address set)
    template<typename Addr, typename Scalar, int Rank, int Dim, typename Cmp>
    TensorField<Addr, Scalar, Rank, Dim, Cmp>
        operator+(const TensorField<Addr, Scalar, Rank, Dim, Cmp>& a,
            const TensorField<Addr, Scalar, Rank, Dim, Cmp>& b) {
        TensorField<Addr, Scalar, Rank, Dim, Cmp> result;
        for (const auto& [addr, val] : a) {
            result.set(addr, val + b.at(addr));
        }
        return result;
    }

    // Left scalar multiplication
    template<typename Addr, typename Scalar, int Rank, int Dim, typename Cmp>
    TensorField<Addr, Scalar, Rank, Dim, Cmp>
        operator*(const Scalar& s, const TensorField<Addr, Scalar, Rank, Dim, Cmp>& f) {
        TensorField<Addr, Scalar, Rank, Dim, Cmp> result;
        for (const auto& [addr, val] : f) {
            result.set(addr, s * val);
        }
        return result;
    }

    // Right scalar multiplication
    template<typename Addr, typename Scalar, int Rank, int Dim, typename Cmp>
    TensorField<Addr, Scalar, Rank, Dim, Cmp>
        operator*(const TensorField<Addr, Scalar, Rank, Dim, Cmp>& f, const Scalar& s) {
        return s * f;
    }

    // -------------------------------------------------------------------------
    // Free functions (tensor operations)
    // -------------------------------------------------------------------------

    // Tensor product of two vector fields (rank 1 → rank 2)
    template<typename Addr, typename Scalar, int Dim, typename Cmp>
    TensorField<Addr, Scalar, 2, Dim, Cmp>
        tensor_product(const TensorField<Addr, Scalar, 1, Dim, Cmp>& a,
            const TensorField<Addr, Scalar, 1, Dim, Cmp>& b) {
        TensorField<Addr, Scalar, 2, Dim, Cmp> result;
        for (const auto& [addr, va] : a) {
            const auto& vb = b.at(addr);
            result.set(addr, va * vb.transpose());
        }
        return result;
    }

    // Trace of a matrix field (rank 2 → rank 0)
    template<typename Addr, typename Scalar, int Dim, typename Cmp>
    TensorField<Addr, Scalar, 0, Dim, Cmp>
        trace(const TensorField<Addr, Scalar, 2, Dim, Cmp>& m) {
        TensorField<Addr, Scalar, 0, Dim, Cmp> result;
        for (const auto& [addr, mat] : m) {
            Scalar tr = 0;
            for (int i = 0; i < Dim; ++i) tr += mat(i, i);
            result.set(addr, tr);
        }
        return result;
    }

    // Symmetrization of a matrix field
    template<typename Addr, typename Scalar, int Dim, typename Cmp>
    TensorField<Addr, Scalar, 2, Dim, Cmp>
        symmetrize(const TensorField<Addr, Scalar, 2, Dim, Cmp>& m) {
        TensorField<Addr, Scalar, 2, Dim, Cmp> result;
        for (const auto& [addr, mat] : m) {
            result.set(addr, (mat + mat.transpose()) / Scalar(2));
        }
        return result;
    }

    // Anti‑symmetrization of a matrix field
    template<typename Addr, typename Scalar, int Dim, typename Cmp>
    TensorField<Addr, Scalar, 2, Dim, Cmp>
        antisymmetrize(const TensorField<Addr, Scalar, 2, Dim, Cmp>& m) {
        TensorField<Addr, Scalar, 2, Dim, Cmp> result;
        for (const auto& [addr, mat] : m) {
            result.set(addr, (mat - mat.transpose()) / Scalar(2));
        }
        return result;
    }

    // Lower index of a vector field using a metric (rank 1 → rank 1, but covariant)
    template<typename Addr, typename Scalar, int Dim, typename Cmp>
    TensorField<Addr, Scalar, 1, Dim, Cmp>
        lower_index(const TensorField<Addr, Scalar, 1, Dim, Cmp>& v,
            const TensorField<Addr, Scalar, 2, Dim, Cmp>& g) {
        TensorField<Addr, Scalar, 1, Dim, Cmp> result;
        for (const auto& [addr, vec] : v) {
            result.set(addr, g.at(addr) * vec);
        }
        return result;
    }

    // Raise index of a covector field using the inverse metric
    template<typename Addr, typename Scalar, int Dim, typename Cmp>
    TensorField<Addr, Scalar, 1, Dim, Cmp>
        raise_index(const TensorField<Addr, Scalar, 1, Dim, Cmp>& v,
            const TensorField<Addr, Scalar, 2, Dim, Cmp>& g_inv) {
        TensorField<Addr, Scalar, 1, Dim, Cmp> result;
        for (const auto& [addr, vec] : v) {
            result.set(addr, g_inv.at(addr) * vec);
        }
        return result;
    }

} // namespace delta::geometry

#endif // DELTA_GEOMETRY_TENSOR_FIELD_H