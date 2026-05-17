// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/geometry/tensor_field.h
// ============================================================================
// TENSOR FIELD – POINTWISE TENSOR FIELDS ON A SET OF ADDRESSES
// ============================================================================
//
// This file defines the TensorField class – a container that associates a
// tensor value (scalar, vector, or matrix) with each point (address) in a set.
// The set is implemented as a std::map with a configurable comparator,
// allowing sparse fields (not every address needs to be present).
//
// ----------------------------------------------------------------------------
// TENSOR RANKS
// ----------------------------------------------------------------------------
//
// The TensorField template parametrises by:
//   - Addr: address type (e.g., point, grid index, vertex index)
//   - Scalar: underlying numeric type (e.g., delta::Rational)
//   - Rank: tensor rank (0 = scalar, 1 = vector, 2 = matrix)
//   - Dim: dimension of the vector/matrix (rows = columns = Dim)
//   - Compare: comparator for addresses (default std::less<Addr>)
//
// For rank 0, the value_type is Scalar.
// For rank 1, the value_type is Eigen::Matrix<Scalar, Dim, 1> (vector).
// For rank 2, the value_type is Eigen::Matrix<Scalar, Dim, Dim> (matrix).
//
// Higher ranks (>2) are not currently supported; the library focuses on
// tensors needed for continuum mechanics and DEC (0‑, 1‑, 2‑forms).
//
// ----------------------------------------------------------------------------
// KEY FEATURES
// ----------------------------------------------------------------------------
//
// 1. **Sparse storage** – values are stored only at addresses explicitly set.
// 2. **Grid initialisation** – can be constructed from any object satisfying
//    the Grid concept (provides begin/end iterators over addresses).
// 3. **Algebraic operations** – pointwise addition, scalar multiplication,
//    tensor product, trace, symmetrisation, index raising/lowering.
// 4. **Flexible comparators** – allows custom ordering of addresses
//    (e.g., PointLess for Euclidean points, or custom for p‑adic addresses).
//
// ----------------------------------------------------------------------------
// OPERATIONS AND INVARIANTS
// ----------------------------------------------------------------------------
//
// - Addition (a + b) requires that both fields have exactly the same set of
//   addresses. If an address is missing in either field, at() throws.
// - Scalar multiplication (s * f) keeps the same address set.
// - tensor_product(a, b): for vector fields a and b, creates a matrix field
//   where each matrix is a ⊗ b (outer product).
// - trace(m): sum of diagonal entries of each matrix.
// - symmetrize / antisymmetrize: (M ± Mᵀ)/2.
// - lower_index(v, g): g·v (metric times vector).
// - raise_index(v, g_inv): g⁻¹·v (inverse metric times covector).
//
// All operations are pointwise and independent per address (embarrassingly
// parallel). The library does not provide parallelisation by default, but
// users can easily apply OpenMP within their own loops.
//
// ----------------------------------------------------------------------------
// PERFORMANCE NOTE
// ----------------------------------------------------------------------------
//
// The underlying storage is std::map (ordered tree). For dense fields where
// addresses form a regular grid, an alternative storage (e.g., std::vector
// indexed by grid coordinates) would be more efficient. This class is
// intended for sparse fields on arbitrary address sets (e.g., point clouds,
// random subsets). For regular grids, prefer TensorField specialised for
// ProductGrid (not yet implemented in this version).
//
// ============================================================================

#ifndef DELTA_GEOMETRY_TENSOR_FIELD_H
#define DELTA_GEOMETRY_TENSOR_FIELD_H

#include <map>
#include <stdexcept>
#include <type_traits>
#include <Eigen/Dense>
#include "delta/core/rational.h"   // for compatibility with Rational (optional)
#include "delta/rational/eigen_integration.h"

namespace delta::geometry {

    // -------------------------------------------------------------------------
    // Helper: determine tensor value type by rank
    // -------------------------------------------------------------------------

    /**
     * @brief Type selector for tensor value based on rank and dimension.
     * @tparam Scalar Underlying numeric type.
     * @tparam Rank Tensor rank (0, 1, 2).
     * @tparam Dim Spatial dimension (for vectors and matrices).
     */
    template<typename Scalar, int Rank, int Dim>
    struct TensorType;

    /** @brief Rank 0 → Scalar. */
    template<typename Scalar, int Dim>
    struct TensorType<Scalar, 0, Dim> {
        using type = Scalar;
    };

    /** @brief Rank 1 → Vector of length Dim. */
    template<typename Scalar, int Dim>
    struct TensorType<Scalar, 1, Dim> {
        using type = Eigen::Matrix<Scalar, Dim, 1>;
    };

    /** @brief Rank 2 → Dim × Dim matrix. */
    template<typename Scalar, int Dim>
    struct TensorType<Scalar, 2, Dim> {
        using type = Eigen::Matrix<Scalar, Dim, Dim>;
    };

    // -------------------------------------------------------------------------
    // Main tensor field class
    // -------------------------------------------------------------------------

    /**
     * @class TensorField
     * @brief Sparse field of tensors (scalar, vector, or matrix) over a set of addresses.
     *
     * @tparam Addr Address type (e.g., point, grid index, vertex index).
     * @tparam Scalar Underlying numeric type.
     * @tparam Rank Tensor rank (0, 1, 2).
     * @tparam Dim Spatial dimension (for vectors and matrices).
     * @tparam Compare Comparator for ordering addresses (default std::less<Addr>).
     */
    template<typename Addr, typename Scalar, int Rank, int Dim,
        typename Compare = std::less<Addr>>
        class TensorField {
        static_assert(Rank >= 0, "Rank must be non-negative");
        static_assert(Dim > 0, "Dimension must be positive");

        public:
            /// @brief Type of the tensor stored at each address.
            using value_type = typename TensorType<Scalar, Rank, Dim>::type;
            /// @brief Type of the address (key).
            using address_type = Addr;
            /// @brief Comparator type for ordering addresses.
            using comparator_type = Compare;

            /**
             * @brief Default constructor – creates an empty field.
             */
            TensorField() = default;

            /**
             * @brief Construct a field from a grid, initialising all addresses with a constant value.
             * @tparam Grid A type that satisfies the Grid concept (provides begin/end over addresses).
             * @param grid The underlying grid (addresses are taken from grid iteration).
             * @param init_val The initial tensor value for every address (default zero).
             */
            template<typename Grid>
            explicit TensorField(const Grid& grid, const value_type& init_val = value_type{}) {
                for (const auto& addr : grid) {
                    set(addr, init_val);
                }
            }

            /**
             * @brief Access value at address (const).
             * @param addr Address to query.
             * @return Const reference to the tensor value.
             * @throws std::out_of_range if address not present.
             */
            const value_type& at(const Addr& addr) const {
                auto it = values_.find(addr);
                if (it == values_.end()) {
                    throw std::out_of_range("TensorField::at: address not found");
                }
                return it->second;
            }

            /**
             * @brief Set value at address (inserts if new, overwrites if exists).
             * @param addr Address to set.
             * @param val Tensor value to assign.
             */
            void set(const Addr& addr, const value_type& val) {
                values_[addr] = val;
            }

            /**
             * @brief Check whether an address exists in the field.
             * @param addr Address to query.
             * @return true if present, false otherwise.
             */
            bool contains(const Addr& addr) const {
                return values_.find(addr) != values_.end();
            }

            /**
             * @brief Number of stored addresses.
             */
            std::size_t size() const { return values_.size(); }

            // ---------------------------------------------------------------------
            // Iterators
            // ---------------------------------------------------------------------

            /** @brief Const iterator to the beginning. */
            auto begin() const { return values_.begin(); }
            /** @brief Const iterator to the end. */
            auto end() const { return values_.end(); }
            /** @brief Mutable iterator to the beginning. */
            auto begin() { return values_.begin(); }
            /** @brief Mutable iterator to the end. */
            auto end() { return values_.end(); }

            /**
             * @brief Access the comparator.
             * @return Const reference to the comparator object.
             */
            const Compare& comparator() const { return values_.key_comp(); }

        private:
            std::map<Addr, value_type, Compare> values_;
    };

    // -------------------------------------------------------------------------
    // Tensor field operators (pointwise)
    // -------------------------------------------------------------------------

    /**
     * @brief Pointwise addition of two tensor fields of same rank.
     * @pre Both fields must have the same set of addresses.
     * @throws std::out_of_range if an address is missing in either field.
     */
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

    /**
     * @brief Left scalar multiplication (s * f).
     */
    template<typename Addr, typename Scalar, int Rank, int Dim, typename Cmp>
    TensorField<Addr, Scalar, Rank, Dim, Cmp>
        operator*(const Scalar& s, const TensorField<Addr, Scalar, Rank, Dim, Cmp>& f) {
        TensorField<Addr, Scalar, Rank, Dim, Cmp> result;
        for (const auto& [addr, val] : f) {
            result.set(addr, s * val);
        }
        return result;
    }

    /**
     * @brief Right scalar multiplication (f * s) – delegates to left multiplication.
     */
    template<typename Addr, typename Scalar, int Rank, int Dim, typename Cmp>
    TensorField<Addr, Scalar, Rank, Dim, Cmp>
        operator*(const TensorField<Addr, Scalar, Rank, Dim, Cmp>& f, const Scalar& s) {
        return s * f;
    }

    // -------------------------------------------------------------------------
    // Free functions (tensor operations)
    // -------------------------------------------------------------------------

    /**
     * @brief Tensor (outer) product of two vector fields.
     * @param a First vector field (rank 1).
     * @param b Second vector field (rank 1).
     * @return Matrix field (rank 2) where each matrix = a ⊗ b.
     */
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

    /**
     * @brief Trace of a matrix field (sum of diagonal entries).
     * @param m Matrix field (rank 2).
     * @return Scalar field (rank 0) with tr(M) at each address.
     */
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

    /**
     * @brief Symmetrisation of a matrix field: (M + Mᵀ)/2.
     */
    template<typename Addr, typename Scalar, int Dim, typename Cmp>
    TensorField<Addr, Scalar, 2, Dim, Cmp>
        symmetrize(const TensorField<Addr, Scalar, 2, Dim, Cmp>& m) {
        TensorField<Addr, Scalar, 2, Dim, Cmp> result;
        for (const auto& [addr, mat] : m) {
            result.set(addr, (mat + mat.transpose()) / Scalar(2));
        }
        return result;
    }

    /**
     * @brief Anti‑symmetrisation of a matrix field: (M - Mᵀ)/2.
     */
    template<typename Addr, typename Scalar, int Dim, typename Cmp>
    TensorField<Addr, Scalar, 2, Dim, Cmp>
        antisymmetrize(const TensorField<Addr, Scalar, 2, Dim, Cmp>& m) {
        TensorField<Addr, Scalar, 2, Dim, Cmp> result;
        for (const auto& [addr, mat] : m) {
            result.set(addr, (mat - mat.transpose()) / Scalar(2));
        }
        return result;
    }

    /**
     * @brief Lower index of a vector field using a metric g (covariant).
     * @param v Vector field (contravariant, rank 1).
     * @param g Metric tensor field (rank 2, symmetric positive definite).
     * @return Covector field (rank 1) where (v_♭)_i = g_{ij} v^j.
     * @note The result is still stored as a column vector (Eigen format);
     *       interpretation as covector is left to the user.
     */
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

    /**
     * @brief Raise index of a covector field using the inverse metric g⁻¹.
     * @param v Covector field (rank 1, stored as column vector).
     * @param g_inv Inverse metric tensor field (rank 2).
     * @return Vector field (contravariant) where (v^♯)^i = (g⁻¹)^{ij} v_j.
     */
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