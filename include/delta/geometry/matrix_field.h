// include/delta/geometry/matrix_field.h
#ifndef DELTA_GEOMETRY_MATRIX_FIELD_H
#define DELTA_GEOMETRY_MATRIX_FIELD_H

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>  // for exp() and log()
#include "delta/geometry/tensor_field.h"

namespace delta::geometry {

    /**
     * @brief Matrix field (rank‑2 tensor) with additional matrix‑specific operations.
     *
     * Inherits from TensorField and provides pointwise matrix multiplication,
     * transpose, determinant, commutator, exponential and logarithm.
     *
     * @tparam Addr    address type (must be ordered)
     * @tparam Scalar  scalar type (e.g. Rational)
     * @tparam Dim     dimension of the square matrices
     * @tparam Compare comparison functor for addresses (default std::less<Addr>)
     */
    template<typename Addr, typename Scalar, int Dim, typename Compare = std::less<Addr>>
    class MatrixField : public TensorField<Addr, Scalar, 2, Dim, Compare> {
    private:
        using Base = TensorField<Addr, Scalar, 2, Dim, Compare>;

    public:
        using value_type = typename Base::value_type;          // Eigen::Matrix<Scalar, Dim, Dim>
        using address_type = typename Base::address_type;

        // Inherit constructors
        using Base::Base;

        // Default constructor
        MatrixField() = default;

        // Construct from a grid (with optional initial value)
        template<typename Grid>
        explicit MatrixField(const Grid& grid, const value_type& init_val = value_type{})
            : Base(grid, init_val) {
        }

        // ---------------------------------------------------------------------
        // Matrix-specific operations
        // ---------------------------------------------------------------------

        /**
         * @brief Pointwise transpose.
         * @return new matrix field with transposed matrices
         */
        MatrixField transpose() const {
            MatrixField result;
            for (const auto& [addr, mat] : *this) {
                result.set(addr, mat.transpose());
            }
            return result;
        }

        /**
         * @brief Pointwise determinant (only for Dim == 2 or 3).
         * @return scalar field (rank 0) with determinant values
         */
        TensorField<Addr, Scalar, 0, Dim, Compare> determinant() const {
            static_assert(Dim == 2 || Dim == 3,
                "determinant() is only implemented for 2x2 and 3x3 matrices");
            TensorField<Addr, Scalar, 0, Dim, Compare> result;
            for (const auto& [addr, mat] : *this) {
                result.set(addr, mat.determinant());
            }
            return result;
        }

        /**
         * @brief Pointwise commutator [A, B] = A*B - B*A.
         * @param other another matrix field
         * @return new matrix field with commutators
         */
        MatrixField comm(const MatrixField& other) const {
            MatrixField result;
            for (const auto& [addr, mat] : *this) {
                const auto& other_mat = other.at(addr);
                result.set(addr, mat * other_mat - other_mat * mat);
            }
            return result;
        }

        /**
         * @brief Pointwise matrix exponential.
         * @return new matrix field with exp(mat) at each point
         */
        MatrixField exp() const {
            MatrixField result;
            for (const auto& [addr, mat] : *this) {
                result.set(addr, mat.exp());
            }
            return result;
        }

        /**
         * @brief Pointwise matrix logarithm.
         * @return new matrix field with log(mat) at each point
         */
        MatrixField log() const {
            MatrixField result;
            for (const auto& [addr, mat] : *this) {
                result.set(addr, mat.log());
            }
            return result;
        }
    };

    // -------------------------------------------------------------------------
    // Free operator for matrix multiplication (enables ADL)
    // -------------------------------------------------------------------------
    template<typename Addr, typename Scalar, int Dim, typename Compare>
    inline MatrixField<Addr, Scalar, Dim, Compare>
        operator*(const MatrixField<Addr, Scalar, Dim, Compare>& a,
            const MatrixField<Addr, Scalar, Dim, Compare>& b) {
        MatrixField<Addr, Scalar, Dim, Compare> result;
        for (const auto& [addr, mat] : a) {
            result.set(addr, mat * b.at(addr));
        }
        return result;
    }

} // namespace delta::geometry

#endif // DELTA_GEOMETRY_MATRIX_FIELD_H