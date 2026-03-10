// include/delta/geometry/matrix_field.h
#pragma once

#include "tensor_field.h"

namespace delta::geometry{

    /**
     * @brief Matrix field (rank-2 tensor field) with additional matrix operations.
     */
    template<typename Addr, typename Scalar, int Dim>
    class MatrixField : public TensorField<Addr, Scalar, 2, Dim> {
        using Base = TensorField<Addr, Scalar, 2, Dim>;
    public:
        using Base::Base;
        using typename Base::value_type;

        // Умножение матриц (поэлементное? нет, матричное умножение для каждой пары)
        // Можно определить оператор * для умножения двух матричных полей, возвращающий новое поле
        // с результатом умножения соответствующих матриц.
        MatrixField operator*(const MatrixField& other) const {
            MatrixField result;
            for (const auto& [addr, mat] : *this) {
                if (other.contains(addr)) {
                    result[addr] = mat * other.at(addr);
                }
            }
            return result;
        }

        // Транспонирование всех матриц
        MatrixField transpose() const {
            MatrixField result;
            for (const auto& [addr, mat] : *this) {
                result[addr] = mat.transpose();
            }
            return result;
        }
    };

    // Для удобства можно определить alias для общего случая
    template<typename Addr, typename Scalar, int Dim>
    using GeneralMatrixField = MatrixField<Addr, Scalar, Dim>;

} // namespace delta::geometry