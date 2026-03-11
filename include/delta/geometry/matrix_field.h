// include/delta/geometry/matrix_field.h
#pragma once

#include <complex>
#include "tensor_field.h"


namespace delta::geometry {

    // Специализация для ранга 2 с дополнительными методами
    template<typename Addr, typename Scalar, int Dim>
    class MatrixField : public TensorField<Addr, Scalar, 2, Dim> {
        using Base = TensorField<Addr, Scalar, 2, Dim>;
    public:
        using typename Base::value_type;
        using typename Base::address_type;
        using Base::set;
        using Base::at;
        using Base::contains;
        using Base::begin;
        using Base::end;

        // Поточечное матричное умножение (не тензорное произведение!)
        MatrixField operator*(const MatrixField& other) const {
            MatrixField result;
            for (const auto& [addr, mat] : *this) {
                if (other.contains(addr)) {
                    result.set(addr, mat * other.at(addr));
                }
            }
            return result;
        }

        // Транспонирование всех матриц
        MatrixField transpose() const {
            MatrixField result;
            for (const auto& [addr, mat] : *this) {
                result.set(addr, mat.transpose());
            }
            return result;
        }

        // След (возвращает скалярное поле)
        auto trace() const {
            return trace(*this);
        }

        // Определитель (для размерностей 2 или 3)
        auto determinant() const {
            TensorField<Addr, Scalar, 0, Dim> result;
            for (const auto& [addr, mat] : *this) {
                if constexpr (Dim == 2 || Dim == 3) {
                    result.set(addr, mat.determinant());
                }
                else {
                    static_assert(Dim == 2 || Dim == 3, "determinant only implemented for Dim=2 or 3");
                }
            }
            return result;
        }

        // Коммутатор с другой матрицей
        MatrixField comm(const MatrixField& other) const {
            MatrixField result;
            for (const auto& [addr, mat] : *this) {
                if (other.contains(addr)) {
                    result.set(addr, mat * other.at(addr) - other.at(addr) * mat);
                }
            }
            return result;
        }

        // Сопряжённое (для комплексных матриц, пока заглушка)
        MatrixField adjoint() const {
            static_assert(!std::is_same_v<Scalar, std::complex<double>>,
                "adjoint not implemented for complex, use transpose() and conjugate manually");
            return transpose();
        }
    };

    // Фабричная функция для создания MatrixField из TensorField<2>
    template<typename Addr, typename Scalar, int Dim>
    MatrixField<Addr, Scalar, Dim> as_matrix_field(const TensorField<Addr, Scalar, 2, Dim>& t) {
        MatrixField<Addr, Scalar, Dim> mf;
        for (const auto& [addr, mat] : t) {
            mf.set(addr, mat);
        }
        return mf;
    }

} // namespace delta::geometry