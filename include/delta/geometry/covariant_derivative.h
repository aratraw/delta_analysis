#pragma once

#include "connection.h"
#include "tensor_field.h"
#include "delta/core/regulative_idea.h"
#include <array>

namespace delta::geometry {

    /**
     * @brief Ковариантная производная скалярного поля вдоль ребра.
     * @param field Скалярное поле (ранг 0, любой размерности).
     * @param conn  Связность (не используется, но оставлена для единообразия).
     * @param edge  Ребро (from -> to), задаётся как std::array<Addr,2> (индексы вершин).
     * @param mesh  Симплициальный комплекс, содержащий вершины.
     * @param metric Метрика для вычисления длины ребра (применяется к точкам).
     * @return Производная (∇_e f)(to) = (f(to) - f(from)) / length.
     */
    template<typename Addr, typename Scalar, int Dim, typename Complex, typename Connection, typename Metric>
    Scalar covariant_derivative(const TensorField<Addr, Scalar, 0, Dim>& field,
        const Connection& conn,
        const std::array<Addr, 2>& edge,
        const Complex& mesh,
        const Metric& metric) {
        (void)conn;
        Scalar len = metric(mesh.vertex(edge[0]), mesh.vertex(edge[1]));
        return (field.at(edge[1]) - field.at(edge[0])) / len;
    }

    /**
     * @brief Ковариантная производная векторного поля вдоль ребра.
     * @param field Векторное поле (ранг 1).
     * @param conn  Связность.
     * @param edge  Ребро (from -> to) (индексы вершин).
     * @param mesh  Симплициальный комплекс.
     * @param metric Метрика для длины.
     * @return Производная в точке to (вектор).
     */
    template<typename Addr, typename Scalar, int Dim, typename Complex, typename Connection, typename Metric>
    Eigen::Matrix<Scalar, Dim, 1>
        covariant_derivative(const TensorField<Addr, Scalar, 1, Dim>& field,
            const Connection& conn,
            const std::array<Addr, 2>& edge,
            const Complex& mesh,
            const Metric& metric) {
        Scalar len = metric(mesh.vertex(edge[0]), mesh.vertex(edge[1]));
        auto U = conn.get_transport(edge[0], edge[1]); // параллельный перенос from->to
        auto v_from = field.at(edge[0]);
        auto v_to = field.at(edge[1]);
        // ∇_e v (to) = (v(to) - U v(from)) / len
        return (v_to - U * v_from) / len;
    }

    /**
     * @brief Ковариантная производная матричного поля (ранг 2, смешанного) вдоль ребра.
     * @param field Матричное поле.
     * @param conn  Связность.
     * @param edge  Ребро (индексы вершин).
     * @param mesh  Симплициальный комплекс.
     * @param metric Метрика.
     * @return Производная в точке to (матрица).
     */
    template<typename Addr, typename Scalar, int Dim, typename Complex, typename Connection, typename Metric>
    Eigen::Matrix<Scalar, Dim, Dim>
        covariant_derivative(const TensorField<Addr, Scalar, 2, Dim>& field,
            const Connection& conn,
            const std::array<Addr, 2>& edge,
            const Complex& mesh,
            const Metric& metric) {
        Scalar len = metric(mesh.vertex(edge[0]), mesh.vertex(edge[1]));
        auto U = conn.get_transport(edge[0], edge[1]);
        auto M_from = field.at(edge[0]);
        auto M_to = field.at(edge[1]);
        // Для (1,1)-тензора: ∇ M (to) = (U^{-1} M(to) U - M(from)) / len
        return (U.inverse() * M_to * U - M_from) / len;
    }

} // namespace delta::geometry