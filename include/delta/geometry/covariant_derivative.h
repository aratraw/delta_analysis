#pragma once

#include "connection.h"
#include "tensor_field.h"
#include "delta/core/regulative_idea.h"

namespace delta::geometry {

    /**
     * @brief Ковариантная производная скалярного поля вдоль ребра.
     * @param field Скалярное поле.
     * @param conn  Связность.
     * @param edge  Ориентированное ребро (from -> to).
     * @param metric Метрика для вычисления длины ребра.
     * @return Производная (∇_e f)(to) = (f(to) - f(from)) / length.
     */
    template<typename Addr, typename Scalar, typename Connection, typename Metric>
    Scalar covariant_derivative(const TensorField<Addr, Scalar, 0, 0>& field,
        const Connection& conn,
        const std::pair<Addr, Addr>& edge,
        const Metric& metric) {
        auto [from, to] = edge;
        Scalar len = metric(from, to);
        return (field.at(to) - field.at(from)) / len;
    }

    /**
     * @brief Ковариантная производная векторного поля вдоль ребра.
     * @param field Векторное поле (ранг 1).
     * @param conn  Связность.
     * @param edge  Ребро (from -> to).
     * @param metric Метрика для длины.
     * @return Производная в точке to (вектор).
     */
    template<typename Addr, typename Scalar, int Dim, typename Connection, typename Metric>
    Eigen::Matrix<Scalar, Dim, 1>
        covariant_derivative(const TensorField<Addr, Scalar, 1, Dim>& field,
            const Connection& conn,
            const std::pair<Addr, Addr>& edge,
            const Metric& metric) {
        auto [from, to] = edge;
        Scalar len = metric(from, to);
        auto U = conn.get_transport(from, to); // параллельный перенос from->to
        auto v_from = field.at(from);
        auto v_to = field.at(to);
        // ∇_e v (to) = (U^{-1} v(to) - v(from)) / len  или (U v(from) - v(to))/len?
        // Определим как (v(to) - U v(from)) / len (переносим v(from) в to и вычитаем)
        return (v_to - U * v_from) / len;
    }

    /**
     * @brief Ковариантная производная матричного поля (ранг 2, смешанного) вдоль ребра.
     * @param field Матричное поле.
     * @param conn  Связность.
     * @param edge  Ребро.
     * @param metric Метрика.
     * @return Производная в точке to (матрица).
     */
    template<typename Addr, typename Scalar, int Dim, typename Connection, typename Metric>
    Eigen::Matrix<Scalar, Dim, Dim>
        covariant_derivative(const TensorField<Addr, Scalar, 2, Dim>& field,
            const Connection& conn,
            const std::pair<Addr, Addr>& edge,
            const Metric& metric) {
        auto [from, to] = edge;
        Scalar len = metric(from, to);
        auto U = conn.get_transport(from, to);
        auto M_from = field.at(from);
        auto M_to = field.at(to);
        // Для (1,1)-тензора: ∇ M (to) = (U^{-1} M(to) U - M(from)) / len
        return (U.inverse() * M_to * U - M_from) / len;
    }

} // namespace delta::geometry