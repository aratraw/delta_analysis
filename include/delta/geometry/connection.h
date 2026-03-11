// include/delta/geometry/connection.h
#pragma once

#include <map>
#include <vector>
#include <utility>
#include <Eigen/Core>
#include "matrix_field.h"

namespace delta::geometry {

    /**
     * @class Connection
     * @brief Дискретная связность на ориентированных рёбрах графа.
     *
     * Хранит матрицы параллельного переноса для каждого ориентированного ребра.
     * Для обратного ребра возвращается обратная матрица.
     *
     * @tparam Addr    Тип адреса (вершины). Должен быть копируемым и сравниваемым.
     * @tparam Scalar  Тип скаляра (например, Rational, double).
     * @tparam Dim     Размерность матриц (должна быть фиксированной).
     * @tparam Group   Тип матрицы (по умолчанию Eigen::Matrix<Scalar, Dim, Dim>).
     */
    template<typename Addr, typename Scalar, int Dim,
        typename Group = Eigen::Matrix<Scalar, Dim, Dim>>
        class Connection {
        public:
            using edge_type = std::pair<Addr, Addr>;
            using matrix_type = Group;

            // Установить матрицу транспорта для ориентированного ребра (from -> to)
            void set_transport(const Addr& from, const Addr& to, const matrix_type& mat) {
                transport_[{from, to}] = mat;
            }

            // Получить матрицу транспорта для ориентированного ребра.
            // Если запрашивается обратное направление, возвращает обратную матрицу.
            matrix_type get_transport(const Addr& from, const Addr& to) const {
                auto it = transport_.find({ from, to });
                if (it != transport_.end()) {
                    return it->second;
                }
                // Проверим обратное направление
                auto it_rev = transport_.find({ to, from });
                if (it_rev != transport_.end()) {
                    return it_rev->second.inverse();
                }
                // Если ребро не задано, можно вернуть единичную матрицу или кинуть исключение
                // По умолчанию вернём identity (что соответствует тривиальной связности)
                return matrix_type::Identity();
            }

            // Параллельный перенос вектора вдоль ребра
            template<typename Vector>
            Vector parallel_transport(const Addr& from, const Addr& to, const Vector& v) const {
                return get_transport(from, to) * v;
            }

            // Голономия вдоль пути, заданного списком рёбер (каждое ребро как пара вершин)
            matrix_type holonomy(const std::vector<edge_type>& path) const {
                matrix_type result = matrix_type::Identity();
                for (const auto& e : path) {
                    result = get_transport(e.first, e.second) * result;
                }
                return result;
            }

            // Проверка согласованности при подразделении:
            // Для каждого грубого ребра (coarse_edge) задан список составляющих его мелких рёбер (fine_edges).
            // Произведение транспортов на мелких должно равняться транспорту на грубом (с точностью до tolerance).
            template<typename EdgeList>
            bool is_consistent(const Connection& fine,
                const std::vector<std::pair<edge_type, EdgeList>>& coarse_to_fine,
                double tolerance = 1e-12) const {
                for (const auto& [coarse_edge, fine_edges] : coarse_to_fine) {
                    matrix_type prod = fine.holonomy(fine_edges);
                    matrix_type coarse_mat = get_transport(coarse_edge.first, coarse_edge.second);
                    if (!prod.isApprox(coarse_mat, tolerance)) {
                        return false;
                    }
                }
                return true;
            }

            // Количество рёбер
            std::size_t size() const { return transport_.size(); }

            // Итераторы для обхода
            auto begin() const { return transport_.begin(); }
            auto end() const { return transport_.end(); }

        private:
            std::map<edge_type, matrix_type> transport_;
    };

    // Вспомогательная функция для создания связности из MatrixField на рёбрах
    template<typename Addr, typename Scalar, int Dim>
    Connection<Addr, Scalar, Dim>
        make_connection(const MatrixField<std::pair<Addr, Addr>, Scalar, Dim>& field) {
        Connection<Addr, Scalar, Dim> conn;
        for (const auto& [edge, mat] : field) {
            conn.set_transport(edge.first, edge.second, mat);
        }
        return conn;
    }

} // namespace delta::geometry