// include/delta/geometry/connection.h
#pragma once

#include <map>
#include <vector>
#include <utility>
#include <Eigen/Core>
#include "matrix_field.h"
#include "delta/geometry/simplicial_complex.h" // для SubdivisionMap

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

            // -------------------------------------------------------------------------
            // Конструкторы и основные методы
            // -------------------------------------------------------------------------
            Connection() = default;

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
                // Если ребро не задано, возвращаем единичную матрицу (тривиальная связность)
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

            // -------------------------------------------------------------------------
            // Создание тривиальной связности
            // -------------------------------------------------------------------------
            /**
             * @brief Создаёт тривиальную связность на заданном наборе рёбер.
             * @param edges Вектор рёбер (каждое ребро задаётся парой вершин).
             * @return Connection с единичными матрицами на всех рёбрах.
             */
            static Connection trivial(const std::vector<edge_type>& edges) {
                Connection conn;
                for (const auto& e : edges) {
                    conn.set_transport(e.first, e.second, matrix_type::Identity());
                }
                return conn;
            }

            // -------------------------------------------------------------------------
            // Построение из калибровочного поля
            // -------------------------------------------------------------------------
            /**
             * @brief Создаёт связность из калибровочного поля (поле групповых элементов).
             * @tparam GaugeField Тип калибровочного поля (должен предоставлять метод matrix()).
             * @param gf Калибровочное поле.
             * @param mesh Симплициальный комплекс (нужен для доступа к рёбрам).
             * @return Connection с матрицами, равными групповым элементам (преобразованным в матрицы).
             */
            template<typename GaugeField, typename Complex>
            static Connection from_gauge_field(const GaugeField& gf, const Complex& mesh) {
                Connection conn;
                for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
                    auto [v0, v1] = mesh.edge_at(e);
                    conn.set_transport(v0, v1, gf.link(e).matrix());
                }
                return conn;
            }

            /**
             * @brief Преобразует связность в калибровочное поле.
             * @tparam GaugeField Тип калибровочного поля (должен иметь конструктор от матрицы).
             * @param mesh Симплициальный комплекс.
             * @return GaugeField, заполненное матрицами из связности.
             */
            template<typename GaugeField, typename Complex>
            GaugeField to_gauge_field(const Complex& mesh) const {
                GaugeField gf(mesh);
                for (std::size_t e = 0; e < mesh.num_edges(); ++e) {
                    auto [v0, v1] = mesh.edge_at(e);
                    auto mat = get_transport(v0, v1);
                    gf.link(e) = typename GaugeField::group_type(mat);
                }
                return gf;
            }

            // -------------------------------------------------------------------------
            // Проверка согласованности при подразделении
            // -------------------------------------------------------------------------
            /**
             * @brief Проверяет, что связность на мелкой сетке согласована с грубой.
             * @param fine Связность на мелкой сетке.
             * @param coarse_to_fine Отображение: грубое ребро (в виде пары вершин) -> список мелких рёбер (каждое мелкое ребро задаётся парой вершин).
             * @param tolerance Допуск.
             * @return true, если для каждого грубого ребра произведение транспортов на мелких рёбрах равно транспорту на грубом.
             */
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

            /**
             * @brief Перегрузка для использования SubdivisionMap (карта подразделения симплициального комплекса).
             * @param fine Связность на мелкой сетке.
             * @param subdiv_map Карта подразделения (исходный симплекс -> список новых симплексов).
             * @param coarse_mesh Грубая сетка.
             * @param fine_mesh Мелкая сетка.
             * @param tolerance Допуск.
             * @return true, если для каждого грубого ребра произведение транспортов на мелких рёбрах равно транспорту на грубом.
             */
            bool is_consistent(const Connection& fine,
                const SubdivisionMap& subdiv_map,
                const SimplicialComplex<Dim, Scalar>& coarse_mesh,
                const SimplicialComplex<Dim, Scalar>& fine_mesh,
                double tolerance = 1e-12) const {
                // Собираем отображение грубое ребро -> список мелких рёбер
                std::vector<std::pair<edge_type, std::vector<edge_type>>> coarse_to_fine;

                // Перебираем все грубые рёбра (размерность 1)
                for (std::size_t idx = 0; idx < coarse_mesh.num_edges(); ++idx) {
                    auto coarse_edge = coarse_mesh.edge_at(idx);
                    SimplexKey key{ 1, idx };
                    auto it = subdiv_map.find(key);
                    if (it == subdiv_map.end()) continue;

                    std::vector<edge_type> fine_edges;
                    for (const auto& new_key : it->second) {
                        if (new_key.dim != 1) continue; // только рёбра
                        auto fine_edge = fine_mesh.edge_at(new_key.idx);
                        // Определяем ориентацию: если мелкое ребро совпадает по направлению с грубым, оставляем как есть,
                        // иначе берём обратный элемент? Пока будем считать, что все мелкие рёбра ориентированы так же,
                        // как грубое (это требует проверки). Для простоты оставляем как есть.
                        fine_edges.push_back({ fine_mesh.vertex(fine_edge[0]), fine_mesh.vertex(fine_edge[1]) });
                    }
                    coarse_to_fine.emplace_back(
                        edge_type{ coarse_mesh.vertex(coarse_edge[0]), coarse_mesh.vertex(coarse_edge[1]) },
                        std::move(fine_edges)
                    );
                }

                return is_consistent(fine, coarse_to_fine, tolerance);
            }

            // -------------------------------------------------------------------------
            // Итераторы и размер
            // -------------------------------------------------------------------------
            std::size_t size() const { return transport_.size(); }
            auto begin() const { return transport_.begin(); }
            auto end() const { return transport_.end(); }

        private:
            std::map<edge_type, matrix_type> transport_;
    };

    // -------------------------------------------------------------------------
    // Вспомогательные функции
    // -------------------------------------------------------------------------

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