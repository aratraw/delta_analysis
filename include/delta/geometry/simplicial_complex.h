#pragma once

#include <vector>
#include <array>
#include <unordered_map>
#include <cstddef>
#include <algorithm>
#include <stdexcept>
#include <boost/container_hash/hash.hpp>
#include <Eigen/Dense>
#include "delta/core/rational.h"

namespace delta::geometry {

    // Хешер для вектора индексов (используется для быстрого поиска симплексов)
    struct SimplexHasher {
        std::size_t operator()(const std::vector<std::size_t>& v) const {
            return boost::hash_range(v.begin(), v.end());
        }
    };

    /**
     * @brief Симплициальный комплекс фиксированной размерности.
     *
     * Хранит вершины с координатами типа Coord и все симплексы (рёбра, треугольники, тетраэдры).
     * Обеспечивает комбинаторные операции (добавление, поиск, инцидентность).
     * Удовлетворяет концепту OrderedGrid (может использоваться как сетка в OperationalFunction).
     *
     * @tparam Dim  Размерность пространства (1,2,3,...)
     * @tparam Coord Тип координат (по умолчанию Rational)
     */
    template<int Dim, typename Coord = Rational>
    class SimplicialComplex {
    public:
        using point_type = Eigen::Matrix<Coord, Dim, 1>;
        using scalar_type = Coord;
        using vertex_index = std::size_t;
        using simplex = std::vector<vertex_index>;

        // Для соответствия SimpleGrid
        using value_type = point_type;

        // Типы, необходимые для удобства и для соответствия концепту SimplicialComplex из grid_concept.h
        using edge_type = std::array<vertex_index, 2>;
        using triangle_type = std::array<vertex_index, 3>;
        using tetrahedron_type = std::array<vertex_index, 4>;

        static constexpr int Dimension = Dim;

        // -------------------------------------------------------------------------
        // Конструкторы
        // -------------------------------------------------------------------------
        SimplicialComplex() = default;

        // -------------------------------------------------------------------------
        // Вершины
        // -------------------------------------------------------------------------
        vertex_index add_vertex(const point_type& p) {
            vertex_index idx = vertices_.size();
            vertices_.push_back(p);
            return idx;
        }

        const point_type& vertex(vertex_index i) const {
            if (i >= vertices_.size()) throw std::out_of_range("Vertex index out of range");
            return vertices_[i];
        }

        std::size_t num_vertices() const noexcept { return vertices_.size(); }

        // Для совместимости с OrderedGrid (сетка из вершин)
        std::size_t size() const noexcept { return vertices_.size(); }
        const point_type& operator[](std::size_t i) const { return vertex(i); }
        auto begin() const { return vertices_.begin(); }
        auto end() const { return vertices_.end(); }

        // Компаратор для лексикографического порядка точек (нужен для OrderedGrid)
        struct Compare {
            bool operator()(const point_type& a, const point_type& b) const noexcept {
                for (int i = 0; i < Dim; ++i) {
                    if (a(i) < b(i)) return true;
                    if (b(i) < a(i)) return false;
                }
                return false; // равны
            }
        };
        using comparator_type = Compare;
        Compare comparator() const noexcept { return Compare{}; }

        // -------------------------------------------------------------------------
        // Добавление симплексов
        // -------------------------------------------------------------------------
        bool add_simplex(const std::vector<vertex_index>& indices) {
            // Проверка минимального размера
            if (indices.size() < 2) return false;
            int dim = static_cast<int>(indices.size()) - 1;
            if (dim > Dim) return false;

            // Проверка, что все индексы вершин существуют
            for (auto idx : indices) {
                if (idx >= vertices_.size()) return false;
            }

            // Проверка невырожденности (пока заглушка, всегда true)
            if (!is_non_degenerate(indices)) return false;

            // Сортируем для канонического представления
            std::vector<vertex_index> sorted = indices;
            std::sort(sorted.begin(), sorted.end());

            auto& map = simplices_map_[dim];
            if (map.find(sorted) != map.end()) return false; // уже есть

            std::size_t new_idx = simplices_[dim].size();
            simplices_[dim].push_back(sorted);
            map[sorted] = new_idx;
            return true;
        }

        bool add_edge(vertex_index v0, vertex_index v1) {
            return add_simplex({ v0, v1 });
        }

        bool add_triangle(vertex_index v0, vertex_index v1, vertex_index v2) {
            return add_simplex({ v0, v1, v2 });
        }

        bool add_tetrahedron(vertex_index v0, vertex_index v1, vertex_index v2, vertex_index v3) {
            return add_simplex({ v0, v1, v2, v3 });
        }

        // -------------------------------------------------------------------------
        // Доступ к симплексам
        // -------------------------------------------------------------------------
        std::size_t num_simplices(int dim) const {
            auto it = simplices_.find(dim);
            return (it == simplices_.end()) ? 0 : it->second.size();
        }

        const simplex& get_simplex(int dim, std::size_t idx) const {
            auto it = simplices_.find(dim);
            if (it == simplices_.end() || idx >= it->second.size())
                throw std::out_of_range("Simplex index out of range");
            return it->second[idx];
        }

        // Поиск симплекса по вершинам (возвращает индекс или -1)
        std::ptrdiff_t find_simplex(int dim, const std::vector<vertex_index>& vertices) const {
            auto it = simplices_map_.find(dim);
            if (it == simplices_map_.end()) return -1;
            std::vector<vertex_index> sorted = vertices;
            std::sort(sorted.begin(), sorted.end());
            auto jt = it->second.find(sorted);
            return (jt == it->second.end()) ? -1 : static_cast<std::ptrdiff_t>(jt->second);
        }

        // Удобные методы для распространённых размерностей
        std::size_t num_edges()      const { return num_simplices(1); }
        std::size_t num_triangles()  const { return num_simplices(2); }
        std::size_t num_tetrahedra() const { return num_simplices(3); }

        edge_type edge_at(std::size_t idx) const {
            const auto& s = get_simplex(1, idx);
            if (s.size() != 2) throw std::logic_error("Not an edge");
            return { s[0], s[1] };
        }

        triangle_type triangle_at(std::size_t idx) const {
            const auto& s = get_simplex(2, idx);
            if (s.size() != 3) throw std::logic_error("Not a triangle");
            return { s[0], s[1], s[2] };
        }

        tetrahedron_type tetrahedron_at(std::size_t idx) const {
            const auto& s = get_simplex(3, idx);
            if (s.size() != 4) throw std::logic_error("Not a tetrahedron");
            return { s[0], s[1], s[2], s[3] };
        }

    private:
        std::vector<point_type> vertices_;
        std::unordered_map<int, std::vector<simplex>> simplices_;
        std::unordered_map<int, std::unordered_map<simplex, std::size_t, SimplexHasher>> simplices_map_;

        // Вспомогательная проверка невырожденности (заглушка)
        bool is_non_degenerate(const std::vector<vertex_index>& /*indices*/) const {
            // TODO: Реализовать проверку через определитель Грама (позже, с метрикой)
            return true;
        }
    };

    // -----------------------------------------------------------------------------
    // Свободная функция: получение инцидентных граней (кодимерности 1)
    // -----------------------------------------------------------------------------
    template<int Dim, typename Coord>
    std::vector<std::pair<std::size_t, int>>
        incident_faces(const SimplicialComplex<Dim, Coord>& mesh,
            int top_dim, std::size_t idx, int low_dim) {
        if (low_dim != top_dim - 1) {
            throw std::invalid_argument("incident_faces: only codimension 1 supported");
        }

        const auto& top_simp = mesh.get_simplex(top_dim, idx);
        std::vector<std::pair<std::size_t, int>> result;

        // Особый случай: для top_dim = 1 (ребро) и low_dim = 0 (вершины)
        // вершины не хранятся как симплексы, поэтому возвращаем их индексы напрямую.
        if constexpr (Dim >= 1) {  // условие не обязательно, просто для ясности
            if (top_dim == 1 && low_dim == 0) {
                // Ребро: две вершины. Знаки: удаление первой вершины (i=0) даёт вторую вершину со знаком +1,
                // удаление второй вершины (i=1) даёт первую вершину со знаком -1.
                result.emplace_back(top_simp[1], 1);  // вершина 1
                result.emplace_back(top_simp[0], -1); // вершина 0
                return result;
            }
        }

        // Общий случай для low_dim >= 1
        for (std::size_t i = 0; i < top_simp.size(); ++i) {
            std::vector<typename SimplicialComplex<Dim, Coord>::vertex_index> face_vertices;
            for (std::size_t j = 0; j < top_simp.size(); ++j) {
                if (j != i) face_vertices.push_back(top_simp[j]);
            }
            std::ptrdiff_t face_idx = mesh.find_simplex(low_dim, face_vertices);
            if (face_idx == -1) {
                throw std::logic_error("incident_faces: face not found in complex");
            }
            int sign = (i % 2 == 0) ? 1 : -1;
            result.emplace_back(static_cast<std::size_t>(face_idx), sign);
        }
        return result;
    }

} // namespace delta::geometry