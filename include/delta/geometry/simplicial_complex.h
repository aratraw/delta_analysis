// include/delta/geometry/simplicial_complex.h
#pragma once

#include <vector>
#include <array>
#include <unordered_map>
#include <cstddef>
#include <algorithm> // для std::minmax
#include <Eigen/Dense>
#include <boost/container_hash/hash.hpp>

namespace delta::geometry {

    /**
     * @brief 2D simplicial complex (triangulation) with vertices, edges, triangles.
     */
    template<typename Coord = double>
    class SimplicialComplex2D {
    public:
        using point_type = Eigen::Vector<Coord, 2>;
        using vertex_index = std::size_t;
        using edge = std::array<vertex_index, 2>;
        using triangle = std::array<vertex_index, 3>;

    private:
        // Вспомогательный алиас для краткости
        using edge_key = std::pair<vertex_index, vertex_index>;

    public:
        SimplicialComplex2D() = default;

        // Добавляет вершину, возвращает её индекс
        vertex_index add_vertex(const point_type& p) {
            vertex_index idx = points_.size();
            points_.push_back(p);
            return idx;
        }

        // Добавляет ребро (v0, v1). Если уже есть — ничего не делает.
        void add_edge(vertex_index v0, vertex_index v1) {
            auto [low, high] = std::minmax(v0, v1);

            // try_emplace эффективнее: не создает пару, если ключ уже есть
            auto [it, inserted] = edge_index_.try_emplace({ low, high }, edges_.size());
            if (inserted) {
                edges_.push_back({ low, high });
            }
        }

        // Добавляет треугольник. Автоматически нормализует и добавляет ребра.
        void add_triangle(vertex_index v0, vertex_index v1, vertex_index v2) {
            triangles_.push_back({ v0, v1, v2 });
            add_edge(v0, v1);
            add_edge(v1, v2);
            add_edge(v2, v0);
        }

        // Поиск индекса ребра. Возвращает -1, если не найдено.
        std::ptrdiff_t find_edge(vertex_index v0, vertex_index v1) const {
            auto [low, high] = std::minmax(v0, v1);
            auto it = edge_index_.find({ low, high });
            return (it == edge_index_.end()) ? -1 : static_cast<std::ptrdiff_t>(it->second);
        }

        // Accessors
        const std::vector<point_type>& points() const { return points_; }
        const std::vector<edge>& edges() const { return edges_; }
        const std::vector<triangle>& triangles() const { return triangles_; }

        std::size_t num_vertices() const { return points_.size(); }
        std::size_t num_edges() const { return edges_.size(); }
        std::size_t num_triangles() const { return triangles_.size(); }

    private:
        std::vector<point_type> points_;
        std::vector<edge> edges_;
        std::vector<triangle> triangles_;

        // Хеш-таблица для быстрого поиска индекса ребра по паре вершин
        std::unordered_map<
            edge_key,
            std::size_t,
            boost::hash<edge_key>
        > edge_index_;
    };

} // namespace delta::geometry
