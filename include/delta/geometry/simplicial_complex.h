// include/delta/geometry/simplicial_complex.h
#pragma once

#include <vector>
#include <array>
#include <unordered_map>
#include <cstddef>
#include <algorithm>
#include <Eigen/Dense>
#include <boost/container_hash/hash.hpp>
#include "delta/core/rational.h"

namespace delta::geometry {

    template<typename Coord = Rational>
    class SimplicialComplex2D {
    public:
        using point_type = Eigen::Vector<Coord, 2>;
        // Типы для концептов
        using vertex_type = point_type;
        using value_type = point_type;
        using vertex_index = std::size_t;
        using edge = std::array<vertex_index, 2>;
        using edge_type = edge;
        using triangle = std::array<vertex_index, 3>;
        using triangle_type = triangle;

    private:
        using edge_key = std::pair<vertex_index, vertex_index>;

    public:
        SimplicialComplex2D() = default;

        vertex_index add_vertex(const point_type& p) {
            vertex_index idx = points_.size();
            points_.push_back(p);
            return idx;
        }

        void add_edge(vertex_index v0, vertex_index v1) {
            auto [low, high] = std::minmax(v0, v1);
            auto [it, inserted] = edge_index_.try_emplace({ low, high }, edges_.size());
            if (inserted) {
                edges_.push_back({ low, high });
            }
        }

        void add_triangle(vertex_index v0, vertex_index v1, vertex_index v2) {
            triangles_.push_back({ v0, v1, v2 });
            add_edge(v0, v1);
            add_edge(v1, v2);
            add_edge(v2, v0);
        }

        std::ptrdiff_t find_edge(vertex_index v0, vertex_index v1) const {
            auto [low, high] = std::minmax(v0, v1);
            auto it = edge_index_.find({ low, high });
            return (it == edge_index_.end()) ? -1 : static_cast<std::ptrdiff_t>(it->second);
        }

        // ----- методы для VertexGrid / SimpleGrid -----
        std::size_t size() const { return points_.size(); }
        const point_type& operator[](std::size_t i) const { return points_[i]; }
        const point_type& vertex(std::size_t i) const { return points_[i]; }
        auto begin() const { return points_.begin(); }
        auto end() const { return points_.end(); }

        // ----- методы для SimplicialComplex -----
        std::size_t num_vertices() const { return points_.size(); }   // добавлено
        std::size_t num_edges() const { return edges_.size(); }
        std::size_t num_triangles() const { return triangles_.size(); }

        // Доступ к отдельным элементам (переименовано для избежания конфликта с типами)
        const edge& edge_at(std::size_t i) const { return edges_[i]; }
        const triangle& triangle_at(std::size_t i) const { return triangles_[i]; }

        // Доступ ко всем данным
        const std::vector<point_type>& points() const { return points_; }
        const std::vector<edge>& edges() const { return edges_; }
        const std::vector<triangle>& triangles() const { return triangles_; }

    private:
        std::vector<point_type> points_;
        std::vector<edge> edges_;
        std::vector<triangle> triangles_;
        std::unordered_map<edge_key, std::size_t, boost::hash<edge_key>> edge_index_;
    };

} // namespace delta::geometry