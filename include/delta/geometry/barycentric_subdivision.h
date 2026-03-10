// include/delta/geometry/barycentric_subdivision.h
#pragma once

#include "simplicial_complex.h"
#include <unordered_map>
#include <boost/container_hash/hash.hpp>

namespace delta::geometry {

    template<typename Coord>
    SimplicialComplex2D<Coord> barycentric_subdivide(const SimplicialComplex2D<Coord>& complex) {
        SimplicialComplex2D<Coord> result;

        // Копируем существующие вершины
        for (const auto& p : complex.points()) {
            result.add_vertex(p);
        }

        // Тип ключа для ребра: (v0, v1) с v0 ≤ v1
        using EdgeKey = std::pair<std::size_t, std::size_t>;
        // Хеш-таблица: ребро -> индекс новой вершины (середина)
        std::unordered_map<EdgeKey, std::size_t, boost::hash<EdgeKey>> edge_midpoint;
        // Хеш-таблица: индекс исходного треугольника -> индекс центроида
        std::unordered_map<std::size_t, std::size_t> triangle_centroid;

        // Создаём середины рёбер
        for (std::size_t e = 0; e < complex.edges().size(); ++e) {
            auto [v0, v1] = complex.edges()[e];
            auto p0 = complex.points()[v0];
            auto p1 = complex.points()[v1];
            typename SimplicialComplex2D<Coord>::point_type mid = (p0 + p1) / 2.0;
            std::size_t mid_idx = result.add_vertex(mid);
            edge_midpoint[{std::min(v0, v1), std::max(v0, v1)}] = mid_idx;
        }

        // Создаём центроиды треугольников
        for (std::size_t t = 0; t < complex.triangles().size(); ++t) {
            auto [v0, v1, v2] = complex.triangles()[t];
            auto p0 = complex.points()[v0];
            auto p1 = complex.points()[v1];
            auto p2 = complex.points()[v2];
            typename SimplicialComplex2D<Coord>::point_type centroid = (p0 + p1 + p2) / 3.0;
            std::size_t c_idx = result.add_vertex(centroid);
            triangle_centroid[t] = c_idx;
        }

        // Разбиваем каждый треугольник на 6 новых
        for (std::size_t t = 0; t < complex.triangles().size(); ++t) {
            auto [v0, v1, v2] = complex.triangles()[t];
            std::size_t c = triangle_centroid[t];

            auto get_mid = [&](std::size_t a, std::size_t b) -> std::size_t {
                EdgeKey key = { std::min(a, b), std::max(a, b) };
                auto it = edge_midpoint.find(key);
                if (it == edge_midpoint.end()) throw std::runtime_error("Edge midpoint not found");
                return it->second;
                };

            std::size_t m01 = get_mid(v0, v1);
            std::size_t m12 = get_mid(v1, v2);
            std::size_t m20 = get_mid(v2, v0);

            // Добавляем 6 треугольников (порядок важен для согласованности ориентации)
            result.add_triangle(v0, m01, c);
            result.add_triangle(m01, v1, c);
            result.add_triangle(v1, m12, c);
            result.add_triangle(m12, v2, c);
            result.add_triangle(v2, m20, c);
            result.add_triangle(m20, v0, c);
        }

        return result;
    }

} // namespace delta::geometry