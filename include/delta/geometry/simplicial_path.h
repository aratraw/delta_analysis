// include/delta/geometry/simplicial_path.h
#pragma once

#include <vector>
#include <cstddef>
#include <stdexcept>
#include "simplicial_complex.h"
#include "barycentric_subdivision.h"

namespace delta::geometry {

    template<typename Coord = Rational>
    class SimplicialDeltaPath {
    public:
        using Complex = SimplicialComplex2D<Coord>;

        explicit SimplicialDeltaPath(Complex initial)
            : current_(std::move(initial)), level_(0) {
        }

        void advance() {
            current_ = barycentric_subdivide(current_);
            ++level_;
        }

        const Complex& current() const { return current_; }
        const Complex& current_grid() const { return current_; }   // для Path концепта
        std::size_t level() const { return level_; }

        std::size_t num_vertices() const { return current_.num_vertices(); }
        std::size_t num_edges() const { return current_.num_edges(); }
        std::size_t num_triangles() const { return current_.num_triangles(); }

        static SimplicialDeltaPath from_level(std::size_t target_level, const Complex& initial) {
            SimplicialDeltaPath path(initial);
            for (std::size_t i = 0; i < target_level; ++i) {
                path.advance();
            }
            return path;
        }

        // Вычисление максимальной длины ребра с использованием заданной метрики
        template<typename Metric>
        auto max_gap(const Metric& metric) const {
            using Distance = decltype(metric(typename Complex::point_type{}, typename Complex::point_type{}));
            Distance max_g{ 0 };
            for (std::size_t i = 0; i < current_.num_edges(); ++i) {
                auto [v0, v1] = current_.edge(i);
                Distance d = metric(current_.vertex(v0), current_.vertex(v1));
                if (d > max_g) max_g = d;
            }
            return max_g;
        }

    private:
        Complex current_;
        std::size_t level_;
    };

} // namespace delta::geometry