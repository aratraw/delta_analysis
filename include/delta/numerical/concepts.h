// include/delta/numerical/concepts.h
#pragma once

#include <concepts>
#include <optional>
#include "delta/geometry/simplicial_complex.h"
#include "delta/core/regulative_idea.h"  // для IsMetric (уже определён, но мы его переиспользуем)

namespace delta::numerical {

    // -------------------------------------------------------------------------
    // Концепт для сетки, используемой в конечно-объёмных методах (2D)
    // Требует наличия методов, добавленных в SimplicialComplex на этапе 3.
    // -------------------------------------------------------------------------
    template<typename G, typename Metric, typename Scalar>
    concept FiniteVolumeGrid2D = requires(G g, std::size_t i, const Metric & m) {
        // Основные методы для количества элементов
        { g.num_triangles() } -> std::convertible_to<std::size_t>;
        { g.num_edges() } -> std::convertible_to<std::size_t>;
        { g.num_vertices() } -> std::convertible_to<std::size_t>;

        // Доступ к вершинам
        { g.vertex(i) } -> std::same_as<typename G::point_type>;

        // Доступ к треугольникам и рёбрам
        { g.triangle_at(i) } -> std::same_as<typename G::triangle_type>;
        { g.edge_at(i) } -> std::same_as<typename G::edge_type>;

        // Поиск симплекса (опционально, но используется)
        { g.find_simplex(1, std::vector<std::size_t>{}) } -> std::convertible_to<std::ptrdiff_t>;

        // Геометрические величины с метрикой
        { g.edge_length(i, m) } -> std::convertible_to<Scalar>;
        { g.edge_center(i) } -> std::same_as<typename G::point_type>;
        { g.cell_center(i) } -> std::same_as<typename G::point_type>;
        { g.cell_volume(i, m) } -> std::convertible_to<Scalar>;
        { g.edge_normal(i, m) } -> std::same_as<typename G::point_type>;

        // Соседи: левый треугольник всегда существует, правый может отсутствовать
        { g.edge_neighbors(i) } -> std::same_as<std::pair<std::size_t, std::optional<std::size_t>>>;
    };

    // -------------------------------------------------------------------------
    // Концепт для сетки, используемой в конечно-элементных методах (2D/3D)
    // -------------------------------------------------------------------------
    template<typename G>
    concept FiniteElementGrid = requires(G g, std::size_t i) {
        typename G::point_type;
        typename G::scalar_type;
        typename G::vertex_index;

        { g.num_vertices() } -> std::convertible_to<std::size_t>;
        { g.vertex(i) } -> std::same_as<typename G::point_type>;
        { g.num_triangles() } -> std::convertible_to<std::size_t>;
        { g.triangle_at(i) } -> std::same_as<typename G::triangle_type>;
        // Для 3D можно добавить tetrahedron_at, но оставим опциональным
    };

    // -------------------------------------------------------------------------
    // Концепт для метрики (уже есть в delta::core, но продублируем для удобства)
    // -------------------------------------------------------------------------
    template<typename M, typename Addr, typename Scalar>
    concept IsMetric = requires(M m, const Addr & a, const Addr & b) {
        { m(a, b) } -> std::convertible_to<Scalar>;
    };

} // namespace delta::numerical