// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/core/grid_concept.h
#pragma once

#include <concepts>
#include <cstddef>
#include <iterator>

namespace delta {

    // -----------------------------------------------------------------------------
    // SimpleGrid: минимальный интерфейс сетки
    // -----------------------------------------------------------------------------
    template<typename G>
    concept SimpleGrid = requires(G g, const G cg, std::size_t i) {
        typename G::value_type;
        { cg.size() } -> std::convertible_to<std::size_t>;
        { cg[i] } -> std::convertible_to<typename G::value_type>;
        { cg.begin() } -> std::input_or_output_iterator;
        { cg.end() } -> std::input_or_output_iterator;
    };

    // -----------------------------------------------------------------------------
    // OrderedGrid: сетка с компаратором (строгий порядок)
    // -----------------------------------------------------------------------------
    template<typename G>
    concept OrderedGrid = SimpleGrid<G> && requires(G g, const G cg) {
        { cg.comparator() } -> std::invocable<typename G::value_type, typename G::value_type>;
        requires std::same_as<std::invoke_result_t<decltype(cg.comparator()),
        typename G::value_type,
            typename G::value_type>, bool>;
    };

    // -----------------------------------------------------------------------------
    // VertexGrid: сетка, где элементы являются вершинами (доступ по индексу к вершинам)
    // -----------------------------------------------------------------------------
    template<typename G>
    concept VertexGrid = SimpleGrid<G> && requires(G g, const G cg, std::size_t i) {
        typename G::vertex_type;
        { cg.vertex(i) } -> std::convertible_to<typename G::vertex_type>;
    };

    // -----------------------------------------------------------------------------
    // SimplicialComplex: сетка с рёбрами и треугольниками (2D симплициальный комплекс)
    // -----------------------------------------------------------------------------
    template<typename G>
    concept SimplicialComplex = VertexGrid<G> && requires(G g, const G cg, std::size_t i) {
        typename G::edge_type;
        typename G::triangle_type;
        { cg.num_edges() } -> std::convertible_to<std::size_t>;
        { cg.edge(i) } -> std::convertible_to<typename G::edge_type>;
        { cg.num_triangles() } -> std::convertible_to<std::size_t>;
        { cg.triangle(i) } -> std::convertible_to<typename G::triangle_type>;
    };

} // namespace delta