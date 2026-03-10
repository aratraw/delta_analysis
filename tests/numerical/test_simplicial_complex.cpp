#include <gtest/gtest.h>
#include "delta/geometry/simplicial_complex.h"
// Предполагаем, что эти хедеры есть, как в вашем примере
// #include "delta/geometry/barycentric_subdivision.h"
// #include "delta/geometry/discrete_forms.h"

using namespace delta::geometry;

TEST(SimplicialComplexTest, CreateTriangle) {
    SimplicialComplex2D<> complex;
    auto v0 = complex.add_vertex({ 0.0, 0.0 });
    auto v1 = complex.add_vertex({ 1.0, 0.0 });
    auto v2 = complex.add_vertex({ 0.0, 1.0 });
    complex.add_triangle(v0, v1, v2);

    EXPECT_EQ(complex.num_vertices(), 3);
    EXPECT_EQ(complex.num_edges(), 3);
    EXPECT_EQ(complex.num_triangles(), 1);

    // Теперь поиск не зависит от порядка аргументов
    EXPECT_GE(complex.find_edge(v0, v1), 0);
    EXPECT_EQ(complex.find_edge(v0, v1), complex.find_edge(v1, v0));
}

TEST(DiscreteFormsTest, ExteriorDerivativeZeroForm) {
    SimplicialComplex2D<> complex;
    auto v0 = complex.add_vertex({ 0.0, 0.0 }); // f=0
    auto v1 = complex.add_vertex({ 1.0, 0.0 }); // f=1
    auto v2 = complex.add_vertex({ 0.0, 1.0 }); // f=0
    complex.add_triangle(v0, v1, v2);

    // Имитируем работу DiscreteForm
    // Важно: ваша реализация exterior_derivative_0 должна знать, 
    // что ребро (v_a, v_b) в памяти всегда отсортировано как (min, max)

    auto e01 = complex.find_edge(v0, v1); // Хранится как (v0, v1), т.к. 0 < 1
    auto e12 = complex.find_edge(v1, v2); // Хранится как (v1, v2) или (v2, v1) зависит от индексов
    auto e20 = complex.find_edge(v2, v0); // Хранится как (v0, v2), т.к. 0 < 2

    // В тестах проверяем логику: df = f(v_second) - f(v_first) 
    // где v_first и v_second — порядок в векторе edges_

    auto get_df = [&](std::size_t edge_idx, const std::vector<double>& f_vals) {
        const auto& edge = complex.edges()[edge_idx];
        return f_vals[edge[1]] - f_vals[edge[0]];
        };

    std::vector<double> f = { 0.0, 1.0, 0.0 };

    // df на ребре (0,1): f[1] - f[0] = 1 - 0 = 1
    EXPECT_DOUBLE_EQ(get_df(e01, f), 1.0);

    // df на ребре (1,2): f[2] - f[1] = 0 - 1 = -1 (если v1 < v2)
    EXPECT_DOUBLE_EQ(get_df(e12, f), -1.0);

    // df на ребре (0,2): f[2] - f[0] = 0 - 0 = 0
    EXPECT_DOUBLE_EQ(get_df(e20, f), 0.0);
}
