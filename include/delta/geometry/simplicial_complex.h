// include/delta/geometry/simplicial_complex.h
#pragma once

#include <vector>
#include <array>
#include <unordered_map>
#include <map>
#include <unordered_set>
#include <cstddef>
#include <algorithm>
#include <optional>
#include <cmath>
#include <stdexcept>
#include <Eigen/Dense>
#include <boost/container_hash/hash.hpp>
#include "delta/core/rational.h"
#include "delta/core/regulative_idea.h"   // для LinearAddress

namespace delta::geometry {

    // -----------------------------------------------------------------------------
    // Хешер для вектора индексов (используется для быстрого поиска симплексов)
    // -----------------------------------------------------------------------------
    struct SimplexHasher {
        std::size_t operator()(const std::vector<std::size_t>& v) const {
            return boost::hash_range(v.begin(), v.end());
        }
    };

    // -----------------------------------------------------------------------------
    // SimplicialComplex – симплициальный комплекс фиксированной размерности Dim
    // -----------------------------------------------------------------------------
    template<int Dim, typename Coord = Rational>
    class SimplicialComplex {
    public:
        using point_type = Eigen::Matrix<Coord, Dim, 1>;
        using scalar_type = Coord;
        using vertex_index = std::size_t;
        using simplex = std::vector<vertex_index>;

        // Типы для удобства и совместимости с концептами
        using value_type = point_type;
        using edge_type = std::array<vertex_index, 2>;
        using triangle_type = std::array<vertex_index, 3>;
        using tetrahedron_type = std::array<vertex_index, 4>;

        // -------------------------------------------------------------------------
        // Конструкторы
        // -------------------------------------------------------------------------
        SimplicialComplex() = default;

        // -------------------------------------------------------------------------
        // Добавление вершин
        // -------------------------------------------------------------------------
        vertex_index add_vertex(const point_type& p) {
            vertex_index idx = vertices_.size();
            vertices_.push_back(p);
            return idx;
        }

        const point_type& vertex(vertex_index i) const { return vertices_.at(i); }
        std::size_t num_vertices() const { return vertices_.size(); }

        // Для совместимости с GridConcept
        std::size_t size() const { return vertices_.size(); }
        const point_type& operator[](std::size_t i) const { return vertices_[i]; }
        auto begin() const { return vertices_.begin(); }
        auto end() const { return vertices_.end(); }

        // -------------------------------------------------------------------------
        // Добавление симплексов
        // -------------------------------------------------------------------------
        bool add_simplex(const std::vector<vertex_index>& indices) {
            if (indices.size() < 2) return false;            // 0-симплексы не храним отдельно
            int dim = static_cast<int>(indices.size()) - 1;
            if (dim > Dim) return false;

            // Проверка невырожденности (евклидов случай через определитель Грама)
            if (!is_non_degenerate(indices)) return false;

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

        // -------------------------------------------------------------------------
        // Вспомогательные методы
        // -------------------------------------------------------------------------
        const std::vector<point_type>& points() const { return vertices_; }

        // -------------------------------------------------------------------------
        // НОВЫЕ МЕТОДЫ для конечно-объёмных расчётов (Этап 3)
        // -------------------------------------------------------------------------

        // Длина ребра через метрику
        template<typename Metric>
        auto edge_length(std::size_t edge_idx, const Metric& metric) const {
            auto [v0, v1] = edge_at(edge_idx);
            return metric(vertex(v0), vertex(v1));
        }

        // Центр ребра (требует LinearAddress)
        point_type edge_center(std::size_t edge_idx) const {
            static_assert(LinearAddress<point_type, scalar_type>,
                "edge_center requires LinearAddress");
            auto [v0, v1] = edge_at(edge_idx);
            return (vertex(v0) + vertex(v1)) / scalar_type{ 2 };
        }

        // Центр треугольника (требует LinearAddress)
        point_type cell_center(std::size_t tri_idx) const {
            static_assert(LinearAddress<point_type, scalar_type>,
                "cell_center requires LinearAddress");
            auto tri = triangle_at(tri_idx);
            return (vertex(tri[0]) + vertex(tri[1]) + vertex(tri[2])) / scalar_type{ 3 };
        }

        // Площадь треугольника (2D) или объём тетраэдра (3D) через метрику
        template<typename Metric>
        auto cell_volume(std::size_t cell_idx, const Metric& metric) const {
            if constexpr (Dim == 2) {
                auto tri = triangle_at(cell_idx);
                return triangle_volume_impl(vertex(tri[0]), vertex(tri[1]), vertex(tri[2]), metric);
            }
            else if constexpr (Dim == 3) {
                auto tet = tetrahedron_at(cell_idx);
                return tetrahedron_volume_impl(vertex(tet[0]), vertex(tet[1]), vertex(tet[2]), vertex(tet[3]), metric);
            }
            else {
                static_assert(Dim == 2 || Dim == 3, "cell_volume only implemented for 2D and 3D");
                return scalar_type{ 0 };
            }
        }

        // Нормаль к ребру в 2D (вектор, длина равна длине ребра, направление выбрано так,
        // что для левого треугольника (с положительной ориентацией) указывает наружу)
        template<typename Metric>
        point_type edge_normal(std::size_t edge_idx, const Metric& metric) const {
            static_assert(Dim == 2, "edge_normal only for 2D");
            auto [v0, v1] = edge_at(edge_idx);
            const point_type& p0 = vertex(v0);
            const point_type& p1 = vertex(v1);
            // Вектор ребра от v0 к v1 (v0 < v1)
            point_type e = p1 - p0;
            // Поворот на -90 градусов: (dx, dy) -> (dy, -dx)
            point_type normal(e.y(), -e.x());
            // Масштабируем, чтобы длина normal равнялась метрической длине ребра
            scalar_type euclidean_len = e.norm();
            if (euclidean_len > 0) {
                scalar_type metric_len = metric(p0, p1);
                normal *= (metric_len / euclidean_len);
            }
            return normal;
        }

        // Соседи ребра: (левый треугольник, правый треугольник)
        // Левый треугольник – тот, в котором ребро ориентировано от v0 к v1 (где v0<v1)
        // Правый – противоположная ориентация.
        std::pair<std::size_t, std::optional<std::size_t>> edge_neighbors(std::size_t edge_idx) const {
            ensure_edge_to_triangles();
            const auto& entry = (*edge_to_triangles_)[edge_idx];
            return { entry.first, entry.second };
        }

        // -------------------------------------------------------------------------
        // Методы incident_faces (уже были)
        // -------------------------------------------------------------------------
        std::vector<std::pair<std::size_t, int>> incident_faces(int top_dim, std::size_t idx, int low_dim) const;

    private:
        // -------------------------------------------------------------------------
        // Поля
        // -------------------------------------------------------------------------
        std::vector<point_type> vertices_;
        std::unordered_map<int, std::vector<simplex>> simplices_;
        std::unordered_map<int, std::unordered_map<simplex, std::size_t, SimplexHasher>> simplices_map_;

        // -------------------------------------------------------------------------
        // НОВЫЕ ПОЛЯ для кэширования информации о соседях рёбер
        // -------------------------------------------------------------------------
        mutable std::optional<std::vector<std::pair<std::size_t, std::optional<std::size_t>>>> edge_to_triangles_;

        // -------------------------------------------------------------------------
        // Вспомогательные функции для объёмов
        // -------------------------------------------------------------------------
        template<typename Metric>
        static auto triangle_volume_impl(const point_type& a, const point_type& b, const point_type& c, const Metric& metric) {
            auto ab = metric(a, b);
            auto bc = metric(b, c);
            auto ca = metric(c, a);
            auto s = (ab + bc + ca) / scalar_type{ 2 };
            using delta::sqrt;
            return sqrt(s * (s - ab) * (s - bc) * (s - ca));
        }

        template<typename Metric>
        static auto tetrahedron_volume_impl(const point_type& a, const point_type& b, const point_type& c, const point_type& d, const Metric& metric) {
            auto ab = metric(a, b);
            auto ac = metric(a, c);
            auto ad = metric(a, d);
            auto bc = metric(b, c);
            auto bd = metric(b, d);
            auto cd = metric(c, d);

            Eigen::Matrix<scalar_type, 5, 5> M;
            M << 0, 1, 1, 1, 1,
                1, 0, ab* ab, ac* ac, ad* ad,
                1, ab* ab, 0, bc* bc, bd* bd,
                1, ac* ac, bc* bc, 0, cd* cd,
                1, ad* ad, bd* bd, cd* cd, 0;
            scalar_type det = M.determinant();
            using delta::sqrt;
            return sqrt(det / 288);
        }

        // Построение кэша edge_to_triangles_
        void ensure_edge_to_triangles() const {
            if (edge_to_triangles_.has_value()) return;

            std::size_t n_edges = num_edges();
            // Временное хранилище: для каждого ребра храним два опциональных индекса треугольников
            std::vector<std::pair<std::optional<std::size_t>, std::optional<std::size_t>>> storage(
                n_edges, { std::nullopt, std::nullopt });

            for (std::size_t t = 0; t < num_triangles(); ++t) {
                auto tri = triangle_at(t);
                std::array<std::pair<std::size_t, std::size_t>, 3> edges = {
                    std::make_pair(tri[0], tri[1]),
                    std::make_pair(tri[1], tri[2]),
                    std::make_pair(tri[2], tri[0])
                };
                for (const auto& e : edges) {
                    std::size_t v0 = e.first, v1 = e.second;
                    bool reversed = (v0 > v1);
                    if (reversed) std::swap(v0, v1);
                    std::ptrdiff_t eidx = find_simplex(1, { v0, v1 });
                    if (eidx == -1) continue; // должно быть всегда найдено
                    std::size_t ueidx = static_cast<std::size_t>(eidx);
                    bool positive = !reversed; // ориентация совпадает с (v0,v1)
                    if (positive) {
                        if (!storage[ueidx].first.has_value()) {
                            storage[ueidx].first = t;
                        }
                        // else: уже есть – не должно быть, но игнорируем
                    }
                    else {
                        if (!storage[ueidx].second.has_value()) {
                            storage[ueidx].second = t;
                        }
                    }
                }
            }

            // Преобразуем в итоговый формат: первый элемент всегда существует
            std::vector<std::pair<std::size_t, std::optional<std::size_t>>> result(n_edges);
            for (std::size_t i = 0; i < n_edges; ++i) {
                const auto& [left_opt, right_opt] = storage[i];
                if (left_opt.has_value()) {
                    result[i] = { *left_opt, right_opt };
                }
                else if (right_opt.has_value()) {
                    // Если нет левого, но есть правый – меняем местами (правый становится левым)
                    result[i] = { *right_opt, std::nullopt };
                }
                else {
                    // Ребро без треугольников (не должно быть в триангуляции)
                    result[i] = { 0, std::nullopt }; // fallback
                }
            }
            edge_to_triangles_ = std::move(result);
        }

        // Проверка невырожденности (без изменений)
        bool is_non_degenerate(const std::vector<vertex_index>& indices) const {
            int k = static_cast<int>(indices.size()) - 1;
            if (k == 0) return true;
            if (k > Dim) return false;

            Eigen::Matrix<Coord, Dim, Eigen::Dynamic> M(Dim, k);
            const auto& p0 = vertices_[indices[0]];
            for (int i = 1; i <= k; ++i) {
                M.col(i - 1) = vertices_[indices[i]] - p0;
            }
            Eigen::Matrix<Coord, Eigen::Dynamic, Eigen::Dynamic> MtM = M.transpose() * M;
            auto ldlt = MtM.template selfadjointView<Eigen::Lower>().ldlt();
            if (ldlt.info() != Eigen::Success) return false;
            Coord det = ldlt.vectorD().prod();
            return det > Coord(0);
        }
    };

    // -----------------------------------------------------------------------------
    // Реализация incident_faces
    // -----------------------------------------------------------------------------
    template<int Dim, typename Coord>
    std::vector<std::pair<std::size_t, int>> SimplicialComplex<Dim, Coord>::incident_faces(
        int top_dim, std::size_t idx, int low_dim) const
    {
        if (low_dim != top_dim - 1) {
            throw std::invalid_argument("incident_faces: only codimension 1 supported");
        }
        const auto& top_simp = get_simplex(top_dim, idx);
        std::vector<std::pair<std::size_t, int>> result;
        for (std::size_t i = 0; i < top_simp.size(); ++i) {
            std::vector<vertex_index> face_vertices;
            for (std::size_t j = 0; j < top_simp.size(); ++j) {
                if (j != i) face_vertices.push_back(top_simp[j]);
            }
            std::ptrdiff_t face_idx = find_simplex(low_dim, face_vertices);
            if (face_idx == -1) {
                throw std::logic_error("incident_faces: face not found in complex");
            }
            int sign = (i % 2 == 0) ? 1 : -1; // ориентация
            result.emplace_back(static_cast<std::size_t>(face_idx), sign);
        }
        return result;
    }

    // -----------------------------------------------------------------------------
    // Вспомогательная функция ориентации (была раньше)
    // -----------------------------------------------------------------------------
    inline int orientation_sign(const std::vector<std::size_t>& parent, std::size_t removed_idx) {
        return (removed_idx % 2 == 0) ? 1 : -1;
    }

    // -----------------------------------------------------------------------------
    // Структуры для карты подразделения
    // -----------------------------------------------------------------------------
    struct SimplexKey {
        int dim;
        std::size_t idx;

        bool operator==(const SimplexKey& other) const {
            return dim == other.dim && idx == other.idx;
        }
    };

    struct SimplexKeyHash {
        std::size_t operator()(const SimplexKey& key) const {
            std::size_t h1 = std::hash<int>{}(key.dim);
            std::size_t h2 = std::hash<std::size_t>{}(key.idx);
            return h1 ^ (h2 << 1);
        }
    };

    using SubdivisionMap = std::unordered_map<SimplexKey, std::vector<SimplexKey>, SimplexKeyHash>;

    // -----------------------------------------------------------------------------
    // Трейты для получения параметров шаблона SimplicialComplex
    // -----------------------------------------------------------------------------
    template<typename> struct ComplexTraits;

    template<int D, typename C>
    struct ComplexTraits<SimplicialComplex<D, C>> {
        static constexpr int Dim = D;
        using Coord = C;
    };

    // -----------------------------------------------------------------------------
    // Компаратор для point_type, используемый в std::map
    // -----------------------------------------------------------------------------
    template<int Dim, typename Coord>
    struct PointLess {
        using point_type = Eigen::Matrix<Coord, Dim, 1>;
        bool operator()(const point_type& a, const point_type& b) const {
            for (int i = 0; i < Dim; ++i) {
                if (a(i) < b(i)) return true;
                if (b(i) < a(i)) return false;
            }
            return false; // равны
        }
    };

    // -----------------------------------------------------------------------------
    // Вспомогательная структура для рекурсивного подразделения (detail)
    // -----------------------------------------------------------------------------
    namespace detail {
        template<typename Complex>
        struct SubdivHelper {
            using point_type = typename Complex::point_type;
            using vertex_index = typename Complex::vertex_index;
            using simplex = typename Complex::simplex;

            SubdivisionMap& subdiv_map;
            // Используем std::map с компаратором PointLess, параметризованным размерностью и координатой из Complex
            std::map<point_type, vertex_index,
                PointLess<ComplexTraits<Complex>::Dim, typename ComplexTraits<Complex>::Coord>>&vertex_cache;
            Complex& result;

            // Рекурсивное разбиение симплекса, заданного списком индексов вершин (уже в новом комплексе)
            std::vector<simplex> subdivide_simplex(const std::vector<vertex_index>& vertices, int dim) {
                if (dim == 0) {
                    return { vertices };
                }
                point_type bary = point_type::Zero();
                for (auto vi : vertices) {
                    bary += result.vertex(vi);
                }
                bary /= static_cast<typename point_type::Scalar>(vertices.size());

                auto it = vertex_cache.find(bary);
                vertex_index bary_idx;
                if (it == vertex_cache.end()) {
                    bary_idx = result.add_vertex(bary);
                    vertex_cache[bary] = bary_idx;
                }
                else {
                    bary_idx = it->second;
                }

                std::vector<simplex> new_simplices;
                for (std::size_t i = 0; i < vertices.size(); ++i) {
                    std::vector<vertex_index> face_vertices;
                    for (std::size_t j = 0; j < vertices.size(); ++j) {
                        if (j != i) face_vertices.push_back(vertices[j]);
                    }
                    auto face_simplices = subdivide_simplex(face_vertices, dim - 1);
                    for (auto& face_simplex : face_simplices) {
                        simplex new_simplex = face_simplex;
                        new_simplex.push_back(bary_idx);
                        std::sort(new_simplex.begin(), new_simplex.end());
                        new_simplices.push_back(std::move(new_simplex));
                    }
                }
                return new_simplices;
            }
        };
    } // namespace detail

    // -----------------------------------------------------------------------------
    // Барицентрическое подразделение
    // -----------------------------------------------------------------------------
    template<typename Complex>
    std::pair<Complex, SubdivisionMap> barycentric_subdivide(const Complex& primal) {
        using point_type = typename Complex::point_type;
        using vertex_index = typename Complex::vertex_index;
        using simplex = typename Complex::simplex;
        using Traits = ComplexTraits<Complex>;
        constexpr int Dim = Traits::Dim;
        using Coord = typename Traits::Coord;

        Complex result;
        SubdivisionMap subdiv_map;
        // Используем std::map с компаратором PointLess, параметризованным Dim и Coord
        std::map<point_type, vertex_index, PointLess<Dim, Coord>> vertex_cache;

        // Копируем все вершины исходного комплекса
        for (std::size_t i = 0; i < primal.num_vertices(); ++i) {
            auto p = primal.vertex(i);
            vertex_cache[p] = result.add_vertex(p);
        }

        int max_dim = 0;
        while (primal.num_simplices(max_dim + 1) > 0) ++max_dim;

        detail::SubdivHelper<Complex> helper{ subdiv_map, vertex_cache, result };

        for (int dim = 1; dim <= max_dim; ++dim) {
            std::size_t num = primal.num_simplices(dim);
            for (std::size_t idx = 0; idx < num; ++idx) {
                const auto& orig_simp = primal.get_simplex(dim, idx);
                std::vector<vertex_index> new_vertices;
                for (auto vi : orig_simp) {
                    new_vertices.push_back(vertex_cache[primal.vertex(vi)]);
                }

                auto new_simplices = helper.subdivide_simplex(new_vertices, dim);

                std::vector<SimplexKey> new_keys;
                for (const auto& ns : new_simplices) {
                    std::ptrdiff_t ns_idx = result.find_simplex(dim, ns);
                    if (ns_idx == -1) {
                        result.add_simplex(ns);
                        ns_idx = result.find_simplex(dim, ns);
                        if (ns_idx == -1) throw std::logic_error("Failed to add simplex");
                    }
                    new_keys.push_back({ dim, static_cast<std::size_t>(ns_idx) });
                }
                subdiv_map[{dim, idx}] = std::move(new_keys);
            }
        }

        return { std::move(result), std::move(subdiv_map) };
    }

} // namespace delta::geometry