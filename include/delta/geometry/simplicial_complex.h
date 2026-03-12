// include/delta/geometry/simplicial_complex.h
#pragma once

#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <cstddef>
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <Eigen/Dense>
#include <boost/container_hash/hash.hpp>
#include "delta/core/rational.h"

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
        using vertex_index = std::size_t;
        using simplex = std::vector<vertex_index>;          // список индексов вершин

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

        std::vector<std::pair<std::size_t, int>> incident_faces(int top_dim, std::size_t idx, int low_dim) const;
        const point_type& vertex(vertex_index i) const { return vertices_.at(i); }
        std::size_t num_vertices() const { return vertices_.size(); }

        // Для совместимости с GridConcept (доступ по индексу и итераторы)
        std::size_t size() const { return vertices_.size(); }
        const point_type& operator[](std::size_t i) const { return vertices_[i]; }
        auto begin() const { return vertices_.begin(); }
        auto end() const { return vertices_.end(); }

        // -------------------------------------------------------------------------
        // Добавление симплексов произвольной размерности
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

        // Удобные обёртки для часто используемых размерностей
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

        // Поиск симплекса по списку вершин (возвращает индекс или -1)
        std::ptrdiff_t find_simplex(int dim, const std::vector<vertex_index>& vertices) const {
            auto it = simplices_map_.find(dim);
            if (it == simplices_map_.end()) return -1;
            std::vector<vertex_index> sorted = vertices;
            std::sort(sorted.begin(), sorted.end());
            auto jt = it->second.find(sorted);
            return (jt == it->second.end()) ? -1 : static_cast<std::ptrdiff_t>(jt->second);
        }

        // Методы для совместимости с концептами SimplicialComplex
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

    private:
        std::vector<point_type> vertices_;
        // simplices_[dim] – вектор симплексов данной размерности (каждый симплекс – вектор индексов)
        std::unordered_map<int, std::vector<simplex>> simplices_;
        // simplices_map_[dim] – отображение отсортированного вектора вершин -> индекс
        std::unordered_map<int, std::unordered_map<simplex, std::size_t, SimplexHasher>> simplices_map_;

        // Проверка невырожденности через определитель матрицы Грама (евклидов случай)
        bool is_non_degenerate(const std::vector<vertex_index>& indices) const {
            int k = static_cast<int>(indices.size()) - 1; // размерность симплекса
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
    // Структуры для карты подразделения
    // -----------------------------------------------------------------------------
    struct SimplexKey {
        int dim;          // размерность симплекса
        std::size_t idx;  // индекс в исходном комплексе

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

    // Карта подразделения: исходный симплекс -> список новых симплексов (каждый представлен ключом)
    using SubdivisionMap = std::unordered_map<SimplexKey, std::vector<SimplexKey>, SimplexKeyHash>;

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
            std::unordered_map<point_type, vertex_index, std::hash<point_type>>& vertex_cache;
            Complex& result;

            // Рекурсивное разбиение симплекса, заданного списком индексов вершин (уже в новом комплексе)
            std::vector<simplex> subdivide_simplex(const std::vector<vertex_index>& vertices, int dim) {
                if (dim == 0) {
                    // 0-симплекс (вершина) не разбивается
                    return { vertices };
                }
                // Вычисляем барицентр
                point_type bary = point_type::Zero();
                for (auto vi : vertices) {
                    bary += result.vertex(vi);
                }
                bary /= static_cast<typename point_type::Scalar>(vertices.size());

                // Ищем или создаём вершину для барицентра
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
                // Для каждой грани (все вершины кроме одной)
                for (std::size_t i = 0; i < vertices.size(); ++i) {
                    std::vector<vertex_index> face_vertices;
                    for (std::size_t j = 0; j < vertices.size(); ++j) {
                        if (j != i) face_vertices.push_back(vertices[j]);
                    }
                    auto face_simplices = subdivide_simplex(face_vertices, dim - 1);
                    // Каждый симплекс грани соединяем с барицентром
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

        Complex result;
        SubdivisionMap subdiv_map;
        std::unordered_map<point_type, vertex_index, std::hash<point_type>> vertex_cache;

        // Копируем все вершины исходного комплекса
        for (std::size_t i = 0; i < primal.num_vertices(); ++i) {
            auto p = primal.vertex(i);
            vertex_cache[p] = result.add_vertex(p);
        }

        // Определяем максимальную размерность симплексов в исходном комплексе
        int max_dim = 0;
        for (const auto& kv : primal.simplices_) {
            if (kv.first > max_dim) max_dim = kv.first;
        }

        detail::SubdivHelper<Complex> helper{ subdiv_map, vertex_cache, result };

        // Обрабатываем симплексы в порядке возрастания размерности
        for (int dim = 1; dim <= max_dim; ++dim) {
            std::size_t num = primal.num_simplices(dim);
            for (std::size_t idx = 0; idx < num; ++idx) {
                const auto& orig_simp = primal.get_simplex(dim, idx);
                // Получаем индексы вершин в новом комплексе
                std::vector<vertex_index> new_vertices;
                for (auto vi : orig_simp) {
                    new_vertices.push_back(vertex_cache[primal.vertex(vi)]);
                }

                // Разбиваем симплекс
                auto new_simplices = helper.subdivide_simplex(new_vertices, dim);

                // Добавляем каждый новый симплекс в результат и запоминаем его индекс
                std::vector<SimplexKey> new_keys;
                for (const auto& ns : new_simplices) {
                    // Пытаемся найти, возможно уже добавлен ранее (из соседнего симплекса)
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
    // Вспомогательная функция для вычисления знака ориентации грани.
// Для симплекса, заданного списком вершин в порядке (v0, v1, ..., vk),
// грань, полученная удалением вершины на позиции i, имеет знак (-1)^i.
// Предполагается, что список вершин исходного симплекса упорядочен (канонический порядок).
    inline int orientation_sign(const std::vector<std::size_t>& parent, std::size_t removed_idx) {
        return (removed_idx % 2 == 0) ? 1 : -1;
    }

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
            int sign = orientation_sign(top_simp, i);
            result.emplace_back(static_cast<std::size_t>(face_idx), sign);
        }
        return result;
    }
} // namespace delta::geometry