// include/delta/geometry/dual_complex.h
#pragma once

#include "simplicial_complex.h"
#include <vector>
#include <cmath>

namespace delta::geometry {

    template<typename PrimalComplex>
    class DualComplex {
    public:
        using point_type = typename PrimalComplex::point_type;
        using scalar_type = typename point_type::Scalar;
        using vertex_index = typename PrimalComplex::vertex_index;

        explicit DualComplex(const PrimalComplex& primal) : primal_(primal) {
            compute_dual_volumes();
        }

        scalar_type dual_volume(int dim, std::size_t idx) const {
            if (dim < 0 || dim > 3) return scalar_type{ 0 };
            return dual_volumes_[dim][idx];
        }

    private:
        const PrimalComplex& primal_;
        std::vector<std::vector<scalar_type>> dual_volumes_[4]; // индексы 0..3

        void compute_dual_volumes() {
            int ambient_dim = primal_.points()[0].size();
            if (ambient_dim == 2) compute_2d();
            else if (ambient_dim == 3) compute_3d();
            // иначе оставляем нули (для размерностей >3)
        }

        void compute_2d() {
            std::size_t nv = primal_.num_vertices();
            std::size_t ne = primal_.num_edges();
            std::size_t nt = primal_.num_triangles();

            dual_volumes_[0].assign(nv, scalar_type{ 0 }); // площади Вороного для вершин
            dual_volumes_[1].assign(ne, scalar_type{ 0 }); // длины дуальных рёбер
            // dual_volumes_[2] для треугольников не используется (2D)

            // Вычислим площади треугольников
            std::vector<scalar_type> tri_area(nt, scalar_type{ 0 });
            for (std::size_t t = 0; t < nt; ++t) {
                auto tri = primal_.triangle_at(t);
                auto a = primal_.vertex(tri[0]);
                auto b = primal_.vertex(tri[1]);
                auto c = primal_.vertex(tri[2]);
                // Площадь через формулу Герона (обобщённая метрика)
                scalar_type ab = (a - b).norm(); // временно евклидово, позже метрика
                scalar_type bc = (b - c).norm();
                scalar_type ca = (c - a).norm();
                scalar_type s = (ab + bc + ca) / 2;
                tri_area[t] = std::sqrt(s * (s - ab) * (s - bc) * (s - ca));
            }

            // Для каждой вершины: сумма 1/3 площадей инцидентных треугольников
            for (std::size_t t = 0; t < nt; ++t) {
                auto tri = primal_.triangle_at(t);
                scalar_type area = tri_area[t];
                scalar_type third = area / scalar_type{ 3 };
                dual_volumes_[0][tri[0]] += third;
                dual_volumes_[0][tri[1]] += third;
                dual_volumes_[0][tri[2]] += third;
            }

            // Для каждого ребра: длина дуального ребра = (площадь левого + правого треугольника) / длина ребра
            // Предварительно накопим сумму площадей треугольников, инцидентных ребру
            std::vector<scalar_type> tri_sum(ne, scalar_type{ 0 });
            for (std::size_t t = 0; t < nt; ++t) {
                auto tri = primal_.triangle_at(t);
                // три ребра треугольника
                std::array<std::size_t, 3> edge_idxs;
                edge_idxs[0] = primal_.find_simplex(1, { tri[0], tri[1] });
                edge_idxs[1] = primal_.find_simplex(1, { tri[1], tri[2] });
                edge_idxs[2] = primal_.find_simplex(1, { tri[2], tri[0] });
                for (int i = 0; i < 3; ++i) {
                    if (edge_idxs[i] != std::size_t(-1)) {
                        tri_sum[edge_idxs[i]] += tri_area[t];
                    }
                }
            }

            for (std::size_t e = 0; e < ne; ++e) {
                auto edge = primal_.edge_at(e);
                scalar_type len = (primal_.vertex(edge[1]) - primal_.vertex(edge[0])).norm();
                if (len > 0) {
                    dual_volumes_[1][e] = tri_sum[e] / len;
                }
            }
        }

        void compute_3d() {
            std::size_t nv = primal_.num_vertices();
            std::size_t ne = primal_.num_edges();
            std::size_t nf = primal_.num_triangles(); // в 3D это грани
            std::size_t nt = primal_.num_tetrahedra();

            dual_volumes_[0].assign(nv, scalar_type{ 0 }); // объём Вороного вершины
            dual_volumes_[1].assign(ne, scalar_type{ 0 }); // площадь дуальной грани (для ребра)
            dual_volumes_[2].assign(nf, scalar_type{ 0 }); // длина дуального ребра (для грани)

            // Вычислим объёмы тетраэдров
            std::vector<scalar_type> tet_vol(nt, scalar_type{ 0 });
            for (std::size_t t = 0; t < nt; ++t) {
                auto tet = primal_.tetrahedron_at(t);
                auto a = primal_.vertex(tet[0]);
                auto b = primal_.vertex(tet[1]);
                auto c = primal_.vertex(tet[2]);
                auto d = primal_.vertex(tet[3]);
                // Объём через смешанное произведение (евклидово, временно)
                tet_vol[t] = std::abs((b - a).dot((c - a).cross(d - a))) / scalar_type{ 6 };
            }

            // Для вершин: сумма 1/4 объёмов инцидентных тетраэдров
            for (std::size_t t = 0; t < nt; ++t) {
                auto tet = primal_.tetrahedron_at(t);
                scalar_type vol = tet_vol[t];
                scalar_type fourth = vol / scalar_type{ 4 };
                for (int i = 0; i < 4; ++i) {
                    dual_volumes_[0][tet[i]] += fourth;
                }
            }

            // Для рёбер: накопим сумму объёмов тетраэдров, инцидентных ребру
            std::vector<scalar_type> tet_sum(ne, scalar_type{ 0 });
            for (std::size_t t = 0; t < nt; ++t) {
                auto tet = primal_.tetrahedron_at(t);
                // 6 рёбер тетраэдра
                std::vector<std::pair<int, int>> edges = { {0,1},{0,2},{0,3},{1,2},{1,3},{2,3} };
                for (auto [i, j] : edges) {
                    std::size_t eidx = primal_.find_simplex(1, { tet[i], tet[j] });
                    if (eidx != std::size_t(-1)) {
                        tet_sum[eidx] += tet_vol[t];
                    }
                }
            }

            // Площадь дуальной грани для ребра = (сумма объёмов инцидентных тетраэдров) / длина ребра
            for (std::size_t e = 0; e < ne; ++e) {
                auto edge = primal_.edge_at(e);
                scalar_type len = (primal_.vertex(edge[1]) - primal_.vertex(edge[0])).norm();
                if (len > 0) {
                    dual_volumes_[1][e] = tet_sum[e] / len;
                }
            }

            // Для граней (2-симплексов): длина дуального ребра = (сумма объёмов инцидентных тетраэдров) / площадь грани
            std::vector<scalar_type> face_vol_sum(nf, scalar_type{ 0 });
            std::vector<scalar_type> face_area(nf, scalar_type{ 0 });
            for (std::size_t t = 0; t < nt; ++t) {
                auto tet = primal_.tetrahedron_at(t);
                // 4 грани тетраэдра
                std::vector<std::array<int, 3>> faces = { {1,2,3},{0,2,3},{0,1,3},{0,1,2} };
                for (const auto& fverts : faces) {
                    std::array<vertex_index, 3> tri = { tet[fverts[0]], tet[fverts[1]], tet[fverts[2]] };
                    std::size_t fidx = primal_.find_simplex(2, { tri[0], tri[1], tri[2] });
                    if (fidx != std::size_t(-1)) {
                        face_vol_sum[fidx] += tet_vol[t];
                        if (face_area[fidx] == 0) {
                            // вычислим площадь грани один раз
                            auto a = primal_.vertex(tri[0]);
                            auto b = primal_.vertex(tri[1]);
                            auto c = primal_.vertex(tri[2]);
                            // формула Герона
                            scalar_type ab = (a - b).norm();
                            scalar_type bc = (b - c).norm();
                            scalar_type ca = (c - a).norm();
                            scalar_type s = (ab + bc + ca) / 2;
                            face_area[fidx] = std::sqrt(s * (s - ab) * (s - bc) * (s - ca));
                        }
                    }
                }
            }

            for (std::size_t f = 0; f < nf; ++f) {
                if (face_area[f] > 0) {
                    dual_volumes_[2][f] = face_vol_sum[f] / face_area[f];
                }
            }
        }
    };

} // namespace delta::geometry