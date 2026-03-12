// include/delta/geometry/discrete_forms.h
#pragma once

#include "simplicial_complex.h"
#include "dual_complex.h"
#include "hat_basis.h"
#include <vector>
#include <unordered_map>
#include <type_traits>
#include <stdexcept>
#include <cmath>
#include <optional>

namespace delta::geometry {

    namespace detail {
        // Вспомогательная функция для вычисления объёма k-симплекса (использует метрику)
        template<int k, typename Complex, typename Metric>
        typename Complex::point_type::Scalar
            primal_volume(const Complex& mesh, std::size_t idx, const Metric& metric) {
            using Scalar = typename Complex::point_type::Scalar;
            const auto& simp = mesh.get_simplex(k, idx);
            if constexpr (k == 0) {
                return Scalar{ 1 };
            }
            else if constexpr (k == 1) {
                auto a = mesh.vertex(simp[0]);
                auto b = mesh.vertex(simp[1]);
                return metric(a, b);
            }
            else if constexpr (k == 2) {
                auto a = mesh.vertex(simp[0]);
                auto b = mesh.vertex(simp[1]);
                auto c = mesh.vertex(simp[2]);
                // формула Герона
                Scalar ab = metric(a, b);
                Scalar bc = metric(b, c);
                Scalar ca = metric(c, a);
                Scalar s = (ab + bc + ca) / Scalar{ 2 };
                using std::sqrt;
                return sqrt(s * (s - ab) * (s - bc) * (s - ca));
            }
            else if constexpr (k == 3) {
                // объём тетраэдра через формулу Кэли-Менгера
                auto a = mesh.vertex(simp[0]);
                auto b = mesh.vertex(simp[1]);
                auto c = mesh.vertex(simp[2]);
                auto d = mesh.vertex(simp[3]);
                Scalar ab = metric(a, b);
                Scalar ac = metric(a, c);
                Scalar ad = metric(a, d);
                Scalar bc = metric(b, c);
                Scalar bd = metric(b, d);
                Scalar cd = metric(c, d);
                Eigen::Matrix<Scalar, 5, 5> M;
                M << 0, 1, 1, 1, 1,
                    1, 0, ab* ab, ac* ac, ad* ad,
                    1, ab* ab, 0, bc* bc, bd* bd,
                    1, ac* ac, bc* bc, 0, cd* cd,
                    1, ad* ad, bd* bd, cd* cd, 0;
                Scalar det = M.determinant();
                using std::sqrt;
                return sqrt(det / 288);
            }
            else {
                static_assert(k <= 3, "primal_volume for k>3 not implemented");
                return Scalar{ 0 };
            }
        }
    } // namespace detail

    // -----------------------------------------------------------------------------
    // DiscreteForm: дискретная k-форма на симплициальном комплексе
    // -----------------------------------------------------------------------------
    template<int k, typename Value, typename Complex>
    class DiscreteForm {
        static_assert(k >= 0, "Form degree must be non-negative");
    public:
        using value_type = Value;
        using complex_type = Complex;
        using vertex_index = typename Complex::vertex_index;

        // Конструктор: ассоциирует форму с комплексом, значения инициализируются нулём
        explicit DiscreteForm(const Complex& mesh)
            : mesh_(mesh), values_(mesh.num_simplices(k), Value{ 0 }) {
        }

        // Доступ к значению на симплексе по индексу
        Value& at(std::size_t idx) { return values_.at(idx); }
        const Value& at(std::size_t idx) const { return values_.at(idx); }

        void set(std::size_t idx, const Value& val) { values_[idx] = val; }

        std::size_t size() const { return values_.size(); }
        const Complex& mesh() const { return mesh_; }

        // -------------------------------------------------------------------------
        // Внешняя производная d : k-form -> (k+1)-form
        // -------------------------------------------------------------------------
        DiscreteForm<k + 1, Value, Complex> d() const {
            static_assert(k + 1 <= Complex::dim(), "Cannot apply d: degree exceeds dimension");
            DiscreteForm<k + 1, Value, Complex> result(mesh_);
            for (std::size_t i = 0; i < mesh_.num_simplices(k + 1); ++i) {
                auto faces = mesh_.incident_faces(k + 1, i, k);
                Value sum{ 0 };
                for (const auto& [face_idx, sign] : faces) {
                    sum += static_cast<Value>(sign) * this->at(face_idx);
                }
                result.at(i) = sum;
            }
            return result;
        }

        // -------------------------------------------------------------------------
        // Звёздочка Ходжа : k-form -> (n-k)-form (возвращает форму на том же комплексе
        // с использованием двойственных объёмов). Требуется DualComplex.
        // -------------------------------------------------------------------------
        template<typename Metric>
        DiscreteForm<k, Value, Complex> star(const DualComplex<Complex, Metric>& dual, const Metric& metric) const {
            static_assert(k <= Complex::dim(), "Invalid degree for Hodge star");
            DiscreteForm<k, Value, Complex> result(mesh_);
            for (std::size_t i = 0; i < values_.size(); ++i) {
                Value vol_primal = detail::primal_volume<k>(mesh_, i, metric);
                Value vol_dual = dual.dual_volume(k, i);
                if (vol_primal == Value{ 0 }) {
                    result.at(i) = Value{ 0 };
                }
                else {
                    result.at(i) = values_[i] * (vol_dual / vol_primal);
                }
            }
            return result;
        }

        // -------------------------------------------------------------------------
        // Интерполяция при подразделении
        // -------------------------------------------------------------------------
        template<typename NewComplex, typename Metric>
        DiscreteForm<k, Value, NewComplex> refine(
            const NewComplex& new_mesh,
            const SubdivisionMap& subdiv_map,
            const Metric& metric) const {

            DiscreteForm<k, Value, NewComplex> result(new_mesh);

            if constexpr (k == 0) {
                // Для 0-форм: используем HatBasis для интерполяции значений в новых вершинах
                HatBasis<NewComplex> basis(new_mesh);
                for (std::size_t i = 0; i < new_mesh.num_vertices(); ++i) {
                    auto p = new_mesh.vertex(i);
                    Value val{ 0 };
                    bool found = false;
                    // Перебираем все исходные симплексы, ищем содержащий точку p
                    for (int dim = Complex::dim(); dim >= 0 && !found; --dim) {
                        for (std::size_t j = 0; j < mesh_.num_simplices(dim); ++j) {
                            auto bary = basis.locate_point_in_simplex(p, mesh_, dim, j);
                            if (bary.has_value()) {
                                const auto& simp = mesh_.get_simplex(dim, j);
                                val = Value{ 0 };
                                for (std::size_t vi = 0; vi < simp.size(); ++vi) {
                                    val += (*bary)[vi] * this->at(simp[vi]);
                                }
                                found = true;
                                break;
                            }
                        }
                    }
                    result.at(i) = found ? val : Value{ 0 };
                }
            }
            else if constexpr (k == 1) {
                // Для 1-форм: распределяем значение родительского ребра между рёбрами-потомками
                // пропорционально их длинам (вычисленным через метрику).
                for (const auto& [old_key, new_keys] : subdiv_map) {
                    if (old_key.dim == 1) {  // только рёбра
                        Value old_val = this->at(old_key.idx);
                        // Собираем все рёбра-потомки размерности 1
                        std::vector<std::size_t> edge_descendants;
                        for (const auto& new_key : new_keys) {
                            if (new_key.dim == 1) {
                                edge_descendants.push_back(new_key.idx);
                            }
                        }
                        if (edge_descendants.empty()) continue;

                        // Вычисляем сумму длин потомков (для нормализации)
                        Value total_len = Value{ 0 };
                        std::vector<Value> lengths;
                        lengths.reserve(edge_descendants.size());
                        for (std::size_t eidx : edge_descendants) {
                            auto edge = new_mesh.edge_at(eidx);
                            Value len = metric(new_mesh.vertex(edge[0]), new_mesh.vertex(edge[1]));
                            lengths.push_back(len);
                            total_len += len;
                        }
                        // Распределяем значение пропорционально длинам
                        if (total_len != Value{ 0 }) {
                            for (std::size_t j = 0; j < edge_descendants.size(); ++j) {
                                result.at(edge_descendants[j]) = old_val * (lengths[j] / total_len);
                            }
                        }
                        else {
                            // Все длины нулевые – распределяем поровну
                            Value each = old_val / static_cast<Value>(edge_descendants.size());
                            for (std::size_t eidx : edge_descendants) {
                                result.at(eidx) = each;
                            }
                        }
                    }
                }
                // Для остальных рёбер (не являющихся потомками) значения остаются нулевыми (по умолчанию)
            }
            else {
                // Для k>=2: простое копирование (значение родительского симплекса копируется во все потомки той же размерности)
                for (const auto& [old_key, new_keys] : subdiv_map) {
                    if (old_key.dim == k) {
                        Value old_val = this->at(old_key.idx);
                        for (const auto& new_key : new_keys) {
                            if (new_key.dim == k) {
                                result.at(new_key.idx) = old_val;
                            }
                        }
                    }
                }
            }
            return result;
        }

    private:
        const Complex& mesh_;
        std::vector<Value> values_;
    };

    // -------------------------------------------------------------------------
    // Внешнее произведение (wedge) – реализовано только для 0-форм
    // -------------------------------------------------------------------------
    template<int p, int q, typename Value, typename Complex>
    DiscreteForm<p + q, Value, Complex> wedge(const DiscreteForm<p, Value, Complex>& a,
        const DiscreteForm<q, Value, Complex>& b) {
        if constexpr (p == 0 && q == 0) {
            // 0-формы: поточечное умножение
            DiscreteForm<0, Value, Complex> result(a.mesh());
            for (std::size_t i = 0; i < result.size(); ++i) {
                result.at(i) = a.at(i) * b.at(i);
            }
            return result;
        }
        else {
            static_assert(p == 0 && q == 0,
                "wedge product for these degrees is not implemented (only 0-forms are supported)");
            return DiscreteForm<p + q, Value, Complex>(a.mesh()); // never reached
        }
    }

    // -------------------------------------------------------------------------
    // Обратная звезда Ходжа (вспомогательная)
    // -------------------------------------------------------------------------
    template<int k, typename Value, typename Complex, typename Metric>
    DiscreteForm<k, Value, Complex> inverse_star(const DiscreteForm<k, Value, Complex>& omega,
        const DualComplex<Complex, Metric>& dual,
        const Metric& metric) {
        static_assert(k <= Complex::dim(), "Invalid degree for inverse Hodge star");
        DiscreteForm<k, Value, Complex> result(omega.mesh());
        for (std::size_t i = 0; i < omega.size(); ++i) {
            Value vol_primal = detail::primal_volume<k>(omega.mesh(), i, metric);
            Value vol_dual = dual.dual_volume(k, i);
            if (vol_dual == Value{ 0 }) {
                result.at(i) = Value{ 0 };
            }
            else {
                result.at(i) = omega.at(i) * (vol_primal / vol_dual);
            }
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Кодифференциал δ = (-1)^{n(k-1)+1} * star^{-1} d star
    // -------------------------------------------------------------------------
    template<int k, typename Value, typename Complex, typename Metric>
    DiscreteForm<k - 1, Value, Complex> codifferential(const DiscreteForm<k, Value, Complex>& omega,
        const DualComplex<Complex, Metric>& dual,
        const Metric& metric) {
        static_assert(k >= 1, "Cannot apply codifferential to a 0-form");
        constexpr int n = Complex::dim();
        constexpr int sign = ((n * (k - 1) + 1) % 2 == 0) ? 1 : -1;

        auto star_omega = omega.star(dual, metric);               // (n-k)-форма на примальном
        auto d_star_omega = star_omega.d();                       // (n-k+1)-форма
        auto star_d_star_omega = inverse_star(d_star_omega, dual, metric); // (k-1)-форма

        DiscreteForm<k - 1, Value, Complex> result(omega.mesh());
        for (std::size_t i = 0; i < result.size(); ++i) {
            result.at(i) = static_cast<Value>(sign) * star_d_star_omega.at(i);
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Лапласиан Δ = dδ + δd
    // -------------------------------------------------------------------------
    template<int k, typename Value, typename Complex, typename Metric>
    DiscreteForm<k, Value, Complex> laplacian(const DiscreteForm<k, Value, Complex>& omega,
        const DualComplex<Complex, Metric>& dual,
        const Metric& metric) {
        auto d_omega = omega.d();                                  // (k+1)-форма
        auto delta_omega = codifferential(omega, dual, metric);    // (k-1)-форма

        auto d_delta_omega = (k > 0) ? delta_omega.d() : DiscreteForm<0, Value, Complex>(omega.mesh());
        auto delta_d_omega = codifferential(d_omega, dual, metric);

        DiscreteForm<k, Value, Complex> result(omega.mesh());
        for (std::size_t i = 0; i < result.size(); ++i) {
            Value val{ 0 };
            if (k > 0) val += d_delta_omega.at(i);
            val += delta_d_omega.at(i);
            result.at(i) = val;
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Ротор (curl) для 1-форм в 3D
    // -------------------------------------------------------------------------
    template<typename Value, typename Complex, typename Metric>
    DiscreteForm<1, Value, Complex> curl(const DiscreteForm<1, Value, Complex>& omega,
        const DualComplex<Complex, Metric>& dual,
        const Metric& metric) {
        static_assert(Complex::dim() == 3, "curl only defined in 3D");
        // curl ω = ⋆ d ω
        auto d_omega = omega.d();                         // 2-форма
        auto star_d_omega = d_omega.star(dual, metric);   // 1-форма
        return star_d_omega;
    }

} // namespace delta::geometry