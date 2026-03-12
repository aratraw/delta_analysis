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

namespace delta::geometry {

    namespace detail {
        // Вспомогательная функция для вычисления объёма k-симплекса
        template<int k, typename Complex, typename Metric>
        typename Complex::point_type::Scalar
            primal_volume(const Complex& mesh, std::size_t idx, const Metric& metric) {
            const auto& simp = mesh.get_simplex(k, idx);
            if constexpr (k == 0) {
                return typename Complex::point_type::Scalar{ 1 };
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
                auto ab = metric(a, b);
                auto bc = metric(b, c);
                auto ca = metric(c, a);
                auto s = (ab + bc + ca) / 2;
                using std::sqrt;
                return sqrt(s * (s - ab) * (s - bc) * (s - ca));
            }
            else if constexpr (k == 3) {
                // Объём тетраэдра через формулу Кэли-Менгера (использует только длины рёбер)
                auto a = mesh.vertex(simp[0]);
                auto b = mesh.vertex(simp[1]);
                auto c = mesh.vertex(simp[2]);
                auto d = mesh.vertex(simp[3]);
                auto ab = metric(a, b);
                auto ac = metric(a, c);
                auto ad = metric(a, d);
                auto bc = metric(b, c);
                auto bd = metric(b, d);
                auto cd = metric(c, d);
                // Матрица Кэли-Менгера
                Eigen::Matrix<typename Complex::point_type::Scalar, 5, 5> M;
                M << 0, 1, 1, 1, 1,
                    1, 0, ab* ab, ac* ac, ad* ad,
                    1, ab* ab, 0, bc* bc, bd* bd,
                    1, ac* ac, bc* bc, 0, cd* cd,
                    1, ad* ad, bd* bd, cd* cd, 0;
                auto det = M.determinant();
                if (det <= 0) return 0;
                using std::sqrt;
                return sqrt(det / 288);
            }
            else {
                static_assert(k <= 3, "primal_volume for k>3 not implemented");
                return 0;
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
        // с использованием дуальных объёмов). Требуется DualComplex.
        // -------------------------------------------------------------------------
        template<typename Metric>
        DiscreteForm<k, Value, Complex> star(const DualComplex<Complex>& dual, const Metric& metric) const {
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
        template<typename NewComplex>
        DiscreteForm<k, Value, NewComplex> refine(const NewComplex& new_mesh,
            const SubdivisionMap& subdiv_map,
            const DualComplex<NewComplex>& /*new_dual*/) const {
            static_assert(std::is_same_v<typename Complex::point_type,
                typename NewComplex::point_type>,
                "Point types must match for interpolation");

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
            else {
                // Для k>=1: используем карту подразделения – каждый новый симплекс наследует
                // значение от исходного (простое копирование, не сохраняет внешнюю производную).
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
    // Внешнее произведение (wedge) – только для 0-форм (p=q=0)
    // -------------------------------------------------------------------------
    template<typename Value, typename Complex>
    DiscreteForm<0, Value, Complex> wedge(const DiscreteForm<0, Value, Complex>& a,
        const DiscreteForm<0, Value, Complex>& b) {
        DiscreteForm<0, Value, Complex> result(a.mesh());
        for (std::size_t i = 0; i < result.size(); ++i) {
            result.at(i) = a.at(i) * b.at(i);
        }
        return result;
    }

    // Для остальных комбинаций выдаём понятную ошибку компиляции
    template<int p, int q, typename Value, typename Complex>
    auto wedge(const DiscreteForm<p, Value, Complex>& a, const DiscreteForm<q, Value, Complex>& b) {
        static_assert(p == 0 && q == 0,
            "wedge product for these degrees not implemented yet (will be added in future updates)");
        return DiscreteForm<p + q, Value, Complex>(a.mesh());
    }

    // -----------------------------------------------------------------------------
    // Обратная звезда Ходжа (вспомогательная)
    // -----------------------------------------------------------------------------
    template<int k, typename Value, typename Complex, typename Metric>
    DiscreteForm<k, Value, Complex> inverse_star(const DiscreteForm<k, Value, Complex>& omega,
        const DualComplex<Complex>& dual,
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

    // -----------------------------------------------------------------------------
    // Кодифференциал δ = (-1)^{n(k-1)+1} * star^{-1} d star
    // -----------------------------------------------------------------------------
    template<int k, typename Value, typename Complex, typename Metric>
    DiscreteForm<k - 1, Value, Complex> codifferential(const DiscreteForm<k, Value, Complex>& omega,
        const DualComplex<Complex>& dual,
        const Metric& metric) {
        static_assert(k >= 1, "Cannot apply codifferential to a 0-form");
        constexpr int n = Complex::dim();  // размерность пространства
        // Знак: (-1)^{n(k-1)+1}
        constexpr int sign = ((n * (k - 1) + 1) % 2 == 0) ? 1 : -1;

        auto star_omega = omega.star(dual, metric);               // (n-k)-форма на примарном
        auto d_star_omega = star_omega.d();                       // (n-k+1)-форма
        auto star_d_star_omega = inverse_star(d_star_omega, dual, metric); // (k-1)-форма

        DiscreteForm<k - 1, Value, Complex> result(omega.mesh());
        for (std::size_t i = 0; i < result.size(); ++i) {
            result.at(i) = static_cast<Value>(sign) * star_d_star_omega.at(i);
        }
        return result;
    }

    // -----------------------------------------------------------------------------
    // Лапласиан Δ = dδ + δd
    // -----------------------------------------------------------------------------
    template<int k, typename Value, typename Complex, typename Metric>
    DiscreteForm<k, Value, Complex> laplacian(const DiscreteForm<k, Value, Complex>& omega,
        const DualComplex<Complex>& dual,
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

} // namespace delta::geometry