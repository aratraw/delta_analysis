// include/delta/numerical/boundary_conditions.h
#pragma once

#include <vector>
#include <functional>
#include <cstddef>
#include <unordered_map>
#include <variant>
#include <optional>
#include <Eigen/Sparse>
#include <Eigen/Dense>

namespace delta::numerical {

    // -----------------------------------------------------------------------------
    // Типы граничных условий
    // -----------------------------------------------------------------------------
    enum class BCType {
        Dirichlet,      // u = g
        Neumann,        // ∂u/∂n = g  (для потока: задан поток)
        Robin,          // α u + β ∂u/∂n = g
        Periodic        // u(left) = u(right) (связь степеней свободы)
    };

    // -----------------------------------------------------------------------------
    // Значение граничного условия (может быть константой или функцией времени/индекса)
    // -----------------------------------------------------------------------------
    template<typename Scalar>
    class BCValue {
    public:
        // Константа
        BCValue(Scalar constant) : value_(constant) {}
        // Функция от времени и индекса узла (или ребра)
        BCValue(std::function<Scalar(double, std::size_t)> func) : value_(func) {}

        // Получить значение для заданного времени t и индекса i
        Scalar operator()(double t, std::size_t i) const {
            if (std::holds_alternative<Scalar>(value_)) {
                return std::get<Scalar>(value_);
            }
            else {
                return std::get<std::function<Scalar(double, std::size_t)>>(value_)(t, i);
            }
        }

    private:
        std::variant<Scalar, std::function<Scalar(double, std::size_t)>> value_;
    };

    // -----------------------------------------------------------------------------
    // Граничные условия для всей сетки (поддержка вершин и рёбер)
    // -----------------------------------------------------------------------------
    template<typename Scalar>
    class BoundaryConditions {
    public:
        using size_type = std::size_t;

        // ----- Условия на вершинах (по индексу вершины) -----
        void set(size_type idx, BCType type, const BCValue<Scalar>& value) {
            vertex_conditions_[idx] = { type, value };
        }

        void set(const std::vector<size_type>& indices, BCType type, const BCValue<Scalar>& value) {
            for (auto idx : indices) {
                set(idx, type, value);
            }
        }

        bool is_boundary_vertex(size_type idx) const {
            return vertex_conditions_.find(idx) != vertex_conditions_.end();
        }

        bool get_vertex_condition(size_type idx, BCType& type, BCValue<Scalar>& value) const {
            auto it = vertex_conditions_.find(idx);
            if (it == vertex_conditions_.end()) return false;
            type = it->second.first;
            value = it->second.second;
            return true;
        }

        // ----- Условия на рёбрах (для конечно-объёмных схем) -----
        void set_edge_condition(size_type edge_idx, BCType type, const BCValue<Scalar>& value) {
            edge_conditions_[edge_idx] = { type, value };
        }

        void set_edge_conditions(const std::vector<size_type>& edge_indices, BCType type, const BCValue<Scalar>& value) {
            for (auto idx : edge_indices) {
                set_edge_condition(idx, type, value);
            }
        }

        bool is_boundary_edge(size_type edge_idx) const {
            return edge_conditions_.find(edge_idx) != edge_conditions_.end();
        }

        bool get_edge_condition(size_type edge_idx, BCType& type, BCValue<Scalar>& value) const {
            auto it = edge_conditions_.find(edge_idx);
            if (it == edge_conditions_.end()) return false;
            type = it->second.first;
            value = it->second.second;
            return true;
        }

        // -------------------------------------------------------------------------
        // Вычисление потока на граничном ребре для конечно-объёмной схемы
        // согласно upwind-правилу.
        // Параметры:
        //   cell_idx  - индекс ячейки (не используется, но может пригодиться)
        //   edge_idx  - индекс граничного ребра
        //   u_cell    - значение в прилегающей ячейке
        //   vn        - проекция скорости на нормаль (уже содержит знак, длина учтена)
        //   t         - текущее время (для зависящих от времени ГУ)
        // Возвращает поток (знаковая величина), который будет вычтен из ячейки.
        // -------------------------------------------------------------------------
        template<typename Value, typename Scalar2>
        Value boundary_flux(size_type /*cell_idx*/, size_type edge_idx,
            Value u_cell, Scalar2 vn, double t = 0.0) const {
            // Проверяем, есть ли условие для этого ребра
            auto it = edge_conditions_.find(edge_idx);
            if (it == edge_conditions_.end()) {
                // Нет заданного условия – считаем, что это открытая граница (outflow)
                // Поток определяется только состоянием ячейки.
                return Value(vn) * u_cell;
            }

            BCType type = it->second.first;
            const BCValue<Scalar>& bc_val = it->second.second;

            if (type == BCType::Dirichlet) {
                // Для границы Дирихле применяем upwind:
                // если поток направлен из ячейки (vn >= 0), используем u_cell,
                // иначе используем заданное граничное значение.
                if (vn >= Scalar2(0)) {
                    return Value(vn) * u_cell;
                }
                else {
                    Scalar u_boundary = bc_val(t, edge_idx);
                    return Value(vn) * Value(u_boundary);
                }
            }
            else if (type == BCType::Neumann) {
                // Для Неймана задан непосредственно поток (предполагается, что значение bc_val уже есть поток).
                // Возвращаем его как есть, игнорируя vn (поток уже задан).
                return Value(bc_val(t, edge_idx));
            }
            else {
                // Другие типы (Robin, Periodic) не реализованы для граничных рёбер.
                // В простейшем случае возвращаем нулевой поток.
                return Value(0);
            }
        }

        // -------------------------------------------------------------------------
        // Поддержка периодических граничных условий (для вершин)
        // -------------------------------------------------------------------------
        void set_periodic_pair(size_type left, size_type right) {
            periodic_pairs_.push_back({ left, right });
        }

        const std::vector<std::pair<size_type, size_type>>& periodic_pairs() const {
            return periodic_pairs_;
        }

    private:
        std::unordered_map<size_type, std::pair<BCType, BCValue<Scalar>>> vertex_conditions_;
        std::unordered_map<size_type, std::pair<BCType, BCValue<Scalar>>> edge_conditions_;
        std::vector<std::pair<size_type, size_type>> periodic_pairs_;
    };

    // -----------------------------------------------------------------------------
    // Общая функция применения граничных условий для FEM-решателей
    // Поддерживает Dirichlet и Neumann. Robin и Periodic требуют доработки.
    // -----------------------------------------------------------------------------
    template<typename Value, typename BC>
    void apply_boundary_conditions(Eigen::SparseMatrix<Value>& A,
        Eigen::Matrix<Value, Eigen::Dynamic, 1>& b,
        const Eigen::Matrix<Value, Eigen::Dynamic, 1>& lumpedM,
        std::size_t n,
        const BC& bc,
        double t,
        const std::vector<std::size_t>& dof_map = {}) {
        using Index = int;

        // Обработка Dirichlet и Neumann
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t global_i = dof_map.empty() ? i : dof_map[i];
            BCType type;
            BCValue<Value> bc_val;
            if (!bc.get_vertex_condition(global_i, type, bc_val)) continue;

            switch (type) {
            case BCType::Dirichlet: {
                Value g = bc_val(t, global_i);
                // Обнуляем строку i
                for (int k = 0; k < A.outerSize(); ++k) {
                    for (typename Eigen::SparseMatrix<Value>::InnerIterator it(A, k); it; ++it) {
                        if (it.row() == static_cast<Index>(i)) {
                            it.valueRef() = 0;
                        }
                    }
                }
                A.coeffRef(static_cast<Index>(i), static_cast<Index>(i)) = 1.0;
                b(i) = g;
                break;
            }
            case BCType::Neumann: {
                Value g = bc_val(t, global_i);
                b(i) += lumpedM(i) * g;  // вклад в правую часть
                // Матрица не изменяется
                break;
            }
            case BCType::Robin:
            case BCType::Periodic:
                // TODO: implement
                break;
            }
        }

        // Периодические условия (упрощённо, пока не реализованы)
        const auto& pairs = bc.periodic_pairs();
        if (!pairs.empty()) {
            // TODO: Implement periodic conditions by linking degrees of freedom
            // For now, ignore or throw.
        }
    }

} // namespace delta::numerical