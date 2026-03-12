// include/delta/numerical/boundary_conditions.h
#pragma once

#include <vector>
#include <functional>
#include <cstddef>
#include <unordered_map>
#include <variant>

namespace delta::numerical {

    // -----------------------------------------------------------------------------
    // Типы граничных условий
    // -----------------------------------------------------------------------------
    enum class BCType {
        Dirichlet,      // u = g
        Neumann,        // ∂u/∂n = g
        Robin,          // α u + β ∂u/∂n = g
        Periodic        // u(left) = u(right) (связь степеней свободы)
    };

    // -----------------------------------------------------------------------------
    // Значение граничного условия (может быть константой или функцией времени/координат)
    // -----------------------------------------------------------------------------
    template<typename Scalar>
    class BCValue {
    public:
        // Константа
        BCValue(Scalar constant) : value_(constant) {}
        // Функция от времени и индекса узла (или от координат)
        BCValue(std::function<Scalar(double, std::size_t)> func) : value_(func) {}

        // Получить значение для заданного времени t и индекса узла i
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
    // Граничные условия для всей сетки
    // -----------------------------------------------------------------------------
    template<typename Scalar>
    class BoundaryConditions {
    public:
        using size_type = std::size_t;

        // Установка условия на одном узле (по индексу)
        void set(size_type idx, BCType type, const BCValue<Scalar>& value) {
            conditions_[idx] = { type, value };
        }

        // Установка условия на группе узлов (например, все узлы на границе)
        void set(const std::vector<size_type>& indices, BCType type, const BCValue<Scalar>& value) {
            for (auto idx : indices) {
                set(idx, type, value);
            }
        }

        // Проверка, является ли узел граничным (т.е. для него задано условие)
        bool is_boundary(size_type idx) const {
            return conditions_.find(idx) != conditions_.end();
        }

        // Получить тип и значение для узла (если не граничный, возвращает false)
        bool get(size_type idx, BCType& type, BCValue<Scalar>& value) const {
            auto it = conditions_.find(idx);
            if (it == conditions_.end()) return false;
            type = it->second.first;
            value = it->second.second;
            return true;
        }

        // Для периодических условий нужно хранить пары связей
        void set_periodic_pair(size_type left, size_type right) {
            periodic_pairs_.push_back({ left, right });
        }

        const std::vector<std::pair<size_type, size_type>>& periodic_pairs() const {
            return periodic_pairs_;
        }

    private:
        std::unordered_map<size_type, std::pair<BCType, BCValue<Scalar>>> conditions_;
        std::vector<std::pair<size_type, size_type>> periodic_pairs_; // для периодических условий
    };

} // namespace delta::numerical