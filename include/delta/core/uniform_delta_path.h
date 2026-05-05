// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/core/uniform_delta_path.h
#pragma once

// =============================================================================
// UniformDeltaPath – равномерный Δ‑путь с dyadic‑измельчением.
// =============================================================================
//
// В текущей архитектуре DeltaPath всегда возвращает ListGrid, даже если
// начальная сетка была UniformGrid — это следствие универсальности refine.
// Для солверов и других алгоритмов, требующих равномерной структуры (step(),
// count()), такое поведение неудобно: после первого же advance мы теряем
// информацию о регулярности.
//
// UniformDeltaPath решает эту проблему **временным** выделением в отдельный
// класс.  Он хранит сетку как UniformGrid и при advance вставляет середины
// всех интервалов, сохраняя равномерность.
//
// -----------------------------------------------------------------------------
// План будущего рефакторинга:
//   При первой же возможности (достижении библиотекой такого уровня
//   абстракции, который позволяет без оверхеда специализировать DeltaPath
//   по шаблонным параметрам) UniformDeltaPath должен быть заменён на
//   typedef (или using) над DeltaPath с подходящей стратегией и оператором.
//
//   Например:
//     template<typename Scalar>
//     using UniformDeltaPath = DeltaPath<
//         Scalar, Scalar, Scalar,
//         LessBetweenness, EuclideanMetric, EuclideanValueMetric,
//         StaticStrategy<MidpointOperator>,
//         std::less<Scalar>
//     >;
//
//   Это устранит дублирование, но сейчас такое решение невозможно из-за
//   жёсткой привязки DeltaPath к ListGrid и сложности частичной
//   специализации под конкретную стратегию.
//
//   До тех пор используем данный класс как честную реализацию концепта Path
//   для равномерного случая.
// =============================================================================

#include "uniform_grid.h"
#include "regulative_idea.h"       // LessBetweenness, EuclideanMetric
#include <functional>

namespace delta {

    template<typename Scalar, typename Compare = std::less<Scalar>>
    class UniformDeltaPath {
    public:
        // Типы, требуемые концептом Path и ProductDeltaPath
        using GridType = UniformGrid<Scalar, Compare>;
        using grid_type = GridType;                 // для единообразия с DeltaPath
        using addr_type = Scalar;
        using value_type = Scalar;                   // минимальное требование
        using metric_type = EuclideanMetric;
        using betweenness_type = LessBetweenness;
        using distance_type = Scalar;

        // ---------------------------------------------------------------------
        // Конструкторы
        // ---------------------------------------------------------------------

        /// Создание из готовой равномерной сетки.
        explicit UniformDeltaPath(GridType grid)
            : grid_(std::move(grid)), level_(0) {
        }

        /// Удобный конструктор: start, step, count, компаратор.
        UniformDeltaPath(Scalar start, Scalar step, std::size_t count,
            Compare comp = Compare{})
            : grid_(start, step, count, std::move(comp)), level_(0) {
        }

        // ---------------------------------------------------------------------
        // Интерфейс Path
        // ---------------------------------------------------------------------

        /// Текущая равномерная сетка.
        const GridType& current_grid() const noexcept { return grid_; }

        /// Уровень измельчения (число вызовов advance).
        std::size_t level() const noexcept { return level_; }

        /// Максимальный зазор по внешней метрике (для совместимости).
        template<typename ExtMetric>
        auto max_gap(const ExtMetric& m) const {
            if (grid_.count() < 2) return Scalar(0);
            // Все зазоры одинаковы, берём первый.
            return m(grid_[0], grid_[1]);
        }

        /// Метрика (для ProductDeltaPath).
        const metric_type& metric() const { return metric_; }

        /// Отношение betweenness (для ProductDeltaPath).
        const betweenness_type& betweenness() const { return betweenness_; }

        // ---------------------------------------------------------------------
        // Продвижение (advance)
        // ---------------------------------------------------------------------

        /**
         * @brief Вставить середины всех интервалов, сохраняя равномерность.
         *
         * Новая сетка имеет шаг step/2, количество точек = 2·count − 1.
         * Например, из [0,1] получится [0,0.5,1].
         */
        void advance() {
            std::size_t n = grid_.count();
            if (n < 2) return;   // нечего измельчать
            std::size_t new_count = 2 * n - 1;
            Scalar start = grid_.start();
            Scalar half_step = grid_.step() / Scalar(2);
            grid_ = GridType(start, half_step, new_count, grid_.comparator());
            ++level_;
        }

        /**
         * @brief Перегрузка advance с функцией — игнорирует функцию,
         *        так как равномерное измельчение не зависит от значений.
         */
        template<typename Func>
        void advance(const Func& /*f*/) { advance(); }

    private:
        GridType grid_;                     ///< Текущая равномерная сетка.
        std::size_t level_ = 0;             ///< Счётчик шагов.
        metric_type metric_{};              ///< Евклидова метрика.
        betweenness_type betweenness_{};    ///< Обычный порядок.
    };

} // namespace delta