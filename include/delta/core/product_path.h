// include/delta/core/product_path.h
#pragma once

#include <tuple>
#include <cstddef>
#include <iterator>
#include "delta/core/path_concept.h"
#include "delta/core/grid_concept.h"

namespace delta {

    // -----------------------------------------------------------------------------
    // ProductGrid: декартово произведение нескольких сеток (ленивое вычисление)
    // -----------------------------------------------------------------------------
    template<typename... Grids>
        requires (SimpleGrid<Grids> && ...)
    class ProductGrid {
    public:
        using value_type = std::tuple<typename Grids::value_type...>;
        using size_type = std::size_t;
        using const_iterator = class iterator;  // defined below

        // Конструктор принимает кортеж ссылок на составляющие сетки
        explicit ProductGrid(std::tuple<const Grids&...> grids)
            : grids_(grids)
            , sizes_(compute_sizes(grids))
        {
        }

        // -------------------------------------------------------------------------
        // Основные методы, требуемые SimpleGrid
        // -------------------------------------------------------------------------
        size_type size() const noexcept {
            // Произведение всех размеров
            return std::apply([](const auto&... g) {
                return (static_cast<size_type>(g.size()) * ...);
                }, grids_);
        }

        // Доступ по линейному индексу (возвращает кортеж адресов)
        value_type operator[](size_type idx) const {
            if (idx >= size())
                throw std::out_of_range("ProductGrid index out of range");
            return compute_tuple(idx);
        }

        const_iterator begin() const noexcept { return const_iterator(this, 0); }
        const_iterator end() const noexcept { return const_iterator(this, size()); }

        // -------------------------------------------------------------------------
        // Итератор (ленивый forward iterator)
        // -------------------------------------------------------------------------
        class const_iterator {
        public:
            using iterator_category = std::forward_iterator_tag;
            using value_type = ProductGrid::value_type;
            using difference_type = std::ptrdiff_t;
            using pointer = const value_type*;
            using reference = const value_type&;

            const_iterator() = default;
            const_iterator(const ProductGrid* grid, size_type idx) : grid_(grid), idx_(idx) {}

            value_type operator*() const { return grid_->compute_tuple(idx_); }

            const_iterator& operator++() { ++idx_; return *this; }
            const_iterator operator++(int) { auto tmp = *this; ++*this; return tmp; }

            bool operator==(const const_iterator& other) const { return idx_ == other.idx_; }
            bool operator!=(const const_iterator& other) const { return idx_ != other.idx_; }

        private:
            const ProductGrid* grid_ = nullptr;
            size_type idx_ = 0;
        };

    private:
        std::tuple<const Grids&...> grids_;
        std::tuple<size_type...> sizes_;  // размеры каждой сетки для быстрого доступа

        // Вычисление кортежа размеров
        static auto compute_sizes(const std::tuple<const Grids&...>& grids) {
            return std::apply([](const auto&... g) {
                return std::tuple<size_type...>(g.size()...);
                }, grids);
        }

        // Вспомогательная функция для вычисления кортежа адресов по линейному индексу
        value_type compute_tuple(size_type idx) const {
            value_type result;
            compute_tuple_impl(idx, result, std::index_sequence_for<Grids...>{});
            return result;
        }

        // Рекурсивное заполнение кортежа с помощью свёртки
        template<std::size_t... I>
        void compute_tuple_impl(size_type idx, value_type& result, std::index_sequence<I...>) const {
            // Распаковываем размеры и соответствующие сетки
            size_type remaining = idx;
            // Для каждого I вычисляем локальный индекс в сетке I
            (([&] {
                auto size = std::get<I>(sizes_);
                size_type local_idx = remaining % size;  // берём остаток
                remaining /= size;
                // Получаем адрес из сетки и сохраняем в кортеж
                std::get<I>(result) = std::get<I>(grids_)[local_idx];
                }()), ...);
            // После цикла remaining должно стать 0 (проверка необязательна)
        }
    };

    // -----------------------------------------------------------------------------
    // ProductPath: путь, порождающий декартово произведение путей
    // -----------------------------------------------------------------------------
    template<typename... Paths>
        requires (Path<Paths, typename Paths::metric_type> && ...)  // используем метрику из пути
    class ProductPath {
    public:
        using metric_type = std::tuple_element_t<0, std::tuple<Paths...>>::metric_type;
        using grid_type = ProductGrid<typename Paths::grid_type...>;

        // Конструктор от кортежа путей
        explicit ProductPath(std::tuple<Paths...> paths)
            : paths_(std::move(paths))
        {
        }

        // Продвинуть все пути на один уровень
        void advance() {
            std::apply([](auto&... p) {
                (p.advance(), ...);
                }, paths_);
        }

        // Текущая сетка (декартово произведение текущих сеток путей)
        grid_type current_grid() const {
            // Собираем ссылки на текущие сетки
            auto grid_refs = std::apply([](const auto&... p) {
                return std::tuple<const typename Paths::grid_type&...>(p.current_grid()...);
                }, paths_);
            return grid_type(std::move(grid_refs));
        }

        // Уровень (берём уровень первого пути, предполагаем синхронизацию)
        std::size_t level() const {
            return std::get<0>(paths_).level();
        }

        // Максимальный разрыв (заглушка – возвращаем 0)
        template<typename Metric>
        auto max_gap(const Metric& /*metric*/) const {
            using Distance = decltype(std::declval<Metric>()(
                std::declval<typename Paths::grid_type::value_type>(),
                std::declval<typename Paths::grid_type::value_type>()));
            return Distance{ 0 };
        }

    private:
        std::tuple<Paths...> paths_;
    };

} // namespace delta