// include/delta/core/product_path.h
#pragma once

#include <array>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <functional>
#include "delta/core/path_concept.h"
#include "delta/core/grid_concept.h"
//DeltaPath ДЛЯ СВОЕГО advance() ТРЕБУЕТ ФУНКЦИЮ. СООТВЕТСТВЕННО МЫ ОБЯЗАНЫ ЕМУ ЕЁ ПЕРЕДАВАТЬ
//ИНАЧЕ НИХЕРА НЕ ЗАВЕДЁТСЯ И КОМПИЛЯТОР ОПЯТЬ БУДЕТ ПЛЕВАТЬСЯ.
namespace delta {

    // -----------------------------------------------------------------------------
    // ProductGrid – декартово произведение N сеток одного типа
    // -----------------------------------------------------------------------------
    template<typename Grid, std::size_t N>
    class ProductGrid {
        static_assert(N > 0, "ProductGrid requires at least one grid");
        static_assert(SimpleGrid<Grid>, "Grid must satisfy SimpleGrid concept");

    public:
        using value_type = std::array<typename Grid::value_type, N>;
        using size_type = std::size_t;
        class const_iterator;

        explicit ProductGrid(std::array<Grid, N> grids)
            : grids_(std::move(grids)) {
            for (std::size_t i = 0; i < N; ++i) {
                sizes_[i] = grids_[i].size();
            }
        }

        size_type size() const noexcept {
            size_type total = 1;
            for (std::size_t i = 0; i < N; ++i) {
                total *= sizes_[i];
            }
            return total;
        }

        value_type operator[](size_type idx) const {
            if (idx >= size()) throw std::out_of_range("ProductGrid::operator[]");
            return compute_tuple(idx);
        }

        const_iterator begin() const noexcept { return const_iterator(this, 0); }
        const_iterator end() const noexcept { return const_iterator(this, size()); }

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
        std::array<Grid, N> grids_;
        std::array<size_type, N> sizes_;

        value_type compute_tuple(size_type idx) const {
            value_type result;
            size_type remaining = idx;
            for (std::size_t i = N; i-- > 0; ) {
                size_type local_idx = remaining % sizes_[i];
                remaining /= sizes_[i];
                result[i] = grids_[i][local_idx];
            }
            return result;
        }
    };

    // -----------------------------------------------------------------------------
    // ProductPath – путь, являющийся декартовым произведением N путей одного типа
    // -----------------------------------------------------------------------------
    template<typename Path, std::size_t N>
    class ProductPath {
        static_assert(N > 0, "ProductPath requires at least one path");

    public:
        using Addr = typename Path::addr_type;
        using Value = typename Path::value_type;
        using grid_type = ProductGrid<typename Path::grid_type, N>;

        // Тип функции для каждого отдельного пути: принимает адрес, возвращает значение
        using SingleFunc = typename Path::Func;

        // Тип массива функций для всех путей
        using FuncArray = std::array<SingleFunc, N>;

        explicit ProductPath(std::array<Path, N> paths) : paths_(std::move(paths)) {}

        // Версия advance с массивом функций (каждая функция для своего пути)
        void advance(const FuncArray& funcs) {
            for (std::size_t i = 0; i < N; ++i) {
                paths_[i].advance(funcs[i]);
            }
        }

        grid_type current_grid() const {
            std::array<typename Path::grid_type, N> grids;
            for (std::size_t i = 0; i < N; ++i) {
                grids[i] = paths_[i].current_grid();
            }
            return grid_type(std::move(grids));
        }

        std::size_t level() const {
            return paths_[0].level();
        }

        template<typename ExtMetric>
        auto max_gap(const ExtMetric& metric) const {
            auto grid = current_grid();
            const std::size_t n = grid.size();
            if (n < 2) {
                using Distance = decltype(metric(grid[0], grid[0]));
                return Distance{ 0 };
            }

            auto max_d = metric(grid[0], grid[0]);
            for (std::size_t i = 0; i + 1 < n; ++i) {
                auto d = metric(grid[i], grid[i + 1]);
                if (d > max_d) max_d = d;
            }
            return max_d;
        }

    private:
        std::array<Path, N> paths_;
    };

} // namespace delta