// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/core/operational_function.h
#pragma once

#include <map>
#include <functional>
#include <cassert>
#include <type_traits>
#include <concepts>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include "list_grid.h"
#include "uniform_grid.h"
#include "grid_concept.h"
#include "grid_refine.h"
#include "rational.h"

namespace delta {

    namespace detail {
        /**
         * @brief Compute the index of an address in a uniform grid.
         *
         * Assumes that the address is exactly of the form start + k * step for some integer k.
         *
         * @tparam Addr Address type (must support subtraction and division).
         * @tparam Grid UniformGrid type.
         * @param addr The address to locate.
         * @param grid The uniform grid.
         * @return The index k such that grid[k] == addr.
         * @throws std::runtime_error if addr is not exactly on the grid (non‑integer index).
         * @throws std::out_of_range if the computed index is out of bounds.
         */
         //КОММЕНТАРИЙ: КАКОГО ЧЁРТА ОПЕРАЦИОННАЯ ФУНКЦИЯ ТРЕБУЕТ ДЛЯ АДРЕСА ЗНАМЕНАТЕЛЬ?
         // А ЕСЛИ МЫ ХОТИМ ОПЕРАЦИОННУЮ ФУНКЦИЮ, ОПРЕДЕЛЁННУЮ НА ПОЛЕ БИНАРНЫХ СТРОК ИЛИ МАТРИЦ?!
         // КОГДА ТЕСТЫ ПОЗЕЛЕНЕЮТ - РАЗОБРАТЬСЯ И ПРИ НЕОБХОДИМОСТИ ПЕРЕПИСАТЬ НАХРЕН НОРМАЛЬНО. КАРАУЛ!
        template<typename Addr, typename Grid>
        std::size_t uniform_index(const Addr& addr, const Grid& grid) {
            auto idx = (addr - grid.start()) / grid.step();
            if (idx.denominator() != Rational(1)) {
                throw std::runtime_error("Address does not belong to uniform grid (non-integer index)");
            }
            std::size_t uidx = static_cast<std::size_t>(idx.numerator().convert_to<long long>());
            if (uidx >= grid.size()) {
                throw std::out_of_range("Index out of bounds");
            }
            return uidx;
        }
    }
    // -------------------------------------------------------------------------
    // Primary template (for arbitrary grids) – uses std::map with a comparator
    // -------------------------------------------------------------------------

    /**
     * @class OperationalFunction
     * @brief A function defined on a grid, with the ability to extend to refined grids.
     *
     * Stores values for all addresses of a given grid. When the grid is refined,
     * new addresses can be added by interpolating from the old values.
     *
     * @tparam Addr      Address type.
     * @tparam Value     Function value type.
     * @tparam Grid      Grid type (must satisfy GridConcept<Grid, Addr>).
     * @tparam Compare   Comparison functor for addresses (default std::less<Addr>).
     */
    template<typename Addr, typename Value, typename Grid,
        typename Compare = std::less<Addr>>
        class OperationalFunction {
        public:
            /// Type of the interpolation callable: (left, right, f(left), f(right)) -> value at midpoint.
            using Interpolator = std::function<Value(const Addr&, const Addr&,
                const Value&, const Value&)>;

            /**
             * @brief Construct an operational function from a grid and an initial value generator.
             *
             * @tparam Func Callable with signature Value(const Addr&).
             * @param grid The grid on which the function is defined.
             * @param initial Function to compute the value at each address.
             */
            template<typename Func>
                requires OrderedGrid<Grid>
            OperationalFunction(const Grid& grid, Func&& initial)
                : values_(grid.comparator()) // use the grid's comparator for ordering
            {
                for (const auto& addr : grid) {
                    values_[addr] = initial(addr);
                }
            }

            /**
             * @brief Extend the function to a refined grid.
             *
             * For every new address in `new_grid` that was not present in `old_grid`,
             * its value is computed using the provided interpolator. The old grid must
             * be a subset of the new grid (i.e., every address of `old_grid` appears in `new_grid`).
             *
             * @tparam OldGrid Type of the old grid (must satisfy GridConcept<OldGrid, Addr>).
             * @param old_grid  The previous (coarser) grid.
             * @param new_grid  The refined grid.
             * @param interpolate Interpolator used to compute values at new addresses.
             */
            template<typename OldGrid>
                requires OrderedGrid<OldGrid>
            void extend(const OldGrid& old_grid, const Grid& new_grid,
                Interpolator interpolate) {
                const std::size_t old_size = old_grid.size();
                const std::size_t new_size = new_grid.size();

                std::size_t old_idx = 0;
                for (std::size_t new_idx = 0; new_idx < new_size; ++new_idx) {
                    const Addr& addr = new_grid[new_idx];

                    if (values_.find(addr) == values_.end()) {
                        assert(old_idx + 1 < old_size && "No interval for new address");
                        const Addr& left = old_grid[old_idx];
                        const Addr& right = old_grid[old_idx + 1];
                        Value val = interpolate(left, right,
                            values_.at(left), values_.at(right));
                        values_[addr] = std::move(val);
                    }

                    if (old_idx + 1 < old_size && addr == old_grid[old_idx + 1]) {
                        ++old_idx;
                    }
                }

                assert(old_idx == old_size - 1 && "Did not consume all old addresses");
            }

            /**
             * @brief Retrieve the value at a given address.
             * @param addr The address (must be present in the function).
             * @return Const reference to the stored value.
             * @throws std::out_of_range if the address is not found.
             */
            const Value& operator()(const Addr& addr) const {
                auto it = values_.find(addr);
                if (it == values_.end()) {
                    throw std::out_of_range("Address not found in operational function");
                }
                return it->second;
            }

            /**
             * @brief Check whether a value is stored for the given address.
             */
            bool contains(const Addr& addr) const {
                return values_.find(addr) != values_.end();
            }

        private:
            std::map<Addr, Value, Compare> values_;  ///< Map from address to value, ordered by comparator.
    };

    // -------------------------------------------------------------------------
    // Specialization for UniformGrid (uniformly spaced grids)
    // -------------------------------------------------------------------------

    /**
     * @brief Specialization of OperationalFunction for uniform grids.
     *
     * Uses a vector for O(1) lookup by index instead of a map. Also handles
     * alignment for Eigen::MatrixXd values automatically.
     *
     * @tparam Addr    Address type (must be LinearAddress).
     * @tparam Value   Function value type.
     * @tparam Compare Comparison functor (used only for compatibility, not for storage).
     */
    template<typename Addr, typename Value, typename Compare>
    class OperationalFunction<Addr, Value, UniformGrid<Addr, Compare>> {
    public:
        using Grid = UniformGrid<Addr, Compare>;
        using Interpolator = std::function<Value(const Addr&, const Addr&,
            const Value&, const Value&)>;

        /// Storage type with correct alignment for Eigen::MatrixXd if needed.
        using StorageType = std::conditional_t<
            std::is_same_v<Value, Eigen::MatrixXd>,
            std::vector<Value, Eigen::aligned_allocator<Value>>,
            std::vector<Value>
        >;

        /**
         * @brief Construct from a uniform grid and an initial value generator.
         *
         * Values are stored in a vector in the same order as the grid points.
         *
         * @tparam Func Callable with signature Value(const Addr&).
         * @param grid The uniform grid.
         * @param initial Function to compute the value at each address.
         */
        template<typename Func>
        OperationalFunction(const Grid& grid, Func&& initial)
            : grid_(grid)
        {
            values_.reserve(grid.size());
            for (const auto& addr : grid) {
                values_.push_back(initial(addr));
            }
        }

        /**
         * @brief Extend the function to a finer uniform grid.
         *
         * The new grid must be a refinement of the old one, i.e., contain all old points
         * plus additional ones in between. New values are computed by interpolation.
         *
         * @param old_grid The previous (coarser) uniform grid.
         * @param new_grid The refined uniform grid.
         * @param interpolate Interpolator used to compute values at new addresses.
         */
        void extend(const Grid& old_grid, const Grid& new_grid,
            Interpolator interpolate) {
            std::size_t old_n = old_grid.size();
            std::size_t new_n = new_grid.size();

            StorageType new_values;
            new_values.reserve(new_n);

            std::size_t old_idx = 0;
            for (std::size_t new_idx = 0; new_idx < new_n; ++new_idx) {
                Addr addr = new_grid[new_idx];
                if (old_idx < old_n && addr == old_grid[old_idx]) {
                    // Existing point – copy old value.
                    new_values.push_back(values_[old_idx]);
                    ++old_idx;
                }
                else {
                    // New point – interpolate between the two surrounding old points.
                    assert(old_idx > 0 && old_idx < old_n && "Invalid interpolation interval");
                    const Addr& left = old_grid[old_idx - 1];
                    const Addr& right = old_grid[old_idx];
                    Value val = interpolate(left, right,
                        values_[old_idx - 1], values_[old_idx]);
                    new_values.push_back(std::move(val));
                }
            }
            assert(old_idx == old_n && "Did not consume all old addresses");

            values_ = std::move(new_values);
            grid_ = new_grid;
        }

        /**
         * @brief Retrieve the value at a given address.
         * @param addr The address (must be exactly on the grid).
         * @return Const reference to the stored value.
         * @throws std::runtime_error if addr is not exactly on the grid.
         * @throws std::out_of_range if the computed index is out of bounds.
         */
        const Value& operator()(const Addr& addr) const {
            std::size_t idx = detail::uniform_index(addr, grid_);
            return values_[idx];
        }

        /**
         * @brief Check whether the address belongs to the grid.
         */
        bool contains(const Addr& addr) const {
            try {
                detail::uniform_index(addr, grid_);
                return true;
            }
            catch (...) {
                return false;
            }
        }

        /// Returns the underlying vector of values (in grid order).
        const StorageType& values() const { return values_; }

    private:
        Grid grid_;           ///< The current uniform grid.
        StorageType values_;  ///< Values in the same order as grid points.
    };

    // -------------------------------------------------------------------------
    // FieldTraits для получения информации о поле
    // -------------------------------------------------------------------------

    template<typename Field>
    struct FieldTraits;

    template<typename Addr, typename Value, typename Grid, typename Compare>
    struct FieldTraits<OperationalFunction<Addr, Value, Grid, Compare>> {
        using address_type = Addr;
        using value_type = Value;
        using grid_type = Grid;
    };

    template<typename Addr, typename Value, typename Compare>
    struct FieldTraits<OperationalFunction<Addr, Value, UniformGrid<Addr, Compare>>> {
        using address_type = Addr;
        using value_type = Value;
        using grid_type = UniformGrid<Addr, Compare>;
    };

    // -------------------------------------------------------------------------
    // Concept Field для проверки, что тип является полем
    // -------------------------------------------------------------------------

    template<typename F, typename Addr>
    concept Field = requires(F f, const F cf, Addr a) {
        typename F::value_type;
        { cf(a) } -> std::convertible_to<typename F::value_type>;
        { cf.contains(a) } -> std::convertible_to<bool>;
    };

} // namespace delta