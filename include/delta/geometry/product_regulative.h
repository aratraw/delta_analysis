// include/delta/geometry/product_regulative.h
#pragma once

/**
 * @file product_regulative.h
 * @brief Произведение регулятивных идей и путей.
 *
 * ⚠️ ВАЖНО ⚠️
 * ВСЕ РЕГУЛЯТИВНЫЕ ИДЕИ В ОДНОМ ПРОИЗВЕДЕНИИ ДОЛЖНЫ БЫТЬ ОДНОГО ТИПА!
 * - В ProductRegulativeIdea типы RI1 и RI2 обязаны совпадать.
 * - В ProductDeltaPath все пути обязаны быть одного типа.
 *
 * НЕ ДОПУСКАЕТСЯ комбинирование идей на матрицах и бинарных строках в одном произведении.
 */

#include <tuple>
#include <array>
#include <utility>
#include <algorithm>
#include <functional>
#include <type_traits>
#include "delta/core/regulative_idea.h"
#include "delta/core/product_grid.h"

namespace delta::geometry {

    // -------------------------------------------------------------------------
    // Вспомогательные шаблоны для комбинирования betweenness и metric
    // -------------------------------------------------------------------------

    namespace detail {

        /**
         * @brief Комбинированное отношение "между" для произведения двух идей.
         *
         * Точка y находится между x и z тогда и только тогда, когда
         * это выполняется для каждой координаты по отдельности.
         */
        template<typename B1, typename B2>
        struct ProductBetweenness {
            ProductBetweenness() = default;
            ProductBetweenness(const B1& b1, const B2& b2) : b1_(b1), b2_(b2) {}

            // Универсальный operator() - принимает ЛЮБЫЕ типы с .first и .second
            template<typename T1, typename T2, typename T3>
            bool operator()(const T1& x, const T2& y, const T3& z) const {
                return b1_(x.first, y.first, z.first) &&
                    b2_(x.second, y.second, z.second);
            }

        private:
            B1 b1_;
            B2 b2_;
        };

        /**
         * @brief Комбинированная метрика для произведения двух идей.
         *
         * Использует max-метрику: расстояние = max( distance1, distance2 ).
         * Это естественный выбор для произведения метрических пространств.
         */
        template<typename M1, typename M2>
        struct ProductMetric {
            ProductMetric() = default;
            ProductMetric(const M1& m1, const M2& m2) : m1_(m1), m2_(m2) {}

            // Возвращаемый тип выводится автоматически
            template<typename T1, typename T2>
            auto operator()(const T1& a, const T2& b) const {
                auto d1 = m1_(a.first, b.first);
                auto d2 = m2_(a.second, b.second);
                return (d1 > d2) ? d1 : d2;
            }

        private:
            M1 m1_;
            M2 m2_;
        };
        /**
         * @brief Комбинированное отношение "между" для массива идей (n копий).
         */
        template<typename B, std::size_t N>
        struct PowerBetweenness {
            using address_type = std::array<typename B::first_argument_type, N>;
            using result_type = bool;

            PowerBetweenness() = default;

            explicit PowerBetweenness(const B& b) : b_(b) {}

            bool operator()(const address_type& x,
                const address_type& y,
                const address_type& z) const {
                for (std::size_t i = 0; i < N; ++i) {
                    if (!b_(x[i], y[i], z[i])) {
                        return false;
                    }
                }
                return true;
            }

            B b_;
        };

        /**
         * @brief Комбинированная метрика для массива идей (n копий).
         *
         * Использует max-метрику по всем координатам.
         */
        template<typename M, std::size_t N>
        struct PowerMetric {
            using address_type = std::array<typename M::first_argument_type, N>;
            using result_type = typename M::result_type;

            PowerMetric() = default;

            explicit PowerMetric(const M& m) : m_(m) {}

            result_type operator()(const address_type& a,
                const address_type& b) const {
                result_type max_dist = 0;
                for (std::size_t i = 0; i < N; ++i) {
                    auto dist = m_(a[i], b[i]);
                    if (dist > max_dist) {
                        max_dist = dist;
                    }
                }
                return max_dist;
            }

            M m_;
        };

    } // namespace detail

    // -------------------------------------------------------------------------
    // ProductRegulativeIdea - произведение двух регулятивных идей
    // -------------------------------------------------------------------------

    /**
     * @brief Произведение двух регулятивных идей.
     *
     * Объединяет две регулятивные идеи в одну, работающую с парами адресов.
     * Адресом является std::pair<Addr1, Addr2>.
     * Betweenness выполняется покоординатно.
     * Метрика - максимум покоординатных расстояний (max-метрика).
     *
     * @tparam RI1 Первая регулятивная идея
     * @tparam RI2 Вторая регулятивная идея
     */
    template<typename RI1, typename RI2>
    class ProductRegulativeIdea {
        static_assert(std::is_same_v<RI1, RI2>,
            "ProductRegulativeIdea: обе идеи должны быть одного типа");
    public:
        // Типы составляющих идей
        using idea1_type = RI1;
        using idea2_type = RI2;
        using address1_type = typename RI1::address_type;
        using address2_type = typename RI2::address_type;
        using betweenness1_type = typename RI1::betweenness_type;
        using betweenness2_type = typename RI2::betweenness_type;
        using metric1_type = typename RI1::metric_type;
        using metric2_type = typename RI2::metric_type;

        // Типы результата
        using address_type = std::pair<address1_type, address2_type>;
        using betweenness_type = detail::ProductBetweenness<betweenness1_type, betweenness2_type>;
        using metric_type = detail::ProductMetric<metric1_type, metric2_type>;

        /**
         * @brief Конструктор от двух идей.
         * @param ri1 Первая идея
         * @param ri2 Вторая идея
         */
        ProductRegulativeIdea(const RI1& ri1 = RI1(), const RI2& ri2 = RI2())
            : ri1_(ri1), ri2_(ri2)
            , betweenness_(betweenness_type(ri1_.betweenness, ri2_.betweenness))
            , metric_(metric_type(ri1_.metric, ri2_.metric)) {
        }
        /**
         * @brief Доступ к комбинированному отношению betweenness.
         */
        const betweenness_type& betweenness() const {
            return betweenness_;
        }

        /**
         * @brief Доступ к комбинированной метрике.
         */
        const metric_type& metric() const {
            return metric_;
        }

        /**
         * @brief Доступ к первой идее.
         */
        const RI1& idea1() const {
            return ri1_;
        }

        /**
         * @brief Доступ ко второй идее.
         */
        const RI2& idea2() const {
            return ri2_;
        }

    private:
        RI1 ri1_;
        RI2 ri2_;
        betweenness_type betweenness_;
        metric_type metric_;
    };

    // -------------------------------------------------------------------------
    // PowerRegulativeIdea - n копий одной регулятивной идеи
    // -------------------------------------------------------------------------

    /**
     * @brief Степень регулятивной идеи (n копий).
     *
     * Создаёт регулятивную идею для ℝⁿ из одной идеи для ℝ.
     * Адресом является std::array<Addr, N>.
     *
     * @tparam RI Базовая регулятивная идея (для 1D)
     * @tparam N Количество копий (размерность)
     */
    template<typename RI, int N>
    class PowerRegulativeIdea {
        static_assert(N > 0, "PowerRegulativeIdea requires positive N");
        static_assert(N <= 10, "PowerRegulativeIdea supports up to 10 dimensions");

    public:
        using base_idea_type = RI;
        using base_address_type = typename RI::address_type;
        using base_betweenness_type = typename RI::betweenness_type;
        using base_metric_type = typename RI::metric_type;

        using address_type = std::array<base_address_type, N>;
        using betweenness_type = detail::PowerBetweenness<base_betweenness_type, N>;
        using metric_type = detail::PowerMetric<base_metric_type, N>;

        /**
         * @brief Конструктор от базовой идеи.
         * @param ri Базовая идея (по умолчанию сконструированная)
         */
        explicit PowerRegulativeIdea(const RI& ri = RI())
            : ri_(ri)
            , betweenness_(ri_.betweenness())
            , metric_(ri_.metric()) {
        }

        /**
         * @brief Доступ к комбинированному отношению betweenness.
         */
        const betweenness_type& betweenness() const {
            return betweenness_;
        }

        /**
         * @brief Доступ к комбинированной метрике.
         */
        const metric_type& metric() const {
            return metric_;
        }

        /**
         * @brief Доступ к базовой идее.
         */
        const RI& base_idea() const {
            return ri_;
        }

    private:
        RI ri_;
        betweenness_type betweenness_;
        metric_type metric_;
    };

    // -------------------------------------------------------------------------
    // ProductDeltaPath - произведение путей
    // -------------------------------------------------------------------------

    /**
     * @brief Произведение нескольких путей.
     *
     * Объединяет несколько путей в один, сеткой которого является
     * декартово произведение сеток составляющих путей.
     *
     * @tparam Paths Типы путей (должны быть все одинаковы)
     */
    template<typename... Paths>
    class ProductDeltaPath {
        static_assert(sizeof...(Paths) > 0, "ProductDeltaPath requires at least one path");

        // Проверяем, что все пути одного типа
        using FirstPath = std::tuple_element_t<0, std::tuple<Paths...>>;
        static_assert((std::is_same_v<FirstPath, Paths> && ...),
            "All paths in ProductDeltaPath must have the same type");

    public:
        using Addr = typename FirstPath::addr_type;
        using Value = typename FirstPath::value_type;

        // Тип функции для произведения путей: принимает массив адресов, возвращает массив значений
        using Func = std::function<std::array<Value, sizeof...(Paths)>(const std::array<Addr, sizeof...(Paths)>&)>;

        // Типы составляющих путей
        using path_types = std::tuple<Paths...>;
        static constexpr std::size_t num_paths = sizeof...(Paths);

        // Тип сетки - произведение сеток (все одного типа)
        using grid_type = delta::ProductGrid<typename FirstPath::grid_type, num_paths>;

        // Тип метрики - берём из первого пути (все метрики одинаковы)
        using metric_type = typename FirstPath::metric_type;

        /**
         * @brief Конструктор от набора путей.
         * @param paths Кортеж путей
         */
        explicit ProductDeltaPath(Paths... paths)
            : paths_(std::move(paths)...) {
        }

        /**
         * @brief Конструктор от кортежа путей.
         * @param paths Кортеж путей
         */
        explicit ProductDeltaPath(std::tuple<Paths...> paths)
            : paths_(std::move(paths)) {
        }

        /**
         * @brief Выполнить один шаг подразделения для всех путей.
         *
         * @param func Функция, которая принимает массив адресов (по одному из каждого пути)
         *             и возвращает массив значений (по одному для каждого пути).
         */
        void advance(const Func& func) {
            // Получаем текущие адреса из всех путей
            auto current_addrs = current_addresses();

            // Для каждого пути создаём функцию, которая:
            // - принимает новый адрес для этого пути
            // - формирует полный массив адресов (фиксируя остальные из current_addrs)
            // - вызывает func и возвращает соответствующее значение
            std::apply([&](auto&... p) {
                [&] <std::size_t... Is>(std::index_sequence<Is...>) {
                    (p.advance([&](const Addr& new_addr) -> Value {
                        std::array<Addr, num_paths> addrs = current_addrs;
                        addrs[Is] = new_addr;
                        return func(addrs)[Is];
                        }), ...);
                }(std::index_sequence_for<Paths...>{});
                }, paths_);
        }

        /**
         * @brief Получить текущую сетку (произведение сеток всех путей).
         */
        grid_type current_grid() const {
            auto grids = std::apply([](const auto&... p) {
                return std::array<typename FirstPath::grid_type, num_paths>{ p.current_grid()... };
                }, paths_);
            return grid_type(std::move(grids));
        }

        /**
         * @brief Получить текущие адреса из каждого пути (последний элемент сетки).
         */
        std::array<Addr, num_paths> current_addresses() const {
            return std::apply([](const auto&... p) {
                return std::array<Addr, num_paths>{
                    p.current_grid()[p.current_grid().size() - 1]...
                };
                }, paths_);
        }

        /**
         * @brief Получить текущий уровень подразделения.
         */
        std::size_t level() const {
            return std::get<0>(paths_).level();
        }

        /**
         * @brief Вычислить максимальный разрыв в метрике.
         */
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

        /**
         * @brief Доступ к кортежу путей.
         */
        const std::tuple<Paths...>& paths() const {
            return paths_;
        }

        /**
         * @brief Доступ к метрике первого пути.
         */
        const metric_type& metric() const {
            return std::get<0>(paths_).metric();
        }

    private:
        std::tuple<Paths...> paths_;
    };

    // -------------------------------------------------------------------------
    // Вспомогательные функции для создания ProductDeltaPath
    // -------------------------------------------------------------------------

    /**
     * @brief Создать ProductDeltaPath из двух путей.
     */
    template<typename Path1, typename Path2>
    ProductDeltaPath<Path1, Path2> make_product_path(Path1 p1, Path2 p2) {
        static_assert(std::is_same_v<Path1, Path2>,
            "make_product_path: paths must be of the same type");
        return ProductDeltaPath<Path1, Path2>(std::move(p1), std::move(p2));
    }

    /**
     * @brief Создать ProductDeltaPath из трёх путей.
     */
    template<typename Path1, typename Path2, typename Path3>
    ProductDeltaPath<Path1, Path2, Path3> make_product_path(Path1 p1, Path2 p2, Path3 p3) {
        static_assert(std::is_same_v<Path1, Path2> && std::is_same_v<Path1, Path3>,
            "make_product_path: all paths must be of the same type");
        return ProductDeltaPath<Path1, Path2, Path3>(std::move(p1), std::move(p2), std::move(p3));
    }

    // -------------------------------------------------------------------------
    // Трейты для определения типов
    // -------------------------------------------------------------------------

    /**
     * @brief Трейт для определения типа адреса продукта идей.
     */
    template<typename... RIs>
    struct product_address_type;

    template<typename RI1, typename RI2>
    struct product_address_type<RI1, RI2> {
        using type = std::pair<typename RI1::address_type, typename RI2::address_type>;
    };

    template<typename RI, int N>
    struct product_address_type<PowerRegulativeIdea<RI, N>> {
        using type = std::array<typename RI::address_type, N>;
    };

    /**
     * @brief Трейт для определения типа метрики продукта идей.
     */
    template<typename... RIs>
    struct product_metric_type;

    template<typename RI1, typename RI2>
    struct product_metric_type<RI1, RI2> {
        using type = detail::ProductMetric<
            typename RI1::metric_type,
            typename RI2::metric_type
        >;
    };

    template<typename RI, int N>
    struct product_metric_type<PowerRegulativeIdea<RI, N>> {
        using type = detail::PowerMetric<typename RI::metric_type, N>;
    };

} // namespace delta::geometry