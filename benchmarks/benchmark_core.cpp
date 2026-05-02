// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0
//benchmarks/benchmark_core.cpp

// =============================================================================
// INTERPRETATION OF BENCHMARK RESULTS (Release build, modest dual-core 2.6 GHz)
// =============================================================================
//
// The following benchmarks demonstrate the practical performance of core
// Δ‑analysis components. Absolute timings reflect the modest hardware;
// relative comparisons are the robust guide.
//
// -----------------------------------------------------------------------------
// 1. Riemann sum performance (f(x)=x²)
// -----------------------------------------------------------------------------
// Four strategies are compared:
//   A) Dyadic (MidpointOperator) – uniform refinement.
//   B) FixedLambda (λ=1/3) – non‑uniform but static refinement.
//   C) AdaptiveOperator – changes insertion point but still refines EVERY interval.
//   D) AdaptiveDeltaPath – true adaptive path (priority queue, refines only
//      intervals with high deviation from linearity).
//
// Results (times, steps 5/10/15):
//   Dyadic:               31.8 μs → 1.12 ms → 56.8 ms
//   FixedLambda:          33.7 μs → 1.84 ms → 92.8 ms
//   AdaptiveOperator:    181   μs → 7.12 ms → 227  ms
//   AdaptiveDeltaPath:      4.7 μs →  8.8 μs →  12.8 μs
//
// Interpretation:
//   • A, B, C all refine every interval at each step → number of points grows
//     exponentially (2^steps+1). Hence the drastic time increase.
//   • AdaptiveOperator is slower than Dyadic because it computes extra
//     metrics (deviation, clamping) even though it still refines all intervals.
//   • AdaptiveDeltaPath refines only intervals where the deviation exceeds
//     a threshold. For f(x)=x², high curvature is localised near x=0.5.
//     Number of points grows slowly → time grows almost linearly.
//   • At 15 steps, AdaptiveDeltaPath is ~4500× faster than Dyadic.
//
// Significance:
//   True adaptivity (AdaptiveDeltaPath) is essential for functions with
//   localised high curvature. The simpler AdaptiveOperator does NOT reduce
//   the number of intervals; it only shifts the insertion point, and thus
//   cannot overcome exponential blow‑up.
//
// -----------------------------------------------------------------------------
// 2. OperationalFunction access: ListGrid vs UniformGrid
// -----------------------------------------------------------------------------
//   ListGrid (std::map):        290 ns (8) → 1280 ns (8192) – O(log n) growth.
//   UniformGrid (vector index): 900–1100 ns constant – O(1).
//
// Significance:
//   For uniformly spaced grids, the specialised OperationalFunction provides
//   constant‑time access, critical for performance in inner loops.
//
// -----------------------------------------------------------------------------
// 3. Overhead of advance() (MidpointOperator vs AdaptiveOperator)
// -----------------------------------------------------------------------------
//   MidpointOperator:        41 ms (15 steps)
//   AdaptiveOperator:        81 ms (15 steps) – about 2× slower.
//
// Interpretation:
//   • AdaptiveOperator performs extra work (value_metric, comparisons,
//     clamping, bounds checks) even when it falls back to the midpoint.
//   • This overhead is the price of flexibility. However, inside a true
//     adaptive path (AdaptiveDeltaPath), the per‑interval overhead is small
//     compared to the exponential reduction in the number of intervals.
//
// -----------------------------------------------------------------------------
// 4. Uniform vs Adaptive Δ‑paths for five test functions
// -----------------------------------------------------------------------------
//
// 4.1 |x-0.5| (single corner)
//     Uniform:    0.11 ms (ε=0.1) → 110 ms (ε=1e-4)
//     Adaptive:   constant ~0.11 ms (all ε)
//     → Adaptive ~1000× faster at ε=1e-4.
//     Only intervals crossing the corner are refined.
//
// 4.2 exp(-1000*(x-0.5)²) (narrow Gaussian peak)
//     Uniform:    6.7 ms (ε=0.1) → 5.8 s (ε=1e-4)
//     Adaptive:   0.28 ms → 10.4 ms
//     → Adaptive ~560× faster at ε=1e-4.
//
// 4.3 sin(100πx) (high‑frequency oscillations)
//     Uniform:    constant ~12 μs
//     Adaptive:   23 ms → 12.8 s (ε=1e-4) – catastrophic!
//     → Uniform up to 1,000,000× faster.
//     Explanation: The function oscillates uniformly; every interval has large
//     deviation → adaptive path refines everything but with huge overhead.
//     Uniform refinement is the correct choice here.
//
// 4.4 |x-0.25|+|x-0.75| (two corners)
//     Uniform:    0.36 ms → 430 ms (ε=1e-4)
//     Adaptive:   constant ~0.14 ms
//     → Adaptive ~3000× faster.
//
// 4.5 (x-0.5)³ (smooth cubic)
//     Uniform:    65 μs → 81 ms (ε=1e-4)
//     Adaptive:   125 μs → 1.2 ms
//     → Adaptive ~56× faster.
//
// -----------------------------------------------------------------------------
// CONCLUSIONS
// -----------------------------------------------------------------------------
// • AdaptiveDeltaPath works as designed: it concentrates points where the
//   function deviates from linearity, achieving massive speedups for functions
//   with localised features. Its runtime often becomes independent of ε.
// • The AdaptiveOperator is NOT a substitute for true adaptivity; it only
//   changes the insertion point but still refines all intervals.
// • Uniform refinement is superior for functions with uniform variation (e.g.,
//   high‑frequency sine). The library leaves the choice to the user.
// • The specialised UniformGrid OperationalFunction provides O(1) access,
//   essential for large uniform grids.
// • These benchmarks, run on modest hardware, confirm correctness and
//   illustrate the practical trade‑offs.
//
// =============================================================================

#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <vector>
#include <set>
#include <benchmark/benchmark.h>

// Delta headers
#include "delta/core/rational.h"
#include "delta/core/delta_path.h"
#include "delta/core/delta_strategy.h"
#include "delta/core/adaptive_delta_path.h"
#include "delta/core/list_grid.h"
#include "delta/core/regulative_idea.h"
#include "delta/core/value_metric.h"
#include "delta/core/delta_operator.h"
#include "delta/core/operational_function.h"
#include "delta/core/uniform_grid.h"
#include "delta/calculus/riemann_sum.h"

using namespace delta;
using delta::operator""_r;

// -----------------------------------------------------------------------------
// Common type aliases
// -----------------------------------------------------------------------------
using Addr = Rational;
using Val = Rational;
using Dist = Rational;
using Between = LessBetweenness;
using AddrMetric = EuclideanMetric;
using ValMetric = EuclideanValueMetric;
using Compare = std::less<Addr>;

// =============================================================================
// PART 1 – Riemann sum performance on different grids
// =============================================================================

template<typename Set, typename Func>
Rational left_riemann_sum_set(const Set& points, Func&& func) {
    if (points.size() < 2) return 0_r;
    auto it = points.begin();
    auto next = std::next(it);
    Rational sum = 0_r;
    while (next != points.end()) {
        sum += func(*it) * (*next - *it);
        ++it; ++next;
    }
    return sum;
}

static void BM_RiemannSumDyadic(benchmark::State& state) {
    ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
    MidpointOperator op;
    auto strategy = StaticStrategy<MidpointOperator>(op);
    auto func = [](const Addr& x) { return x * x; };
    DeltaPath<Addr, Val, Dist, Between, AddrMetric, ValMetric,
        decltype(strategy), Compare>
        path(grid0, strategy, Between{}, AddrMetric{}, ValMetric{});

    for (int i = 0; i < state.range(0); ++i) path.advance(func);
    const auto& final_grid = path.current_grid();
    for (auto _ : state) {
        Rational sum = calculus::left_riemann_sum(final_grid, func);
        benchmark::DoNotOptimize(sum);
    }
}

static void BM_RiemannSumFixedLambda(benchmark::State& state) {
    ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
    FixedLambdaOperator op(1_r / 3_r);
    auto strategy = StaticStrategy<FixedLambdaOperator>(op);
    auto func = [](const Addr& x) { return x * x; };
    DeltaPath<Addr, Val, Dist, Between, AddrMetric, ValMetric,
        decltype(strategy), Compare>
        path(grid0, strategy, Between{}, AddrMetric{}, ValMetric{});

    for (int i = 0; i < state.range(0); ++i) path.advance(func);
    const auto& final_grid = path.current_grid();
    for (auto _ : state) {
        Rational sum = calculus::left_riemann_sum(final_grid, func);
        benchmark::DoNotOptimize(sum);
    }
}

static void BM_RiemannSumAdaptiveOperator(benchmark::State& state) {
    ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
    AdaptiveOperator op(1_r / 10_r, 1_r / 10_r);
    auto strategy = StaticStrategy<AdaptiveOperator>(op);
    auto func = [](const Addr& x) { return x * x; };
    DeltaPath<Addr, Val, Dist, Between, AddrMetric, ValMetric,
        decltype(strategy), Compare>
        path(grid0, strategy, Between{}, AddrMetric{}, ValMetric{});

    for (int i = 0; i < state.range(0); ++i) path.advance(func);
    const auto& final_grid = path.current_grid();
    for (auto _ : state) {
        Rational sum = calculus::left_riemann_sum(final_grid, func);
        benchmark::DoNotOptimize(sum);
    }
}

static void BM_RiemannSumAdaptivePath(benchmark::State& state) {
    std::vector<Addr> init = { 0_r, 1_r };
    MidpointOperator op;
    ValMetric vm;
    Dist threshold = 1_r / 1000_r;
    auto path = AdaptiveDeltaPath<Addr, Val, Dist, Between, AddrMetric, ValMetric,
        MidpointOperator, Compare>::from_uniform(
            init, [](const Addr& x) { return x * x; }, op, 0, threshold,
            Between{}, AddrMetric{}, vm);
    for (int i = 0; i < state.range(0); ++i) path.advance();
    const auto& pts = path.points();
    for (auto _ : state) {
        Rational sum = left_riemann_sum_set(pts, [](const Addr& x) { return x * x; });
        benchmark::DoNotOptimize(sum);
    }
}

BENCHMARK(BM_RiemannSumDyadic)->Arg(5)->Arg(10)->Arg(15);
BENCHMARK(BM_RiemannSumFixedLambda)->Arg(5)->Arg(10)->Arg(15);
BENCHMARK(BM_RiemannSumAdaptiveOperator)->Arg(5)->Arg(10)->Arg(15);
BENCHMARK(BM_RiemannSumAdaptivePath)->Arg(5)->Arg(10)->Arg(15);

// =============================================================================
// PART 2 – OperationalFunction access: ListGrid vs UniformGrid
// =============================================================================

static void BM_OpFuncMap(benchmark::State& state) {
    std::size_t n = state.range(0);
    std::vector<Addr> points;
    for (std::size_t i = 0; i < n; ++i) points.push_back(Addr(static_cast<int64_t>(i)));
    ListGrid<Addr, Compare> grid(points.begin(), points.end());
    OperationalFunction<Addr, Val, decltype(grid)> func(grid,
        [](const Addr& x) { return x; });
    Addr mid = Addr(static_cast<int64_t>(n / 2));
    for (auto _ : state) {
        Val v = func(mid);
        benchmark::DoNotOptimize(v);
    }
}

static void BM_OpFuncUniform(benchmark::State& state) {
    std::size_t n = state.range(0);
    UniformGrid<Addr, Compare> grid(0_r, 1_r, n);
    OperationalFunction<Addr, Val, decltype(grid)> func(grid,
        [](const Addr& x) { return x; });
    Addr mid = Addr(static_cast<int64_t>(n / 2));
    for (auto _ : state) {
        Val v = func(mid);
        benchmark::DoNotOptimize(v);
    }
}

BENCHMARK(BM_OpFuncMap)->Range(8, 8 << 10);
BENCHMARK(BM_OpFuncUniform)->Range(8, 8 << 10);

// =============================================================================
// PART 3 – Overhead of advance() method (MidpointOperator vs AdaptiveOperator)
// =============================================================================

static void BM_AdvanceOverheadMidpoint(benchmark::State& state) {
    ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
    MidpointOperator op;
    auto strategy = StaticStrategy<MidpointOperator>(op);
    auto func = [](const Addr& x) { return x; };
    for (auto _ : state) {
        DeltaPath<Addr, Val, Dist, Between, AddrMetric, ValMetric,
            decltype(strategy), Compare>
            path(grid0, strategy, Between{}, AddrMetric{}, ValMetric{});
        for (int i = 0; i < state.range(0); ++i) path.advance(func);
        benchmark::DoNotOptimize(path.current_grid().size());
    }
}

static void BM_AdvanceOverheadAdaptiveOperator(benchmark::State& state) {
    ListGrid<Addr, Compare> grid0({ 0_r, 1_r });
    AdaptiveOperator adapt_op(1_r / 100_r, 1_r / 100_r);
    auto strategy = StaticStrategy<AdaptiveOperator>(adapt_op);
    auto func = [](const Addr& x) { return x; };
    for (auto _ : state) {
        DeltaPath<Addr, Val, Dist, Between, AddrMetric, ValMetric,
            decltype(strategy), Compare>
            path(grid0, strategy, Between{}, AddrMetric{}, ValMetric{});
        for (int i = 0; i < state.range(0); ++i) path.advance(func);
        benchmark::DoNotOptimize(path.current_grid().size());
    }
}

BENCHMARK(BM_AdvanceOverheadMidpoint)->Arg(5)->Arg(10)->Arg(15);
BENCHMARK(BM_AdvanceOverheadAdaptiveOperator)->Arg(5)->Arg(10)->Arg(15);

// =============================================================================
// PART 4 – Comparison of Uniform and Adaptive Δ-paths for test functions
// =============================================================================

Val test_function_abs(const Addr& x) {
    Rational half = 1_r / 2_r;
    return (x < half) ? half - x : x - half;
}

// use double for the sake of the speed of the benchmark. 
// We only need comparative speed between high-level approaches, thus ruling out
// the low-level speed with which we calculate Rational Transcendentals should be irrelevant.
Val test_function_peak(const Addr& x) {
    double t = (x - 1_r / 2_r).convert_to<double>();
    double val = std::exp(-1000.0 * t * t);
    return Rational(static_cast<int64_t>(val * 1e12), 1e12);
}
Val test_function_osc(const Addr& x) {
    double t = x.convert_to<double>();
    double val = std::sin(100.0 * M_PI * t);
    return Rational(static_cast<int64_t>(val * 1e12), 1e12);
}
Val test_function_two_corners(const Addr& x) {
    Rational q1 = 1_r / 4_r, q2 = 3_r / 4_r;
    Rational part1 = (x < q1) ? (q1 - x) : (x - q1);
    Rational part2 = (x < q2) ? (q2 - x) : (x - q2);
    return part1 + part2;
}
Val test_function_cubic(const Addr& x) {
    Rational mid = x - 1_r / 2_r;
    return mid * mid * mid;
}

template<typename Grid>
Dist max_oscillation(const Grid& grid, const std::function<Val(Addr)>& func, const ValMetric& vm) {
    Dist max_osc = 0_r;
    for (std::size_t i = 0; i + 1 < grid.size(); ++i) {
        Dist d = vm(func(grid[i]), func(grid[i + 1]));
        if (d > max_osc) max_osc = d;
    }
    return max_osc;
}

#define UNIFORM_BENCHMARK(name, func) \
static void BM_UniformToEpsilon_##name(benchmark::State& state) { \
    Dist epsilon = 1_r / static_cast<int64_t>(state.range(0)); \
    ListGrid<Addr, Compare> grid0({0_r, 1_r}); \
    MidpointOperator op; \
    auto strategy = StaticStrategy<MidpointOperator>(op); \
    ValMetric vm; \
    for (auto _ : state) { \
        DeltaPath<Addr, Val, Dist, Between, AddrMetric, ValMetric, \
                  decltype(strategy), Compare> \
            path(grid0, strategy, Between{}, AddrMetric{}, vm); \
        Dist osc; \
        do { \
            path.advance(func); \
            osc = max_oscillation(path.current_grid(), func, vm); \
        } while (osc > epsilon); \
        benchmark::DoNotOptimize(osc); \
    } \
} \
BENCHMARK(BM_UniformToEpsilon_##name)->Arg(10)->Arg(100)->Arg(1000)->Arg(10000);

#define ADAPTIVE_BENCHMARK(name, func) \
static void BM_AdaptiveToEpsilon_##name(benchmark::State& state) { \
    Dist epsilon = 1_r / static_cast<int64_t>(state.range(0)); \
    std::vector<Addr> init = {0_r, 1_r}; \
    MidpointOperator op; \
    ValMetric vm; \
    const std::size_t uniform_levels = 3; \
    for (auto _ : state) { \
        auto path = AdaptiveDeltaPath<Addr, Val, Dist, Between, AddrMetric, ValMetric, \
                                      MidpointOperator, Compare>::from_uniform( \
                init, func, op, uniform_levels, epsilon, \
                Between{}, AddrMetric{}, vm); \
        while (path.advance()) {} \
        benchmark::DoNotOptimize(path.points().size()); \
    } \
} \
BENCHMARK(BM_AdaptiveToEpsilon_##name)->Arg(10)->Arg(100)->Arg(1000)->Arg(10000);

UNIFORM_BENCHMARK(Abs, test_function_abs)
ADAPTIVE_BENCHMARK(Abs, test_function_abs)

UNIFORM_BENCHMARK(Peak, test_function_peak)
ADAPTIVE_BENCHMARK(Peak, test_function_peak)

UNIFORM_BENCHMARK(Osc, test_function_osc)
ADAPTIVE_BENCHMARK(Osc, test_function_osc)

UNIFORM_BENCHMARK(TwoCorners, test_function_two_corners)
ADAPTIVE_BENCHMARK(TwoCorners, test_function_two_corners)

UNIFORM_BENCHMARK(Cubic, test_function_cubic)
ADAPTIVE_BENCHMARK(Cubic, test_function_cubic)

// =============================================================================
//                          CUSTOM REPORTER
// =============================================================================

class GroupReporter : public benchmark::ConsoleReporter {
    std::set<std::string> printed_groups_;

    std::string group_of(const std::string& name) {
        if (name.find("BM_RiemannSum") != std::string::npos) return "RiemannSum";
        if (name.find("BM_OpFunc") != std::string::npos) return "OpFunc";
        if (name.find("BM_Advance") != std::string::npos) return "Advance";
        if (name.find("BM_UniformToEpsilon") != std::string::npos ||
            name.find("BM_AdaptiveToEpsilon") != std::string::npos) return "AdaptiveVsUniform";
        return "";
    }
    void print_description(const std::string& group) {
        if (group == "RiemannSum") {
            std::cout << R"(
================================================================================
         Riemann sum computation performance on different grids
================================================================================
What is measured: time to compute the left Riemann sum of f(x)=x^2
on a grid obtained after a fixed number of refinement steps using
four different delta-strategies:
  - Dyadic (MidpointOperator) – uniform refinement
  - FixedLambda (lambda=1/3) – non-uniform but static
  - AdaptiveOperator – insertion point adapts to function values
  - AdaptiveDeltaPath – truly adaptive path (priority = deviation)
The number of steps is 5, 10, 15 (starting from grid {0,1}).
For AdaptiveDeltaPath, the threshold is set to 1/1000 to force refinement.

Expected behavior:
  - Dyadic and FixedLambda grids are deterministic and grow exponentially;
    sum computation time should increase dramatically with step count.
  - AdaptiveOperator produces similar exponential growth (all intervals refined).
  - AdaptiveDeltaPath concentrates points near the centre (high curvature),
    so grid size stays small even after many steps -> sum time remains low.
================================================================================

)";
        }
        else if (group == "OpFunc") {
            std::cout << R"(
================================================================================
         OperationalFunction access performance: ListGrid vs UniformGrid
================================================================================
What is measured: time to retrieve a function value at a given address
for two different grid implementations:
  - ListGrid: general OperationalFunction uses std::map (O(log n) lookup)
  - UniformGrid: specialized version uses vector and direct index (O(1) access)
The function is identity f(x)=x, and the queried address is the middle point.
Grid sizes range from 8 to 8192 points.

Expected behavior:
  - ListGrid version should show increasing time with grid size (logarithmic).
  - UniformGrid version should be nearly constant, independent of size.
  - The difference demonstrates the importance of the specialization
    for regularly spaced grids.
================================================================================

)";
        }
        else if (group == "Advance") {
            std::cout << R"(
================================================================================
         Measuring the overhead of the advance() method
================================================================================
What is measured: execution time of a fixed number of advance() calls
for two different Delta-operators inside the same DeltaPath (non-adaptive path):
  - MidpointOperator (simple arithmetic mean)
  - AdaptiveOperator (computes insertion point based on interval info)
The function is linear f(x)=x in both cases, so the AdaptiveOperator always
falls back to the midpoint (since max_oscillation=0 or df <= threshold).
Thus we measure purely the extra computations performed by the adaptive
operator (value_metric calls, divisions, comparisons) per interval.
Parameter n = 5, 10, 15 - number of consecutive advance() calls.

Expected behavior:
  - Time grows exponentially with n because each advance() processes
    all intervals of the current grid (whose number doubles each step).
  - AdaptiveOperator should be slower due to extra logic.
  - The slowdown factor shows the cost of the adaptive operator's logic
    relative to a simple midpoint.

These results help quantify the price of using a more sophisticated operator
within a uniform refinement scheme. In combination with the true adaptive path
(AdaptiveDeltaPath), this overhead is outweighed by the reduction in the total
number of intervals needed to achieve a given accuracy.
================================================================================

)";
        }
        else if (group == "AdaptiveVsUniform") {
            std::cout << R"(
================================================================================
         Comparison of Uniform and Adaptive delta-paths
================================================================================
What is measured: time (in nanoseconds) spent to achieve a given
accuracy epsilon (maximum oscillation) for various test functions.
epsilon takes values: 0.1, 0.01, 0.001, 0.0001 (corresponding to arguments 10,100,1000,10000).

Uniform path: at each step, midpoints of all intervals are inserted.
Adaptive path: refines only those intervals where the deviation from
linear interpolation (deviation) exceeds epsilon. Initial uniform exploration: 3 levels.

Expected behavior:
  - For functions with localized features (corner, narrow peak) the adaptive path
    will be significantly faster, especially for small epsilon.
  - For uniformly oscillating functions (sin(100*pi*x)) there will be no gain, possibly
    even slowdown due to overhead.
  - For functions with two corners, adaptivity is also efficient, but the number of points
    will be about twice as many as for one corner.
  - For smooth functions with increased curvature (cubic) adaptivity may provide
    a moderate gain, since curvature is distributed over the entire interval.

Test functions:
  1. Abs          - |x-0.5| (corner at center)
  2. Peak         - exp(-1000*(x-0.5)^2) (narrow Gaussian peak)
  3. Osc          - sin(100*pi*x) (high-frequency oscillations)
  4. TwoCorners   - |x-0.25| + |x-0.75| (two corners)
  5. Cubic        - (x-0.5)^3 (smooth cubic)
================================================================================

)";
        }
    }
public:
    void ReportRuns(const std::vector<Run>& reports) override {
        for (const auto& run : reports) {
            std::string group = group_of(run.benchmark_name());
            if (!group.empty() && printed_groups_.find(group) == printed_groups_.end()) {
                print_description(group);
                printed_groups_.insert(group);
            }
        }
        benchmark::ConsoleReporter::ReportRuns(reports);
    }
};

// =============================================================================
//                                   MAIN
// =============================================================================

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv); 
    GroupReporter reporter;
    benchmark::RunSpecifiedBenchmarks(&reporter);
    return 0;
}