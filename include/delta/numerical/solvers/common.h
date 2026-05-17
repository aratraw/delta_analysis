// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// include/delta/numerical/solvers/common.h
// ============================================================================
// COMMON FACILITIES FOR NUMERICAL SOLVERS
// ============================================================================
//
// This header collects types, constants and helper functions that are shared
// by two or more solvers.  It is intended to grow as the library evolves.
//
// • Put an entity here ONLY IF at least two solver modules need it.
// ----------------------------------------------------------------------------
// CURRENT CONTENTS
// ----------------------------------------------------------------------------
//   - TimeScheme enumeration (used by heat, wave, advection solvers).
//
// ----------------------------------------------------------------------------
// PLANNED FUTURE ADDITIONS
// ----------------------------------------------------------------------------
//   1. Assembly helpers
//        template<typename Scalar, typename Grid2D>
//        Eigen::SparseMatrix<Scalar> assemble_laplacian_5pt(const Grid2D& grid);
//
//        template<typename Scalar, typename Grid2D>
//        Eigen::Matrix<Scalar, Dynamic, 1> lumped_mass_vector(const Grid2D& grid);
//
//      These are currently duplicated in poisson_solver and heat_solver.
//      Extracting them will eliminate duplication and simplify new solvers.
//
//   2. Time‑stepping utilities
//      – compute_num_steps(T, dt)  (handles rounding and last‑step adjustment).
//      – adaptive time‑step selectors for stiff problems.
//
//   3. Common linear algebra wrappers
//      – thin wrappers around Eigen::SparseLU / Eigen::BiCGSTAB that throw
//        meaningful exceptions and optionally log performance.
//
//   4. Solver traits and concepts
//      – concept StaticSolver, TimeDependentSolver, etc. (once C++20 concepts
//        are fully available and the architecture stabilises).
//
//   5. Default parameters
//      – default_alpha, default_max_iterations, …
//
// ============================================================================
// =============================================================================
// РАЦИОНАЛЬНАЯ АРИФМЕТИКА И ЧИСЛЕННЫЕ СВОЙСТВА СОЛВЕРОВ
// =============================================================================
//
// Все солверы библиотеки работают с типом delta::Rational, который реализует
// точную рациональную арифметику (сокращение дробей, произвольная точность).
// Трансцендентные функции (sin, cos, exp, pi) вычисляются с контролируемой
// абсолютной погрешностью, задаваемой через delta::default_eps() (по умолчанию
// ~1e-30). Это фундаментально меняет численное поведение по сравнению с
// плавающей точкой (double/float):
//
// 1. ОТСУТСТВИЕ ШУМА ОКРУГЛЕНИЯ
//    Каждое арифметическое действие выполняется точно. Нет «потерянных битов»
//    мантиссы, нет накопления машинного эпсилона. Благодаря этому:
//    • Симметрии и законы сохранения выполняются с машинной точностью.
//    • Паразитные высокочастотные моды, возбуждаемые шумом в double,
//      здесь не возникают – начальные данные, содержащие только низкие
//      гармоники, не порождают высокочастотных осцилляций.
//    • Явные схемы, превысившие порог устойчивости, могут НЕ показывать
//      катастрофического роста, если начальные данные строго принадлежат
//      устойчивому подпространству. Рост будет ограничен или проявится
//      лишь через очень большое число шагов.
//
// 2. ОШИБКА ОТСЕЧЕНИЯ ТРАНСЦЕНДЕНТНЫХ ФУНКЦИЙ
//    Начальные и граничные условия, а также точные решения, записанные
//    через sin, exp, pi, являются рациональными приближениями с гарантиро-
//    ванной точностью eps. Эта погрешность:
//    • Детерминирована и гладка – не является случайным шумом.
//    • Пренебрежимо мала при стандартном eps = 1e-30.
//    • Может быть дополнительно уменьшена пользователем (set_default_eps).
//    В тестах на устойчивость эта погрешность может проецироваться на
//    неустойчивые моды, но с амплитудой порядка eps, что делает её влияние
//    незначительным на временах порядка сотен шагов.
//
//    ВАЖНО: композиция трансцендентных функций увеличивает ошибку
//    предсказуемым, детерминированным образом. Например, вычисление
//    sin(2 * delta::pi(eps)) с eps=1e-30 не даст точного нуля, хотя
//    математически sin(2π)=0. Ошибка масштабируется, но её величину
//    можно оценить заранее, зная реализацию каждого вызова. Если когда‑либо
//    потребуется точное значение sin(2π)=0, достаточно включить простые
//    Expression Templates (или использовать уже существующий LazyRational),
//    который выполняет алгебраические упрощения до численного вычисления.
//    В текущей версии такие упрощения не применяются, и пользователь должен
//    учитывать эту особенность при задании аналитических решений.
//
// 3. ОТСУТСТВИЕ ДНА ТОЧНОСТИ
//    В отличие от double, ошибка дискретизации может убывать сколь угодно
//    далеко при измельчении сетки и уменьшении шага по времени, ограничиваясь
//    лишь вычислительными затратами (ростом знаменателей). Поэтому:
//    • Оценка порядка сходимости по двум-трём сеткам может давать
//      нерегулярные результаты (суперсходимость) для специальных решений.
//    • Наиболее надёжным инвариантом является монотонное убывание ошибки
//      при измельчении дискретизации.
//
// 4. ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ К ТЕСТИРОВАНИЮ
//    • Тесты на устойчивость: сравнивайте ошибку при устойчивом и неустойчивом
//      шаге – ошибка неустойчивого расчёта должна быть значимо больше.
//      Не ожидайте астрономических чисел (взрыва) при точной арифметике.
//    • Тесты на сходимость по времени: проверяйте, что при уменьшении Δt
//      ошибка монотонно убывает (строго меньше для меньшего Δt).
//      Конкретный порядок не гарантирован.
//    • Тесты на сходимость по пространству: при фиксированном достаточно
//      мелком Δt ошибка на более подробной сетке должна быть меньше.
//    • При долгих расчётах (тысячи шагов) погрешность трансцендентных
//      может накопиться – контролируйте её через delta::set_default_eps.
//
// =============================================================================

// =============================================================================
// BEYOND RATIONALS: THE PROMISE OF p‑ADIC SOLVERS
// =============================================================================
//
// The library already ships with p‑adic metrics and betweenness relations
// (PAdicMetric<p>, PAdicMidpointOperator, etc.), making it possible to
// formulate Δ‑paths over ultrametric address spaces.  Extending the solver
// framework to p‑adic (ultrametric) regularity ideas is a natural next step
// that promises several **qualitative** benefits beyond what rational
// arithmetic alone can offer:
//
// 1. **Built‑in multilevel hierarchy**
//    A p‑adic expansion naturally forms a tree.  Each refinement step
//    corresponds to appending a digit to the p‑adic string – the grid is
//    not a flat array but a prefix tree (trie).  This gives Adaptive Mesh
//    Refinement (AMR) “for free”: refining a region means adding children
//    to a node, without perturbing the global indexing.  Domain decomposition
//    for parallel solvers (MPI/OpenMP) becomes trivial because ultrametric
//    balls are either disjoint or nested – no partial overlaps.
//
// 2. **Hensel lifting instead of iterative Newton**
//    In a p‑adic solver, one first solves the problem modulo p (coarse
//    solution) and then lifts the solution to p², p⁴, … via Hensel’s lemma
//    in a single analytical step per doubling of precision.  This is
//    quadratic convergence *in digits*, not just in error norm, and the
//    result is an exact p‑adic integer built digit‑by‑digit, not an
//    approximation with a tolerance.
//
// 3. **Stiffness immunity**
//    The strong triangle inequality |x + y|ₚ ≤ max(|x|ₚ, |y|ₚ) guarantees
//    that errors (and perturbations) do NOT accumulate through addition:
//    the sum is never larger than the largest summand.  Hard multi‑scale
//    problems where float solvers lose small scales under large ones find a
//    natural home in p‑adic arithmetic – the information about microscopic
//    features is preserved in lower‑order digits without being crushed.
//
// 4. **Operator adaptation**
//    Classical differential operators must be replaced by p‑adic analogues
//    (e.g., the Vladimirov derivative), which measure change not as a slope
//    but as a variation with respect to the “depth” of detail.  The
//    Δ‑analysis framework can accommodate this seamlessly: the path and
//    operator interfaces are already parameterised by the regulative idea,
//    so plugging in a p‑adic regulative idea automatically provides the
//    correct betweenness, metric and refinement operators.
//
// -------------------------------------------------------------------------
// STATUS AND ROAD MAP
//   The foundational components (PAdicMetric, PAdicMidpointOperator) exist.
//   What remains to be done to realise p‑adic solvers:
//     • Implement Vladimirov‑type discrete operators for p‑adic addresses.
//     • Adapt assemble_laplacian_5pt (or its equivalent) to the ultrametric
//       adjacency structure.
//     • Design a Hensel‑lifting workflow that uses the existing path
//       hierarchy: solve modulo p → lift to p² → lift to p⁴ → …
//     • Provide solver specialisations for elliptic and parabolic problems
//       on p‑adic product grids.
//   This is a medium‑term goal that builds directly on the current
//   architecture and is fully aligned with the Δ‑analysis credo:
//   the continuum is what emerges from a *process*, and the process can
//   equally well be an ultrametric one.
// =============================================================================
#pragma once

#include <cstddef>
#include <Eigen/Sparse>
#include <array>
#include <vector>
#include <stdexcept>
#include "delta/core/uniform_grid.h"
#include "delta/core/product_grid.h"
#include "delta/rational/eigen_integration.h"

namespace delta::numerical::solvers {

    // =========================================================================
    // Time discretisation schemes
    // =========================================================================
    enum class TimeScheme {
        EXPLICIT_EULER,
        IMPLICIT_EULER,
        CRANK_NICOLSON
        // Leapfrog, Newmark, BDF2, … will be added when needed.
    };

    // =========================================================================
    // Assembly helpers
    // =========================================================================
    /**
     * @brief Assemble the five‑point Laplacian stiffness matrix on a
     *        ProductGrid<UniformGrid<Scalar>, 2>.
     *
     * The matrix is *unscaled*:  entry (i,j) = 4 on diagonal, -1 on neighbours.
     * Scaling by 1/(hx*hy) is left to the caller.
     */
    template<typename Scalar>
    Eigen::SparseMatrix<Scalar> assemble_laplacian_5pt(
        std::size_t nx, std::size_t ny)
    {
        std::size_t n = nx * ny;
        Eigen::SparseMatrix<Scalar> A(n, n);
        std::vector<Eigen::Triplet<Scalar>> triplets;
        triplets.reserve(n * 5);
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) continue;
                std::size_t row = i + j * nx;
                triplets.emplace_back(row, row, 4_r);
                triplets.emplace_back(row, row - 1, -1_r);
                triplets.emplace_back(row, row + 1, -1_r);
                triplets.emplace_back(row, row - nx, -1_r);
                triplets.emplace_back(row, row + nx, -1_r);
            }
        }
        A.setFromTriplets(triplets.begin(), triplets.end());
        return A;
    }

    /**
     * @brief Return the lumped mass vector for a uniform product grid.
     *
     * Each entry equals hx * hy for interior and boundary nodes alike.
     */
    template<typename Scalar>
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> lumped_mass_vector(
        std::size_t nx, std::size_t ny, Scalar hx, Scalar hy)
    {
        std::size_t n = nx * ny;
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> M(n);
        M.setConstant(hx * hy);
        return M;
    }

    template<typename Scalar>
    Eigen::SparseMatrix<Scalar> assemble_laplacian_1d(std::size_t n) {
        Eigen::SparseMatrix<Scalar> L(n, n);
        std::vector<Eigen::Triplet<Scalar>> triplets;
        triplets.reserve(3 * n);
        for (std::size_t i = 0; i < n; ++i) {
            if (i > 0)     triplets.emplace_back(i, i - 1, -1_r);
            triplets.emplace_back(i, i, 2_r);
            if (i + 1 < n) triplets.emplace_back(i, i + 1, -1_r);
        }
        L.setFromTriplets(triplets.begin(), triplets.end());
        return L;
    }

} // namespace delta::numerical::solvers