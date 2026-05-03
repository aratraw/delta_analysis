# Δ‑analysis Library

[![DOI (paper)](https://zenodo.org/badge/DOI/10.5281/zenodo.18761044.svg)](https://doi.org/10.5281/zenodo.18761044)
[![DOI (software)](https://zenodo.org/badge/DOI/10.5281/zenodo.18934082.svg)](https://doi.org/10.5281/zenodo.18934082)
[![CI](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage->95%-brightgreen)]()
[![License](https://img.shields.io/badge/license-PolyForm%20Small%20Business%201.0.0-blue)]()

A C++20 library for **exact, constructive mathematical analysis** where the continuum is the *limit of a refinement process*, not a pre‑existing set of points. It provides grids, adaptive paths, discrete exterior calculus, and an advanced lazy rational engine – all parametric over the address space (rationals, matrices, binary strings, p‑adic numbers, …).

> **Version 0.2 is stable and feature‑rich.** The next release (0.3) will add symbolic differentiation, differential geometry on forms, and solvers. See the [future milestones](#-future) below.

---

## 🔥 What the Library Already Does

- **Exact rational arithmetic** with arbitrary precision and zero floating‑point error. Algebraic identities (e.g., d² = 0) hold *exactly*.
- **Adaptive refinement** that concentrates points where a function deviates from linearity. For localised features it can be **1000× faster** than uniform grids.
- **Discrete Exterior Calculus (DEC)** – exterior derivative, Hodge star, codifferential, Laplacian, wedge product – all with exact invariants.
- **Lazy rational engine** – a mutable, move‑only expression tree with global hash‑consing, automatic garbage collection, and algebraic simplification. Summation of 500 000 terms runs **2–6× faster** than eager Boost rationals.
- **Parametric analysis** – change the address space or metric, and the same code works for p‑adic analysis, matrix‑valued functions, or ultra‑metric trees.
- **Hundreds of tests** that double as executable documentation and usage examples.

For a deep dive see the [documentation table](#-documentation).

---

## 🚀 Quick Example

```cpp
#include "delta/core/rational.h"
#include "delta/core/adaptive_delta_path.h"
#include "delta/core/delta_operator.h"
#include <iostream>

using namespace delta;
using Addr = Rational;

int main() {
    // f(x) = |x - 0.5|  (a sharp kink at the centre)
    auto func = [](const Addr& x) -> Rational {
        return abs(x - Rational(1, 2));
    };

    AdaptiveOperator adapt_op(Rational(1,10), Rational(1,20));
    std::vector<Addr> initial = {0_r, 1_r};

    auto path = AdaptiveDeltaPath<Addr,Rational,Rational,
        LessBetweenness,EuclideanMetric,EuclideanValueMetric,
        AdaptiveOperator>(initial, func, adapt_op, Rational(1,100));

    while (path.advance()) {}   // refine until the queue is empty

    for (const auto& p : path.points())
        if (p > Rational(45,100) && p < Rational(55,100))
            std::cout << p << " ";
}
```

This clusters points near the corner without any external heuristics – adaptivity is built into the Δ‑path.

---

## 🧪 Tests Are the Primary Documentation

We have **≈400 test cases** organised into ~45 files; the code volume of the test suite is roughly on par with the library headers.  
Every test verifies a fundamental mathematical invariant, edge case, or cross‑component integration. They serve as **executable examples** – if you want to learn how to use a feature, go to the corresponding test file.

- `tests/calculus/test_riemann_sum.cpp` – Riemann sums on dyadic and non‑uniform grids
- `tests/geometry/discrete_forms_test.cpp` – `d∘d=0`, Hodge star, Laplacian, wedge product
- `tests/numerical/discrete_operators_test.cpp` – finite differences, convergence tests
- `tests/rational/lazy_rational_contract_tests.cpp` – the complete mutation contract of `LazyRational`
- `tests/rational/transcendentals_correctness.cpp` – π, sin, cos, exp, log up to 100 digits

Full coverage report: [docs/test_coverage.md](docs/test_coverage.md).

The tests always pass (`ctest` returns 0), and they are the ultimate guarantee of correctness.

---

## ⚙️ The Lazy Rational Engine

The library’s numerics are powered by an advanced lazy evaluation system:

- **Move‑only mutable trees** – `a + b` mutates `a` in place, O(1) per term.
- **Global hash‑consed pool** – structurally identical sub‑expressions are shared.
- **Automatic garbage collection** – when the pool fills up, all live clean roots are evaluated to constants and the pool is rebuilt. GC is **part of the computational model** – it’s the moment deferred evaluation is forced.
- **Pyramidal compact reduction (PCR)** – sums are reduced in batches of 32 to avoid intermediate fraction swell.
- **Algebraic simplification** – detects `Exp(Log(x)) → x`, folds `A+A → 2*A`, distributive `a*b + a*c → a*(b+c)`, up to **1000× speedup**.
- **One step away from symbolic differentiation** – the same expression tree can be differentiated automatically (planned for v0.3).

Learn more in [docs/optimal_coding_guideline.md](docs/optimal_coding_guideline.md) and [docs/architecture.md](docs/architecture.md).

---

## 📁 Documentation

| Document | Description |
|----------|-------------|
| [**User Manual**](docs/user_manual.md) | Getting started, basic types, grids, paths, DEC, numerical operators, and `LazyRational`. |
| [**Architecture Overview**](docs/architecture.md) | Rational backend, lazy engine, pool & GC, core & calculus layers, geometry & numerical modules, modularity principles. |
| [**Performance Benchmarks**](docs/benchmark_results.md) | Methodology and interpretation for rational arithmetic, transcendentals, simplification, and core adaptivity. |
| [**Optimal Coding Guidelines**](docs/optimal_coding_guideline.md) | Performance‑critical patterns, when to simplify, GC behaviour, and pool management. |
| [**Test Coverage Report**](docs/test_coverage.md) | Detailed walk‑through of every test suite, what it validates, and non‑obvious aspects. |

---

## 📊 Benchmarks (overview)

All benchmarks include correctness checks. Full report: [docs/benchmark_results.md](docs/benchmark_results.md).

| Scenario | Delta | Reference | Speedup |
|----------|-------|-----------|---------|
| **Harmonic series (50 000 terms)** |  lazy 487 ms  | Boost et_off 2860 ms | **5.9×** |
| **Random rationals (500 000 terms)** | lazy 794 ms  | Boost et_off 1847 ms | **2.33×** |
| **sin(1.234…) at ε=1e-80** | 247 µs  | naive series 1646 µs | **6.7×** |
| **π at ε=1e-80 (cached)** | 2 µs  | naive series 547 µs | **273×** |
| **Adaptive vs uniform (kink, ε=1e-4)** |  adaptive 96 µs  | uniform 98 ms  | **1021×** |
| **Adaptive vs uniform (narrow peak, ε=1e-4)** | adaptive 8.6 ms  | uniform 5.3 s  | **618×** |

> **Important:** Micro‑benchmarks can be misleading (e.g., a faster `sqrt` may produce bloated rationals that kill downstream performance). Read the [Benchmarking Philosophy](docs/benchmark_results.md#1-benchmarking-philosophy-and-global-remarks) before drawing conclusions.

---

## 🔧 Building

**Requirements:** CMake ≥ 3.15, C++20 compiler, [vcpkg](https://vcpkg.io/) (recommended), Boost, Eigen3, Abseil, fmt, Google Test, Google Benchmark.

```bash
# Configure with presets
cmake --preset x64-release
cmake --build out/build/x64-release

# Run tests
cd out/build/x64-release
ctest --output-on-failure
cmake --build . --target benchmarks   # build all benchmarks
```

---

## 🌌 Philosophy & Scientific Background

Δ‑analysis rebuilds analysis from a single premise: *between any two addresses a third can be inserted*. Iterative refinement generates a sequence of finite grids that converge to a continuum – but the continuum is **never postulated**; it remains a regulative idea.

Originally developed in a 920‑page research monograph ([Zenodo](https://doi.org/10.5281/zenodo.18761044)), the theory:

- Constructs ℝ without actual infinity,
- Derives Einstein equations from a discrete insertion rule,
- Explains dark matter/energy as topological complexity,
- Argues that the Navier–Stokes Millennium Problem is physically meaningless at finite energy – and gives an explicit constructive solution at any finite scale.

This library is the computational companion to that work. It realises the constructive core of the theory in C++20, letting you experiment with the concepts directly.

---

## 🔮 Future (v0.3)

The existing codebase is stable and already huge, but the roadmap for v0.3 includes:

- **Symbolic differentiation** – automatic differentiation on `LazyRational` trees.
- **Differential geometry on discrete forms** – full DEC with circumcentric duals, 3D wedge products, and generalised N‑forms.
- **Solvers** – template‑based PDE solvers (Poisson, wave, elasticity) decoupled from the discretisation, using the generic Δ‑path interface.
- Further performance optimisations and additional metrics.

Development will preserve the strict separation of layers; no breaking changes are expected in the core modules

---

## 📄 License

**PolyForm Small Business License 1.0.0**.  
For uses beyond this license, please contact: timohaishimcev@gmail.com

---

## 📚 Citation

```bibtex
@misc{ishimtsev_2026_18761044,
  author       = {Ishimtsev, Timofey and Echo},
  title        = {General Delta-Theory of the Discrete Continuum:
                  Refounding Analysis to Unify Relativity and Quantum Gravity},
  month        = feb,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18761044},
  url          = {https://doi.org/10.5281/zenodo.18761044}
}
```

---

## 🙏 Acknowledgements

- [Boost.Multiprecision](https://www.boost.org/doc/libs/release/libs/multiprecision/) – arbitrary‑precision backend.
- [Eigen](https://eigen.tuxfamily.org/) – linear algebra and tensor operations.
- [Abseil](https://abseil.io/) – high‑performance containers.
- [Google Test / Benchmark](https://github.com/google) – testing and benchmarking.
- [fmt](https://fmt.dev/) – formatting.

---

**Explore the discrete foundations of analysis and physics with Δ‑analysis.**  
Questions, ideas, commercial licensing: open an issue or email the author.