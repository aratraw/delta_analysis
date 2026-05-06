# Δ‑analysis Library

[![CI](https://img.shields.io/badge/build-passing-brightgreen)]()
[![Tests](https://img.shields.io/badge/tests-400%20passing-brightgreen)](https://github.com/aratraw/delta_analysis/tree/main/tests)
[![Coverage](https://img.shields.io/badge/coverage-%3E95%25-brightgreen)](https://github.com/aratraw/delta_analysis/tree/main/tests)
[![License](https://img.shields.io/badge/license-PolyForm%20Small%20Business%201.0.0-blue)]()
[![DOI (paper)](https://zenodo.org/badge/DOI/10.5281/zenodo.18761044.svg)](https://doi.org/10.5281/zenodo.18761044)
[![DOI (software)](https://zenodo.org/badge/DOI/10.5281/zenodo.18934082.svg)](https://doi.org/10.5281/zenodo.18934082)

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

## ⚙️ The Lazy Rational Engine

The library’s numerics are powered by an advanced and meticulously crafted lazy evaluation system:

- **Move‑only mutable trees** – `a + b` mutates `a` in place, O(1) per term.
- **Global hash‑consed pool** – structurally identical sub‑expressions are shared.
- **Automatic garbage collection** – when the pool fills up, all live clean roots are evaluated to constants and the pool is rebuilt. GC is **part of the computational model** – it’s the moment deferred evaluation is forced.
- **Pyramidal compact reduction (PCR)** – sums are reduced in batches of 32 to avoid intermediate fraction swell.
- **Algebraic simplification** – detects `Exp(Log(x)) → x`, folds `A+A → 2*A`, distributive `a*b + a*c → a*(b+c)`, up to **1000× speedup**.
- **One step away from symbolic differentiation** – the same expression tree can be differentiated automatically (planned for v0.3).

Learn more in [docs/optimal_coding_guideline.md](docs/optimal_coding_guideline.md) and [docs/architecture.md](docs/architecture.md).

---

## Competitive Landscape

Δ‑analysis occupies a **new niche** at the intersection of four domains that are usually served by separate, unintegrated tools:

1. **Exact rational arithmetic & symbolic manipulation** – Boost.Multiprecision, GiNaC, Mathematica, SymPy  
2. **Discrete Exterior Calculus (DEC) & simplicial meshes** – PyDEC, DecLib, geometry components of CGAL  
3. **Adaptive mesh refinement & numerical analysis** – deal.II, p4est, libMesh (usually floating‑point only)  
4. **Constructive mathematics & formal verification** – Coq, Agda, NuPRL (not designed for industrial‑scale computation)

No existing library bridges all four. The table below shows how Δ‑analysis compares against the most relevant tools across the key capability domains.

### 1. Arithmetical Core — the raw computational engine

This table compares Δ‑analysis against dedicated arbitrary‑precision libraries on arithmetic capabilities, lazy evaluation, and performance.

| Capability | Δ‑analysis | Boost.MP | GMP | CLN |
|------------|------------|----------|-----|-----|
| **Arbitrary‑precision rationals** | ✅ (native, via Boost) | ✅ | ✅ | ✅ |
| **Lazy evaluation with built‑in GC** | ✅ (unique) | ❌ (et_on only, no GC) | ❌ | ❌ |
| **Algebraic simplification** (flatten, fold, distribute, cancel) | ✅ (up to 1000× speedup) | ❌ | ❌ | ❌ |
| **Transcendental functions** with absolute error guarantee | ✅ (hybrid float/series, cached π) | ❌ | ❌ | ❌ |
| **Smart Summation** (pyramidal compact reduction) | ✅ (2–6× faster than Boost) | ❌ | ❌ | ❌ |
| **Eigen integration** (NumTraits, ADL transcendental) | ✅ | ❌ | ❌ | ❌ |
| **Source Available** | ✅ PolyForm Small Business | ✅ Boost | ✅ LGPL/GPL | ✅ GPL |

*Δ‑analysis wraps Boost.Multiprecision as its integer backend, then adds a full lazy‑evaluation layer with GC, simplification, and transcendental caching. You can always drop down to raw Boost rationals within the same codebase, but the lazy engine provides substantial speed and convenience for any non‑trivial workload.*

### 2. Mathematical Framework — high‑level discrete analysis

This table compares Δ‑analysis against systems that integrate multiple mathematical domains (symbolic, geometric, numerical) under one roof.

| Capability | Δ‑analysis | Mathematica | GiNaC | CGAL | deal.II / libMesh | SymPy / NumPy |
|------------|------------|-------------|-------|------|-------------------|---------------|
| **Discrete Exterior Calculus** (d, ⋆, δ, Δ, ∧) with exact invariants | ✅ exact, metric‑aware | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Simplicial complexes & barycentric dual** | ✅ (2D, 3D) | ❌ (general, but not exact rational) | ❌ | ✅ (meshes, but no DEC) | ✅ (floating‑point) | ❌ |
| **Adaptive mesh refinement** (priority‑queue, deviation‑based) | ✅ (up to 1000× gain) | ⚠️ (general, not exact rational) | ❌ | ❌ | ✅ (floating‑point) | ❌ |
| **Parametric over address space & metric** (ℝⁿ, p‑adic, matrices, trees) | ✅ | ❌ (fixed ℝ) | ❌ | ❌ | ❌ | ❌ |
| **Tensor & matrix fields** with algebraic ops | ✅ | ✅ | ❌ | ❌ | ✅ (tensors) | ✅ (NumPy) |
| **Finite‑difference operators** (grad, div, curl, Laplacian) on product grids | ✅ (1D–4D, metric‑aware) | ✅ | ❌ | ❌ | ✅ | ⚠️ (NumPy only via extensions) |
| **Hat basis functions & interpolation** on simplices | ✅ (2D, 3D) | ❌ | ❌ | ✅ | ✅ | ❌ |
| **Algebraic simplification engine** | ✅ (symbolic, up to 1000×) | ✅ (closed) | ✅ | ❌ | ❌ | ✅ (SymPy) |
| **Source Available** | ✅ (PolyForm) | ❌ (closed kernel) | ✅ (GPL) | ✅ (GPL/LGPL) | ✅ (LGPL) | ✅ (BSD) |

### What we do that nobody else does

1. **Exact constructive analysis built on refinement processes**  
   We treat the continuum as the limit of a discretization process, not a pre‑existing set. Algebraic identities (d²=0, Green’s identities) hold *exactly* on every finite grid, thanks to rational arithmetic. This eliminates floating‑point noise as an error class.

2. **Discrete Exterior Calculus with arbitrary metric, fully integrated**  
   Our DEC operators (exterior derivative, Hodge star, codifferential, Laplacian, wedge product) work with any user‑supplied metric. Switch from Euclidean to p‑adic or matrix metric, and the operators remain valid – something no other DEC library provides.

3. **Parametric regulative idea**  
   Addresses, metric, betweenness, and refinement strategies are template parameters. The same path, continuity check, and Riemann sum code works for classical ℝⁿ, p‑adic numbers, matrices, binary trees – without modification. This is a completely novel architectural abstraction.

4. **Lazy rational engine where GC is part of the computational model**  
   Our move‑only mutable expression trees accumulate lazily; when the global pool fills, a garbage collector forces evaluation of all living roots into constants. GC is not a cleanup afterthought – it is the moment deferred computation is realized, achieving 2–6× speedups over eager evaluation.

5. **Adaptive refinement driven by a priority queue**  
   We insert new points only where the function deviates from linearity. For localized features (kinks, narrow peaks) this yields 600–1000× speedups over uniform grids. The logic is generic and decoupled from the grid type.

6. **Transcendental functions with absolute accuracy and caching**  
   All elementary functions guarantee |f(x) – result| < ε, with a hybrid float/series path and caching of constants per epsilon. This combination of speed, precision, and correctness is unmatched in open‑source rational libraries.

7. **Test suite as executable specification**  
   Our >400 tests verify fundamental mathematical invariants. They are not mere unit tests – they are the operational proof that the library’s algebraic claims hold exactly. The test code volume is on par with the library headers.

### Our aspiration: the open alternative to Mathematica’s kernel

Systems like Mathematica integrate symbolic manipulation, numerics, geometry, and visualization into a seamless experience – but their core is **closed, proprietary, and hidden**.  
**Δ‑analysis aims to provide a comparable integrated mathematical computing platform, built from the ground up on exact, constructive principles, with a fully open, inspectable, and embeddable C++20 core.**  
We are not there yet – v0.2 already delivers a unique combination of exact DEC, adaptive refinement, lazy symbolic simplification, and parametric geometry that no other open‑source library offers. The roadmap to v0.3 brings symbolic differentiation, full differential geometry on forms, and PDE solvers, pushing us further toward that vision.

The difference? **You can see every line of code, understand every mathematical choice, and embed the exact same machinery into your own applications (under the PolyForm Small Business license).**

### In summary

Δ‑analysis is not just another numerics library – it is a **framework for exact constructive mathematical modelling**. It is the first open‑source tool to unify exact rational arithmetic, discrete exterior calculus, adaptive refinement, and a symbolic simplification engine under a single, parametric architecture, whose unique theoretical backbone is algorithm- and discrete-native, where finite meshes are **first-class citizens of the source approach to mathematial analysis itself**.

**There is no comparable open‑source project.**

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

This code clusters points near the corner without any external heuristics – adaptivity is built into the Δ‑path.

---

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

## 🧪 Tests Are the Primary Documentation

We have **≈400 test cases** organised into ~45 files; the code volume of the test suite is roughly on par with the library headers.  
Every test verifies a fundamental mathematical invariant, edge case, or cross‑component integration. They serve as **executable examples** – if you want to learn how to use a feature, go to the corresponding test file.

- `tests/calculus/test_riemann_sum.cpp` – Riemann sums on dyadic and non‑uniform grids
- `tests/geometry/discrete_forms_test.cpp` – `d∘d=0`, Hodge star, Laplacian, wedge product
- `tests/numerical/discrete_operators_test.cpp` – finite differences, convergence tests
- `tests/rational/lazy_rational_contract_tests.cpp` – the complete mutation contract of `LazyRational`
- `tests/rational/transcendentals_correctness.cpp` – π, sin, cos, exp, log up to 100 digits

Full coverage report: [docs/test_coverage.md](docs/test_coverage.md).

All tests pass, and they are the ultimate guarantee of correctness.

---

# 🌌 Philosophy & Scientific Background

Δ‑analysis theory rebuilds mathematical analysis from a single premise: *between any two addresses a third can be inserted*. Iterative refinement generates a sequence of finite grids that converge to a continuum – but the continuum is **never postulated**; it remains a regulative idea.

Originally developed in a 920‑page research monograph ([Zenodo](https://doi.org/10.5281/zenodo.18761044)), the theory:

- Constructs ℝ without actual infinity,
- Derives Einstein equations from a discrete insertion rule,
- Explains dark matter/energy as topological complexity,
- Argues that the Navier–Stokes Millennium Problem is physically meaningless at finite energy – and gives an explicit constructive solution at any finite scale.

This library is the computational companion to that work. It realises the constructive core of the theory in C++20, letting you experiment with the concepts directly.

## Why Rational Numbers Are NOT a Niche Choice

Every other numerical library uses floating‑point arithmetic. It’s fast, it’s familiar, it’s “good enough for most cases.”  

**But in double precision, you cannot even guarantee `0.1 + 0.2 == 0.3`.**  
We consider that not just a minor nuisance, but a **ridiculous** foundation for any serious mathematical work. When a simulation violates a conservation law by 1e‑16 and you shrug and say “floating‑point error”, you’ve stopped doing mathematics and started doing heuristics. We refuse to accept that.

The immediate objection is always performance: “Rationals are too slow; this library is a niche toy.”

**We disagree, and here is why.**

1. **Raw rational arithmetic will never beat double in a fist‑fight.**  
   But we don’t fight that fight. We change the rules of engagement entirely.

2. **The lazy engine with pyramidal compact reduction** sums 500 000 random terms **2.3x faster** than eager Boost rationals – *while using the same Boost backend underneath*.  
   The speedup comes from knowing how fractions grow, and preventing intermediate swell through batched, hierarchical reduction. We don’t optimise the low-level arithmetic; we optimise the *algorithmic structure* of how the arithmetic is conducted, to do it the smart way.

3. **And the benefit scales with problem size.**  
   For harmonic‑series‑like workloads (the worst case for rational numbers), our advantage grows to **5.9 ×** on 50 000 terms.  
   The more operands you have, the more our smart summation outruns the naive one. Meanwhile, double‑based code would already be drowning in rounding noise from the start, while we march on with exact results.

4. **Now add adaptive refinement on top of that lazy engine.**  
   A uniform double‑precision grid can spend 5 seconds resolving a narrow peak to ε=1e‑4.  
   Our adaptive path finishes the same job in **8 milliseconds**, and the result is **exact**.  
   That’s not a niche trade‑off. That’s a qualitative leap in efficiency *and* correctness.

So no, Δ‑analysis is not a slow, academic curiosity.  
It’s a deliberate bet that **algorithmic intelligence beats brute‑force arithmetic**, and that exactness is not the enemy of performance – it’s the foundation on which genuinely robust numerical methods can be built.

**Stop optimising your multiplication. Start optimising your asymptotics.**  
That’s what we do. And the outcome is a library that competes with floating‑point frameworks on speed, while delivering results that are mathematically impeccable.

*Because speed does not matter if the result is weather on Mars (unless you explicitly compute weather on Mars)*

### The Core Philosophy (what separates us from the 50‑year industrial status quo)

**1. Mathematical determinism over faith in IEEE 754**
- We reject artificial noise. The IEEE 754 floating‑point standard introduces irreducible rounding error. In complex systems, this noise becomes an independent physical force — a ghost in the machine that can dominate dynamics.
- **Exact rational truth.** Using `Rational` as the foundational numeric type guarantees absolute accuracy. We never “approximate” a solution; we compute its exact discrete value. If a test fails, the algorithm is wrong — never the processor’s floating‑point peculiarities.
- **Provability.** Since every algebraic identity (d² = 0, Green’s identities, etc.) holds exactly on a finite grid, tests serve as mathematical proof, not statistical plausibility checks.

**2. Intelligent light‑weighting vs. the “tax of fear”**
- **Safety margins are poison.** The industrial habit of adding a 300 % safety factor is a direct consequence of not trusting the calculations. Extra mass increases structural load, reduces efficiency, and complicates the entire system.
- **Computation is cheaper than over‑engineering.** 24 hours of CPU time on exact rational arithmetic costs less than tons of surplus metal, wasted fuel, and years of physical prototyping. When you know the true value, you can design to the physical limit of the material, not to the limit of an engineer’s fear of rounding error.
- **Purity of design.** Exact computation enables structures that operate at the genuine boundary of material capability — lighter, faster, and more efficient — because every gram and every Newton is accounted for.

**3. Algorithmic hierarchy over brute‑force hardware**
- **Complexity beats flops.** Instead of buying 1000 GPUs to power a blind Monte‑Carlo sweep, we invest in developing algorithms for adaptive mesh refinement (AMR), lazy symbolic evaluation, and pyramidal compact reduction. The same problem that requires peta‑flops of noisy floating‑point can often be solved exactly with a laptop and the right algorithm.
- **Focus on what matters.** We spend the “expensive” rational arithmetic exclusively where it counts — near singularities, sharp gradients, and topological boundaries — while the lazy engine ensures that simple regions remain computationally cheap.
- **N‑dimensional universality.** The Δ‑analysis framework is identical whether you are simulating heat transfer in an engine, logical connectivity in an expert system, or causal structure in a quantum‑gravity model. The mathematics does not care about the domain.

**4. The unity of logic and physics**
- **One metric for all.** In the Delta system there is no distinction between “divergence of a temperature field” and “divergence of a semantic field.” Any object with a well‑defined betweenness relation, metric, and connection can be analysed with absolute precision — whether it lives in physical space, phase space, or an abstract knowledge graph.
- **A hallucination‑free backend for AI.** By providing a strictly rational, algebraically closed foundation, we are building the substrate on which Symbolic AI and expert systems can operate. A reasoning engine that runs on exact arithmetic cannot “hallucinate” due to floating‑point drift; its conclusions are as trustworthy as the axioms they rest on.

*Not all of this is implemented today; the current release is a stable, powerful foundation. But this is our roadmap — a deliberate, step‑by‑step construction of a unified computational platform where logic, physics, and engineering speak the same exact language.*

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

**PolyForm Small Business License 1.0.0**

| Your case | Allowed? |
|-----------|----------|
| Non-commercial: personal learning, research, hobby projects, etc. | ✅ Yes, free |
| Student / academic projects (non-commercial) | ✅ Yes, free |
| Commercial startup with revenue < $1M/year | ✅ Yes, free |
| Commercial business with revenue ≥ $1M/year | ❌ No, requires commercial license |
| Proprietary closed-source software (no source distribution) | ❌ No, requires commercial license |

> **For use outside the PolyForm Small Business License**  
> (e.g., large enterprise, proprietary integration, OEM licensing)  
> **contact:** `timohaishimcev@gmail.com`  
> Usually reply within 1–2 business days.

### Why this license?

This library is the result of an enormous research and implementation effort (920‑page source monograph, ~40k lines of code, hundreds of tests).  
The PolyForm Small Business License allows **free use of Δ‑analysis in education, research, and small commercial projects**, while requiring fair compensation when a large business derives substantial value from it.

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
