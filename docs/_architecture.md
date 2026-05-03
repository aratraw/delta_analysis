## Architecture Overview: Foundation

This document explains the foundational layers of the Δ‑analysis library: the integer and rational arithmetic backends, the design rationale behind the scalar type, and the lazy evaluation engine. These are the building blocks upon which grids, paths, operators, and the discrete exterior calculus are constructed.

---

### 1. Integer Backend (`storage.h` + `utils.h`)

#### `dumb_int` – The Workhorse Integer

```cpp
namespace delta::internal {
    using dumb_int = boost::multiprecision::number<
        boost::multiprecision::cpp_int_backend<>,
        boost::multiprecision::et_off   // CRITICAL
    >;
}
```

`dumb_int` is a `cpp_int` with **expression templates disabled** (`et_off`). The library already has its own lazy evaluation layer (`LazyRational`); allowing Boost’s expression templates to generate intermediate lazy objects would only add indirection and make arithmetic 2–3× slower. With `et_off`, integer operations happen immediately and predictably.

This type serves as the numerator/denominator of every rational number, and is used wherever a raw, eager integer is needed.

#### `Value` – The Rational Number

```cpp
using Value = boost::multiprecision::number<
    boost::multiprecision::rational_adaptor<
        boost::multiprecision::cpp_int_backend<
            128,                                        // MinBits
            0,                                          // MaxBits (unlimited)
            boost::multiprecision::signed_magnitude,
            boost::multiprecision::unchecked,
            std::allocator<boost::multiprecision::limb_type>  // DO NOT CHANGE TO void
        >
    >,
    boost::multiprecision::et_off
>;
```

The `Value` type is a rational number with arbitrary precision. The backend parameters are fixed and must not be altered:

- **`MinBits = 128`** : numbers fitting in 128 bits (≈38 decimal digits) are stored directly inside the object (no heap allocation).
- **`MaxBits = 0`** : unlimited size when needed.
- **`unchecked`** : disables runtime checks for performance.
- **`std::allocator<limb_type>`** : **never replace with `void`** – doing so causes subtle Heisenbugs.

**Why Boost and not a custom small‑big integer?**  
A custom implementation (small‑object optimisation with fallback to heap) was tried and benchmarked; it was **12% slower** than the naive Boost backend. Boost’s limb operations are highly optimised, often down to assembly, and the `MinBits` parameter already provides stack allocation for small numbers. The pragmatic decision was to abandon the custom backend and rely entirely on Boost.

**GMP backend** – technically possible by replacing `cpp_int_backend` with `gmp_int`, but GMP is LGPL/GPL licensed. The default backend uses the permissive Boost license and imposes no such obligations. Using GMP is entirely the user’s own responsibility.

Additional utility functions (`is_zero`, `is_one`, `numerator`, `denominator`, `to_double`) operate directly on the backend for efficiency.

---

### 2. Rational Numbers (`rational.h`)

The `rational.h` header is the single entry point for all rational arithmetic in the library. It unifies the eager and lazy number types, transcendental functions, and integration with Eigen.

#### Eager Rational (`Rational`)

`Rational` is an **eager, arbitrary‑precision rational** that wraps a `Value`. All arithmetic operations are performed immediately and return new `Rational` objects. It is the primary scalar type for the entire library.

**Why `Rational` and not `double`?**

1. **Unbounded refinement** – Doubles have 53 bits of mantissa; after ~50 dyadic refinement steps coordinates become indistinguishable. `Rational` stores exact binary fractions (k/2^m) without bound; refinement can continue arbitrarily deep.

2. **Exact algebraic invariants** – Discrete operators rely on identities like d²=0, summation by parts, and Green’s identities. With doubles, rounding errors destroy these cancellations; with `Rational` they hold *exactly*.

3. **Predictable comparisons** – `a == b` is a well‑defined, exact predicate. There is no need for fuzzy epsilon‑based comparisons except where transcendental functions are involved.

4. **Decoupling error sources** – Measurement noise, discretisation error, and iterative solver tolerance are the only approximations. No arithmetic noise contaminates the results.

5. **Constructive core compatibility** – Addresses must be actualisable (dyadic rationals, finite decimals). `Rational` represents them exactly; double cannot even store 0.1 exactly.

6. **Performance compromise** – Speed is secondary during development and verification. Once correctness is proven, the same template code can be instantiated with `double` for production runs (compile‑time decision).

#### Lazy Rational (`LazyRational`)

`LazyRational` is a **move‑only, mutable expression graph**. It accumulates operations (arithmetic and transcendental) without evaluating them, then performs a single evaluation at the end.

- **Mutable design** – `a + b` mutates `a` in place, absorbing `b`’s tree. Accumulation is O(1) per term; the whole tree is evaluated once in O(N). This is the key performance advantage over immutable libraries.

- **Dirty vs Clean** – A lazy expression starts *dirty* (mutable local tree). When needed, it can be *canonicalised* into a *clean* node in the global hash‑consed pool.

- **Simplification** – The canonicalisation step applies algebraic rewrites (e.g., `x + NEG(x) → 0`, `a*b + a*c → a*(b+c)`, folding of identical terms). Simplification is **not a default** and should be explicitly requested (`simplify_inplace()` or `eval()` without `skip_simplify`).

- **Evaluation** – `eval()` computes the rational value (canonicalising first if needed). `eval_inplace(true)` performs a destructive, simplification‑free evaluation: it tears down the dirty tree, applies the efficient pyramidal compact reduction, and replaces the object with a single constant result. This direct path is 2–6× faster than eager sequential summation for large workloads.

- **Global pool and GC** – Clean nodes are stored in a thread‑local, hash‑consed pool with automatic garbage collection. When the pool reaches its threshold, all live clean roots are evaluated to constants, and the pool is rebuilt. GC is part of the computational model – it is the moment deferred evaluation is forced.

- **Interaction with Rational** – `Rational::as_lazy()` wraps a constant into a dirty `LazyRational`. Conversely, evaluating a lazy expression yields a `Rational`.

#### Transcendental Functions

All transcendental functions (`sqrt`, `exp`, `log`, `sin`, `cos`, `acos`, `asin`, `atan`, `tan`, `pi`, `e`, `pow`) accept an explicit epsilon parameter for absolute error control. They exist in both eager (returning `Rational`) and lazy (creating `LazyRational` nodes) versions.

Internally, a **hybrid approach** is used: for coarse epsilon (≥ 1e‑35), a fast float‑path via `cpp_dec_float_100` is taken; for finer precision, a purely rational series path (e.g., Chudnovsky for π, binary splitting for sin/cos, scaling‑and‑squaring for exp) is used. Certain functions (sqrt, log, e) always use the series path because the float‑path offers no speed benefit.

The **default epsilon** is 1e‑30, stored in a thread‑local global variable and changeable via `set_default_eps()`.

#### Eigen Integration

`Eigen::NumTraits<delta::Rational>` is specialised so that `Rational` can be used as a scalar type in Eigen matrices. Transcendental functions are found via ADL.

---

### 3. Upward from the Foundation

On top of these numeric types, the library builds:

- **Grids** (`ListGrid`, `UniformGrid`, `ProductGrid`) – ordered sets of addresses.
- **Δ‑paths** (`DeltaPath`, `AdaptiveDeltaPath`, `TreeDeltaPath`) – sequences of refined grids driven by delta operators.
- **Moduli of continuity/differentiability** – concepts for checking analytical properties.
- **Operational functions** – functions defined on grids that can be extended upon refinement.
- **Geometry & DEC** – simplicial complexes, barycentric dual, discrete forms (exterior derivative, Hodge star, Laplacian), tensor/matrix fields.
- **Numerical operators** – finite‑difference gradient, divergence, curl, Laplacian on product grids; integration and Green’s identities.

This architecture ensures that every higher‑level component can operate on exact rational arithmetic, preserving the constructive philosophy of the library from the bottom up.

## Architecture Overview (continued)

This section extends the foundation with the global node pool and garbage collection infrastructure, then moves upward into the core computational layers: grids, Δ‑paths, operational functions, and the calculus module.

---

### 4. Global Node Pool, Garbage Collection, and Caches

When a `LazyRational` is canonicalised (either explicitly or implicitly during evaluation), its expression tree is moved into a **thread‑local, hash‑consed node pool**. The pool is the central repository of all immutable (clean) expression nodes. It is also the substrate on which the garbage collector operates.

#### 4.1 Pool Structure

```cpp
struct NodePool {
    size_t max_size = 1'000'000;         // soft limit
    size_t gc_threshold;                 // 0.9 * max_size
    std::vector<Node> nodes;             // all nodes, some may be free
    std::vector<Value> values;           // shared constants & epsilons
    std::vector<int> refcount;           // 0 = free/unused
    size_t next_free_index = 0;          // allocation hint

    // Hash‑consing caches
    absl::flat_hash_map<Value, int, ...> value_cache;
    absl::flat_hash_map<Value, int, ...> constant_cache;
    absl::flat_hash_map<SumProductKey, int, ...> sum_product_cache;
    absl::flat_hash_map<UnaryKey, int, ...> unary_cache;
};
```

- **`constant_cache`** maps a constant `Value` to the index of its unique `CONST` node.
- **`sum_product_cache`** stores `SUM` and `PRODUCT` nodes, keyed by their canonicalised operand sets.
- **`unary_cache`** handles all other operations (`NEG`, `RECIP`, `SQRT`, `EXP`, …, `POW`), keyed by operation type, children, and epsilon index.
- **`value_cache`** deduplicates constant values stored in the pool’s `values` vector; this includes both numeric constants and epsilon values used by transcendental nodes.

These caches guarantee that **structurally identical sub‑expressions are represented by a single node**. This is the essence of hash‑consing: it saves memory and allows O(1) structural equality checks via pointer comparison.

#### 4.2 Allocation Strategy

The pool is **append‑only between GC cycles**. Individual slots are never reused in‑place; instead, when the pool grows too large, a full garbage collection cycle creates a completely new pool. Between GC cycles, allocation works as follows:

- `next_free_index` is a hint for the next likely‑free slot. The allocator scans forward from this index; if a slot is already occupied, it continues scanning.
- If no free slot is found, the pool’s vectors are expanded in **chunks of 4096 nodes**. This avoids pre‑allocating the full `max_size` while keeping reallocation overhead low.
- When `next_free_index` reaches `gc_threshold` (90% of `max_size`) and GC is not temporarily disabled, `collect_garbage()` is triggered automatically.

The append‑only design means that once a node is created, its index is stable until the next GC cycle. This is important because clean `LazyRational` objects hold only an integer index into the pool. Those indices remain valid across many canonicalisation operations.

#### 4.3 Reference Counting

Every canonicalised node carries a reference count (`refcount`). The count tracks how many clean `LazyRational` objects currently point to this node.

- `increment_ref(idx)` is called when a `LazyRational` clones a clean subtree or when canonicalisation creates a new root.
- `decrement_ref(idx)` decreases the refcount; when it reaches zero, the children’s refcounts are recursively decremented. **The node’s data is intentionally not cleared** – this would be wasteful because the entire pool will be replaced at the next GC.

#### 4.4 Garbage Collection Algorithm

When the pool occupancy exceeds the threshold, `collect_garbage()` runs:

1. **Root Snapshot** – A list of all clean `LazyRational` objects is obtained from the global **clean object registry** (`g_clean_rationals`, a thread‑local `unordered_set`). These objects are the roots of all live expression trees.

2. **Evaluation** – For each live root, the entire subtree is evaluated to a single `Value`. This is the moment when **deferred computation is forced**: the garbage collector is not just a memory manager; it performs the actual numerical evaluation that the lazy strategy postponed.

3. **Pool Replacement** – A new `NodePool` is created, sized to the maximum root index plus one. Each evaluated value is stored as a `CONST` node at the *same index* as the original root. Thus, all existing `LazyRational` objects keep their `clean_index_` unchanged, but now point to a simple constant instead of a complex DAG.

4. **Reset** – The old pool is discarded, and `next_free_index` is placed at the first free slot.

Because all live expressions are collapsed to constants, the new pool becomes almost entirely empty except for those root indices. This is why the pool may appear **sparse** (e.g., indices 0, 1000, and 50000 occupied). This sparseness is **not a problem**: the next allocation cycle will fill free slots contiguously from the low end, quickly re‑densifying the pool.

#### 4.5 Clean Object Registry

To enable GC, every clean `LazyRational` registers itself in the global set `g_clean_rationals`. When the object transitions to dirty state or is destroyed, it unregisters. The registry provides the only complete list of live clean roots; without it, GC would not know which pool indices are still in use.

#### 4.6 Full Pool Reset

`internal::reset_pool()` performs a **complete teardown**:

- All clean objects in the registry are reinitialised in place as dirty zero. This destroys the clean trees and removes them from the registry.
- The node pool is replaced with a brand‑new, empty instance.
- The π cache and the clean registry are cleared.

After `reset_pool()`, all `LazyRational` instances become dirty zero. No dangling references remain. This is useful for testing and for reclaiming memory between independent computational phases.

#### 4.7 π Cache

The value of π is cached per epsilon in a thread‑local `std::map<Value, Value>`. When `pi(eps)` is called, the cache is checked first. If a value exists, it is returned immediately; otherwise, the Chudnovsky algorithm computes π from scratch. The cache is cleared by `reset_pi_cache()` and by `reset_pool()`.

---
## Architecture Overview: Core & Calculus Layer

This section describes the architectural principles behind the `core` and `calculus` modules—the heart of Δ‑analysis. The architecture is built on a radical separation of concerns: the discretisation and limiting processes used for continuity, differentiability, and integration are **completely agnostic to the nature of the underlying space**. This is achieved through a system of C++20 concepts, templated type parameters, and composable abstractions that decouple every geometric and metric notion from the algorithms themselves.

The result is a framework where the same code that works for classical real analysis on ℝⁿ can also work, without change, for p‑adic spaces, matrix‑valued functions, binary trees, or any future regulative idea—simply by plugging in the appropriate types that satisfy the required concepts.

---

### 1. The Parameterisation Principle

Every central component (grids, paths, operators, moduli) is templated on a set of type parameters that collectively define a **regulative idea**:

- `Addr` – the type of spatial addresses (points, strings, matrices, …).
- `Value` – the type of function values (rationals, matrices, …).
- `Distance` – the scalar type for measuring distances between addresses or values.
- `Betweenness` – a ternary relation defining the order structure.
- `Metric` – a distance function on addresses.
- `ValueMetric` – a distance function on function values.

The algorithms (refinement, oscillation calculation, Riemann sums, continuity checks) **never hardcode Euclidean geometry or real numbers**. Instead, they operate through these parameters and the operations they support. This is the architectural realisation of the library’s constructive philosophy: “the continuum is the limit of a refinement process, and the nature of that process is specified by the regulative idea, not imposed by the library.”

Concrete regulative ideas are assembled by providing implementations of the relevant concepts. For instance, the classical real line is obtained with:

- `Addr = Rational`
- `Betweenness = LessBetweenness`
- `Metric = EuclideanMetric`
- `ValueMetric = EuclideanValueMetric`

To switch to a p‑adic analysis, one only replaces `Metric` with `PAdicMetric<p>`; all paths and calculus checks adapt automatically because they use the metric through the `Metric` concept.

---

### 2. Grid Concepts and Their Role

Grids are finite ordered sets of addresses that approximate a continuum. Architecturally, grids are defined through concepts that abstract away the storage and access patterns:

- **`SimpleGrid<G>`** — the minimal interface: random access, iteration, size query. Any type satisfying this concept can be used with grid‑based algorithms.
- **`OrderedGrid<G>`** — adds a comparator for strict total order.
- **`SubtractableAddress<Addr>`** — required when computing grid gaps or Riemann sums.

Algorithms such as `max_gap`, `max_oscillation`, and the Riemann sum functions are **templated on a grid type `Grid`** and constrained only by the required concepts. This means they work uniformly for `ListGrid`, `UniformGrid`, `ProductGrid`, `TreeGrid`, or any user‑defined grid.

The grid refinement function `refine_grid` dispatches on the concrete grid type (via `if constexpr`) but always returns a `ListGrid`—the most general grid type—ensuring that further refinement steps can be applied without knowing the original grid’s internal structure. This design preserves genericity while allowing specialisations for efficiency (e.g., `UniformGrid` provides O(1) memory representation).

---

### 3. Delta Paths as Abstraction of Refinement Sequences

A **Δ‑path** is the architectural embodiment of the idea that a continuum limit is approached through a sequence of refined grids. The class template `DeltaPath` is parametrised on the full regulative idea (Addr, Value, Betweenness, Metric, ValueMetric), the refinement Strategy, and a Comparator. Internally, it stores a `ListGrid<Addr, Compare>` and applies the strategy’s operator to each interval at every `advance()` call.

The path **does not** know how new points are chosen—it only requires that the strategy provides an operator satisfying `DeltaOperator`. The operator receives an `IntervalInfo` object containing the endpoints, function values, the current maximum oscillation, and references to the betweenness, metric, and value metric. This context enables operators to be as sophisticated as needed (e.g., adapting to local variation) while remaining decoupled from the path mechanics.

Double buffering and optional caching of function values are internal implementation details that improve performance but are hidden from the architecture. The only contract is that after `advance(func)`, the current grid is refined according to the regulative idea and the operator’s rule.

The **TreeDeltaPath** is a specialised path that bypasses the operator entirely—refinement simply adds the children of all leaves. This demonstrates that the path abstraction is flexible enough to accommodate refinement processes that are not based on a point‑insertion rule.

---

### 4. Adaptive Paths as a Priority‑Driven Process

`AdaptiveDeltaPath` extends the concept of a path by introducing a **priority queue of intervals**. Instead of refining every interval, it selects the interval with the highest deviation from linearity and refines only that one. The priority is computed using the value metric; thus the adaptive behaviour is automatically tuned to the chosen regulative idea’s notion of distance between function values.

The adaptive path is initialised either from a set of points or from a uniform path (via `from_uniform`). This hybrid approach—a few uniform levels followed by adaptive refinement—is a common pattern that demonstrates how different strategies can be composed within the same architectural framework.

---

### 5. Abstracting Refinement Strategies and Operators

The separation of “which point to insert” from “how to refine” is realised through two layers:

- **`DeltaOperator<Op, Addr, Value, …>`** — a concept for a callable that, given endpoints and `IntervalInfo`, returns a new address between them.
- **`DeltaStrategyConcept<S, …>`** — a concept for an object that provides an operator for a given refinement level.

The strategy can be static (same operator at every level), dynamic (a pre‑defined sequence), or factory‑based (creating operators on demand). This stratification allows the refinement rule to be level‑dependent, which is essential for non‑stationary iterative methods (e.g., decreasing λ in a dynamic λ‑operator to achieve uniform density in the limit).

The architecture thus enables a user to define a custom analysis by supplying only a metric, a betweenness, and a strategy for point insertion. All the control flow—the loop over intervals, the computation of oscillation, the storage of grids—is provided by the generic path classes.

---

### 6. Calculus Module: Generic Analytical Checks

The calculus module (`continuity.h`, `differentiability.h`, `modulus.h`, `riemann_sum.h`) encapsulates the limiting processes of classical analysis, but without any commitment to real numbers or Euclidean structure.

- **Modulus** is a concept `Modulus<M, T>` that maps a distance (typically the grid’s maximum gap) to an error bound. Predefined moduli (`PowerModulus`, `LogarithmicModulus`) can be instantiated with any scalar type, including `Rational` or a user‑defined number system.

- **Continuity** is checked by `check_continuity_level(grid, func, vm, modulus)`, which uses only the value metric, the grid, and the modulus. It works for any grid type satisfying `SimpleGrid` and any function type callable on addresses.

- **Differentiability** is checked by `check_differentiability(grids, addr, func, D, modulus, first_level)`, which compares left and right difference quotients against the expected derivative. It requires the address type to be subtractable, but imposes no other geometric assumptions.

- **Riemann sums** are computed via `left_riemann_sum`, `right_riemann_sum`, and `tagged_riemann_sum`. These functions are fully generic over grid types and function types; they only need `SubtractableAddress` and the ability to evaluate the function at grid points. The specialised `tree_riemann_sum` integrates over tree‑structured grids using the uniform measure 2^{–level}, extending the same concept to non‑Euclidean address spaces.

All these checks and operations are designed to be used together: one builds a path (uniform or adaptive), obtains a sequence of grids, and applies the calculus functions to verify convergence or compute integrals—all within the same regulative idea that the path was built with.

---

### 7. Operational Functions as Field Storage

`OperationalFunction` bridges the gap between a grid and function evaluation. It stores function values on a grid and allows them to be extended to a refined grid via interpolation. Architecturally, it is a template specialised for different grid types:

- The general version uses an ordered map (`std::map<Addr, Value, Compare>`), offering O(log n) lookup for arbitrary grids.
- The specialisation for `UniformGrid` stores values in a vector indexed by the ordinal position, providing O(1) access and enabling efficient composition with other uniform‑grid algorithms.

This specialisation is a prime example of the library’s approach: the same generic function can be used with any grid, but a faster path exists for the common case of uniform grids, selected automatically at compile time.

---

### 8. Summary of Dependencies and Extensibility

The core and calculus modules form a layered architecture:

1. **Concepts** (Grid, Betweenness, Metric, DeltaOperator, Modulus) define the interfaces.
2. **Grids and paths** implement the discretisation process, using only those interfaces.
3. **Calculus functions** consume grids and paths to perform analytical checks, again through the concepts.
4. **Operational functions** provide persistent storage and interpolation.

The entire system is open for extension. Adding a new regulative idea (e.g., a graph metric, a quantum‑mechanical phase space) requires only:

- Defining an address type,
- Implementing the metric and betweenness functors,
- Optionally providing a specialised delta operator or strategy.

All existing algorithms—refinement, oscillation, continuity, differentiability, Riemann sums—then become available for that new idea **without any modification** to the library code. This is the foundational architectural promise of Δ‑analysis: the mathematical analysis is parametric over the regulative idea, and the code reflects that parametrisation directly.

## Architecture Overview: Geometry and Numerical Modules (v0.2)

The `geometry` and `numerical` modules constitute the upper, most application‑facing layer of the library. They provide discretisation tools for partial differential equations, discrete exterior calculus (DEC), tensor fields, and finite‑difference operators on product grids. **This layer is under active development** and is currently at a **feature‑complete but architecturally intermediate state**. While the lower levels (rationals, core paths, calculus) are stable and about 85% of their final form, the geometry and numerical modules are intentionally designed to be extended, generalised, and refactored in subsequent versions without breaking the foundational layers.

The guiding principle is **absolute modularity**: each architectural layer is oblivious to the implementation details of the layers below and imposes only minimal, concept‑based requirements on the layers above. This ensures that future generalisations—to N‑dimensional unstructured meshes, arbitrary metrics, and alternative discretisation schemes—can be introduced transparently.

---

### 1. Architectural Layering and Independence

The dependency chain is strictly upward:

```
  Solvers (future)
       ↓
  Numerical Operators, Integrals, DEC
       ↓
  Geometry (Simplicial Complex, Dual Complex, Tensor Fields, Hat Basis)
       ↓
  Core (Grids, Paths, Operational Functions, Calculus)
       ↓
  Rational (Eager/Lazy, Transcendentals)
       ↓
  Storage (Value, dumb_int)
```

Each layer depends only on the concepts and types exposed by the layer directly below it. For instance:

- **Grids do not know about arithmetic.** They store addresses and provide access patterns, but never perform arithmetic operations on them. Arithmetic is the responsibility of the path’s operator or the user’s function.
- **Paths are agnostic to the concrete grid type.** `DeltaPath` works with any grid satisfying `SimpleGrid` (or `OrderedGrid` for sorting guarantees), and its double‑buffering algorithm makes no assumptions about the grid’s internal storage.
- **Geometry will be made agnostic to paths and grids.** Currently, `SimplicialComplex` and `DualComplex` depend on specific metric types, but the architecture already enforces this through the `Metric` concept rather than hard‑coded Euclidean assumptions. Future refactors will abstract these further to accept arbitrary regulative ideas.
- **Solvers (future) will be agnostic to the discretisation geometry.** By interacting only with discrete operators and forms, solvers will treat the underlying mesh (structured or unstructured, simplicial or polyhedral) as a black box providing matrix and vector assembly.

This stratified design enables the library to evolve incrementally: stable lower layers can be optimised and tested independently, while the upper layers undergo repeated abstraction and enrichment without destabilising the core.

---

### 2. Geometry Module: Current State and Architectural Role

The geometry module currently provides:

- **`SimplicialComplex<Dim, Coord>`** – storage and query of simplicial meshes (vertices, edges, triangles, tetrahedra) with incidence relations, barycentric subdivision, and metric‑aware geometric queries (edge length, area, volume).
- **`DualComplex<PrimalComplex, Metric>`** – barycentric dual complex; computes dual volumes and establishes primal‑to‑dual bijections for DEC.
- **`DiscreteForm<k, Value, Complex>`** – discrete differential k‑forms on simplicial complexes, with exterior derivative `d`, Hodge star `⋆`, codifferential `δ`, and Hodge Laplacian `Δ`.
- **`HatBasis<Complex>`** – piecewise linear Lagrange basis functions; provides interpolation, gradient, and point location on triangular and tetrahedral meshes.
- **`TensorField<Addr, Scalar, Rank, Dim, Compare>`** and **`MatrixField`** – sparse fields of scalar, vector, or matrix values over an address set, with algebraic and transcendental operations (matrix exponential, logarithm).
- **`ProductRegulativeIdea`** and **`ProductDeltaPath`** – tools for building higher‑dimensional regular grids from 1D components.

These components are fully capable of performing DEC on small 2D and 3D simplicial meshes. All geometric computations (lengths, areas, volumes, normals) are performed through the supplied `Metric` object, making them formally independent of Euclidean assumptions.

However, several architectural limitations exist in v0.2:

- **Metric‑blind storage** – `TensorField` and `OperationalFunction` currently do not carry metric information; they store values at addresses but cannot recompute geometry internally. The metric must be passed explicitly to every operation.
- **Fixed dimension** – `SimplicialComplex` is templated on a compile‑time dimension; runtime‑varying dimension is not supported (though this is rarely needed in practice).
- **Grid dependence is implicit** – `DiscreteForm` holds a reference to its mesh, but the form operations themselves do not abstract over grid types; they are hard‑wired to `SimplicialComplex`.
- **Primal‑dual bijection assumption** – The current `DualComplex` implementation uses a strict one‑to‑one mapping, which holds for the barycentric dual but may not hold for circumcentric or other dual types.

These limitations are not design flaws; they represent deliberate staging. The next development phases will:

- **Abstract `DiscreteForm` over the grid type** – decouple forms from `SimplicialComplex` so that they can be defined on any cell complex satisfying a `ComplexConcept` (primal mesh + incidence + metric).
- **Introduce `ComplexConcept` and `DualComplexConcept`** – formalise the requirements for primal and dual meshes, enabling plug‑in circumcentric or polyhedral duals.
- **Extend to higher dimensions** – generalise wedge product, Hodge star, and subdivision to N‑simplices.
- **Make `ProductDeltaPath` fully metric‑aware** – embed the regulative idea into the product path so that all operations (refinement, gap computation) use the product metric without manual intervention.

These refactors will take place **without modifying the core layer**; the new concepts and interfaces will be added in the geometry module, and existing code paths will be gradually migrated.

---

### 3. Numerical Module: Discretisation Operators and Integration

The `numerical` module currently contains:

- **`discrete_operators.h`** – finite‑difference gradient, divergence, curl (2D and 3D), and Laplacian on `UniformGrid`, `ListGrid`, and `ProductGrid`. All operators are metric‑aware and support second‑order central differences with fallback to one‑sided differences at boundaries.
- **`integrals.h`** – cell volume computation, numerical integration (Riemann‑style summation), and verification of Green’s identities (1D and 2D). The 2D Green’s checks currently use a FEM stiffness matrix for bilinear elements on rectangular grids.
- **`cotangent_laplacian.h`** – construction of the cotangent Laplacian and lumped mass matrix for 2D triangle meshes, using exact rational arithmetic and metric‑based edge lengths.

The architecture of these operators mirrors the modularity of the core: they are templated on the grid type and metric, and they return `TensorField` objects keyed by the grid’s address type. Parallelism is enabled via OpenMP when the number of grid points exceeds a threshold (default 1000).

**Why a separate numerical module?**  
The DEC operators in the geometry module compute discrete exterior calculus on simplicial meshes and are exact (up to rational arithmetic). The finite‑difference operators in the numerical module provide a complementary approach for structured grids and are optimised for speed, with adjustable difference schemes. Both modules are architecturally parallel; they do not depend on each other but share the same core concepts (grids, metrics, tensor fields). In future versions, the numerical operators will be extended to simplicial grids via the DEC framework, unifying the two approaches under a common interface.

**Current limitations and roadmap:**

- **Green’s identity verification** – The 2D checks in `integrals.h` currently derive the boundary term from the volume terms, making the test trivially true. A proper boundary integral computation (using the supplied metric) is planned.
- **Sparse matrix assembly** – The cotangent Laplacian builds a global sparse matrix. Future versions will add similar assembly routines for DEC‑based operators on unstructured meshes.
- **Convergence testing** – The numerical module lacks built‑in convergence analysis; this will be supplied by higher‑level solver or analysis tools.

---

### 4. Interaction Between Geometry and Numerical Modules

Although the two modules are independent, they are designed to interoperate cleanly:

- Both produce `TensorField` objects, which can be consumed by generic post‑processing routines (e.g., error norms, visualisation).
- Discrete forms from DEC can be converted to `TensorField` (scalar or vector) over the mesh vertices/edges/faces, allowing them to be used with the numerical integration and display tools.
- The `ProductGrid` used by the numerical operators is the same `ProductGrid` defined in the core; this grid type can be refined via `ProductDeltaPath`, producing sequences that could, in principle, be fed into the DEC operators (once those operators are generalised to product grids).

The long‑term vision is that a user will be able to select a discretisation strategy (structured finite‑difference, unstructured DEC, or a hybrid) from a common interface, and the library will assemble the appropriate operators, residuals, and Jacobians using the same regulative idea and rational arithmetic.

---

### 5. Future Directions: Upward Abstraction

The architecture’s ultimate goal is to support *solvers* that are fully decoupled from the discretisation details. A solver should request:

- A `Grid` (or `Complex`) and a `Path` for refinement,
- A set of `DiscreteForm`s representing the unknown fields,
- A `Metric` and a `Betweenness`,
- A `DeltaStrategy` for adaptive refinement,

and then operate through abstract interfaces (`AssembleStiffnessMatrix`, `ComputeResidual`, `ApplyBoundaryConditions`) that are implemented by the geometry or numerical backends. This will be achieved by introducing `SolverConcept` and `ProblemConcept` interfaces in a future `solvers` module, which is currently a placeholder.

The path to this goal involves:

1. **Stabilising the current numerical and DEC operators** – ensuring exactness, convergence, and correct boundary handling for a representative set of test problems.
2. **Abstracting the geometry** – introducing `CellComplexConcept`, `PrimalFormConcept`, `DualFormConcept` to decouple DEC from `SimplicialComplex`.
3. **Creating solver abstractions** – time‑stepping schemes, nonlinear solvers, and linear algebra backends that treat the discretised operators as black‑box functors.
4. **Performance optimisation** – once the modularity is complete, profiling and optional specialisation (e.g., switch to double arithmetic for time‑critical inner loops) will be introduced without altering the high‑level interfaces.

Throughout this evolution, the foundational layers (rational numbers, grids, paths) will remain unchanged, ensuring backward compatibility and a stable base for experimentation and extension.

## 8. Ultimate Modularity: Orthogonal Refactoring and Interface Invariants

The architecture of the Δ‑analysis library is not merely modular in the conventional sense; it is **radically modular** to a degree that makes large‑scale, cross‑cutting refactors a routine engineering activity rather than a disruptive event. The foundational principle is that every layer interacts with the layers below exclusively through **narrow, stable concept interfaces**, and every layer provides its own set of equally stable interface promises to the layers above.

### Orthogonal Evolution: The Rational Sub‑Library Rewrite

The most striking demonstration of this modularity occurred during the development of the lazy evaluation engine. The original `rational.h` header was a single, monolithic file that defined the `Rational` class and a handful of transcendental functions. Over the course of several iterations, this single header was split into **more than 20 separate files** organised into a dedicated `rational/` sub‑directory:

- `storage.h`, `utils.h` – integer and rational backends
- `rational_class.h`, `rational_impl.h` – eager arithmetic
- `lazy_rational.h`, `lazy_rational_impl.h`, `lazy_nodes.h`, `node_types.h` – lazy expression trees
- `node_pool.h` – global hash‑consed pool
- `simplify_impl.h` – algebraic simplification engine
- `evaluate_impl.h`, `evaluation_core.h`, `reduce.h` – evaluation and pyramidal reduction
- `interval.h`, `context.h`, `global_state.h` – auxiliary infrastructure
- `transcendentals.h`, `eigen_integration.h` – transcendentals and Eigen interop
- and several more.

This was not a cosmetic reorganisation. It introduced a **completely new computational paradigm**—lazy, mutable expression accumulation with automatic garbage collection, canonicalisation, and algebraic simplification—**without requiring a single change in the core, calculus, or geometry modules**. The `core` directory, which depends on `rational.h`, continued to compile and function correctly throughout the refactor because `rational.h` itself was preserved as a thin aggregation header that included all the new sub‑files. The interface guarantees promised by `Rational` and its associated free functions (`sqrt`, `sin`, `+`, `-`, `*`, `/`, `==`, `<`, …) remained identical. The fact that the underlying implementation had been completely rebuilt—with a global node pool, reference counting, thread‑local caches, and a sophisticated GC—was invisible to the rest of the library.

This level of isolation is possible because of a deliberate architectural choice: **each layer depends only on the syntactic and semantic interface of the layer below, never on its implementation details.** The `core` module needs an arithmetic type that supports addition and multiplication; it does not care whether that type is a thin wrapper around Boost rationals or the root of a lazy DAG that defers evaluation to a global pool.

### Interface Invariants as the Enabler

This radical modularity is sustained by a set of **interface invariants**—implicit contracts that each type family must obey, making them predictable enough to serve as stable foundations:

- **Arithmetic types** (`Rational`, future `double` specialisations) always expose the standard arithmetic operators (`+`, `-`, `*`, `/`), comparison operators (`==`, `<`, …), and conversions. As long as these operations exist and behave as a field, any code using them remains correct.
- **Grids** (`ListGrid`, `UniformGrid`, `ProductGrid`, `TreeGrid`) promise `size()`, random access via `operator[]`, iteration via `begin()`/`end()`, a `value_type`, and, for ordered grids, a `comparator()`. Algorithms like `max_gap` or `left_riemann_sum` are written against the `SimpleGrid` concept and thus work for any future grid type that fulfills it.
- **Paths** (`DeltaPath`, `AdaptiveDeltaPath`, `TreeDeltaPath`) expose `advance()`, `current_grid()`, `level()`, and `max_gap()`. A calculus function that computes differentiability on a sequence of grids obtains those grids via `path.current_grid()` and never needs to know whether they were generated by uniform dyadic refinement, a priority‑queue adaptive process, or a tree expansion.
- **Metrics and Betweenness** are simple callables with well‑defined signatures. The core and calculus code calls them through templates and never stores them as anything other than generic types.

Because these invariants are strictly observed, any component can be redesigned or replaced as long as it respects the same contract. The library is not a monolith that must be kept in perpetual balance; it is a layered framework where each layer can be independently modernised, optimised, or even completely re‑implemented.

### Future‑Proofing Through Orthogonality

The plan to extend the library to higher dimensions, unstructured N‑dimensional meshes, arbitrary regulative ideas, and abstract solver interfaces is a direct consequence of this orthogonality:

- **Geometry abstraction** (introducing `CellComplexConcept`, `PrimalFormConcept`) will change the internals of the `geometry` module, but the numerical operators that consume discrete forms will be insulated by the form’s interface (`size()`, `operator[]`, `d()`, `star()`).
- **Adding a new regulative idea** (e.g., a graph metric or an ultrametric on p‑adic numbers) requires only a new metric functor and perhaps a specialised betweenness; all paths and calculus functions will automatically use it because they are templated on those types.
- **Switching the arithmetic backend** (e.g., from `cpp_int_backend` to `gmp_int`) is a one‑line change in `storage.h` that recompiles the entire library without altering a single algorithm.

The architecture thus guarantees that **no future extension is a breaking change**; every addition is orthogonally contained. Refactoring is not a risk to be managed but a normal, healthy process of incremental improvement. The library is designed from the outset to be grown, not merely maintained.

In the end, the Δ‑analysis library is not just a collection of mathematical tools—it is a **framework for constructing mathematical tools**, where the rules of construction (concepts, interface invariants, layered dependencies) are themselves part of the architecture. This meta‑level structure is what distinguishes the library from a typical scientific codebase: it is built to evolve gracefully, absorbing new ideas and new mathematics without ever requiring a rewrite of the existing foundations.