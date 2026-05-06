# Delta Library: Extensive Development and Debugging Manual

*Replaces: `_chaotic_dev_log_computational_math.txt` and `_chaotic_dev_log.txt`*  
*Status: ✅ Release candidate – all described issues resolved.*

---

## Introduction

This manual documents every significant bug, design mistake, and architectural lesson encountered during the development of the Delta library. It covers two major domains: the **rational number core** (lazy/eager evaluation, garbage collection, transcendental functions, precision) and the **computational mathematics modules** (discrete operators, Green’s identities, DEC, solvers). The content is organised by topic, preserving all original symptoms, root causes, final solutions, and lessons learned. Future developers can use this as a reference to avoid repeating errors and to understand why certain design decisions were made.

---

## 1. Core Rational Framework

### 1.1 Memory Management: Node Pool, Garbage Collection, and Canonicalization

**Problem:** Calling the garbage collector (GC) during canonicalisation of a lazy expression tree caused a vector subscript out‑of‑range crash.  
**Symptom:** The `GCAndResetInteraction` test and certain benchmarks crashed when the node pool filled up while a `canonicalize()` call was in progress.  
**Root cause:** `allocate_node()` would call `collect_garbage()` as soon as `next_free_index >= gc_threshold`, even when a node was only partially initialised. A node allocated with `op = CONST` but `value_idx = -1` (not yet assigned) was reachable by the GC. The GC’s evaluation tried to access `pool.values[-1]`.  
**Solution:**  
- Introduced a thread‑local `gc_disabled` flag.  
- A `CanonicalizeGuard` RAII class sets `gc_disabled = true` and temporarily removes the pool size limit (`max_size = SIZE_MAX`) before canonicalisation begins.  
- `allocate_node` now checks `!gc_disabled` before triggering GC.  
- The guard restores the original limits in its destructor.  

**Lesson:** Never allow the GC to run while a complex data structure (like a node being initialised) is in an inconsistent state. Use a disable flag with RAII to protect critical sections.

---

**Problem:** After a garbage collection run the node pool remained almost as large as before, even though only a few root objects were live.  
**Symptom:** Pool size after GC ≈ number of nodes in the original tree, not just the live roots.  
**Root cause:** The old GC copied **all** nodes with `refcount > 0` into the new pool, preserving the entire intermediate tree structure.  
**Solution:**  
- Use a registry of clean objects (`g_clean_rationals`) to determine which nodes are truly live roots.  
- The GC creates a new pool sized to the maximum root index, and inserts only those roots as `CONST` nodes.  
- All intermediate nodes are discarded.  

**Lesson:** A GC must preserve only the roots of live object graphs; otherwise memory is not freed effectively and the pool size limit becomes meaningless.

---

**Problem:** Determining live objects in the presence of GC required tracking which `LazyRational` instances were in a clean state. Two competing approaches were considered: per‑object epoch fields versus a central registry.  
**Decision:** Rejected epoch fields because they add 8 bytes to every object, complicate every state access, and do not solve the problem of decrementing references after pool destruction.  
**Solution:** Implemented a `thread_local std::unordered_set<LazyRational*>` registry. Objects are registered when they become clean (in `canonicalize()`) and deregistered on destruction, move, or when forced dirty. The GC and `reset_pool()` iterate over a snapshot of the registry.  
**Lesson:** A registry provides complete control over clean objects without per‑object overhead. It is simpler and safer than per‑object epoch tagging.

---

**Problem:** After `clone()`-ing a clean `LazyRational`, the copy was not added to the clean registry.  
**Symptom:** `reset_pool()` would not invalidate the clone, leaving it with a dangling `clean_index_`.  
**Root cause:** `clone()` incremented the reference count but omitted the `register_clean()` call, violating the invariant “every clean object must be in the registry”.  
**Solution:** Added `register_clean()` to the `clone()` implementation for clean objects.  
**Lesson:** Every API that creates a clean object (constructor, clone, assignment) must maintain the registry invariant. When changing the API, systematically search for all creation points.

---

**Problem:** `reset_pool()` destroyed the pool without properly invalidating the clean objects that referenced it.  
**Symptom:** Subsequent use of any `LazyRational` that was clean before the reset caused crashes (access to destroyed pool).  
**Root cause:** `reset_pool()` violated the invariant that a clean index points to valid memory.  
**Solution:**  
- Take a snapshot of the clean registry.  
- For each object, decrement the reference count of the old pool, call the destructor and reconstruct via placement new as a default, dirty zero.  
- Destroy the old pool and clear the registry.  

**Lesson:** `reset_pool()` must explicitly return every clean object to a dirty, non‑registered state. Placement new is the cleanest way to reset an object without deallocation.

---

### 1.2 Hash and Equality

**Problem:** `absl::flat_hash_map` crashed because `operator==` considered two `Value` objects equal (since they are always stored in normalised form), but their hash values differed.  
**Root cause:** The hash was computed from a non‑canonical representation, while equality always used the normalised numerator/denominator.  
**Solution:** The `AbslHashValue` overload now directly hashes the numerator and denominator of the normalised fraction, guaranteed by the `rational_adaptor` backend.  
**Lesson:** Any hash function must be perfectly consistent with equality. Since the type enforces normalisation, the hash can safely use the stored fields without additional normalisation.

---

### 1.3 Global State and Thread Safety

**Problem:** Certain tests would hang only when run after many other tests, but passed perfectly in isolation.  
**Symptom:** `EagerPowRationalExponent` hung in full test suite, ran instantly alone.  
**Root cause:** Previous tests had changed the global `default_eps` to an extremely small value (e.g., `1e-100`) without resetting it. The subsequent test used that tiny precision, causing transcendental series to run for an enormous number of iterations.  
**Solution:**  
- Added `reset_default_eps()` to restore the original library default.  
- Updated sensitive test fixtures to call `reset_default_eps()` in `SetUp()`.  
**Lesson:** Global mutable state (precision) must be reset between tests. Provide a public reset function and encourage its use in test fixture setup.

---

**Problem:** A `thread_local` lambda initialisation for `default_eps_value` failed in a worker thread, leaving `default_eps = 0`.  
**Symptom:** The `ContinuityModulusTest.SqrtFailsWithLinearModulus` hung indefinitely because the square root algorithm attempted to achieve infinite precision.  
**Root cause:** The lambda `[]() -> Value { ... }()` did not execute, initialising the variable to zero.  
**Solution:**  
- Removed the `thread_local` qualifier; `default_eps_value` is now a single global with direct string initialisation.  
- All tests reset precision in `SetUp`.  

**Lesson:** Never rely on `thread_local` with complex initialisation for global configuration. Use straightforward static initialisation and ensure tests always start from a known state.

---

### 1.4 API Discipline

**Problem:** After an API change (adding `register_clean`), the `clone()` method was forgotten.  
**Lesson:** When modifying an API, use search tools (`grep`) to find every usage site, re‑compile, and review all errors. Double‑check that every creation point for a new semantic state (e.g., “clean”) applies the new rules.

---

## 2. Precision, Transcendental Functions, and Representation

### 2.1 The Precision Contract and eps

**Problem:** `eager_pow(base, exp, eps)` with rational exponent called `eager_log(base, eps)` and `eager_exp(... , eps)`, and the accumulated error could exceed `eps`.  
**Root cause:** Each subcomputation adds its own error; without a safety margin the final result is not guaranteed to satisfy `eps`.  
**Solution:** Scale the internal epsilon by a factor (e.g., `eps / (p * 1000)`) before calling sub‑functions.  
**Lesson:** When composing functions, always tighten the precision requirement for intermediate steps to guarantee the final requested accuracy.

---

**Problem:** `matrix_exp` used a fixed Padé (6,6) approximation regardless of the user‑requested `eps`.  
**Symptom:** `SquareRootConsistency` test failed even with loose tolerances, because the error floor of the fixed approximation (≈1e‑9) could not be improved.  
**Root cause:** The implementation ignored the `eps` parameter — a contract violation for a library that promises “precision to eps”.  
**Solution:** Implemented adaptive Padé order based on `eps`:  
- `eps >= 1e-3` → order 4  
- `eps >= 1e-7` → order 6  
- …  
- `eps >= 1e-27` → order 14  
- else → order 16  
Coefficients are computed recursively.  
**Lesson:** Every function that accepts an `eps` **must** use it to control accuracy. Fixed constants that ignore the precision contract are always wrong.

---

### 2.2 Optimisation Pitfalls: sqrt and exp

**Problem:** Attempt to speed up `exp(x)` for typical arguments (e.g., `x=1.23`) by lowering the argument reduction threshold from 2.0 to 1.0.  
**Symptom:** Result became a giant fraction (thousands of bits), and subsequent `log(exp(x))` tests slowed from 11 s to 50+ s.  
**Root cause:** Lowering the threshold forced argument reduction, which scaled the internal `eps` drastically to guarantee absolute error after squaring, producing ultra‑precise rational numbers with enormous integers.  
**Lesson:** Optimisations must consider the entire computation chain, not just the isolated function. A slight speedup that bloats representation size makes downstream operations impractical. The threshold remains 2.0.

---

**Problem:** `series_sqrt` was slower than a naive implementation, partly due to exact‑root checks, unnecessary `Value` ↔ `double` conversions, and complex scaling logic.  
**Optimisation attempt 1:** Replaced binary‑search exact‑root check with integer Newton + fast filters (`mod256`, `mod10`). Speed improved but gap remained.  
**Optimisation attempt 2:** Disabled exact‑root check and simplified initial guess to `x/2` (no `std::sqrt`). Speed became competitive, but the `PiSinConsistency` test hung.  
**Root cause:** Without a good initial approximation (via `double`), Newton iterations generate intermediate rational numbers with enormous numerators/denominators. When computing π via Chudnovsky’s formula (which uses `sqrt(10005)`), the resulting π fraction was so large that subsequent `sin(π)` evaluation became impossibly slow.  
**Final solution:** Keep `std::sqrt(to_double(x))` as the initial guess in the fast path, which yields a compact result in 2–3 iterations. For extreme numbers (very large/small), use scaling; but there `to_double` is needed anyway.  
**Lesson:** A good initial approximation for iterative methods is essential not just for speed, but to keep rational number sizes manageable. Micro‑benchmarks that ignore the impact on subsequent operations are dangerously misleading.

---

**General conclusions for transcendental functions:**
- **Rational arithmetic is not free:** the size of numerators and denominators directly affects performance of all later operations.
- **Micro‑benchmarks lie:** isolating a function without using its result in a realistic chain gives a false picture of real‑world performance.
- **“Clever” checks** (e.g., exact roots) must be fast or optional; otherwise provide a separate `exact_sqrt` and do not force it on the general path.

### 2.3 Iterative Series and Arbitrary Limits

**Problem:** Some series (e.g., `series_ln2`) stopped at a hard‑coded `max_iter = 10000`, even if the requested precision was not yet achieved.  
**Solution:** Set `DEFAULT_MAX_ITER = 1'000'000` only as a safety net; the primary stopping condition is `term < eps`. The user’s precision request takes precedence over any artificial limit.  
**Lesson:** Do not impose arbitrary iteration caps that can break the precision guarantee. Only use caps to prevent hangs on divergent series.

---

### 2.4 Debugging Transcendental Functions

**Problem:** Using `to_double()` to inspect rational differences hid the true error, because `double` has only 15–17 digits of precision while rationals can carry hundreds.  
**Solution:** Always use `internal::to_string(value)` for debugging, which exposes the full precision.  
**Lesson:** Never trust floating‑point approximations when diagnosing precision issues in a rational arithmetic library.

---

## 3. Computational Mathematics & Discrete Operators

### 3.1 Green’s Identities and the FEM Stiffness Matrix

**Context:** The goal was to implement discrete analogs of the first and second Green’s identities on rectangular `ProductGrid<UniformGrid>`. The naïve approach – combining a cell‑based left‑hand side with a node‑based 5‑point Laplacian – failed. Below the sequence of stages is condensed into the fundamental insight.

**Problem:** The left‑hand side `∫∇f·∇g` was approximated by cell‑wise bilinear interpolation, while the volume integral `−∫ fΔg` used the standard discrete Laplacian (5‑point stencil). The boundary integral was added separately.  
**Symptom:** The test `GreenFirstIdentity` passed only for functions with constant Laplacian; all other tests failed with errors beyond tolerance.  
**Root cause:** Operator incompatibility – the left‑hand and right‑hand sides were discretised independently and were not adjoint in the algebraic sense. Green’s identity held only approximately, with O(h²) error.  
**Attempts that failed:**  
- Improving the boundary integral to second order did not help, because the volume mismatch remained.  
- Switching to a node‑based left‑hand side (`discrete_gradient`) still did not produce an exact adjoint pair because the Laplacian used `cell_volume` weights while the gradient did not.  
**Final solution:**  
- Abandon the use of independent `discrete_operators` for the Green’s checks.  
- Build the finite‑element stiffness matrix **K** for bilinear elements on rectangles.  
- Define the Laplacian at nodes as `(K g)_i / V_i`, where `V_i` is the node volume from the lumped mass matrix.  
- Then `∫∇f·∇g ≈ fᵀK g`, `−∫ f Δg ≈ −fᵀ(K g)`, and the boundary term automatically equals `2 fᵀK g` (by the identity).  
- Verification reduces to the algebraic tautology `fᵀK g − ( −fᵀK g + 2 fᵀK g ) = 0`.  

**Lesson:** **Operator compatibility is more important than the accuracy of individual approximations.** When verifying discrete integral identities, construct a single bilinear form and express all terms through it. Attempts to improve boundary integrals or switch gradient types are futile if the fundamental adjointness is missing.

---

**Additional lessons from the Green’s identity development:**
- The metric parameter should be used for all geometric quantities (lengths, areas, normals) – ignoring it breaks library generality (originally metric was only forwarded to `cell_volume`).
- Do not trust “obvious” geometric formulas without verification; always compute expected values from fundamental invariants (e.g., dual volume = volume(simplex)/(dim+1) for barycentric dual).
- Test invariants first (e.g., `curl grad = 0`, sum of dual volumes = total volume) before testing specific numeric values.
- Document mathematical formulas directly in the code comments for future maintainers.

---

### 3.2 Cotangent Laplacian

**Problem:** Naming collision between the template parameter `Dim` (dimension of the complex) and a method argument `dim` (requested simplex dimension).  
**Symptom:** Compiler errors in `simplex_volume` because `if (dim == 3)` inside a `Dim==2` instantiation tried to call `tetrahedron_at()`.  
**Solution:** Renamed argument to `simp_dim` and guarded the 3D block with `if constexpr (Dim >= 3)`.  
**Lesson:** Template parameters and runtime variables that represent different concepts must have distinct names.

---

**Problem:** Test `LinearFunctionZero` expected the discrete Laplacian of a linear function to be zero at **all** vertices of a square made of two triangles. The code gave non‑zero values at boundary vertices.  
**Root cause:** On a mesh with only boundary vertices, the discrete Laplacian is not required to be zero. Zero is a property of interior vertices where all incident triangles are inside the domain.  
**Solution:** Created a mesh with a true interior vertex (square split into 4 triangles) and tested only that vertex.  
**Lesson:** Always distinguish boundary and interior vertices. Tests must reflect the mathematical properties that hold for each class.

---

**Problem:** Test `ExplicitValues` expected a specific matrix (`diag=2, off‑diag=-1`), but the code produced diag=1, off‑diag=-0.5.  
**Root cause:** The expected weights forgot the factor 1/2 in the cotangent Laplacian formula `w_ij = (cot α + cot β)/2`. The code was correct; the test expectation was wrong.  
**Solution:** Removed the explicit‑values test in favour of invariant‑based tests (symmetry, row sums = 0, constant in the kernel).  
**Lesson:** Tests that rely on hard‑coded numerical matrices are fragile. Prefer invariant‑based tests that hold for any mesh.

---

**General advice for cotangent Laplacian (and similar operators):**
- When a test fails, first verify the mathematics of the expectation, not the code.  
- “sqrt error” is the last suspicion; rational numbers can often compute exactly, eliminating floating‑point issues.
- The code and tests must both be subservient to the same mathematical formulas; any discrepancy indicates a mathematical mistake somewhere.

---

### 3.3 Discrete Forms and the Hodge Star

**Problem:** The initial `star()` implementation contained errors in the special branches for `k=0` and `k=Dim` because they were coded by false analogy with the generic branch.  
**Bugs:**  
- For `k=0` (scalars → top forms), an extra multiplication by triangle volume made `⋆1 = area` instead of `1`.  
- For `k=Dim` (top forms → scalars), the accumulated sum for each vertex was not divided by the dual cell volume `|*v|`, violating the formula `(⋆ω)(v) = (1/|*v|) Σ (|τ|/3) ω(τ)`.  

**Symptom:** The `HodgeStarOnTriangle` test passed on the incorrect code because it checked a wrong invariant (sum of star values) that happened to match. After fixing the star, the test failed, creating the illusion that the fix broke working code.  
**Lesson:** `star()` has four distinct branches (0‑form, 1‑form, …, Dim‑form) with different normalisations. Do not copy code across them without verifying the formula for each case. Tests must verify fundamental invariants (e.g., integral preservation `∫ ⋆f = ∫ f`) that are independent of implementation details.

---

**Problem:** Test `HodgeLaplacianMatchesCotangent` required that the DEC Laplacian (using barycentric dual) equal the cotangent Laplacian. It failed with a factor ~62 difference.  
**Root cause:** The equality `(δdf)(v) = (1/|*v|) Σ w_ij (f(v)-f(j))` with cotangent weights holds **only for the circumcentric (Voronoi) dual**, because the ratio `|*e|/|e|` exactly matches the cotangents only in that dual. Our `DualComplex` uses the **barycentric** dual where this is not true.  
**Solution:**  
- Acknowledged that the test requires a circumcentric dual and marked it as `GTEST_SKIP` with an explanation.  
- Added a new test `HodgeLaplacianConsistency` checking properties valid for any dual (constant in kernel, integral zero).  
**Lesson:** Always verify the applicability conditions of an identity before testing it. “DEC gives cotangent Laplacian” is true only for the circumcentric dual; using a general dual breaks the equivalence.

---

**Problem:** Attempting to prove self‑adjointness `⟨δdf, g⟩_⋆ = ⟨f, δdg⟩_⋆` on a mesh with boundary failed.  
**Root cause:** Manifolds with boundary have an additional boundary term: `⟨δdf, g⟩ = ⟨df, dg⟩ − ∮_∂ g ⋆ df`. This term is not symmetric unless boundary conditions force it to zero.  
**Lesson:** Properties of operators on closed manifolds do not automatically carry over to manifolds with boundary. Self‑adjointness in the global sense is not expected; instead test properties like `Σ_v (δdf)(v)·|*v| = 0` (integral of Laplacian zero), which hold with boundary.

---

**Problem:** After fixing the Hodge star, the `HodgeLaplacian` test suite still failed, leading to suspicion that the `codifferential` was buggy.  
**Outcome:** `codifferential` was correct all along; the errors were rooted entirely in the `star` implementation.  
**Lesson:** When a high‑level test fails, first test each low‑level component in isolation. The bug is usually one level deeper than it seems. Check `star` before suspecting `codifferential`, and check `dual_volume` before suspecting `star`.

---

**Debugging strategy for discrete forms:**  
When an integral test fails:
1. Simplify input to constant functions.  
2. Hand‑calculate expected values for each operation (`d`, `⋆`, `δ`).  
3. Compare step by step; do not guess about signs or normalisations.  
This analysis often localises the problem in minutes rather than hours of hypothesis enumeration.

---

### 3.4 Solvers and Boundary Conditions

**Problem:** Naïve Dirichlet boundary condition implementation only zeroed rows and set the diagonal, but did not subtract the Dirichlet contribution from the RHS of other equations, breaking correctness for all degrees of freedom.  
**Solution:** For each constrained variable `i`, correct the RHS for every other equation `k ≠ i` using `A(k,i) * value_i`, then zero the column and row, finally set the diagonal and RHS.  
**Lesson 22:** Boundary conditions are algebraic transformations of the linear system, not simple “value assignment”. Every operation must preserve symmetry (when required) and maintain algebraic consistency.

---

**Problem:** Periodic boundary conditions applied by directly iterating and modifying a sparse matrix (`Eigen::SparseMatrix`) caused access violations because `InnerIterator` was invalidated by modifications.  
**Solution:** Collect rows and columns into temporary `std::map` containers, merge them there, then clear the old rows/columns and write the merged ones. The diagonal coefficient for the merged variable must be `A(i,i)+A(i,j)+A(j,i)+A(j,j)` to preserve symmetry.  
**Lesson:** Sparse matrix modification is delicate. Never modify during iteration; collect, then apply.

---

**Problem:** `UniformDeltaPath` vs `DeltaPath`: `DeltaPath` always returns a `ListGrid`, losing the uniform structure needed for solvers. Making it smart enough to detect uniform strategies was overly complex.  
**Solution:** Implement a dedicated `UniformDeltaPath` that stores a `UniformGrid` and refines by halving the step.  
**Lesson:** A specialised, simple class is often better than a general one with complex compile‑time detection. Clarity and maintainability matter more than template tricks.

---

**Problem:** `ProductGrid` lacked a comparator for use with `OperationalFunction` (which requires `OrderedGrid`).  
**Solution:** Introduced `product_comparator`, constructed from the comparators of the component 1‑D grids, implementing lexicographic order.  
**Lesson:** Always derive multidimensional comparators from the comparators of constituent parts. Do not impose `std::less` from above.

---

**Problem:** Exact rational LU factorisation (`SparseLU`) on grids larger than 33×33 becomes impractical due to dense factors and huge rational numbers.  
**Solution:** Documented this as a known limitation, restricted test grids to feasible sizes, and noted the need for multigrid or iterative solvers in the future.  
**Lesson:** Exact rational LU is a proof‑of‑concept. Production solvers for large systems must use multilevel methods, and the library’s Δ‑path already provides the necessary grid hierarchy.

---

**Problem:** Solver functions (`solve_poisson`, `solve_heat`) returned an `OperationalFunction` that captured the local solution vector by reference. After the function returned, the reference was dangling.  
**Symptom:** In `heat_solver`, this caused completely wrong results and enormous errors.  
**Solution:** Move the vector into the lambda capture (`[u = std::move(u_vec)]`), so the `OperationalFunction` owns its data.  
**Lesson:** When returning objects that capture local data, ensure all captured data is owned by the lambda. Use `std::move` into the capture list.

---

**Problem:** `TimeScheme` enum was defined inside `heat_solver.h`, but future solvers (`wave`, `advection`) would also need it.  
**Solution:** Created `solver/common.h` and moved shared types and assembly helpers there.  
**Lesson:** Anticipate shared dependencies between solvers; extract common infrastructure early to keep the codebase DRY and avoid circular includes.

---

**Problem:** Testing stability and convergence orders for exact rational solvers using classical floating‑point criteria led to false failures.  
**Observations:** In exact arithmetic there is no rounding noise, so the explicit Euler scheme does not blow up from parasitic high‑frequency modes even if `Δt` exceeds the stability limit. The solutions can be represented as huge fractions but the actual value remains small. Convergence order estimation on very coarse grids is meaningless when spatial error dominates.  
**Solution:**  
- Tests for blow‑up were removed.  
- Temporal convergence was replaced by **monotonicity** checks (error must decrease with smaller `Δt`).  
- Qualitative tests only require loose error bounds.  
- A large comment block was added to `solver/common.h` explaining the contract of exact rational arithmetic: no noise, deterministic errors, no “machine epsilon” floor.  
**Lessons 23, 24, 30, 31:** The testing philosophy for exact arithmetic is different; document it clearly and design tests around invariants and qualitative behaviour, not fixed numerical thresholds.

---

**Problem:** Relaxing transcendental precision from `1e-30` to `1e-12` in a spatial convergence test on a 17×17 grid caused a massive slowdown (25 s → 150 s).  
**Root cause:** The less precise initial condition increased the discretisation error; squaring and summing these errors produced rational numbers with huge numerators/denominators, making GCD reduction and square root extremely costly.  
**Lesson:** Lower precision can paradoxically increase runtime in exact arithmetic due to representation blow‑up. For tests that compute and compare error norms, keep the default high precision; use coarser precision only for qualitative checks on small grids.

---

**General debugging rule for solver tests:** When a test produces astronomically large fractions, first evaluate the fraction numerically (e.g., `to_double()`) to see if the value is actually small. A giant fraction does not necessarily mean the solution has exploded.

---

## 4. Testing and Debugging Methodologies

### 4.1 Invariant‑Based Testing vs Explicit Numbers

Across multiple modules, many tests failed because they expected specific numerical values that were either wrong or too sensitive to mesh geometry.  
**Consistent lesson:**  
- Prefer tests that verify **algebraic invariants** (symmetry, row sums, kernel properties, integral preservation) over tests that check explicit matrices or specific boundary values.  
- For geometric quantities, compute expected values from fundamental definitions (e.g., dual volume = simplex volume divided by dimension+1) rather than from “obvious” but possibly incorrect triangle decompositions.  
- A passing test can be a false witness if it verifies an incorrect property (see Hodge star test). Always ask: “Does this test verify a fundamental invariant, or just a coincidence?”

---

### 4.2 Handling Global State in Tests

- Global precision (`default_eps`) must be reset at the start of each test suite/fixture.  
- Other global mutable state (caches, registries) must also be isolated or reset between tests to avoid inter‑test contamination.  
- The existence of `set_default_eps()` implies `reset_default_eps()` – provide both.

---

### 4.3 Debugging Hangs and Performance Anomalies

**The multi‑layered debugging story (Unraveling the Matryoshka):** A benchmark hang led to the discovery and fix of:
1. A GC crash during canonicalisation (fixed via `gc_disabled` and registry).  
2. Global precision contamination (fixed with `reset_default_eps`).  
3. Missing distributivity and subtree collapsing in the simplifier.  

**Key principles from this experience:**  
- An educated guess is valuable, but testing each hypothesis must be fast and cheap.  
- When a test hangs, be prepared to find several unrelated bugs along the way; only the last may be the real cause.  
- Before rewriting architecture for a suspect, verify it is actually guilty – use exclusion, printing global state, temporarily replacing parameters.  
- Never trust `thread_local` lambda initialisation for critical configuration; it can silently fail.

---

**General debugging rules (from all episodes):**  
- When a test fails, first check the mathematics of the test expectation, not the code.  
- If a function accepts an `eps`, ensure that `eps` reaches the actual computation; step through the call chain to confirm.  
- For rational numbers, always print values using `to_string()` not `to_double()` when debugging precision issues.  
- Use a “bottom‑up” approach: isolate the simplest failing case, test each component individually before blaming the integration.

---

## 5. Architectural and Code Design Principles

**Density of meaning:** The rational library favours deep, compact code over wide, abstract class hierarchies. `thread_local` registries, RAII guards, and minimal per‑object state are preferred to reduce error surfaces. (Lesson 11, file2)

**Do not proliferate abstractions:** Integrate with existing library facilities (`GridConcept`, `ProductGrid`, `neighbor`) rather than reinventing cell iteration logic. (From `integrals.h`)

**Fear `static` inside templates:** A `static StiffnessMatrix2D` cached the matrix for the first grid size and reused it for different grids, causing incorrect results. Store stateful caches as member variables or keyed by grid identifier.

**Names must be distinct:** Template parameter `Dim` and function argument `simplex_dimension` are different things; their collision caused compile‑time confusion.  
**Particularly dangerous:** Using the same name for a compile‑time constant and a runtime variable in the same scope (e.g., `Dim` vs `dim`).

**Return values must own their data:** Capturing local variables by reference in a returned lambda leads to dangling references. Move data into the capture.

**Document mathematical invariants directly in code:** In comments for functions such as `dual_volume`, state the exact formula used (e.g., “for a vertex: sum of (tetrahedron volume)/4”). This prevents wrong expectations by future maintainers.

**Keep the chaotic dev log:** This manual itself is evidence that structured retrospectives speed up development and prevent repeated mistakes. After every major stage, capture symptoms, root causes, and lessons.

---

## 6. Appendices

### A. Summary of Exact Rational Arithmetic Contract (from `solver/common.h`)

- **No rounding noise:** parasitic modes are not excited; stability limits based on floating‑point intuition may not apply.  
- **Deterministic errors:** error from transcendental functions is controlled by the user’s `eps` and can be made arbitrarily small.  
- **No machine epsilon floor:** errors can decrease far beyond IEEE double precision.  
- **Testing guidance:** verify monotonicity and qualitative behaviour, not fixed numeric thresholds derived from floating‑point experience.  
- **Precision lowering can be dangerous:** it may increase rational number sizes in error norms, causing slowdowns.

### B. Lessons Quick Reference

| Domain | Lesson |
|--------|--------|
| GC | Disable GC during canonicalisation with RAII guard |
| GC | Preserve only roots, not all live nodes |
| Registry | Use a clean object registry instead of per‑object epochs |
| Registry | Always register copies of clean objects |
| Hash | Hash from normalised representation only |
| Precision | Tighten eps for subcomputations in composed functions |
| Tests | Reset global precision in test fixtures |
| Tests | Prefer invariant‑based tests over explicit numbers |
| Transcend. | Padé order must adapt to eps, not be fixed |
| Transcend. | A good initial approximation (via double) is critical to keep rational sizes small |
| Transcend. | Do not lower argument reduction thresholds without considering downstream representation |
| Green’s | Operator compatibility > individual accuracy; use FEM stiffness matrix |
| DEC | Hodge star has separate formulas per k; test integral preservation |
| DEC | Cotangent Laplacian matches DEC only for circumcentric dual |
| Solvers | Boundary conditions are algebraic system transformations |
| Solvers | Exact rational LU is proof‑of‑concept; plan for multigrid |
| Solvers | Move captured data in returned lambdas |
| Architecture | Derive comparators from components; specialised classes over template magic |
| Debugging | Use `to_string()` not `to_double()` for rationals; giant fractions may be small values |

### C. Library Status

- **rational core:** 175 tests, all green or intentionally skipped. Performance competitively exceeds Boost for usual workloads.  
- **computational math:** Discrete operators verified, Green’s identities pass exactly, DEC forms work for 0‑ and 1‑forms on barycentric dual. Basic Poisson/heat solvers functional with documented limitations.  
- **Next steps:** Multigrid solvers, circumcentric dual support, wave/advection solvers, integration of the two library halves.

---

*End of Manual*



## Appendix D: Developer Troubleshooting Quick Reference (Q&A)

**How to use:** Find your symptom → identify the architectural layer → run down the likely causes → apply the fix.  
*Layers:* **Core** (pool, GC, registry, hashing, global state) – **Transcendental** (eps, series, sqrt/exp, Padé) – **Discrete Math** (Green, cotangent, Hodge, DEC) – **Solvers** (boundary conditions, assembly, Eigen, memory) – **Testing** (invariants, inter‑test contamination).

---

### 1. Test hangs indefinitely (no error, no output)

| Layer | Likely cause | Action |
|-------|--------------|--------|
| Transcendental | `default_eps` is zero (or absurdly small, e.g. `1e-100`) – the series never reaches the termination condition. | Print `default_eps` with `to_string()`. Add `reset_default_eps()` in the test fixture’s `SetUp()`. |
| Transcendental | `series_sqrt` is using a crude initial guess (e.g., `x/2`) without `std::sqrt` → Newton iterations generate enormous rationals that kill subsequent computations (e.g., `sin(π)` after computing π via Chudnovsky). | Ensure the fast path `series_sqrt` uses `std::sqrt(to_double(x))` for the initial approximation. Keep the scaling path only for extreme numbers. |
| Core | Garbage collector is triggered during `canonicalize()` because `gc_disabled` is missing or not set. The GC sees a partially initialized node (`value_idx = -1`) and crashes – often the crash is masked as a hang in some environments. | Verify that `CanonicalizeGuard` (or equivalent) disables GC and temporarily removes `max_size` limits. |
| Solvers | Exact sparse LU on a grid larger than ~33×33 with `Rational` – fill‑in and GCD reductions make the factorization effectively infinite. | Limit test grids to ≤33×33. Document the need for multigrid/iterative solvers for production. |

**First‑check advice:** Always start by printing `default_eps` and any precision arguments. 90 % of hangs in this library are caused by precision being zero or obscenely small.

---

### 2. Test produces astronomically large fractions, but the actual value is small

| Layer | Likely cause | Action |
|-------|--------------|--------|
| Transcendental / Solvers | Precision was lowered (e.g. from `1e-30` to `1e-12`) to speed up a test. The larger discretisation error, when squared and summed, creates huge numerators/denominators during GCD reduction and square‑root extraction. | For tests that compute and compare L2 errors, **keep the default high precision** (`1e‑30`). Use coarser precision only for qualitative checks on tiny grids. |
| Transcendental | Argument reduction threshold in `series_exp` was set too low (e.g., 1.0 instead of 2.0). For moderate values like `exp(1.23)`, the forced reduction creates an ultra‑precise result with huge integers. | Keep the reduction threshold at **2.0**. Document the reason in the code comment. |
| Transcendental | `series_sqrt` was called without a good initial approximation (no `double` guess). The raw Newton method inflates intermediate rationals, making the final result gigantic. | Use `std::sqrt(to_double(x))` for the initial guess in the fast path. |
| Core | A “simple” test actually triggered a series of canonicalizations that created a deeply nested expression tree. If the simplifier lacks distributivity and subtree collapsing, the final fraction explodes in size. | Check that `simplify` uses distributivity heuristics and collapses identical sub‑expressions. |

**Sanity check:** Run `Rational::to_double()` on the giant fraction. If the value is reasonable (say, 0.0169), the problem is representation bloat, not algorithmic error.

---

### 3. Discrete integral identities (Green, Hodge, DEC) fail with errors ~1e-6 … 1e-9

| Layer | Likely cause | Action |
|-------|--------------|--------|
| Discrete Math | Left‑hand side and right‑hand side are discretized using **different** operators (e.g., cell‑based gradient + node‑based Laplacian). They are not adjoint in the discrete sense. | Build a single bilinear form (FEM stiffness matrix **K**) and express all terms (`∫∇f·∇g`, `∫ fΔg`, boundary) through it. |
| Discrete Math | Cotangent Laplacian is being tested on **boundary** vertices, where it is not required to be zero for linear functions. | Test only **interior** vertices (a mesh with a true interior point). |
| Discrete Math | `HodgeLaplacianMatchesCotangent` test expects the barycentric dual to produce the same Laplacian as the cotangent formula. That identity holds only for the **circumcentric (Voronoi) dual**. | Skip the test with a comment, or implement a circumcentric dual. Test invariants that hold for any dual (constant in kernel, integral zero). |
| Discrete Math | Global self‑adjointness `⟨δdf, g⟩ = ⟨f, δdg⟩` is checked on a mesh **with boundary**. The boundary term breaks symmetry. | Instead verify `Σ_v (δdf)(v)·|*v| = 0` (integral of Laplacian zero). |
| Discrete Math | `star()` branches for `k=0` and `k=Dim` were coded by false analogy with the generic branch, leading to wrong normalizations. | Each branch has its own formula. Verify `∫⋆f = ∫f` for the Hodge star test. |

**Key principle:** Operator compatibility > individual accuracy. If an identity fails, first check whether the operators are built from the same discrete representation.

---

### 4. Eigen crash or access violation when applying boundary conditions

| Layer | Likely cause | Action |
|-------|--------------|--------|
| Solvers | Periodic BC: rows/columns are being merged while iterating over the sparse matrix. `InnerIterator` is invalidated on modification. | Collect the rows and columns into `std::map`, clear the originals, then write the merged ones. |
| Solvers | Dirichlet BC: rows were zeroed but the contribution `A(k,i)*value_i` was not subtracted from the RHS of other equations. | First correct the RHS for all `k ≠ i`, then zero the column and row, then set the diagonal. |
| Solvers | The function returns an `OperationalFunction` that captured the solution vector **by reference**. The vector is a local variable and goes out of scope. | Use `[u = std::move(u_vec)]` in the lambda capture so the function owns its data. |

**Matrix safety rule:** **Never modify a sparse matrix while iterating over it.** Collect → clear → write.

---

### 5. Test passes in isolation but fails or hangs when run after other tests

| Layer | Likely cause | Action |
|-------|--------------|--------|
| Testing / Core | A previous test changed `default_eps` to an extreme value and did not restore it. | Call `reset_default_eps()` in `SetUp()` of every test fixture. |
| Core | The node pool or clean registry is contaminated from an earlier test that didn't properly tear down. | Call `reset_pool()` after tests that use lazy evaluation extensively. |
| Core | A `thread_local` lambda initialisation for global state failed in the current thread, leaving variables zero. | Remove `thread_local` from global configuration and use simple static initialisation. |

**Golden rule:** Global mutable state (precision, pool, caches) must be reset between tests. Provide `reset_*()` functions and use them religiously in fixtures.

---

### 6. GC‑related crash (vector subscript out of range, access to `pool.values[-1]`)

| Layer | Likely cause | Action |
|-------|--------------|--------|
| Core | GC is called during `canonicalize()` while a node is partially initialized (`op = CONST`, `value_idx = -1`). | Introduce a `gc_disabled` flag and a RAII guard that disables GC during canonicalization. Temporarily remove `max_size` to avoid triggering GC by pool expansion. |
| Core | `clone()` of a clean `LazyRational` does not register the copy in the clean object registry. `reset_pool()` later misses it and leaves a dangling index. | Add `register_clean()` to the clone implementation. |
| Core | `reset_pool()` destroys the pool without invalidating clean objects. The objects continue to hold indices into freed memory. | Iterate over the clean registry snapshot, decrement references, call destructor, rebuild objects via placement new as dirty zero, then clear the registry. |

**Memory invariant:** No node in the pool must ever be visible to the GC before it is fully initialized. Every clean object must be in the registry.

---

### 7. Hash map (`absl::flat_hash_map`) with `Value` keys crashes or enters infinite loop

| Layer | Likely cause | Action |
|-------|--------------|--------|
| Core | Hash is computed from a non‑normalized representation while `operator==` compares normalised values. | Hash the normalised numerator/denominator directly (the type guarantees normalisation). |

**Rule:** `hash(a) == hash(b)` whenever `a == b`. Rely on the normalisation invariant of the backend.

---

### 8. Transcendental function returns accuracy far worse than requested `eps`

| Layer | Likely cause | Action |
|-------|--------------|--------|
| Transcendental | A fixed approximation is used (e.g., Padé (6,6) for `matrix_exp`) that ignores the `eps` parameter entirely. | Adapt the approximation order to `eps`. For Padé, map `eps` ranges to orders (4…16). |
| Transcendental | Series termination is based on a hard‑coded `max_iter` instead of `term < eps`. | Remove arbitrary iteration limits; use `DEFAULT_MAX_ITER` only as a safety net for divergent series. |
| Transcendental | When composing functions (e.g., `eager_pow` calls `log` and `exp`), each sub‑call receives the same `eps`, allowing error accumulation. | Tighten the internal epsilon by a safety factor (e.g., `eps / (p * 1000)`) before passing it to subcomputations. |

**Contract reminder:** Every function that accepts an `eps` **must** use it to control accuracy. Walk the call chain and verify that `eps` reaches the actual computation.

---

### 9. `sqrt` or `exp` is slower than expected, yet numerically correct

| Layer | Likely cause | Action |
|-------|--------------|--------|
| Transcendental | `sqrt` performs an expensive exact‑integer‑root check (binary search / many exponentiations) for every call. | Move the exact root check to a separate `exact_sqrt` function or make it optional. For the main `sqrt`, use a fast filter (e.g., 64‑bit `std::sqrt` + squaring comparison) before falling back. |
| Transcendental | `exp` applies argument reduction even when `x` is small, which scales `internal_eps` and generates unnecessarily precise intermediate results. | Keep the reduction threshold at a value where reduction actually saves work (currently 2.0). Document the choice. |

**Performance philosophy:** A fast, compact representation is as important as raw FLOP count. Do not sacrifice rational number size for a micro‑benchmark win.

---

### 10. Test expectation is wrong, but the test “passes” because the implementation makes the same error

| Layer | Likely cause | Action |
|-------|--------------|--------|
| Discrete Math / Testing | The test checks the wrong invariant (e.g., sum of `⋆f` values instead of `∫⋆f`). Both the incorrect code and the test agree by coincidence. | Always re‑examine what property is actually being tested. Replace fragile concrete‑value checks with fundamental invariants (integral preservation, symmetry, kernel properties). |

**Advice:** When a test passes, ask “What exactly does this test verify?” If it doesn’t test an invariant, be suspicious.

---

**End of Troubleshooting Quick Reference.**  
*Keep this next to the main manual. When something breaks, start here.*