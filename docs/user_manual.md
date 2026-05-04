*Back to [README](../README.md) | [Documentation Index](../README.md#-documentation)*

# User Manual

## Philosophy: Read the Tests

The test suite of Δ‑analysis is not a collection of trivial unit checks. Each test validates **fundamental mathematical identities** – the discrete version of Green’s theorem, exactness of the exterior derivative, convergence of Riemann sums, nilpotency d²=0, Hodge star consistency, and more. If a test passes, the corresponding mathematical invariant holds exactly (up to rational arithmetic).

Therefore, the most reliable way to learn how to use a particular component is to **look at the test that exercises it**. This manual provides a few annotated examples to get you started, but the full catalogue of working, end‑to‑end usage patterns is in the `tests/` directory. Treat the tests as executable documentation.

---

## 1. Basic Types and Initialization

All examples assume that the necessary headers are included and `using namespace delta;` is in effect (or appropriate qualifications are used).

### Rational Numbers

```cpp
#include "delta/core/rational.h"

// Construction
Rational a = 1_r;              // integer literal
Rational b = "0.5"_r;          // decimal string
Rational c = "1/3"_r;          // fraction
Rational d(2, 3);              // numerator, denominator
Rational e = delta::sqrt(2_r); // transcendental with default epsilon
```

For performance tips, see [LazyRational and evaluation](#lazyrational-and-performance).

### Grids

```cpp
#include "delta/core/list_grid.h"
#include "delta/core/uniform_grid.h"
#include "delta/core/product_grid.h"

// 1D sorted list of points
ListGrid<Rational, std::less<Rational>> grid({0_r, 1_r, 2_r});

// Uniform grid: start, step, number of points
UniformGrid<Rational> ugrid(0_r, 1_r/4_r, 5); // {0, 0.25, 0.5, 0.75, 1}

// 2D product grid from uniform 1D grids
UniformGrid<Rational> gx(0_r, 1_r/3_r, 4);
UniformGrid<Rational> gy(0_r, 1_r/2_r, 3);
ProductGrid<UniformGrid<Rational>, 2> product_grid({gx, gy});
```

### Metrics and Betweenness

These are callable objects that define the geometry of your space:

```cpp
#include "delta/core/regulative_idea.h"

LessBetweenness betweenness;         // for totally ordered addresses
EuclideanMetric euclidean;           // distance |a-b|
PAdicMetric<2> p2_metric;            // 2‑adic distance
StringUltrametric tree_metric;       // for binary tree addresses
EuclideanValueMetric value_metric;   // for measuring differences of function values
```

Combine them into a regulative idea if you need to pass them around as a group:

```cpp
using Idea = RegulativeIdea<Rational, LessBetweenness, EuclideanMetric>;
Idea classical(betweenness, euclidean);
```

---

## 2. Delta Paths and Refinement

A **delta path** generates a sequence of refined grids. You supply a strategy (which operator to use at each level), and the path manages the rest.

### Uniform (Dyadic) Refinement

```cpp
#include "delta/core/delta_path.h"
#include "delta/core/delta_strategy.h"
#include "delta/core/delta_operator.h"

// Midpoint operator always inserts the arithmetic mean
MidpointOperator mid_op;
auto strategy = StaticStrategy<MidpointOperator>(mid_op);

// Create a path on [0,1]
ListGrid<Rational> grid0({0_r, 1_r});
DeltaPath path(grid0, strategy, LessBetweenness{}, EuclideanMetric{}, EuclideanValueMetric{});

// Advance a few steps with a function
auto func = [](const Rational& x) { return x * x; };
path.advance(func);
const auto& current_grid = path.current_grid(); // now contains {0, 0.5, 1}
```

### Adaptive Refinement

```cpp
#include "delta/core/adaptive_delta_path.h"

// Start with adaptive path after 3 uniform levels
auto adaptive_path = AdaptiveDeltaPath<...>::from_uniform(
    {0_r, 1_r}, func, mid_op, 3,          // 3 uniform levels
    Rational(1, 1000),                    // threshold
    LessBetweenness{}, EuclideanMetric{}, EuclideanValueMetric{}
);

// Refine until no interval exceeds the threshold
while (adaptive_path.advance()) {}
const auto& points = adaptive_path.points(); // highly non‑uniform set
```

### Riemann Sums

```cpp
#include "delta/calculus/riemann_sum.h"

// After building a path, compute the integral
Rational integral = left_riemann_sum(path.current_grid(), func);
```

---

## 3. Geometry: Simplicial Complexes and DEC

### Building a Mesh

```cpp
#include "delta/geometry/simplicial_complex.h"

SimplicialComplex<2, Rational> mesh;
auto v0 = mesh.add_vertex({0_r, 0_r});
auto v1 = mesh.add_vertex({1_r, 0_r});
auto v2 = mesh.add_vertex({0_r, 1_r});

mesh.add_edge(v0, v1);
mesh.add_edge(v1, v2);
mesh.add_edge(v2, v0);
mesh.add_triangle(v0, v1, v2);
```

### Dual Complex and Discrete Forms

```cpp
#include "delta/geometry/dual_complex.h"
#include "delta/geometry/discrete_forms.h"

EuclideanMetric metric;
DualComplex<decltype(mesh), EuclideanMetric> dual(mesh, metric);

// Create a 0‑form with value 1 everywhere
DiscreteForm<0, Rational, decltype(mesh)> f(mesh);
for(std::size_t v = 0; v < mesh.num_vertices(); ++v)
    f[v] = 1_r;

// Compute Δf = δ d f
auto lap_f = f.d().codifferential(dual, metric);
// lap_f is a 0‑form; for constant function it should be zero everywhere
```

### Hat Basis (interpolation)

```cpp
#include "delta/geometry/hat_basis.h"

HatBasis<decltype(mesh)> basis(mesh);
auto val = basis.interpolate(Point2D(0.2_r, 0.3_r), vertex_values);
```

---

## 4. Numerical Operators on Product Grids

```cpp
#include "delta/numerical/discrete_operators.h"
#include "delta/numerical/integrals.h"

// 2D product grid as before
auto grad = discrete_gradient(grid, scalar_field, max_metric);
auto div  = discrete_divergence(grid, vector_field, max_metric);
auto lap  = discrete_laplacian(grid, scalar_field, max_metric);
auto curl = discrete_curl_2d(grid, vector_field, max_metric);

// Check Green's identity
bool ok = check_green_first_2d(grid, f, g, metric, tolerance);
```

---

## 5. Tensor and Matrix Fields

```cpp
#include "delta/geometry/tensor_field.h"
#include "delta/geometry/matrix_field.h"

using Field2D = TensorField<Point2D, Rational, 0, 2, PointLess<2>>;
Field2D scalar_field(grid2d);          // rank 0
TensorField<Point2D, Rational, 1, 2, ...> vector_field(grid2d); // rank 1
MatrixField<Point2D, 2> matrix_field(grid2d);

// Example: set matrix values and compute exponential
matrix_field.set(p0, some_matrix);
auto exp_field = matrix_field.exp(eps);
```

---

## 6. LazyRational and Performance

The most efficient way to evaluate a large expression is:

```cpp
LazyRational acc;
for (int i = 0; i < N; ++i) {
    acc + term;   // accumulates lazily
}
acc.eval_inplace(true);       // destroy tree, skip simplification, compute result
Rational result = acc.eval(); // O(1) retrieval from the resulting CONST node
```

- Use `skip_simplify = true` unless you expect algebraic cancellations (e.g., `Sin(x)-Sin(x)`).
- Pool and GC are automatic; you normally don’t need to tune them. For extreme usage, see `set_pool_max_size` and `reset_pool`.

See the full coding guidelines in `docs/coding_guidelines.md` for details.

---

## 7. Where to Go Next

- **Test files** in `tests/` – each file demonstrates a complete workflow. Start with:
  - `tests/calculus/test_riemann_sum.cpp` for integration patterns.
  - `tests/geometry/discrete_forms_test.cpp` for DEC.
  - `tests/numerical/discrete_operators_test.cpp` for finite differences.
- **Test fixtures** (`tests/test_fixtures.h`, `tests/test_fixtures_geometry_numerical.h`) show how to set up common configurations.
- **Doxygen** documentation for detailed API reference.

Remember: the tests are your best friend. They are guaranteed to compile and pass; copy, adapt, and extend them.