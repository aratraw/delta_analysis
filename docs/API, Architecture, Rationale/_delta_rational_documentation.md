# Δ‑Analysis Rational Arithmetic Subsystem – Complete Documentation
(c) 2026 Timofey Ishimtsev.
Licensed under PolyForm Small Business License 1.0.0

---

# WHY Rational, not double, is the PRIMARY SCALAR TYPE FOR Δ‑ANALYSIS

In the Δ‑analysis framework, the continuum is not a pre‑existing set of points. It emerges as the *limit* of an infinite refinement process: grids are refined level by level, and the final continuum objects (ℝⁿ, smooth functions, etc.) are invariants of that process. This design imposes **fundamental requirements** on the underlying scalar type that floating‑point doubles simply cannot satisfy.

## 1. UNBOUNDED REFINEMENT – THE KILLER ARGUMENT AGAINST DOUBLE

A typical Δ‑path uses dyadic or barycentric refinement: from a coarse grid, each refinement level halves the edge lengths. After m levels, the smallest representable coordinate difference is 2^{-m} (or a similar geometric factor).

A double has only 53 bits of mantissa. When m exceeds 53, 2^{-m} becomes smaller than 1e-16 – **the next refinement step adds points that are indistinguishable from existing points** when stored as double. The refinement effectively stops. The continuum limit is never approached beyond ~50 levels.

But Δ‑analysis *requires* the ability to refine without a built‑in bound. The continuum, by definition, is the idealised limit of an infinite process. If the implementation forces a hard stop after 50 refinements, the “continuum” is merely an illusion created by the finite precision of the arithmetic.

Rational numbers have no such limitation. A number k/2^m is stored exactly as a pair of integers (k, 2^m). No matter how large m becomes, the value remains exact. Therefore the refinement can continue arbitrarily far, and the limit behaviour can be studied correctly.

Moreover, Boost.Multiprecision (the backend of our Rational) recognises powers of two and uses **bit shifts** internally for multiplication and division by 2^k. This means that operations like a / 2^m are extremely efficient – often as fast as working with integers. There is no gradual loss of performance as the denominator grows, as long as it stays a power of two.

## 2. EXACT INVARIANTS ARE THE BACKBONE OF Δ‑ANALYSIS

The framework heavily relies on *exact* algebraic identities on every finite grid:

- d(d(ω)) = 0 (nilpotence of exterior derivative)
- ∫ (f Δg - g Δf) dV = ∫_∂ (f ∇g - g ∇f)·n dS (Green’s second identity)
- summation by parts
- curl grad f = 0, div curl v = 0
- consistency under subdivision: e.g. for a 1‑form, ω(e) = ω(e1) + ω(e2) when edge e is split into e1 and e2.

With double arithmetic, none of these identities hold even approximately. Rounding errors accumulate and break the exact cancellations that are built into the discrete operators. Consequently, you can never be sure whether a failed test indicates a real bug in the algorithm or just floating‑point noise.

Rational arithmetic guarantees that the identities hold *exactly* on each finite grid (modulo possible overflow, which is avoided by using arbitrary‑precision integers). This makes debugging and verification possible, and ensures that the entire mathematical machinery of Δ‑analysis is realised faithfully at every finite stage.

## 3. PREDICTABLE COMPARISONS AND TESTING – THE “SPEAKING ERROR” EFFECT

With double, the simple test `EXPECT_EQ(a, b)` is meaningless; you must replace it by fuzzy comparisons with an arbitrarily chosen epsilon. The choice of epsilon is never rigorous and often masks real errors.

With Rational, `a == b` is a well‑defined, deterministic predicate. This enables:

- TDD with strict equality checks.
- Automatic verification of the discrete Green’s identities.
- Detecting unintended modifications of fields during refinement.

But the real power becomes visible when a test *fails*. Suppose you expect a result approximately 1/6, but the code produces a huge irreducible fraction like 1/2. Immediately you know: the discrepancy is not rounding noise, it is structural. The exact value 1/2 tells you that your expectation probably omitted a factor 2 somewhere, or that a contribution is counted twice. If the unexpected result equals the sum of two simple fractions (e.g. 1/4 + 1/8 = 3/8), you can directly look for the code segment that introduces those specific rational numbers (1/4 and 1/8). The error itself points you to the bug.

This “speaking error” property is absent in floating‑point: 0.1666667 vs 0.5 could be anything – rounding, cancellations, or a real mistake. You cannot reverse‑engineer the cause from the numbers.

## 4. DECOUPLING DIFFERENT SOURCES OF UNCERTAINTY IN REAL‑WORLD MODELS

In any realistic application, multiple error sources coexist:

- Modelling error (e.g. simplified physics)
- Measurement noise (input data)
- Discretisation error (grid, time step)
- Iterative solver tolerance
- Rounding errors (if using double)

With double, all these are tangled together. You cannot tell whether a discrepancy of 1e‑8 comes from the grid being too coarse, from a large condition number, or from accumulated rounding.

With Rational (and exact algebraic operations), the only remaining numerical approximations are:

- The **truncation error** of transcendental series (sqrt, exp, log, trig), which is controlled by a user‑supplied epsilon.
- The limitations of the discrete model itself (grid refinement level).

All other sources of “noise” are eliminated. Therefore, when you compare simulation results with reference data, any mismatch can be traced back to *either* the model inadequacy *or* insufficient refinement – never to arithmetic flakiness. This clean separation is invaluable for calibration, validation, and uncertainty quantification.

## 5. SIMPLE INTERACTION WITH THE CONSTRUCTIVE CORE 𝒦

Δ‑analysis explicitly restricts addresses to points whose coordinates are *actualisable* (e.g. terminating decimals, dyadic rationals, or more generally the universal constructive core 𝒦* = ℚ\{0}). Rational numbers are the natural representation for such points: they can be stored exactly, reduced to lowest terms, and tested for membership in the chosen core.

Doubles cannot represent even simple fractions like 1/3 exactly, and they cannot distinguish between a genuine zero coordinate (which is excluded from 𝒦) from a non‑zero coordinate that became zero due to rounding. This breaks the fundamental ontology of Δ‑analysis.

## 6. PERFORMANCE COMPROMISE – BUT FOR THE RIGHT REASONS

Double is undeniably faster. However, in Δ‑analysis speed is a secondary concern during development and verification. Once the algorithms are debugged and the invariants are proven on rationals, one can optionally introduce a template parameter `typename Scalar` and instantiate the same code with `double` for large‑scale production runs. This is a **compile‑time decision**, not a philosophical contradiction.

Therefore, the **primary scalar type** of the library is Rational, because the library’s raison d’être – the rigorous construction of continuum limits from discrete processes – cannot be realised with double. Floating‑point support is a possible optimisation, not the foundation.

## 7. BOTTOM LINE

Double kills the very idea of unbounded refinement, destroys the exact algebraic invariants, and forces fuzzy comparisons that make verification unreliable. Its error contamination prevents clean separation of modelling, discretisation, and arithmetic uncertainties. Δ‑analysis without Rational is not Δ‑analysis – it is just another finite‑difference library with a fancy name.

Hence, **Rational is the targeted, natural, and only defensible scalar type** for the core of the Δ‑analysis library.

---

# WHY RATIONAL, NOT double – THE FIELD CLOSURE PRINCIPLE

A **field** is a set equipped with addition, subtraction, multiplication, and division (by non‑zero elements) such that all results stay within the set. The rational numbers ℚ form a field: if a = p/q and b = r/s (with integers p,q,r,s, q,s ≠ 0), then

- a ± b = (ps ± rq)/(qs)
- a * b = (pr)/(qs)
- a / b = (ps)/(qr)  (b ≠ 0)

are again rational numbers. Our Rational class stores numerator and denominator exactly, using arbitrary‑precision integers. Therefore every arithmetic operation on Rational yields another Rational – **exactly and without approximation**.

Double (binary floating‑point) does **not** have this property. Its representable numbers are a discrete subset of ℚ (numbers of the form m·2^e with a bounded mantissa). For example, 0.1 is not exactly representable; neither are 0.2, 0.3, etc. The sum of two representable numbers often falls outside the set. Hence **double is not even a ring**, let alone a field.

## CONSEQUENCE: WITHOUT FIELD CLOSURE, THERE IS NO GEOMETRY

Geometry deals with points, vectors, coordinates, and transformations:

- Points have coordinates.
- Vectors are added, subtracted, scaled.
- Coordinates are added to vectors to obtain new points.
- Lengths and inner products involve squaring and summing coordinates.

All these operations require closure of the underlying numeric type:

- If you add two coordinates, you must get a coordinate.
- If you multiply a coordinate by a scalar, you must get a coordinate.
- If you compute a squared distance (x₁−x₂)² + (y₁−y₂)², the result must be a valid element of the field.

With double, this fails already at the first step: the sum of two coordinates that are exactly representable may not be representable, forcing rounding. The rounding errors accumulate, break exact algebraic identities (e.g. the parallelogram law, the Pythagorean theorem), and ultimately destroy any hope of a consistent geometric model. You cannot speak of a “vector space” over a set that is not closed under addition and scalar multiplication. You cannot define a metric that respects the field structure. In short, **double does not support geometry** – it only supports approximate, error‑prone simulations that happen to be “close enough” for some engineering purposes.

Δ‑analysis demands a rigorous geometric foundation: points, vectors, and coordinates must belong to a field (or at least a ring) that is closed under all necessary operations. Rational provides exactly that. Double does not, and no amount of rounding or epsilon tuning can fix this fundamental deficiency.

Therefore, **Rational is the only logical choice** for the scalar type in a library that aims to implement a genuine geometric system.

---

# OBJECTION: “transcendental functions break field closure – you cannot have exact √2”

The criticism: “Rational is a field, but you introduce sqrt, sin, exp with tolerance ε. This loses exactness – you cannot compute √2 exactly. Geometry without exact diagonals is meaningless. So you are no better than double.”

This objection presupposes the existence of √2 as a *completed object* that must be represented. In Δ‑analysis we reject that presupposition.

## THERE IS NO “TRUE VALUE” OF AN IRRATIONAL NUMBER

An irrational number (√2, π, e, …) does **not** exist as an independent object in the constructive universe. What exists are:

- A rule that defines a Cauchy sequence of rational numbers.
- At each finite stage, a concrete rational number that approximates according to that rule.
- The limit (the “true” irrational) is a regulative idea, not a constructible point.

Therefore, when we compute `sqrt(2, eps)`, we are **not** approximating some pre‑existing √2 that lives outside the rationals. Instead, we are executing the rule “produce a rational number r such that r^2 is within eps of 2”. The result r is the only meaningful object; there is no hidden “true” value behind it.

## CONSEQUENCE: THE FIELD ℚ IS CONSTRUCTIVELY CLOSED

For any rational inputs, the operations +, −, ×, / produce rational outputs exactly. For transcendental operations, the output is **by definition** a rational number (computed by series, binary splitting, etc.) that is guaranteed to satisfy the requested tolerance. There is no claim that the output equals an abstract irrational object; there is only the rational output itself.

Thus, the field ℚ is *constructively closed* under all operations we define: the result always lands in ℚ. The concept of “error relative to a true value” is a convenient way to reason about the coherence of sequences, but it does not introduce any non‑rational entity into the computational substrate.

## WHY THIS IS FUNDAMENTALLY DIFFERENT FROM double

Double pretends to represent √2 as a fixed binary fraction (≈1.4142135623730951) and implicitly assumes that this is an “approximation” to a pre‑existing real number. The error is hard‑coded, cannot be refined without changing the data type, and the operations (+,-,*,/) on double are not even exact for rationals.

With Rational, `sqrt(2, 1e-6)`, `sqrt(2, 1e-12)`, `sqrt(2, 1e-30)` produce different rational numbers. The sequence is under our control, and the limit (the regulative idea) is never mistaken for an actual object. The arithmetic operations stay exact, and the transcendental operations produce rational results that belong to the same field ℚ. No foreign “real” numbers ever enter the system.

## GEOMETRY WITHOUT “TRUE” IRRATIONAL LENGTHS

The objection that “geometry without exact diagonals is meaningless” implicitly assumes that a perfect square with side length 1 exists in reality and that its diagonal must have length √2 as an element of a pre‑existing continuum. Neither holds in a constructive framework.

- Any physical square is made of finitely many elementary units (atoms, cells). Its side is a rational number given by a measurement with finite precision.
- The diagonal is a rational length (by the Pythagorean theorem applied to rational sides) whose square may not be exactly 2; but we can refine the measurement (or the conceptual construction) to make it as close to 2 as desired.
- The notion of an “ideal square” with exactly rational sides and exactly irrational diagonal is a mathematical fantasy – useful for reasoning, but not a constructible reality.

Δ‑analysis embraces this: geometry is the study of rational approximations and their limits. The field ℚ, together with parametrically accurate transcendental functions, provides all the necessary constructive power.

## CONCLUSION: CONSTRUCTIVE CLOSURE IS THE ONLY RELEVANT CLOSURE

The demand that a field be closed under “taking √2” is the demand for algebraic closure – which ℚ does not have, and which is irrelevant for computational geometry. What matters is that every operation defined on rationals yields a rational result. That holds for +, -, ×, / exactly, and for transcendental functions with a tolerance parameter. The tolerance parameter does not introduce irrational objects; it only quantifies the refinement level of the constructive process.

Therefore, the alleged contradiction disappears. Rational is not an approximation of an ideal continuum; it is the genuine constructive field. Double, on the other hand, fails even at exact addition of simple rationals and embeds a false belief in the existence of “true” real constants. The choice of Rational as the primary scalar type is not a compromise – it is the only coherent choice for a library that takes constructivity seriously.

---

# COMPREHENSIVE TECHNICAL REFERENCE – delta::rational

This header (`delta/core/rational.h`) unifies the entire rational computing sub‑library. Treating it as a black box, the rest of the project can use arbitrary‑precision rational arithmetic, lazy expression trees, and transcendental functions without delving into internal details. However, to use the sub‑library efficiently and correctly you must understand its dual eager/lazy architecture, the design rationale behind move‑only LazyRational, and the performance implications of different usage patterns.

## 1. HIGH‑LEVEL ARCHITECTURE

The library consists of two main public classes:

- **Rational** – a strictly *eager*, arbitrary‑precision rational number (backed by Boost.Multiprecision cpp_int **with expression templates disabled – `et_off`**).
- **LazyRational** – a *lazy*, move‑only expression graph that accumulates operations (arithmetic + transcendental) and is evaluated once, potentially after high‑level algebraic simplification.

Eager functions (`sqrt`, `exp`, `log`, …) return Rational and compute immediately using series expansions (or a fast float‑fallback for modest precision). Lazy versions (`Sqrt`, `Exp`, …) build a graph node and defer computation to evaluation time, enabling symbolic simplifications.

All heavy internal machinery (node pool, caches, series implementations) is hidden in the `delta::internal` namespace, so regular users only interact with `Rational`, `LazyRational`, the free functions and the literal suffix.

---

## 2. EXTERNAL API – COMPLETE REFERENCE

This section lists **every public function, class, and literal** that a user should know. Internal details (node pool, GC, evaluation core, etc.) are omitted.

### 2.1 `delta::Rational` – eager exact rational

#### Constructors

```cpp
Rational() noexcept;                         // 0
Rational(int num);
Rational(long long num);
Rational(unsigned long long num);
Rational(const boost::multiprecision::cpp_int& num);
Rational(const boost::multiprecision::cpp_int& num, const boost::multiprecision::cpp_int& den);
Rational(const std::string& s);              // "123", "123/456", "0.5", "1.23e-4"
Rational(double v);                          // **ONLY for Eigen compatibility** – see warning
// Internal (rarely needed):
explicit Rational(const internal::Value& val);
```

> ⚠️ **`Rational(double)` warning**  
> This constructor exists **only** because some Eigen solvers require a scalar type to be constructible from `double` (they initialise temporaries with `0.0`). It uses a **continued‑fraction algorithm with early stopping** (error ~1e-5) and **does not** guarantee exact representation of arbitrary decimals. For exact input always use string literals: `"0.1"_r`, `"1/3"_r`. Never use `Rational(0.1)` in application code.

#### Literals (`delta::literals`)

```cpp
42_r                // Rational(42)
"3.14"_r            // Rational(157/50)
"22/7"_r            // Rational(22,7)
```

#### Accessors

```cpp
const internal::Value& value() const noexcept;   // raw backend (read‑only)
Rational numerator() const;                      // returns a Rational with denominator 1
Rational denominator() const;                    // always positive
internal::dumb_int numerator_raw() const;
internal::dumb_int denominator_raw() const;
double to_double() const;                        // approximate
std::string to_string() const;                   // exact "num/den"
LazyRational as_lazy() const;                    // wrap constant into lazy tree
internal::Interval approx_interval() const;      // double‑based interval [x,x]
```

#### Arithmetic (eager, return new `Rational`)

```cpp
Rational operator+(const Rational& a, const Rational& b);
Rational operator-(const Rational& a, const Rational& b);
Rational operator*(const Rational& a, const Rational& b);
Rational operator/(const Rational& a, const Rational& b);
Rational operator-(const Rational& a);            // unary minus
```

#### Compound assignment (modify left operand)

```cpp
Rational& operator+=(Rational& a, const Rational& b);
Rational& operator-=(Rational& a, const Rational& b);
Rational& operator*=(Rational& a, const Rational& b);
Rational& operator/=(Rational& a, const Rational& b);
```

#### Comparisons (exact)

```cpp
bool operator==(const Rational& a, const Rational& b);
bool operator!=(const Rational& a, const Rational& b);
bool operator<(const Rational& a, const Rational& b);
bool operator<=(const Rational& a, const Rational& b);
bool operator>(const Rational& a, const Rational& b);
bool operator>=(const Rational& a, const Rational& b);
```

Cross‑type comparisons with `LazyRational` are also provided (see Section 2.3).

#### Batch addition (optimised)

```cpp
Rational batch_add(const std::vector<Rational>& terms);
```

Uses a common denominator to minimise intermediate swell. Much faster than summing in a loop.

#### Absolute value

```cpp
Rational abs(const Rational& x);
```

#### Conversion to numeric types

```cpp
template<typename T> T convert_to() const;
// Supported: T = double, int, long long, internal::dumb_int
// Throws if not integer or out of range.
```

#### Additional utility functions

```cpp
Rational floor(const Rational& x);   // greatest integer ≤ x
```

---

### 2.2 `delta::GaussQi` – exact complex rational (Gaussian rationals)

#### Constructors

```cpp
GaussQi() = default;                             // 0+0i
explicit GaussQi(const Rational& re);            // re + 0i
GaussQi(const Rational& re, const Rational& im);
GaussQi(Rational&& re, Rational&& im);
explicit GaussQi(long long re);                  // explicit – no implicit int conversion
GaussQi(long long re, long long im);
explicit GaussQi(const std::string& str);        // "(re,im)" or simplified forms (see below)
GaussQi(const std::string& re_str, const std::string& im_str);
```

#### Literals (`delta::literals`)

```cpp
"1+2i"_qi          → GaussQi(1,2)
"1/2-3/4i"_qi      → GaussQi(1/2, -3/4)
"0.333i"_qi        → GaussQi(0, 333/1000)
"i"_qi             → GaussQi(0,1)
"-i"_qi            → GaussQi(0,-1)
"2"_qi             → GaussQi(2,0)
"(1,2)"_qi         → alternative comma form
```

#### Accessors

```cpp
const Rational& real() const noexcept;
const Rational& imag() const noexcept;
void real(const Rational& r);
void imag(const Rational& i);
std::string to_string() const;
std::pair<double,double> to_double() const;
```

#### Basic operations

```cpp
Rational norm() const;          // re² + im²
GaussQi conj() const;           // conjugate
```

#### Arithmetic (binary, unary, with `GaussQi`, `Rational`, `int`)

```cpp
GaussQi operator+(const GaussQi&, const GaussQi&);
GaussQi operator-(const GaussQi&, const GaussQi&);
GaussQi operator*(const GaussQi&, const GaussQi&);
GaussQi operator/(const GaussQi&, const GaussQi&);
GaussQi operator-(const GaussQi&);   // unary minus

// Mixed with Rational (and int via explicit conversion)
GaussQi operator+(const GaussQi&, const Rational&);
GaussQi operator+(const Rational&, const GaussQi&);
// similar for -, *, /

// Compound assignment
GaussQi& operator+=(const GaussQi&);
GaussQi& operator-=(const GaussQi&);
GaussQi& operator*=(const GaussQi&);
GaussQi& operator/=(const GaussQi&);
GaussQi& operator+=(const Rational&);
GaussQi& operator-=(const Rational&);
GaussQi& operator*=(const Rational&);
GaussQi& operator/=(const Rational&);
```

#### Comparisons

```cpp
bool operator==(const GaussQi&, const GaussQi&);
bool operator!=(const GaussQi&, const GaussQi&);
```

---

### 2.3 `delta::LazyRational` – mutable, move‑only lazy expression tree

**Move‑only** (copy constructor and copy assignment are **deleted**). Use `.clone()` for explicit deep copies.

#### Construction

```cpp
LazyRational();                         // dirty CONST(0)
explicit LazyRational(const Rational&);
explicit LazyRational(Rational&&);
```

#### State inspection

```cpp
bool is_dirty() const;   // true: tree is local, can be mutated
bool is_clean() const;   // true: node lives in global pool, shared, immutable
```

#### Cloning

```cpp
LazyRational clone() const;   // deep copy (clean: just increment refcount)
```

#### Mutating operators (modify **left operand**, return reference to it)

```cpp
LazyRational& operator+(LazyRational& a, const LazyRational& b);
LazyRational& operator+(LazyRational& a, const Rational& b);
// Similarly for -, *, / (including mixed with Rational on left)
// Example: a + b   → a is mutated, becomes SUM(a,b)
```

**Important**: The expression `a + b` mutates `a`. To keep `a` unchanged, write `a.clone() + b`.

#### Compound assignment (same effect)

```cpp
LazyRational& operator+=(LazyRational&, const LazyRational&);
LazyRational& operator+=(LazyRational&, const Rational&);
// etc.
```

#### Unary minus (returns a **new** `LazyRational`)

```cpp
LazyRational operator-(const LazyRational&);
```

#### Bulk insertion (for performance in loops)

```cpp
void append_values(std::vector<internal::Value>&& values);  // add leaf constants to a SUM
void append_nodes(std::vector<int>&& node_indices);          // add child nodes to a SUM
```

#### Evaluation

```cpp
Rational eval(bool skip_simplify = false) const;
  // If skip_simplify == false: canonicalize (dirty→clean) then evaluate.
  // If skip_simplify == true: evaluate dirty tree directly (faster, no simplification).

void eval_inplace(bool skip_simplify = false);
  // Destroys the tree, replaces *this with a clean CONST node holding the result.
  // After this, is_clean() == true.
```

#### Simplification

```cpp
void simplify_inplace();        // force canonicalization (dirty→clean) without evaluation
LazyRational simplify() const;  // returns a new clean LazyRational (calls clone first)
```

#### Interval approximation (fast double‑based, cached)

```cpp
internal::Interval approx_interval() const;
```

#### Force dirty state (for low‑level manipulation)

```cpp
void ensure_dirty();
```

#### Interactions with global default epsilon

The lazy transcendentals (see Section 2.5) accept an optional `eps` parameter. If omitted, `delta::default_eps()` is used.

---

### 2.4 Context – global default epsilon

```cpp
Rational default_eps();                     // current value (initial 1e-30)
void set_default_eps(const Rational& eps); // change global default (shared across threads!)
void reset_default_eps();                   // restore 1e-30
```

> ✅ **Correction**: The default epsilon is stored as a global `inline Value` (not `thread_local`). Therefore it is **shared across all threads**. Setting it in one thread changes it for all threads – this is by design for reproducible results.

---

### 2.5 Transcendental functions – eager (return `Rational`)

All eager functions take an optional `eps` parameter (default = `delta::default_eps()`). The `eps` is an **absolute** error bound: |true value – computed| ≤ eps.

```cpp
Rational sqrt(const Rational& x, const Rational& eps = default_eps());
Rational exp(const Rational& x, const Rational& eps = default_eps());
Rational log(const Rational& x, const Rational& eps = default_eps());
Rational sin(const Rational& x, const Rational& eps = default_eps());
Rational cos(const Rational& x, const Rational& eps = default_eps());
Rational acos(const Rational& x, const Rational& eps = default_eps());
Rational asin(const Rational& x, const Rational& eps = default_eps());
Rational atan(const Rational& x, const Rational& eps = default_eps());
Rational tan(const Rational& x, const Rational& eps = default_eps());   // uses Lambert continued fraction
Rational pi(const Rational& eps = default_eps());
Rational e(const Rational& eps = default_eps());
Rational pow(const Rational& base, const Rational& exponent, const Rational& eps = default_eps());
Rational pow(const Rational& base, int exponent);   // exact binary exponentiation
```

#### Additional exact‑rational functions (with π‑period)

```cpp
Rational sinpi(const Rational& x, const Rational& eps = default_eps());   // sin(π·x)
Rational cospi(const Rational& x, const Rational& eps = default_eps());   // cos(π·x)
Rational tanpi(const Rational& x, const Rational& eps = default_eps());   // tan(π·x)
Rational asinpi(const Rational& y, const Rational& eps = default_eps());  // asin(y)/π
Rational acospi(const Rational& y, const Rational& eps = default_eps());  // acos(y)/π
Rational atanpi(const Rational& y, const Rational& eps = default_eps());  // atan(y)/π
```

For special arguments (multiples of 1/12, 1/6, 1/4, 1/3, 1/2) these return **exact rational** results (e.g. `sinpi(1/6) = 1/2`). Otherwise they fall back to the general series.

---

### 2.6 Transcendental functions – lazy (return `LazyRational`)

These functions **do not mutate** their arguments (they clone internally). They build a node in the expression graph that will be evaluated later.

```cpp
LazyRational Sqrt(const LazyRational& x, const Rational& eps = default_eps());
LazyRational Sqrt(const Rational& x, const Rational& eps = default_eps());

LazyRational Exp(const LazyRational& x, const Rational& eps = default_eps());
LazyRational Exp(const Rational& x, const Rational& eps = default_eps());

LazyRational Log(const LazyRational& x, const Rational& eps = default_eps());
LazyRational Log(const Rational& x, const Rational& eps = default_eps());

LazyRational Sin(const LazyRational& x, const Rational& eps = default_eps());
LazyRational Sin(const Rational& x, const Rational& eps = default_eps());

LazyRational Cos(const LazyRational& x, const Rational& eps = default_eps());
LazyRational Cos(const Rational& x, const Rational& eps = default_eps());

LazyRational Acos(const LazyRational& x, const Rational& eps = default_eps());
LazyRational Acos(const Rational& x, const Rational& eps = default_eps());

LazyRational Pi(const Rational& eps = default_eps());
LazyRational E(const Rational& eps = default_eps());

// Pow overloads (all possible combinations)
LazyRational Pow(const LazyRational& base, const LazyRational& exponent, const Rational& eps = default_eps());
LazyRational Pow(const Rational& base, const LazyRational& exponent, const Rational& eps = default_eps());
LazyRational Pow(const LazyRational& base, const Rational& exponent, const Rational& eps = default_eps());
LazyRational Pow(const Rational& base, const Rational& exponent, const Rational& eps = default_eps());
LazyRational Pow(const LazyRational& base, int exponent);   // uses integer exponentiation
```

> ℹ️ **Note**: Lazy versions of `Asin`, `Atan`, `Tan` are declared but commented out in the current code (the enum entries exist but the lazy factories are not yet implemented). Eager versions work fine.

---

### 2.7 Transcendental functions for `GaussQi` (eager, return `GaussQi` or `Rational`)

All take an optional `eps` (default = `default_eps()`).

```cpp
Rational abs(const GaussQi& z, const Rational& eps = default_eps());   // sqrt(re²+im²)
Rational arg(const GaussQi& z, const Rational& eps = default_eps());   // atan2(im, re)

GaussQi sqrt(const GaussQi& z, const Rational& eps = default_eps());   // principal branch
GaussQi exp(const GaussQi& z, const Rational& eps = default_eps());
GaussQi log(const GaussQi& z, const Rational& eps = default_eps());    // principal branch
GaussQi sin(const GaussQi& z, const Rational& eps = default_eps());
GaussQi cos(const GaussQi& z, const Rational& eps = default_eps());
GaussQi tan(const GaussQi& z, const Rational& eps = default_eps());
GaussQi asin(const GaussQi& z, const Rational& eps = default_eps());
GaussQi acos(const GaussQi& z, const Rational& eps = default_eps());
GaussQi atan(const GaussQi& z, const Rational& eps = default_eps());

GaussQi pow(const GaussQi& z, const GaussQi& w, const Rational& eps = default_eps());
GaussQi pow(const GaussQi& z, int exponent);   // exact binary exponentiation

// Additional real helper:
Rational atan2(const Rational& y, const Rational& x, const Rational& eps = default_eps());
```

> ⚠️ The lazy subsystem does **not** support `GaussQi` – you cannot build lazy expression trees with complex numbers. Use eager functions or combine two `LazyRational` trees manually.

---

### 2.8 Integration with Eigen3

After including `<delta/rational/eigen_integration.h>` (or the master header), both `delta::Rational` and `delta::GaussQi` become valid scalar types for `Eigen::Matrix`, `Eigen::Array`, and many algorithms.

#### `NumTraits` specialisations

```cpp
template<> struct NumTraits<delta::Rational> {
    // epsilon() and dummy_precision() return delta::default_eps()
    enum { IsInteger = 0, IsSigned = 1, IsComplex = 0, RequireInitialization = 1 };
    static const int ReadCost = 8;
    static const int AddCost = 250;   // rational addition cost
    static const int MulCost = 200;   // rational multiplication cost
};

template<> struct NumTraits<delta::GaussQi> {
    enum { IsInteger = 0, IsSigned = 1, IsComplex = 1, RequireInitialization = 1 };
    static const int ReadCost = 16;
    static const int AddCost = 500;   // (a+c)+(b+d)i
    static const int MulCost = 1200;  // (ac-bd)+(ad+bc)i
};
```

#### Element‑wise transcendentals (ADL, automatic)

Because `delta::sin`, `delta::cos`, etc. live in namespace `delta`, they are found by ADL:

```cpp
Eigen::Matrix<Rational, Dynamic, Dynamic> A;
auto B = A.array().sin();    // element‑wise delta::sin
auto C = A.array().cos();    // element‑wise delta::cos
// similarly .exp(), .log(), .sqrt(), .pow()
```

#### True matrix transcendental functions (namespace `delta`)

These operate on square matrices as a whole (not element‑wise). They accept any Eigen expression, evaluate it to a dense dynamic matrix, and apply the corresponding algorithm (Padé, scaling‑and‑squaring, Newton, etc.).

```cpptemplate<typename Derived>
auto exp(const Eigen::MatrixBase<Derived>& A, const Rational& eps = default_eps());

template<typename Derived>
auto log(const Eigen::MatrixBase<Derived>& A, const Rational& eps = default_eps());

template<typename Derived>
auto sin(const Eigen::MatrixBase<Derived>& A, const Rational& eps = default_eps());

template<typename Derived>
auto cos(const Eigen::MatrixBase<Derived>& A, const Rational& eps = default_eps());

template<typename Derived>
auto sqrt(const Eigen::MatrixBase<Derived>& A, const Rational& eps = default_eps());
```

All functions require square matrices. The scalar type must be `Rational` or `GaussQi`. Diagonal matrices are detected and processed element‑wise for speed.

**Complex (GaussQi) logarithm** uses a special trace‑normalisation algorithm to avoid the catastrophic bit‑length explosion that occurs with scaling by 2 or repeated square roots.

#### Example

```cpp
#include <delta/rational/eigen_integration.h>

Eigen::Matrix<delta::Rational, 2, 2> M;
M << 1_r, 2_r, 3_r, 4_r;
auto expM = delta::exp(M, delta::Rational(1, 1000000000000000000)); // ε=1e-18
std::cout << expM << std::endl;

// Complex rational matrix
Eigen::Matrix<delta::GaussQi, 2, 2> Z;
Z << GaussQi(1,2), GaussQi(0,1), GaussQi(1,0), GaussQi(0,0);
auto logZ = delta::log(Z, delta::default_eps() / 10);
```

---

## 3. UNDER THE HOOD – EVALUATION, SERIES, SIMPLIFICATION, AND GC

*(This section remains largely accurate; only the float‑path description is updated.)*

### 3.1 Multi‑precision float path (bit‑based dispatching)

Instead of a fixed threshold like `1e-35`, the library dynamically selects the precision of the intermediate floating‑point type based on the required number of bits. Three fixed‑width binary float types are used (all with expression templates disabled, `et_off`):

```cpp
using Float256 = cpp_bin_float<256>;   // ~77 decimal digits
using Float512 = cpp_bin_float<512>;   // ~154 digits
using Float1024 = cpp_bin_float<1024>; // ~308 digits
```

For a given call to `sin`, `cos`, `exp`, `atan`, `pi`, etc., the required bit precision is:

```cpp
required_bits = bits_of_abs(x) + precision_bits(eps) + guard_bits
```

where  
- `bits_of_abs(x)` – number of bits in the absolute value of the argument (0 for bounded functions like `acos`),  
- `precision_bits(eps)` – number of bits of the denominator of `eps` (i.e., how many bits of precision are requested),  
- `guard_bits` – a small safety margin (typically 16 for trigonometric functions, 32 for `exp`).

Then `select_float_path(required_bits)` selects the smallest type that can represent the required precision:

- `required_bits ≤ 240` → `Float256`
- `≤ 496` → `Float512`
- `≤ 1008` → `Float1024`
- otherwise → fall back to exact rational series.

The result is converted back to `Rational` using `to_rational_with_eps`, which scales the float by a power of two, rounds to the nearest integer, and forms a fraction. This guarantees that the final rational number is exact (no floating‑point rounding errors are carried over).

### 3.2 Series implementations (rational)

All series (sqrt, exp, log, sin, cos, atan, asin, pi, e) use binary splitting or fast converging methods. In particular:

- **`tan` for `Rational`** uses the **Lambert continued fraction** (`tan_lambert`), which converges faster and avoids the need to compute both `sin` and `cos` separately. For arguments very close to π/2 + kπ, it switches to cotangent of the small difference.
- **`exp`** uses argument reduction by repeated division by 2 and squaring, with internal epsilon scaled accordingly.
- **`log`** reduces to `[1/2,2]` and uses the `artanh` series.
- **`pi`** uses the Chudnovsky series with binary splitting, cached per epsilon.

### 3.3 Canonicalization and simplification

When `LazyRational::canonicalize()` is called (automatically before evaluation unless `skip_simplify` is true), the dirty tree is transformed into a **clean** node in the global pool. The `simplify_tree` function applies algebraic rewrites:

- Flatten nested `SUM`/`PRODUCT`
- Remove zeros/ones
- Group identical constants (`a+a` → `2*a`)
- Fold identical subtrees (`A+A` → `2*A`)
- Distribution (`a*b + a*c` → `a*(b+c)`)
- Cancel `x + (-x)` → 0, `x * (1/x)` → 1
- Cancel inverse function chains (`NEG(NEG(x))` → `x`, `EXP(LOG(x))` → `x`)
- `POW` simplifications (`x^0` → `1`, `0^positive` → `0`, `(x^a)^b` → `x^(a*b)` for integer exponents)

All simplifications are **symbolic** – they construct new nodes, they do not evaluate numeric values.

### 3.4 Node pool and garbage collection

- The global pool (`NodePool`) is **thread‑local** – each thread has its own independent pool and caches.
- Clean nodes are reference‑counted. When a `LazyRational` is destroyed, it decrements the refcount of its root node.
- When the pool occupancy exceeds `gc_threshold` (0.9 × `max_size`), `collect_garbage()` is triggered automatically (unless GC is disabled).
- GC takes a snapshot of all live clean `LazyRational` objects (via the registry `g_clean_rationals`), evaluates each root subtree to a constant, and replaces the entire tree with a single `CONST` node at the same index. The pool is then rebuilt (or compacted) – all dead nodes are discarded.
- The maximum pool size can be set via `internal::set_pool_max_size(size_t)`. Default is 1,000,000 nodes.

### 3.5 Interval arithmetic for fast comparisons

`LazyRational::approx_interval()` computes a double‑based outward‑rounded interval for the whole expression (without canonicalization). This interval is cached and invalidated on mutations. The comparison operators (`==`, `<`, etc.) first check whether the intervals overlap; if they do not, the result is known immediately. Otherwise they fall back to exact evaluation after canonicalization.

---

## 4. QUICK REFERENCE – COMMON USAGE PATTERNS

### Eager one‑shot computation

```cpp
Rational a = "1.5"_r;
Rational b = 2_r;
Rational c = sqrt(a*a + b*b);   // exact rational approximation
```

### Accumulate in a loop with `LazyRational` (O(N) instead of O(N²))

```cpp
LazyRational acc;
for (int i = 0; i < N; ++i) {
    acc + sin(Rational(i+1));
}
Rational total = acc.eval();
```

### Building a complex expression tree (with cloning)

```cpp
LazyRational x = LazyRational("0.5"_r);
auto expr = Sin(x.clone() * 2_r + 1_r)      // x*2+1 kept lazy
          + Cos(x.eval() * 3_r);            // x evaluated to Rational, then lazy Cos
Rational res = expr.eval();
```

### Changing global epsilon

```cpp
set_default_eps("1/10000000000000000000"_r);   // 1e-19
reset_default_eps();
```

### Using `GaussQi`

```cpp
GaussQi z("1/2"_r, "1/3"_r);
GaussQi w = exp(z, "1e-20"_r);
Rational r = abs(w);
std::cout << "|exp(z)| = " << r << std::endl;
```

### Using Eigen with rational matrices

```cpp
#include <delta/rational/eigen_integration.h>

Eigen::Matrix<delta::Rational, 2, 2> M;
M << 1_r, 2_r, 3_r, 4_r;
auto Minv = M.inverse();                     // exact rational inverse
auto expM = delta::exp(M);                   // matrix exponential (Padé)
```

### Using the exact π‑period functions

```cpp
Rational angle = "1/3"_r;
Rational s = sinpi(angle);   // √3/2
Rational c = cospi(angle);   // 1/2
Rational t = tanpi(angle);   // √3
Rational a = asinpi("0.5"_r); // 1/6
```

---

## 5. KNOWN LIMITATIONS & FUTURE WORK

- Lazy versions of `asin`, `atan`, `tan` are declared but not yet implemented (the node types exist, but the factories are commented out). Use eager versions.
- No lazy complex (`GaussQi`) expressions – you must combine two `LazyRational` trees manually if you need lazy complex arithmetic.
- The node pool never shrinks; repeated GC may leave sparse arrays (though compaction is planned).
- Very large rational numbers (thousands of digits) become slow; there are no asymptotic optimisations beyond what Boost provides.
- The `convert_to<T>` method for `Rational` supports `int`, `long long`, `dumb_int`, `double` – no `float` directly.
- Serialisation of `Rational` or `LazyRational` is not provided.

---

## 6. THREAD SAFETY

- **`Rational`** objects are independent and immutable after construction; they are safe to use concurrently in different threads.
- **`LazyRational`** objects are **not** thread‑safe – a single object must not be mutated or evaluated from multiple threads simultaneously.
- All global state (node pool, π cache, clean object registry, GC flags) is **thread‑local**. Each thread has its own independent pool, its own cache, and its own registry. This eliminates locking overhead.
- The **default epsilon** (`delta::default_eps()`) is **global** (shared across all threads). Setting it in one thread changes it for all threads – this is by design (reproducible results).