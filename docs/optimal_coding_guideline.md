*Back to [README](../README.md) | [Documentation Index](../README.md#-documentation)*

## Performance-Optimized Coding Guidelines for LazyRational

This guide focuses on extracting maximum performance from the library’s lazy evaluation system. It assumes you are already familiar with the basics of `Rational` and `LazyRational` (construction, basic operations).  

The core insight: **lazy evaluation is not about delaying computation for abstraction – it is a deliberate performance strategy.** When used correctly, it can outperform traditional eager arithmetic by 2–6×, even for simple summation of hundreds of thousands of terms. The key is to understand when to accumulate, when to simplify, and when to destroy the tree.

---

### 1. The Golden Rule of Accumulation

**Never re‑initialize a term inside a loop if you can reuse it.**

```cpp
// Correct: declare term outside, assign inside. One construction, one destruction.
LazyRational acc, term;
for (int i = 0; i < 80; ++i) {
    term = Sin(Rational(i)) * Cos(Rational(i + 1));  // Builds dirty tree; no evaluation yet.
    acc + term;                                      // Mutates acc, absorbs term's subtree.
}
```

```cpp
// Wrong: recreating term each iteration causes 80 constructor/destructor calls.
LazyRational acc;
for (int i = 0; i < 80; ++i) {
    LazyRational term = Sin(Rational(i).as_lazy()) * Cos(Rational(i + 1).as_lazy()); 
    acc + term;   // Even though term is moved into acc, its construction & destruction cost is paid.
}
```

`acc + term` mutates `acc` in place – it **does not** produce a new object. The tree of `term` is imported into `acc` efficiently. Because of this, `term` can be reused across iterations without any penalty. On the other hand, constructing and destroying a `LazyRational` is not free: it involves local vector allocations and, if the object becomes clean, interactions with the global node pool and reference counting.

**The library forbids `acc = acc + term` at compile time** (copy assignment is deleted). This is intentional: such an expression would force a deep copy of the entire accumulator on every iteration, leading to O(N²) runtime. Use `acc + term;` or `acc += term;` and let the mutations happen in‑place.

---

### 2. The Fast Path: Accumulate, Evaluate, Discard

The simplest and often **fastest** way to compute a value is to build the expression lazily and then destroy it with `eval_inplace(true)`.

```cpp
LazyRational acc;
for (const auto& value : huge_dataset) {
    acc + value;                      // absorb into sum
}
acc.eval_inplace(true);               // evaluate without simplification, destroy dirty tree
Rational result = acc.eval();         // now acc is a CONST node; eval is a trivially cheap getter
```

`eval_inplace(true)`:
- `true` means `skip_simplify = true` – no algebraic simplification, no canonicalization.
- Evaluates the entire dirty tree in‑place, appending no new nodes, performing no copies. After completion, the entire dirty structure is replaced by a single constant node.
- This single evaluation is often 2‑6× faster than the equivalent eager sum (e.g., 500 000 random fractions or harmonic series terms). The magic lies in the **pyramidal compact reduction** (PCR) used internally for SUM nodes – it batches additions into chunks of 32, avoiding intermediate rational swell.

**Why is “dumb” lazy faster than eager?**  
Eager addition `result = a + b + c + ...` computes every step immediately, causing rationals with growing denominators to be built at each step. Lazy evaluation with PCR sums in a tree, minimising the size of intermediate fractions. Even if half your terms are zeros, skipping the simplifier is still faster – the simplifier’s overhead exceeds the minor benefit of removing a few neutral elements. The simplifier is designed for a different purpose (see §3).

---

### 3. When to Use Simplification

Simplification (canonicalization) is **a default stage that you can and should skip where possible**. It is a powerful tool for scenarios where algebraic structure matters, but its cost must be weighed against its benefit. The following benchmarks illustrate when simplification pays off and when it should be avoided.

#### 3.1 Benchmark: Algebraic Identities (Enormous Win)

Consider a chain of `Exp(Log(…))` pairs that collapses to a constant:

```
Depth | With Canon (ms) | Without Canon (ms) | Result
------|-----------------|--------------------|-------
   1  |               0 |                  0 | simplify 12.80x faster
   2  |               0 |                 19 | simplify 245.30x faster
   3  |               0 |                 21 | simplify 403.31x faster
   4  |               0 |                 31 | simplify 867.14x faster
   5  |               0 |                 40 | simplify 1111.72x faster
   6  |               0 |                 49 | simplify 1303.74x faster
   8  |               0 |                 71 | simplify 1788.42x faster
  10  |               0 |                 90 | simplify 249.51x faster
```

Here, simplification reduces the entire nested expression to the seed value in one pass, avoiding many expensive transcendental evaluations. **This is the ideal scenario for the simplifier.**

#### 3.2 Benchmark: Only and Specifically Neutral Element Removal (Often a Loss)

Now consider a sum where half the terms are zeros:

```
  N    | With Canon (ms) | Without Canon (ms) | Result
-------|-----------------|--------------------|-------
   100 |               0 |                  0 | no_simplify 22.35x faster
   500 |               1 |                  0 | no_simplify 7.57x faster
  1000 |               0 |                  0 | no_simplify 3.92x faster
  5000 |               2 |                  1 | no_simplify 2.16x faster
 20000 |               9 |                  5 | no_simplify 1.75x faster
 50000 |              17 |                 11 | no_simplify 1.56x faster
```

Removing zeros from a sum never pays back the cost of simplification for moderate N—the simplifier’s overhead is larger than the work saved by skipping a few additions. Even at 50 000 terms, skipping simplification is still 1.56× faster. **For unstructured sums, simplification is a net loss.**

#### 3.3 The Simplification Sweet Spot

Enable simplification when:
- **Transcendental cancellations** are possible (`Sin(x) – Sin(x)`, `Exp(Log(x)) – x`). These can eliminate the need to compute expensive series entirely.
- **High‑precision transcendentals** are involved, especially if they might cancel. A `sin(1, eps=1e‑100)` that cancels with another identical term is *free* after simplification—it vanishes before any series is evaluated.
- **Deeply nested algebraic identities** are present. Repetitive structures like `(a*b + a*c) / a` can be collapsed into `b + c`, dramatically reducing tree size and evaluation cost.

Disable simplification (`skip_simplify = true`) when:
- You are simply accumulating numbers with no repeating sub‑expressions. The simplifier will only add overhead.
- The expression is a straight sum with fewer than, say, 100 000 terms and contains no transcendentals. Even if there are zeros, the PCR summation handles them efficiently enough.
- You are going to call `eval_inplace(true)` for a one‑shot computation and discard the tree afterwards.

**The verdict:** The best `sin(1, eps=1e‑100)` is the one that was cancelled out by simplification and never computed. Use simplification when your expression has high *potential* for algebraic collapse—transcendentals, repeated sub‑trees, or distributivity. For everything else, the raw lazy evaluation path is faster.

---

### 4. The `eval_inplace` Semi‑Destructive Pattern

`eval_inplace` is the most lightweight evaluation method:
1. It destroys the dirty tree state.
2. It consumes the vector of nodes and leaf values, avoiding any copying.
3. After the call, the `LazyRational` becomes a clean object containing a single `CONST` node.
4. Subsequent `eval()` calls on that clean constant node are O(1) – they simply retrieve the stored rational.

This is ideal for one‑shot computations (“accumulate, compute, output, forget”).

`eval()` (without `inplace`) preserves the original dirty tree, which requires copying data into a working vector. Only use it when you intend to keep the expression for later modification or evaluation with different settings.

---

### 5. Summary: Choosing the Right Strategy

| Scenario | Recommended Pattern | Rationale |
|----------|-------------------|-----------|
| Straight sum/integral, single use | `acc + ... ; acc.eval_inplace(true);` | Maximum speed, no simplification overhead. |
| Expression with many repeated sub‑terms | `acc + ... ; acc.simplify_inplace();` then `eval()` | Reuse identical sub‑expressions, reduce work. |
| Mixed transcendental expressions with possible cancellations | Build dirty tree, then `expr.eval()` (skip_simplify = false by default) | Let the simplifier detect algebraic identities. |
| Reusing the same expression over multiple iterations | Canonicalize once, then clone and evaluate with `skip_simplify = true` | One‑time cost of simplification, fast repeated evaluation. |
| Debugging or inspecting tree structure | Use `simplify_inplace()` or `canonicalize()` and then examine clean nodes | Clean trees are hash‑consed and have stable indices. |

**Final word of the performance‑conscious developer:**  
*Know your data and your expression.* The library gives you fine‑grained control over evaluation cost. A lazy expression is a weapon – use it wisely. When in doubt, benchmark with realistic workloads.

## 6. Garbage Collection and Pool Management

The global node pool and its garbage collector (GC) are designed to be completely transparent in everyday use. You should not need to think about them. However, understanding the internals helps when you push the library to extremes.

### The Node Pool

When a `LazyRational` is canonicalized (via `simplify_inplace()` or an implicit `eval()` that does not skip simplification), its expression tree is moved into a global, thread‑local **node pool**. The pool is hash‑consed: identical sub‑expressions are stored only once and shared among all clean `LazyRational` objects.

- **Default maximum size:** 1 000 000 nodes.  
- **GC threshold:** 90% of `max_size`.  

You can adjust this limit with:

```cpp
internal::set_pool_max_size(5'000'000);  // for extremely large symbolic computations
```

### The GC *Is* Computation

Here is the most important architectural insight you must internalize: **the garbage collector does not just clean up memory – it performs actual computation.** When the pool triggers GC, the system:

1. Identifies all live clean roots (expressions you are still holding).
2. Evaluates every one of those roots to a concrete `Rational` value.
3. Stores the resulting constants back into the pool, replacing the entire tree structure.
4. The original complex DAG is discarded.

In other words, GC is the moment when deferred evaluation is forced upon the live expressions. It is not an external cleanup process bolted onto the side; it is a **natural, integrated phase of the lazy evaluation lifecycle.** You accumulate symbolic expressions lazily, and when the pool fills up, the system says: “Time to settle accounts – compute everything you are still holding, and let’s start fresh.” This is by design. It means that no computation is ever lost or duplicated; it simply gets performed at a different time than you might expect.

Because of this, GC can be thought of as a **global reduction pass**: it crunches all outstanding lazy work into concrete numbers. The price of this pass is proportional to the total complexity of the live trees. If you hold many deeply nested, un‑evaluated trees, GC will spend a corresponding amount of time evaluating them. This is fair and predictable.

### “Sparse” Pool After GC – No Worries

After garbage collection, the new pool might be “sparse”: if you had roots at indices `0`, `1000`, and `50000`, the pool vector will have size 50001 with only three occupied slots. This is **not a memory leak** and **not a performance problem**.

Why? Because the pool allocates in chunks of 4096 nodes. In a real‑world scenario, after GC you will immediately start building new expressions. New nodes fill the pool contiguously using the first free indices, quickly covering any gaps. The “sparse” state is transient and irrelevant.

### What Happens When You Exceed `max_size`

If you canonicalize more live nodes than `max_size`, the library will not crash or lose data. Instead, it will repeatedly perform the global reduction (GC). The process is fully automatic but can become slow:

- Suppose `max_size` is 1 000 000 and your computation creates 2 000 000 unique canonicalized nodes.
- The pool will fill up, trigger GC (evaluating all live trees into constants), then fill up again, trigger GC again, and so on.
- Each GC cycle is an O(pool) operation that walks the entire pool and re‑evaluates subtrees.

**That is the intended trade‑off:** the library is optimized for typical workloads where the number of *simultaneously* clean objects stays well below one million. If you are performing massive symbolic manipulations (e.g., building and canonicalizing millions of distinct sub‑expressions), consider:

- Increasing `max_size` substantially.
- Avoiding unnecessary canonicalization – use `skip_simplify = true` wherever possible.
- Calling `reset_pool()` between independent phases to release memory.

The library will never leak memory, and it will always maintain correctness. It may just spend more time in GC (i.e., computing) than in your explicit evaluation calls. This is fair warning, not a bug.

---

**The bottom line:** For normal use, forget about the pool. Build your expressions, evaluate them, and let the automatic GC handle the rest. The architecture has been designed so that you can “throw nodes at it” without creating dangling references or memory corruption. The garbage collector is not a janitor – it’s your deferred computation engine.