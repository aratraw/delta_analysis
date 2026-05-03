# Benchmarking Methodology and Result Interpretation in the Exact Rational Computation Library

**Key thesis:** Micro-optimizing an isolated function can lead to catastrophic consequences for the entire library if one fails to account for the impact on the size of rational representations and subsequent operations. Benchmarks that measure only execution time while ignoring the "thickness" of the result are **fundamentally unrepresentative** for assessing real performance in complex computational pipelines.

## 1. What is "local optimization" and why is it dangerous?

Local optimization is a change to a function's implementation that speeds up its execution in an isolated call (e.g., in a microbenchmark where the result is not passed anywhere). An example is replacing the initial approximation in `series_sqrt` from `std::sqrt(double)` to `x/2`. Such a `sqrt` starts running 1.5–2 times faster in a microbenchmark, but produces monstrously bloated rational numbers (thousands of bits). When this result is then used as an argument for `pi`, `sin`, `cos`, or `log`, the library either hangs or runs orders of magnitude slower.

**Conclusion:** Local optimization must not degrade the library's global properties – compactness of representation, test stability, and predictability of runtime in real-world scenarios.

## 2. Why are benchmarks themselves blind?

The standard approach: call a function many times with different arguments, measure time, average. This approach **does not see**:
- The **size (bit-length) of the returned rational number**. Yet this is precisely what determines the cost of all subsequent operations on that number.
- The **impact on caches**, on garbage collection (if there is a global node pool).
- The **instability** that only manifests when the result is passed to other functions.

Therefore, the **only reliable way to evaluate a change** is to run **all** correctness tests (especially those where the result is passed further) and **comparative benchmarks on operation chains**, not on an isolated call.

## 3. Criteria for making optimization decisions

Before changing the implementation of any transcendental function, ask yourself:

0. **Am I ready to toss two days of my life out the window** chasing Heisenbugs that will pop up in the most unexpected places?
1. **Am I losing accuracy guarantees?** (e.g., epsilon scaling in `exp`).
2. **Am I increasing the bit-size of a typical result?** (check on `sqrt(2)` with `eps=1e-80` – numerator/denominator should be within a few hundred bits, not thousands).
3. **Do all correctness tests pass?** especially those involving `sin(pi)`, `pi` at high precisions, `log(exp(x))`, `pow` with rational exponents.
4. **Does performance on real tasks improve (or at least not worsen)?** e.g., computing the `sin(pi)` series with high precision, constructing and simplifying large expressions.
5. **Am I creating a "time bomb"** – a situation where the function is fast for some inputs but catastrophically slow for others (only slightly different)?

## 4. Recommendations for conducting benchmarks in this library

- **Never trust an isolated microbenchmark alone.** Always run `TranscendentalCorrectnessTest` (especially `PiSinConsistency`, `SeriesPathHighPrecision`, `PiPrecisionBenchmark`).
- **Measure not only time but also result sizes.** Add debug output for `num.bit_length()`, `den.bit_length()` for key values (e.g., `sqrt(2)`, `pi(1e-80)`). Ensure that after optimization, the bit-size has not grown more than 2x.
- **Compare not against a "naive" implementation, but against the previous version of the library.** A naive implementation may be completely unsuitable for working in chains.
- **Check not only mean/median time but also variance.** Bloated numbers can cause random outliers (hangs).
- **Use real computational pipelines:** e.g., constructing `sin(pi)` at different precisions, computing `exp(log(2))`, etc.

## 5. Practical examples (lessons learned in this library)

| Optimization | Microbenchmark result | Real effect | Conclusion |
|--------------|------------------------|----------------|-------------|
| `series_sqrt` with `guess = x/2` | `sqrt` became 1.5–2x faster | `PiSinConsistency` hung, `SeriesPathHighPrecision` slowed down 5x | Bloated representation killed subsequent operations |
| `series_sqrt` with `guess = std::sqrt(double)` | slower than `x/2`, but compact | all tests pass, acceptable speed | Safe compromise |
| `series_sqrt` with `guess = isqrt(num*den)/den` | comparable to `x/2` in speed | all tests pass, result compact | Ideal solution |
| `series_exp` with reduction threshold 1.0 | `exp` sped up by a few percent | `log(exp(x))` became catastrophically slow due to intermediate fraction bloat | Reduction threshold must be 2.0 |
| Removing `try_exact_nth_root` from `eager_sqrt` | `sqrt` sped up by 20–30% | lost exact roots for perfect squares (but tests passed) | architecturally wrong, later reverted |

## 6. Conclusion

**A benchmark is just one tool, not the ultimate truth.** In an exact rational computation library, **compactness of representation and stability across the entire argument range are more important than microseconds on an isolated function.** Before celebrating a speedup, make sure you haven't undermined the foundation on which the rest of the library rests.

> "If you optimize a function that returns a monster fraction, you haven't optimized anything – you've just moved the monster elsewhere."

---
Here is the English translation of the revised conclusion:

---

## 6. Conclusion

Thus, **a performance degradation of even two-fold in a local function is not in itself a defect**, if behind that degradation lie additional guarantees:
- absolute accuracy (error ≤ eps),
- compactness of the rational representation (bit stability),
- predictable behavior in all subsequent operations.

It is wrong to interpret benchmarks solely by dry numbers without understanding the nuances of the library's architecture and the interconnections between functions. An isolated microbenchmark does not see how the result affects subsequent computations – and that is precisely where catastrophic slowdowns often hide.

Moreover, attempting to "fix" the library through local optimization (e.g., replacing the initial approximation in series_sqrt with `x/2` or lowering the exponential reduction threshold to 1.0) **can break the library in the most unexpected places**: `sin(π)` starts hanging, `exp(log(x))` runs tens of times slower, and correctness tests fail by timeout.

**This is not a problem of the library – this is a problem of mathematics.**  
Rational numbers behave in complex ways: their representation (numerator/denominator) can explode if the initial conditions of iterative methods are not managed. Any optimization in such a library must be evaluated holistically, not solely by the execution time of an isolated function.

**Remember:**  
- Local speed ≠ global efficiency.  
- Compactness of representation is often more important than a few microseconds.  
- Benchmarks that ignore the subsequent use of the result are blind.  
- Correctness tests are your primary problem detector.  
- If you are not prepared to run all tests after an optimization – do not optimize.

> Quote from the internal motto: *"You are not optimizing a function in a vacuum; you are optimizing the entire library, where every result will sooner or later become someone else's argument."*

---