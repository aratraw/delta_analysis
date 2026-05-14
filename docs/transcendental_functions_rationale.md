# Transcendental Functions in delta::Rational

## 1. Goals and Context

The `delta::Rational` library provides **absolutely precise rational computations** for elementary transcendental functions. The user supplies an acceptable absolute error `eps` (as a `Value`, a rational number), and the function returns a `Value` guaranteed to differ from the true mathematical value by no more than `eps`.

Core requirements:
- No dependencies with copyleft licenses (GMP, MPFR — LGPL/GPL, unacceptable).
- Portability without DLLs (Windows, Linux, macOS) — header‑only libraries only.
- High performance for typical engineering precisions (eps ≈ 1e‑15…1e‑50).
- Ability to compute with arbitrary precision without changing the backend.

## 2. Architecture: Two Paths, Three Float Precisions

Every function provides two computational paths:

- **Float path** — evaluation on a fixed‑width `cpp_bin_float` followed by conversion to `Value` via `to_rational_with_eps`. Fast for moderate `eps`.
- **Series path** — pure rational series with argument reduction and binary splitting. Slower but covers any precision.

Path and float width are selected **at runtime, but without dynamic mantissa allocation** — three concrete types are used: `Float256`, `Float512`, `Float1024`. This avoids the runtime‑dispatching overhead typical of GMP.

### 2.1. Why Not GMP / MPFR

**License.** GMP is distributed under the LGPL (or GPL for some components). Static linking requires either opening your source code or meticulously complying with LGPL obligations (e.g., providing object files for re‑linking). For a commercial library under the PolyForm Small Business License this is categorically unacceptable. MPFR uses GMP as a backend and inherits its licensing restrictions.

**Windows build.** GMP requires either compiling from source with MSYS2 or using pre‑built DLLs. Neither option is “out‑of‑the‑box” header‑only, complicating distribution and integration.

**Performance.** In the target precision range (up to ~1000 bits) Boost.Multiprecision with a fixed width is comparable to, and often faster than, GMP because:
- It does not allocate mantissa memory dynamically (size is known at compile time).
- It uses stack allocation for small numbers.
- It eliminates branching on operand size inside arithmetic operations.
- The compiler can aggressively inline and optimize the template classes.

Thus, the decision to rely solely on Boost.Multiprecision with fixed bit widths is a deliberate engineering trade‑off that covers 99% of practical needs without licensing headaches.

### 2.2. Why No Float2048 and Float4096

Adding wider float types causes an exponential increase in compilation cost: each `cpp_bin_float<N>` instantiates many templates and precomputed constant tables (π, ln2, etc.), inflating binary size. For 1024 bits this is still acceptable; for 2048 bits compilation becomes significantly slower and the practical benefit approaches zero: for `eps` around 1e‑300 (decimal), rational series with binary splitting already run faster than float arithmetic with a huge mantissa. Requests for precision beyond 1e‑100 are virtually non‑existent in real applications; when they do occur, they are handled by the series path, which is not limited by a fixed width.

## 3. Dispatch

### 3.1. Trigonometric Functions and π

For `sin`, `cos`, `acos`, `asin`, `atan`, `tan`, `pi` a two‑level threshold is used:

```
eps >= 1e-35  → Float256
eps >= 1e-70  → Float512
otherwise    → series
```

These functions always return a value in `[-1, 1]`, so the absolute error unambiguously determines the required mantissa length. The thresholds `1e-35` and `1e-70` were chosen to guarantee accuracy with a few guard bits.

### 3.2. Exponential — Adaptive Bit‑Aware Dispatcher

`exp(x)` is fundamentally different: the result grows exponentially. For `x=40` the value is ~2.35e17, so an `eps=1e-35` already requires ~150 mantissa bits just for the fractional part. A simple `eps`‑based threshold does not work.

The dispatcher in `eager_exp`:
- Quickly rejects arguments where `|x|` is such that `msb(num) - msb(den) > 15` — this corresponds to `x > 32768`, the result will certainly overflow Float1024; immediately falls back to series.
- For the rest, it computes:
  - `integral_bits ≈ (x_approx * 23) >> 4` — estimate of the integer part of the result in bits (`log2(e) ≈ 1.4375 ≈ 23/16`).
  - `precision_bits = msb(denominator(eps))`.
  - `net_weight_bits = integral_bits + precision_bits`.
- Chooses Float256 when `≤ 240`, Float512 when `≤ 496`, Float1024 when `≤ 1008`, otherwise series.

This scheme guarantees that the float path is used only when the mantissa of the chosen type can safely hold both the integer and fractional parts with the required precision. The adaptivity allows using the fast float path for many values `x > 20` that would previously (with the fixed threshold 20.0) have fallen into the series path.

### 3.3. `sqrt`, `log`, `e`

Float paths for these functions **are absent**. Historically, with `cpp_dec_float_100` and string‑based conversion they were 2–3× slower than the rational analogues. After switching to binary floats and fast power‑of‑two conversion the gap narrowed, but still offered no advantage:
- `sqrt` benefits from an integer initial guess and fast rational arithmetic; a float conversion would add extra steps.
- `log` requires argument reduction, which in rational form is trivial (multiplication/division by 2) but in float would need additional conversions.
- `e` is computed once and cached; the rational series is simple and fast.

Thus, these functions always use the series path — a decision justified both by benchmarks and by simplicity.

## 4. Series Path Implementation

### 4.1. Argument Reduction and Epsilon Scaling

Each series function reduces the argument to a region of rapid convergence, simultaneously adjusting the internal `eps` so that the final error stays within the requested bound.

- **sin/cos**: reduction to `[-π, π]` using `eager_pi(eps)`. The error of π does not exceed `eps`, so no additional scaling is required.
- **exp**: repeatedly divide the argument by `2^k` until `|x| ≤ 2`, then compute the series for the reduced argument and square the result `k` times. **Crucially**: `internal_eps = eps / 2^{exp_bits + k + 2}`, where `exp_bits` is the binary exponent of the result. Without this factor the error after squaring would exceed `eps`.
- **log**: reduction to `[1/2, 2]` using `k * ln2`. Computes `arctanh((m-1)/(m+1))`; the series converges quickly.
- **sqrt**: for numbers with `|log2(x)| ≤ 60` an initial guess `floor(sqrt(num*den))/den` is used, followed by Newton's method. For extreme numbers — scaling by powers of 4 to bring the argument into a safe range.

### 4.2. Binary Splitting

For π, sin, cos, atan, asin the binary splitting technique is employed. Unlike a naive recurrent sum where large rational numbers are multiplied at each step, binary splitting combines terms in pairs, reducing intermediate fractions and limiting the overall growth of numerators/denominators. This avoids “rational explosion” and allows working with `eps` down to 1e‑1000 and beyond.

The binary splitting implementation is recursive, but for small N (determined by `eps`) a recurrent branch is used, since the recursion overhead is not worthwhile for few terms.

### 4.3. Constant Caching

π, e, and ln2 are stored in a static array of `std::map<Value, Value>`, keyed by the `eps` value. On the first request with a given `eps` the constant is computed (via float or series, depending on `eps`) and saved. All subsequent calls to `eager_pi(eps)` etc. receive the precomputed value.

Importantly: `series_sin` and other series functions call `eager_pi(eps)`, which for small `eps` (where the series path is active) is automatically directed to `series_pi` and caches the result. No mixing of π with different precisions occurs, because at small `eps` the float path for π is not used, and at large `eps` the series path for sin/cos is not called.

## 5. Float → Value Conversion: Binary Approach

The old `to_rational_with_eps` implementation:
- Converted `cpp_dec_float_100` to a fixed‑precision string.
- Parsed the string, assembled the numerator, and raised 10 to the power of the denominator length.
- Performed `gcd` reduction.

Problems:
- String parsing and decimal arithmetic are slow.
- The denominator (a power of 10) interacts poorly with other fractions (mixed bases).
- `cpp_dec_float_100` is a decimal type, inefficient in a binary environment.

The new implementation:
- Uses binary `cpp_bin_float` types.
- Computes `k = -exponent(eps) + extra_bits`, multiplies the float by `2^k`, rounds to an integer, and obtains the fraction `rounded / 2^k`.
- The denominator is a power of two, which speeds up subsequent rational arithmetic (shifts instead of multiplications, simpler gcd).
- No strings — everything works at the bit level.

This change accelerated the float path several‑fold and made it applicable to a wider range of functions (though for sqrt/log it still proved unprofitable).

## 6. Lessons Learned During Development

### 6.1. The Catastrophe of Series Vectorisation

An attempt to replace the recurrent loop with precomputation of all series terms and summation via PCR led to a 5‑fold slowdown. The reason: recurrent updating of a term (`term *= x / n`) has O(1) complexity, whereas recomputing factorials from scratch for each element is O(n). Combined with vector allocations and copying of large rational numbers, this completely destroyed performance.

Conclusion: **iterative accumulation with recurrent term update is the only correct way**.

### 6.2. The Importance of the Initial Guess in `sqrt`

Using `x/2` as an initial guess gave fast iterations but led to gigantic numerators/denominators in the final fraction. This did not show up in micro‑benchmarks of sqrt, but destroyed the performance of all subsequent operations with that result (e.g., `sin(π)` stopped finishing in reasonable time). Switching to `integer_floor_sqrt(num*den)/den` solved the problem while keeping the representation compact.

### 6.3. The Reduction Threshold in `exp` = 2.0

Lowering the threshold to 1.0 accelerated series convergence but produced enormous rational numbers due to unnecessary reduction and squaring. The result of `exp(1.234)` became monstrous, and the subsequent `log(exp(x))` in tests took minutes. The threshold 2.0 is the golden mean: most “ordinary” arguments are not reduced and yield compact fractions.

### 6.4. Epsilon Scaling in `exp` Is Non‑Negotiable

Without scaling of `internal_eps`, the error after raising to the power `2^k` grows by a factor of `2^k * exp(x)`. For large `x` this leads to completely incorrect results. No optimization should ever remove this multiplier.

## 7. Performance and Coverage of Scenarios

- **90% of typical tasks** (scientific, engineering) use `eps` in the range 1e‑10 … 1e‑50. They are fully covered by Float256, providing a 2–9× speedup over the series path.
- **9%** — elevated precision 1e‑50 … 1e‑100, served by Float512 or Float1024.
- **1%** — ultra‑high precision (<1e‑100), served by rational series of arbitrary precision. Here computation time is not critical because such requests are rare.

Thus, without involving GMP or dynamic precision, the library covers the entire spectrum of needs while remaining lightweight and license‑clean.

## 8. Conclusion

The presented architecture is the result of a long iterative process, numerous experiments, and bottleneck analysis. It has proven its efficiency in practice and contains no hidden regressions. All important details are documented in the code (file comment) and in this document. Any changes must be made with a full understanding of the principles described here and must be accompanied by a run of the complete test and benchmark suite.