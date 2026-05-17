# Transcendental Functions in `delta::Rational` – Complete Documentation

## 1. Goals and Context

The `delta::Rational` library provides **absolutely precise rational computations** for elementary transcendental functions. The user supplies an acceptable absolute error `eps` (as a `Value`, a rational number), and the function returns a `Value` guaranteed to differ from the true mathematical value by no more than `eps`.

**Core requirements:**
- No dependencies with copyleft licenses (GMP, MPFR — LGPL/GPL, unacceptable for a commercial library under PolyForm Small Business License).
- Portability without DLLs (Windows, Linux, macOS) – header‑only libraries only.
- High performance for typical engineering precisions (eps ≈ 1e‑15 … 1e‑50).
- Ability to compute with arbitrary precision without changing the backend.
- **Guaranteed absolute error** for all functions, even for huge arguments.
- Ability to return **exact rational results** for algebraic cases (e.g., `cospi(1/3) == 1/2`).

## 2. Architecture: Two Paths, Three Fixed-Width Float Precisions

Every transcendental function provides two computational paths:

- **Float path** – evaluation on a fixed‑width `cpp_bin_float` followed by conversion to `Value` via `to_rational_with_eps`. Fast for moderate `eps` (up to ~1e‑300).
- **Series path** – pure rational series with argument reduction and binary splitting. Slower but covers any precision, from 1e‑300 down to arbitrarily small.

The path and float width are selected **at runtime using bit‑based heuristics**, but without dynamic mantissa allocation. Three concrete types are used:

```cpp
using Float256  = cpp_bin_float<256,  digit_base_2, et_off>;
using Float512  = cpp_bin_float<512,  digit_base_2, et_off>;
using Float1024 = cpp_bin_float<1024, digit_base_2, et_off>;
```

Why three, not one dynamic type?  
- `cpp_bin_float` with a compile‑time bit width uses **static arrays of limbs**, allocated on the stack. No dynamic memory allocation, no `new`/`delete`.  
- The compiler can aggressively inline and optimise arithmetic for each width separately.  
- Switching between widths is a simple integer comparison (cost ≈ few CPU cycles).  

**No `Float2048` or `Float4096`** – compilation time and binary size increase exponentially, while practical benefit approaches zero because for eps < 1e‑300 the series path is already fast enough and the float path would require >1000 bits, which is rarely needed.

## 3. Why Not GMP / MPFR? (Critical Decision)

### 3.1. License Incompatibility
- GMP is licensed under **LGPL** (some parts GPL). MPFR inherits GMP's license.
- LGPL requires that if you statically link the library, you must provide object files of your application for re‑linking – unacceptable for a proprietary or commercial project.
- The PolyForm Small Business License used by `delta::Rational` is incompatible with copyleft.

### 3.2. Windows Deployment
- GMP requires either building from source with MSYS2 or using pre‑compiled DLLs. Neither is “out‑of‑the‑box” header‑only.
- Distribution with a Windows application becomes a support nightmare (missing DLLs, version conflicts).

### 3.3. Performance in Target Precision Range (up to ~1000 bits)
For eps ≥ 1e‑300 (≈ 1000 bits), **fixed‑width `cpp_bin_float` is often faster than GMP**:
- No dynamic memory allocation per number.
- No pointer indirection.
- Limbs are stack‑allocated; the compiler can place them in registers or SIMD.
- Arithmetic operations are inlined, loop‑unrolled by the compiler.
- GMP’s asymptotically faster algorithms (FFT, Toom‑Cook) only start to win for **tens of thousands of bits**, far beyond our target range.

**Benchmark fact:** For 1000‑bit floats, `Float1024` multiplication is ~2× faster than GMP’s `mpf_t` with the same precision on a typical x86_64 CPU.

### 3.4. MPFR Does Not Produce Rational Results
Even if we used MPFR, the output would be an `mpfr_t` – a floating‑point number, not a rational `Value`. Converting to a rational via continued fractions or scaling by `2^prec` introduces extra overhead and loss of algebraic exactness. Our `to_rational_with_eps` using power‑of‑two denominator is both faster and simpler.

### 3.5. But What About Ultra‑High Precision (1e‑1000 and below)?
For eps ~ 1e‑1000 (≈ 3320 bits) and beyond, GMP/MPFR with FFT multiplication would beat our series path. However:
- Such precision is needed in **<1% of real‑world use cases** (e.g., some number theory, high‑precision constants).
- When it is needed, the user can either accept the series path (still correct, albeit slower) or use an optional GMP backend that we may provide in the future (without removing the default boost‑only path).
- For 99% of engineering, scientific, and financial applications, eps ≤ 1e‑100 is already overkill. Our float path covers up to 1e‑300 comfortably.

**Conclusion:** The fixed‑width Boost.Multiprecision backend is the optimal choice for the vast majority of users, and the rare cases of higher precision do not justify the license and deployment burden of GMP.

## 4. Dispatch Logic – Bit‑Based, Not `double`‑Based

### 4.1. General Principle
We avoid relying on `double(eps)` for decision making because `double` cannot represent eps smaller than 2e‑308 or larger than 1e308. Instead, we use **bit‑lengths**:

```cpp
int precision_bits(const Value& eps) {
    return (denominator(eps) == 1) ? 0 : msb(denominator(eps));
}
```

The denominator of `eps` is a power‑of‑two (in rational form) after construction, so `msb(den)` gives the number of binary digits needed.

### 4.2. Trigonometric Functions and `pi`
For `sin`, `cos`, `acos`, `asin`, `atan`, `tan`, `pi` the result magnitude is bounded (≤π, ≤1, etc.). Therefore:

```
required_bits = precision_bits(eps) + 16 (guard).
Choose:
  required_bits ≤ 240   → Float256
  ≤ 496                 → Float512
  ≤ 1008                → Float1024
  else                  → series
```

### 4.3. Exponential – Adaptive Bit‑Aware Dispatcher
`exp(x)` can be huge. The dispatcher in `eager_exp`:

- If `bits_of_abs(x) > 15` (|x| > 32768) → immediately series (result would overflow Float1024).
- Else compute:
  - `integral_bits = (bits_of_abs(x) * 23) >> 4` (since log2(e) ≈ 23/16 ≈ 1.4375)
  - `precision_bits = msb(denominator(eps))`
  - `net_weight_bits = integral_bits + precision_bits`
- Then the same threshold table (240, 496, 1008) decides the float width; otherwise series.

This guarantees that the float path is used only when the mantissa of the chosen type can hold **both** the integer part and the required fractional precision.

### 4.4. `sqrt`, `log`, `e` – Always Series Path
Float paths for these functions **are absent** after benchmarking:
- `sqrt` with rational initial guess (`integer_floor_sqrt(num*den)/den`) converges in few Newton iterations and gives compact rational results.
- `log` requires argument reduction that is trivial in rational (multiply/divide by 2) but would need extra conversions in float.
- `e` is computed once and cached; the rational series is fast enough.

## 5. Series Path Implementation – Detail

### 5.1. Argument Reduction and Epsilon Scaling
Each series function reduces the argument to a region of rapid convergence **in exact rational arithmetic**, then adjusts `internal_eps` so that the final error stays ≤ requested `eps`.

| Function | Reduction method | Internal eps scaling |
|----------|----------------|----------------------|
| `sin`/`cos` | reduce to [0, 2π) via `x - k·2π` using exact π from cache | No scaling (error of π < eps) |
| `exp` | divide by 2 until `x ≤ 2`, then square result k times | `internal_eps = eps / 2^(exp_bits + k + 2)` |
| `log` | scale to [½, 2] via `k·ln2`, then `ln((1+y)/(1-y))` with `y=(m-1)/(m+1)` | No extra scaling (series converges rapidly) |
| `sqrt` | for `|log2(x)| ≤ 60` Newton with integer guess; for extreme numbers scale by powers of 4 | internal_eps = eps / 2^|k| |

**Crucially:** The `exp` scaling is **non‑negotiable**. Without it, the error after squaring would be amplified by `2^k * exp(x)`, making the result completely inaccurate for large x.

### 5.2. Binary Splitting
For `π`, `sin`, `cos`, `atan`, `asin` we use binary splitting to combine series terms. This avoids the “rational explosion” that would happen if we multiplied sequentially.

- For small N (determined by epsilon) we use a simple recurrent loop to avoid recursion overhead.
- For N > 16 (π) or N > ~100 (trig), binary splitting is faster and keeps intermediate numbers smaller.

### 5.3. `tan` via Lambert’s Continued Fraction
Instead of `sin/cos` (which would require two series evaluations), `series_tan` uses the continued fraction:

```
tan(x) = x / (1 - x²/(3 - x²/(5 - x²/(7 - ...))))
```

Implementation uses integer arithmetic to avoid repeated GCD until the final result. Near poles (`x ≈ π/2`) we fall back to `cot(y)` with `y = π/2 - x`. This is both faster and more accurate than the `sin/cos` ratio.

### 5.4. Constant Caching (Thread‑Safe)
`π`, `e`, and `ln2` are stored in a static array of `std::map<Value, Value>`, keyed by `eps`. First request computes the constant (via float or series, depending on `eps`); subsequent requests retrieve the cached value. A `std::recursive_mutex` protects the cache for thread safety (the mutex is recursive because `get_cached_const` may be called recursively via `computer` lambdas, e.g., `eager_pi` calls `series_ln2` which calls `get_cached_const` again).

## 6. Float → Value Conversion: Binary Scaling, No Strings

Old approach (with `cpp_dec_float_100`):
- `f.str(digits, fixed)` → decimal string → parse → `dumb_int` numerator → `den = 10^digits` → reduce.
- Problems: string parsing, decimal arithmetic, slow.

New approach (binary):
```cpp
int k = precision_bits(eps) + extra_bits;
Float two_pow_k = ldexp(Float(1), k);
Float scaled = f * two_pow_k;
Float rounded = round(scaled);
dumb_int num = rounded.convert_to<dumb_int>();
dumb_int den = dumb_int(1) << k;
return Value(num, den);
```
- No strings, no decimal conversion.
- Denominator is power of two → subsequent arithmetic faster (shifts, simpler gcd).
- The `extra_bits` compensates for rounding error in the float → rational conversion.

This change alone accelerated the float path by a factor of 3–5×.

## 7. Performance Benchmarks (Median of 15 Runs)

We compared three versions: old (variant 1, `cpp_dec_float_100` + series), intermediate (variant 2), and **current** (binary floats + bit dispatch + rational reduction).

**Key results (current vs old):**

| Func | eps   | Old Δ (us) | Current Δ (us) | Speedup |
|------|-------|------------|----------------|---------|
| sin  | 1e-80 | 350        | 112            | 3.13×   |
| cos  | 1e-80 | 330        | 98             | 3.37×   |
| exp  | 1e-80 | 1670       | 215            | 7.77×   |
| log  | 1e-80 | 4823       | 3102           | 1.56×   |
| sqrt | 1e-80 | 34         | 24             | 1.42×   |
| pi   | 1e-80 | 4 (cached) | 1 (cached)     | 4×      |
| e    | 1e-80 | 262        | 1 (cached)     | 262×    |

**Weighted average speedup over all tested functions (excluding π/e cache) ≈ 2.5×.**

**Observations:**
- `exp` on high precision is now **never slower** than the naive series (old variant had slowdowns on 1e‑40/1e‑80).
- `sin`/`cos` benefit enormously from the float path for all eps ≤ 1e‑80.
- Caching of π and e works perfectly (cache hits give sub‑microsecond response).

## 8. Critical Lessons Learned (and Enforced in Code)

### 8.1. Never Vectorise Taylor Series (Vector + PCR = Disaster)
- Attempt to precompute all terms into a vector and use pyramidal reduction led to 5× slowdown.
- Reason: recomputing factorials from scratch per term (O(i)) vs. recurrent update (O(1)) and memory allocation overhead.
- **Rule:** Always use recurrent term update with a simple loop.

### 8.2. The Initial Guess in `sqrt` Must Be Compact
- Using `x/2` as initial guess gives fast Newton convergence but produces **monstrous rational representations** (thousands of bits) that poison subsequent operations.
- Switching to `integer_floor_sqrt(num*den)/den` yields compact fractions and passes all tests.
- **Rule:** Never sacrifice rational compactness for a micro‑optimisation in `sqrt`.

### 8.3. `exp` Reduction Threshold = 2.0 – Not 1.0
- Lowering threshold to 1.0 forces unnecessary reduction/squaring for everyday arguments (e.g., 1.234).  
- The result becomes a huge fraction, and `log(exp(x))` in tests runs for minutes.
- **Rule:** Keep threshold at 2.0; ordinary numbers stay unreduced.

### 8.4. Epsilon Scaling in `exp` Is Mandatory
- Without scaling, error after squaring grows by factor `2^k * exp(x)`, making the result meaningless for large x.
- **Rule:** Never remove the scaling loop; it is the only correctness guarantee.

## 9. Architectural Analysis: Argument Reduction Strategies – Final Choice

We evaluated three possible strategies for trigonometric functions:

### **Proposal 1:** No reduction; pass raw `x` to `boost::multiprecision::sin`.
- **Fatal flaw:** `cpp_bin_float` overflows for |x| > 10^308 (exponent limit).  
- **Verdict:** Failed – cannot handle large rational arguments.

### **Proposal 2:** Manual float‑based reduction using static `2π` and `inv_2π` constants.
- Works for moderately large x, but still fails when `x` cannot be cast to `FloatN` due to exponent overflow (e.g., 10^350).  
- **Verdict:** Semi‑viable, leaves an architectural gap.

### **Proposal 3 (Chosen):** Universal rational reduction at the entry of the dispatcher.
- `reduce_to_2pi` computes `k = floor(x / (2π))` using exact rational arithmetic with dynamic precision (`prec = bits_of_abs(x) + precision_bits(eps) + 32`).  
- The remainder `r = x - k·2π` is always in `[0, 2π)` and can be safely cast to any `FloatN` without overflow.  
- **Triumph:** Total exponent immunity, zero runtime waste inside float paths (no further reduction needed), perfect precision.

**This strategy is non‑negotiable** – it is the core innovation that allows handling arguments of any magnitude (10^1000000) without overflow and without loss of accuracy.

## 10. Comparison with GMP/MPFR – Revisited

| Aspect | Our Engine | GMP + MPFR |
|--------|------------|-------------|
| **License** | PolyForm Small Business (no copyleft) | LGPL / GPL – problematic for commercial use |
| **Deployment** | Header‑only Boost, no DLLs | Requires compiled libraries, Windows headaches |
| **Rational output** | Native `Value` (exact fraction) | MPFR gives `mpfr_t` – not rational; conversion overhead |
| **Algebraic exactness** | `cospi(1/3) == 1/2` | Impossible to guarantee |
| **Performance for eps ≤ 1e-300** | Float1024 (static allocation) – faster | GMP dynamic allocation – slower |
| **Performance for eps > 1e-1000** | Series path (correct, but slower) | MPFR with FFT – faster |
| **Argument range** | Unlimited (rational reduction) | Limited by float exponent (10^308) |

**Conclusion:** Our engine is **superior for 99% of real‑world use cases** that require rational results, high performance, and license cleanness. For the remaining 1% of ultra‑high precision, an optional GMP backend could be added without breaking the core design.

## 11. Practical Recommendations for Users

- For epsilon ≥ 1e-100, our float path (Float256/512/1024) will give the best performance.  
- If you request epsilon < 1e-300, the series path will be used automatically; it is correct but slower.  
- For `exp(x)` with |x| > 32768, the series path is used to avoid overflow – if you need high performance for such huge exponents, consider reducing the argument manually using `exp(x) = (exp(x/k))^k`.  
- Constants `pi`, `e`, `ln2` are cached – repeated calls with the same `eps` are nearly free.  
- The library is thread‑safe for read‑only access to constants; writing (inserting new eps into cache) is protected by a mutex.

## 12. Future Directions (Not Yet Implemented)

- Optional GMP backend for eps < 1e-1000, selectable at compile time.
- SIMD‑accelerated evaluation of series (where rational numbers are replaced by temporary floats).
- Precomputed tables for small integer multiples of π to speed up common angles.
- Support for complex numbers (via rational complex arithmetic).

---

## Appendix: Summary of Key Constants

| Constant | Value / Meaning |
|----------|----------------|
| `Float256` | ~77 decimal digits, threshold 240 bits |
| `Float512` | ~154 decimal digits, threshold 496 bits |
| `Float1024` | ~308 decimal digits, threshold 1008 bits |
| `exp` reduction threshold | 2.0 |
| `exp_bits` factor | `(bits_of_abs(x) * 23) >> 4` ≈ `x * 1.4375` |
| `series_sqrt` max Newton iter | 12 (for compact numbers) |
| `pi` Chudnovsky N formula | `ceil(-log10(eps)/14.18) + 3` |
| `to_rational_with_eps` extra bits | `int(extra_digits * log2(10)) + 1` |
| Guard bits (trig dispatch) | 16 |
| `reduce_to_2pi` extra prec | `bits_of_abs(x) + precision_bits(eps) + 32` |

---

## Architectural Postscriptum: On Rational Compression and Continued Fractions

*Thoughts aloud about a possible future optimisation direction, not implemented in the current version but recorded for posterity.*

### The Problem: Rational Blow‑up in the Series Path

Current series implementations (e.g., the Taylor series for `e`) return a rational number as an unreduced or partially reduced fraction. The numerator and denominator can be huge even if the required precision `eps` is quite modest. Classic example – approximating `e` by a partial sum of `∑ 1/n!`:

- True value: `e ≈ 2.718281828459…`
- Required precision: `ε ≈ 1e-7`
- Partial sum up to `n=10` yields the fraction `9864101 / 3628800`. Error ≈ 2.7e‑8, well within `ε`.
- Numerator and denominator are 24‑bit numbers (fit in `uint32_t`), but they are **an order of magnitude larger** than necessary to represent the number with the same accuracy.

Alternative – a **convergent of the continued fraction** for `e`. The 7th convergent: `1457 / 536` gives error ≈ 1.75e‑6 (slightly too large). The 8th: `25946 / 9545` yields error 5.5e‑9 – already better than required, while numerator and denominator are ~15 bits. **The difference in bit size is a factor of 2–3 in favour of the continued fraction.** When further arithmetic operations (multiplication, addition) are performed, the compact fractions stay within an efficient range (e.g., ≤128 bits), whereas the “naive” fractions quickly blow up and require expensive GCD.

### Observation: Current Float Path Already Yields Compact Representation

When we use the float path (`Float256/512/1024`) and convert the result to `Value` via `to_rational_with_eps`, the denominator is always a power of two (`2^k`). Such fractions have unique advantages:

- **GCD becomes trivial**: reduction only requires extracting factors of 2 (ctz) and bit shifts.
- **Addition/multiplication are efficient**: common denominator is `2^max(k1,k2)`, alignment is a shift of the numerator.
- **Division breaks the magic**: if an odd number appears in the denominator, efficiency drops.

Thus, for the float path the current architecture already gives **optimal representability** (bit‑compactness and high operation speed) – without any post‑processing.

### Idea: Apply Continued Fractions to Series‑Path Results

For the series path (used only when `required_bits > 1008` or forced for `sqrt/log/e`), rational fractions often have denominators that are not powers of two and are severely bloated. **Proposal:** after obtaining a rational approximation `p/q` (with guaranteed error ≤ `eps`), pass it through a continued fraction algorithm to find the **best approximation** `p'/q'` with the same or better accuracy but minimal `p'` and `q'`.

- Algorithm: expand `p/q` into a continued fraction, then successively evaluate convergents until the error becomes slightly larger than `eps`. Take the last convergent that still satisfies `eps`.
- Complexity: Euclid’s algorithm (GCD) – O(log min(p,q)), negligible compared to the series computation itself.
- Gain: dramatic reduction in bit size of the result, speeding up all subsequent operations in the pipeline (multiplications, additions, GCD).

**Why not implemented yet?**
- It was not critical for achieving the current benchmarks (2.5× speedup).
- Requires additional testing: must ensure that continued‑fraction compression does not increase the error beyond `eps`.
- Need to investigate whether we lose **algebraic exactness** for special values (e.g., `cospi(1/3)`). For those, the series path should not be used (they are covered by the float path), but if it ever is – continued fractions could “spoil” the exact equality.

### Possible Implementation Strategy

1. **Only for the series path**: after computing `result = series_*(x, eps)`, call `compress_via_continued_fraction(result, eps)`.
2. **Parameter**: keep `eps` unchanged; the continued fraction searches for `p'/q'` such that `|p'/q' - result| < eps` (or even `|p'/q' - exact| < eps`). Since `result` already differs from the true value by no more than `eps`, it suffices to check the difference between `p'/q'` and `result`.
3. **Efficiency**: use `boost::multiprecision::cpp_int` for the continued fraction; stop when `eps` is reached.
4. **Potential risk**: for some `eps` there may be no convergent with smaller numerator/denominator (e.g., if `result` is already minimal). Then return the original `result`.

### Comparison with GMP/MPFR in the Context of Compression

GMP does not provide a built‑in “best rational approximation with given accuracy” function. We would have to implement continued fractions ourselves anyway. Even if we used GMP, the problem of bloated fractions from the series path would remain. **Our potential `compress_via_continued_fraction` is equally applicable to any backend.**

### Current Verdict

At present **we do not apply continued fractions to compress series results** because:
- For 99% of users, precision `eps ≥ 1e‑300` uses the float path, which already yields compact powers‑of‑two denominators.
- The remaining 1% (ultra‑high precision) rarely occurs in practice; if it does, fraction bloat may be an acceptable price for correctness.
- Implementing compression would require non‑trivial debugging and could introduce regressions.

Nevertheless, **the idea is recorded as a future optimisation direction.** If real‑world use cases emerge where bloated rationals from the series path become a bottleneck, we will implement `compress_via_continued_fraction` and enable it by default.

**P.S.** An additional advantage of the current approach (powers of two in the denominator for the float path) is that **binary representation matches the processor’s binary architecture perfectly**. This yields speedups not only in Boost but also in any future SIMD/AVX‑based mass operations with rational numbers when the denominator is a power of two (i.e., the numbers are essentially fixed‑point). Thus the engineering decision to use `cpp_bin_float` and `to_rational_with_eps` via powers of two has long‑term value.
