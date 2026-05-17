# Δ‑Analysis Rational Subsystem – Public API Reference

**Purpose:** This document is the **user‑facing API reference** for the `delta::rational` situated in include/delta/rational/... folder 
**Aggregated Master-Headers to include:**
```cpp
#include "delta/core/rational.h"       // master header: Rational, LazyRational, GaussQi, all the transcendentals
#include "delta/core/eigen_integration.h"    // Eigen integration with matrix transcendentals (optional).
```
Why exclude Eigen integration from Master-Header rational.h? To only pay for it when you really need it. Compile time for Eigen is UNFATHOMABLY SLOW.

---

## 1. Core Types

### 1.1 `delta::Rational` – Eager exact rational

**Construction:**
```cpp
Rational();                                // 0
Rational(int n); Rational(long long n); Rational(unsigned long long n);
Rational(const std::string& s);            // "123", "123/456", "0.5".
Rational(double v);                        // ⚠️ Only for Eigen compatibility – use strings, literals or integer constructors for exact input
```

**Literals (namespace `delta::literals`):**
```cpp
42_r                    // Rational(42)
"3.14"_r                // Rational(157/50)
"22/7"_r                // Rational(22,7)
NOTE: format 0.5_r does not compile, however "0.5"_r does.
```

**Accessors:**
```cpp
double to_double() const;
std::string to_string() const;             // "num/den"
Rational numerator() const;
Rational denominator() const;
```

**Arithmetic (return new `Rational`):**
```cpp
Rational operator+(Rational, Rational);
Rational operator-(Rational, Rational);
Rational operator*(Rational, Rational);
Rational operator/(Rational, Rational);
Rational operator-(Rational);               // unary minus
```

**In‑place modification:**
```cpp
Rational& operator+=(Rational&, const Rational&);
// similarly -=, *=, /=
```

**Comparisons (exact):**
```cpp
bool operator==(const Rational&, const Rational&);
bool operator!=(const Rational&, const Rational&);
bool operator<(const Rational&, const Rational&);
// <=, >, >= similarly
```

**Utilities:**
```cpp
Rational abs(const Rational&);
Rational floor(const Rational&);
Rational batch_add(const std::vector<Rational>&);  // faster than loop
```

---

### 1.2 `delta::GaussQi` – Exact complex rational

**Construction:**
```cpp
GaussQi();                                   // 0+0i
explicit GaussQi(const Rational& re);        // re+0i
GaussQi(const Rational& re, const Rational& im);
explicit GaussQi(long long re);              // explicit – no implicit conversion
explicit GaussQi(const std::string& str);    // "(re,im)", "1+2i", "1/2-3/4i"
```

**Literals (namespace `delta::literals`):**
```cpp
"1+2i"_qi           // GaussQi(1,2)
"1/2-3/4i"_qi       // GaussQi(1/2, -3/4)
"i"_qi              // GaussQi(0,1)
"2"_qi              // GaussQi(2,0)
```

**Accessors:**
```cpp
const Rational& real() const;
const Rational& imag() const;
void real(const Rational&);
void imag(const Rational&);
std::string to_string() const;
std::pair<double,double> to_double() const;
```

**Operations:**
```cpp
Rational norm() const;          // re²+im²
GaussQi conj() const;
// +, -, *, / (with GaussQi, Rational, int)
// +=, -=, *=, /= similarly
// ==, !=
```

---

### 1.3 `delta::LazyRational` – Mutable, move‑only lazy expression tree

**⚠️ Move‑only:** copy constructor and assignment are **deleted**. Use `.clone()`.

**Construction:**
```cpp
LazyRational();                         // 0
explicit LazyRational(const Rational&);
explicit LazyRational(Rational&&);
```

**Cloning:**
```cpp
LazyRational clone() const;             // deep copy
```

**Mutating operators (modify left operand, return reference):**
```cpp
LazyRational& operator+(LazyRational& a, const LazyRational& b);  // a becomes SUM(a,b)
LazyRational& operator+(LazyRational& a, const Rational& b);
// similarly -, *, /, and compound assignments +=, -=, *=, /=
// Unary minus: returns new LazyRational
```

**Bulk insertion (optimised for loops):**
```cpp
void append_values(std::vector<internal::Value>&& values);   // add constants to SUM
void append_nodes(std::vector<int>&& node_indices);          // add child nodes
```

**Evaluation:**
```cpp
Rational eval(bool skip_simplify = false) const;   // returns evaluated result
void eval_inplace(bool skip_simplify = false);     // replaces *this with constant
```

**Simplification:**
```cpp
void simplify_inplace();                // canonicalize without evaluation
LazyRational simplify() const;          // return new simplified copy
```

**Inspectors:**
```cpp
bool is_dirty() const;   // local tree, can be mutated
bool is_clean() const;   // in global pool, immutable, shared
```

---

## 2. Transcendental Functions

All functions take an optional `eps` (absolute error bound). Default = `delta::default_eps()` (initial 1e-30).

### 2.1 Eager (return `Rational`)

```cpp
Rational sqrt(const Rational& x, const Rational& eps = default_eps());
Rational exp(const Rational& x, const Rational& eps = default_eps());
Rational log(const Rational& x, const Rational& eps = default_eps());
Rational sin(const Rational& x, const Rational& eps = default_eps());
Rational cos(const Rational& x, const Rational& eps = default_eps());
Rational tan(const Rational& x, const Rational& eps = default_eps());
Rational asin(const Rational& x, const Rational& eps = default_eps());
Rational acos(const Rational& x, const Rational& eps = default_eps());
Rational atan(const Rational& x, const Rational& eps = default_eps());
Rational pi(const Rational& eps = default_eps());
Rational e(const Rational& eps = default_eps());
Rational pow(const Rational& base, const Rational& exp, const Rational& eps = default_eps());
Rational pow(const Rational& base, int exponent);   // exact binary exponentiation
```

**π‑periodic functions (exact rational for multiples of 1/12,1/6,1/4,1/3,1/2):**
```cpp
Rational sinpi(const Rational& x, const Rational& eps = default_eps());   // sin(π·x)
Rational cospi(const Rational& x, const Rational& eps = default_eps());
Rational tanpi(const Rational& x, const Rational& eps = default_eps());
Rational asinpi(const Rational& y, const Rational& eps = default_eps());  // asin(y)/π
Rational acospi(const Rational& y, const Rational& eps = default_eps());
Rational atanpi(const Rational& y, const Rational& eps = default_eps());
```

### 2.2 Lazy (return `LazyRational`, do **not** mutate arguments)

```cpp
LazyRational Sqrt(const LazyRational& x, const Rational& eps = default_eps());
LazyRational Exp(const LazyRational& x, const Rational& eps = default_eps());
LazyRational Log(const LazyRational& x, const Rational& eps = default_eps());
LazyRational Sin(const LazyRational& x, const Rational& eps = default_eps());
LazyRational Cos(const LazyRational& x, const Rational& eps = default_eps());
LazyRational Pi(const Rational& eps = default_eps());
LazyRational Pow(const LazyRational& base, const LazyRational& exp, const Rational& eps = default_eps());
// Overloads for Rational + LazyRational, LazyRational + Rational, Rational + Rational, and int exponent
```

> ℹ️ Lazy `Asin`, `Atan`, `Tan` are not yet exposed – use eager versions.

---

### 2.3 Complex (`GaussQi`) transcendentals (eager)

```cpp
Rational abs(const GaussQi& z, const Rational& eps = default_eps());   // sqrt(re²+im²)
Rational arg(const GaussQi& z, const Rational& eps = default_eps());   // atan2(im,re)
GaussQi sqrt(const GaussQi& z, const Rational& eps = default_eps());
GaussQi exp(const GaussQi& z, const Rational& eps = default_eps());
GaussQi log(const GaussQi& z, const Rational& eps = default_eps());    // principal branch
GaussQi sin(const GaussQi& z, const Rational& eps = default_eps());
GaussQi cos(const GaussQi& z, const Rational& eps = default_eps());
GaussQi tan(const GaussQi& z, const Rational& eps = default_eps());
GaussQi asin(const GaussQi& z, const Rational& eps = default_eps());
GaussQi acos(const GaussQi& z, const Rational& eps = default_eps());
GaussQi atan(const GaussQi& z, const Rational& eps = default_eps());
GaussQi pow(const GaussQi& z, const GaussQi& w, const Rational& eps = default_eps());
GaussQi pow(const GaussQi& z, int exponent);   // exact

Rational atan2(const Rational& y, const Rational& x, const Rational& eps = default_eps());
```

> ⚠️ No lazy complex transcendentals. For lazy complex, combine two `LazyRational` trees manually.

---

## 3. Context – Global Default Epsilon

```cpp
Rational default_eps();                      // current value (initial 1e-30)
void set_default_eps(const Rational& eps);  // changes for **all threads**
void reset_default_eps();                   // restores 1e-30
```

---

## 4. Eigen Integration

**Include:** `#include <delta/rational/eigen.h>`

After including, `delta::Rational` and `delta::GaussQi` are valid scalar types for `Eigen::Matrix`.

### Element‑wise (via `.array()`):
```cpp
Eigen::Matrix<Rational, Dynamic, Dynamic> A;
auto B = A.array().sin();   // delta::sin on each element
auto C = A.array().exp();   // delta::exp on each element
```
Note: DO NOT MISTAKE ELEMENT-WISE with true matrix transcendentals for the general case.

### True matrix functions (namespace `delta`):
```cpp
template<typename Derived>
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

**Requirements:** Square matrix, scalar type `Rational` or `GaussQi`. Diagonal matrices are optimised (element‑wise).

---

## 5. Quick Examples

### Basic arithmetic:
```cpp
Rational a = "1/3"_r;
Rational b = 2_r;
Rational c = a + b;                 // 7/3
Rational d = sqrt(c);               // approximate √(7/3)
```

### Lazy accumulation (O(N) instead of O(N²)):
```cpp
LazyRational acc;
for (int i = 0; i < 1000; ++i)
    acc + sin(Rational(i));
Rational result = acc.eval();
```

### Complex numbers:
```cpp
GaussQi z("1/2"_r, "1/3"_r);
GaussQi w = exp(z, "1e-20"_r);
Rational r = abs(w);
```

### Eigen matrix exponential:
```cpp
#include <delta/rational/eigen.h>
Eigen::Matrix<delta::Rational, 2, 2> M;
M << 1_r, 2_r, 3_r, 4_r;
auto expM = delta::exp(M);
```

---

## 6. Thread Safety

| Type              | Thread‑safe?                                      |
|-------------------|---------------------------------------------------|
| `Rational`        | ✅ Yes (immutable after construction)            |
| `GaussQi`         | ✅ Yes (immutable after construction)            |
| `LazyRational`    | ❌ No – single object must not be used concurrently |
| Global state      | Node pool, caches, registry: **thread‑local**    |
| `default_eps()`   | ⚠️ **Global** – shared across all threads        |

---

## 7. Known Limitations (User‑Visible)

- Lazy `asin`, `atan`, `tan` not yet available – use eager.
- No lazy complex (`GaussQi`) expressions.
- `Rational(double)` constructor is **inaccurate** (because double itself is inaccurate) – use string literals for exact input.
- Serialisation not provided.
- Very large rationals (thousands of digits) become slow, which is to be expected, honestly.

---

## 8. Error Handling

- Division by zero → throws `std::domain_error`.
- `log(0)` or `log(negative)` → throws `std::domain_error`.
- `sqrt(negative)` → throws `std::domain_error`.
- Overflow in rational arithmetic → Boost multiprecision throws (rare, arbitrary precision).

---

**See also:**
- docs/... folder