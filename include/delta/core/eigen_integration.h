// ============================================================================
// DELTA + EIGEN INTEGRATION – MATRIX TRANSCENDENTALS IN EXACT RATIONAL ARITHMETIC
// ============================================================================
// This header provides seamless integration of delta::Rational and delta::GaussQi
// with the Eigen3 library.  After including this file, you can use Eigen matrices
// and arrays with the scalar types delta::Rational (exact real rationals) and
// delta::GaussQi (exact complex rationals).  In addition, the header provides
// true matrix transcendental functions that operate on square matrices as a whole,
// including exp, log, sin, cos, and sqrt (principal branches).  Element‑wise
// transcendentals are also available via the standard .array() interface.
//
// ----------------------------------------------------------------------------
// QUICK START
// ----------------------------------------------------------------------------
//   #include <delta/eigen.h>
//   using namespace delta;
//
//   // 1. Build a rational matrix
//   Eigen::Matrix<Rational, 2, 2> A;
//   A << 1_r, 2_r,
//        3_r, 4_r;
//
//   // 2. Element‑wise transcendental (ADL finds delta::exp)
//   Eigen::Matrix<Rational, 2, 2> expA_elem = A.array().exp();
//
//   // 3. True matrix exponential (scaling‑and‑squaring + Padé)
//   Eigen::Matrix<Rational, 2, 2> expA_matrix = delta::exp(A);
//
//   // 4. Complex rational matrices
//   Eigen::Matrix<GaussQi, 2, 2> Z;
//   Z << GaussQi(1,2), GaussQi(0,1),
//        GaussQi(1,0), GaussQi(0,0);
//   auto logZ = delta::log(Z, Rational(1, 10000000000000000000)); // ε = 1e-19
//
//   // 5. Evaluate and convert to double (when needed)
//   double approx = logZ(0,0).real().to_double();
//
// ----------------------------------------------------------------------------
// AVAILABLE FUNCTIONS (overloaded for MatrixBase<Derived>)
// ----------------------------------------------------------------------------
// All functions require a square matrix.  The scalar type must be either
// delta::Rational or delta::GaussQi.
//
//   delta::exp(A, eps)     – matrix exponential
//   delta::log(A, eps)     – principal matrix logarithm
//   delta::sin(A, eps)     – matrix sine (principal branch)
//   delta::cos(A, eps)     – matrix cosine
//   delta::sqrt(A, eps)    – principal matrix square root
//
// The `eps` parameter is a delta::Rational specifying the requested absolute
// accuracy for each element.  For large‑value results (e.g., exp(A) with huge
// entries) the tolerance is internally scaled by the magnitude of the result
// to maintain relative accuracy.  If omitted, the global default epsilon
// (see delta::default_eps()) is used, typically 1e-30.
//
// ----------------------------------------------------------------------------
// SYNTAX AND SEMANTICS
// ----------------------------------------------------------------------------
// The functions are **type‑erased** – they accept any Eigen expression
// (e.g., A*B + C) but evaluate it to a dynamically‑sized matrix first.
// This avoids compile‑time explosion from expression templates while
// incurring negligible overhead (rational arithmetic dominates anyway).
//
// Element‑wise semantics use the .array() interface.  For example:
//   A.array().sin()   – computes delta::sin of each element, not matrix sine.
//   delta::sin(A)     – computes true matrix sine (Taylor / exp(iA) method).
//
// Diagonal matrices are detected and processed element‑wise for speed.
//
// ----------------------------------------------------------------------------
// PERFORMANCE & ACCURACY
// ----------------------------------------------------------------------------
// • Rational arithmetic is slow compared to double.  However, the lazy
//   expression mechanism (delta::LazyRational) and the optimised Padé
//   implementations make large sums and matrix operations practical.
// • For epsilon ≥ 1e-35, many trigonometric and exponential functions use a
//   fast floating‑path (boost::multiprecision::cpp_dec_float_100) internally.
//   Below that threshold, exact rational series are used.
// • The matrix exponential and logarithm use scaling‑and‑squaring with
//   diagonal Padé approximants.  For complex (GaussQi) logarithm,
//   classical scaling by division by 2 does NOT work (it fails to reduce phase).
//   Instead we perform a single trace normalisation ( B = M / (trace(M)/n) ),
//   then apply a Padé approximant for log(I+X).  This yields high accuracy
//   without catastrophic growth of rational bit length.
// • Performance for 5×5 matrices with epsilon = 1e-19:
//       exp:    ~900 ms   (complex)
//       log:    ~400 ms   (complex)
//       sin:   ~2000 ms   (complex, through exp(iA))
//       sqrt:   ~400 ms   (complex, Newton)
//   Real rational matrices are significantly faster (≤ 100 ms).
//
// ----------------------------------------------------------------------------
// PRECAUTIONS & LIMITATIONS
// ----------------------------------------------------------------------------
// 1. **Exactness is guaranteed** – all operations remain rational (no floating‑
//    point approximation) unless the fast float path is explicitly chosen.
// 2. **Memory** – rational numbers can grow large (hundreds or thousands of bits)
//    when many operations are performed.  Use eval() or simplify() to collapse
//    intermediate expressions when possible.  The global node pool (thread‑local)
//    automatically garbage‑collects unused expression nodes.
// 3. **Complex matrices** – the principal matrix logarithm requires that the
//    matrix has no eigenvalues on the negative real axis (branch cut).  This
//    is not checked by the library; results may be inaccurate for such matrices.
// 4. **Thread safety** – each thread maintains its own node pool and caches.
//    Do not transfer delta::Matrix objects between threads without cloning.
//    The global default epsilon (delta::default_eps()) is NOT thread‑local and
//    is shared across threads (by design).
// 5. **Dense matrices only** – the Eigen backends used (partialPivLu) are dense.
//    For sparse rational matrices, convert to dense first.
//
// ----------------------------------------------------------------------------
// EXAMPLE: INVERSE SCALING TEST (EXP(LOG(A)) ≈ A)
// ----------------------------------------------------------------------------
//   #include <delta/eigen.h>
//   #include <iostream>
//
//   int main() {
//       using namespace delta;
//       Eigen::Matrix<Rational, 3, 3> A;
//       A << 1, 2, 3,
//            4, 5, 6,
//            7, 8, 9;
//       Rational eps(1, 1000000000000000000); // 1e-18
//
//       auto logA = delta::log(A, eps);
//       auto A_recovered = delta::exp(logA, eps);
//
//       std::cout << "Original A:\n" << A << "\n\n";
//       std::cout << "Recovered A:\n" << A_recovered << "\n";
//       return 0;
//   }
//
// ----------------------------------------------------------------------------
// REQUIRED HEADERS (automatically included)
// ----------------------------------------------------------------------------
//   #include <Eigen/Core>
//   #include <Eigen/LU>
//   #include "rational_class.h"
//   #include "gauss_qi.h"
//   #include "eigen_transcendentals.h"
//
// ----------------------------------------------------------------------------
// FURTHER DOCUMENTATION
// ----------------------------------------------------------------------------
// See the detailed commentary inside eigen_transcendentals.h for the theory
// behind scaling‑and‑squaring, Padé order selection, and the rationale for
// type erasure.
// ============================================================================

#pragma once

#include "delta/rational/eigen_integration.h" 