/**
 * \defgroup examples Examples (from tests)
 * \brief Real-world usage scenarios extracted from the test suite.
 *
 * The following examples are taken directly from the test suite. They
 * illustrate key concepts of Δ‑analysis:
 *
 * - Continuity verification with power and logarithmic moduli
 * - Differentiability checks on dyadic grids
 * - Riemann sums and convergence
 * - Adaptive refinement with `AdaptiveDeltaPath`
 * - Construction of √2 as a fundamental sequence
 * - Discrete Exterior Calculus (DEC) – exterior derivative, Hodge star,
 *   Laplacian, and wedge product
 * - Cotangent Laplacian – algebraic properties and action on functions
 *
 * Each file is self-contained and includes both the test and its
 * mathematical justification.
 */

 /**
  * \page examples_snippets Snippet Gallery
  * \brief Selected code snippets illustrating key concepts.
  *
  * These snippets are extracted from the test suite to demonstrate typical
  * usage patterns.
  *
  * \section continuity Continuity check with a power modulus
  * \snippet test_continuity.cpp continuity_identity
  *
  * \section differentiability Differentiability check for a quadratic function
  * \snippet test_differentiability.cpp differentiability_quadratic_at_half
  *
  * \section dec_dsquare_zero Discrete Exterior Calculus: d∘d = 0
  * \snippet discrete_forms_test.cpp dsquare_zero_for_0form
  */