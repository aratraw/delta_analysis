// include/delta/core/rational.h
#pragma once

// Main public header for delta::Rational.

// Forward declarations and basic types (no full definition required)
#include "delta/rational/rational_fwd.h"
#include "delta/rational/interval.h"
#include "delta/rational/utils.h"
#include "delta/rational/context.h"

// The full definition of Rational (includes storage.h internally)
#include "delta/rational/rational_class.h"

// Additional functionality that depends on the full Rational definition
#include "delta/rational/transcendentals.h"
#include "delta/rational/literals.h"
#include "delta/rational/eigen_integration.h"