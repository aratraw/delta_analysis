// include/delta/core/rational.h
#pragma once

// Основной класс Rational и его реализация
#include "delta/rational/rational_class.h"
#include "delta/rational/lazy_rational.h"
// Пользовательские литералы (_r)
#include "delta/rational/literals.h"

// Трансцендентные функции (sqrt, exp, log, sin, cos, acos, pi, e, pow)
#include "delta/rational/transcendentals.h"
#include "delta/rational/context.h"
// Интеграция с Eigen (опционально)
#include "delta/rational/eigen_integration.h"