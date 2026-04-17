// rational_fwd.h
#pragma once

#include <cstddef>

namespace delta {
    class Rational;
    class LazyRational;
}

namespace delta::internal {
    struct SmallStorage;
    struct BigStorage;
    class Interval;
}