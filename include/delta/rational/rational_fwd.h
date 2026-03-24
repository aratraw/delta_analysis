#pragma once

#include <cstddef>
#include <memory>
#include <variant>

namespace delta {
    class Rational;
}

namespace delta::internal {
    struct SmallStorage;
    struct BigStorage;
    class ExpressionRoot;
    class Interval;

    using Value = std::variant<SmallStorage, BigStorage>;
}