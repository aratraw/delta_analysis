// include/delta/rational/rational_fwd.h
#pragma once

#include <cstddef>
#include <memory>
#include <variant>
#include <vector>

namespace delta {

    class Rational;

} // namespace delta

namespace delta::internal {

    // Forward declarations of internal storage types
    struct SmallStorage;
    struct BigStorage;
    struct LazyNode;

    // Storage variant – forward declaration (full definition in storage.h)
    using Storage = std::variant<SmallStorage, BigStorage, std::shared_ptr<LazyNode>>;

    // Interval class (defined in interval.h)
    class Interval;

    // Global context variables (declared in context.h)
    extern thread_local bool global_eager_mode;
    extern thread_local Rational default_eps_value;

    // Helper functions (will be defined in evaluation.h, simplify.h, etc.)
    Rational evaluate(const Rational& r);
    Rational simplify(const Rational& r);

} // namespace delta::internal