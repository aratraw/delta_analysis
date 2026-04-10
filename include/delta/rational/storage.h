// storage.h
#pragma once

#include "rational_fwd.h"
#include "utils.h"

#include <absl/numeric/int128.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/number.hpp>
#include <boost/multiprecision/rational_adaptor.hpp>

#include <cassert>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace delta::internal {

    // ============================================================================
    // ARCHITECTURAL NOTE: WHY WE USE std::unique_ptr FOR BigStorage
    // ============================================================================
    // The Value type must fit into a single cache line (64 bytes) for optimal
    // performance. After moving the `reduced` flag from SmallStorage into Value,
    // sizeof(Value) = 48 bytes, which is well within the cache line.
    // This design eliminates bit‑twiddling bugs (the previous approach stored
    // the flag in the high bit of the denominator, causing loss of precision
    // for large denominators with bit 127 set).
    //
    // Important invariants:
    //   - For ValueType::Big, big_ptr is never null (except after move).
    //   - Copying a Value creates a deep copy of the underlying BigRationalType.
    //   - Moving a Value transfers ownership (no deep copy).
    //   - Equality comparisons are value‑based, not pointer‑based.
    //   - Hash functions also operate on values.
    // ============================================================================

    // ============================================================================
    // SmallStorage – 128‑bit rational, NO internal flag. Normalization flag is
    // stored in the Value structure.
    // ============================================================================
    struct SmallStorage {
        absl::int128 num;
        absl::uint128 den;   // Always the actual denominator (never has a flag bit)

        constexpr SmallStorage() noexcept : num(0), den(1) {}
        constexpr explicit SmallStorage(absl::int128 n) noexcept : num(n), den(1) {}
        constexpr SmallStorage(absl::int128 n, absl::uint128 d) noexcept : num(n), den(d) {}

        // Normalize in‑place. The caller must provide the reduced flag.
        void normalize(bool& reduced) {
            if (reduced) return;
            if (den == 0) throw std::domain_error("SmallStorage: denominator cannot be zero");
            if (num == 0) {
                den = 1;
                reduced = true;
                return;
            }
            absl::uint128 raw_den = den;
            if (raw_den < 0) {   // Should never happen for unsigned, but kept for safety
                raw_den = -raw_den;
                num = -num;
            }
            absl::uint128 abs_num = num < 0 ? static_cast<absl::uint128>(-num) : static_cast<absl::uint128>(num);
            absl::uint128 g = binary_gcd(abs_num, raw_den);
            if (g > 1) {
                abs_num /= g;
                raw_den /= g;
                num = (num < 0) ? -static_cast<absl::int128>(abs_num) : static_cast<absl::int128>(abs_num);
                den = raw_den;
            }
            reduced = true;
        }

        // No is_normalized() method – flag is outside.
    };

    // ============================================================================
    // BigRationalType – без изменений
    // ============================================================================
    using BigRationalType = boost::multiprecision::number<
        boost::multiprecision::rational_adaptor<
        boost::multiprecision::cpp_int_backend<>
        >,
        boost::multiprecision::et_off
    >;

    // ============================================================================
    // BigStorage – contains unique_ptr<BigRationalType>
    // ============================================================================
    struct BigStorage {
        std::unique_ptr<BigRationalType> ptr;

        BigStorage() : ptr(std::make_unique<BigRationalType>(0)) {}
        explicit BigStorage(const dumb_int& n) : ptr(std::make_unique<BigRationalType>(n)) {}
        BigStorage(const dumb_int& n, const dumb_int& d) {
            if (d == 0) throw std::domain_error("BigStorage: denominator cannot be zero");
            ptr = std::make_unique<BigRationalType>(n, d);
        }
        explicit BigStorage(const BigRationalType& v) : ptr(std::make_unique<BigRationalType>(v)) {}
        explicit BigStorage(BigRationalType&& v) : ptr(std::make_unique<BigRationalType>(std::move(v))) {}

        BigStorage(const BigStorage& other) : ptr(std::make_unique<BigRationalType>(*other.ptr)) {}
        BigStorage(BigStorage&& other) noexcept : ptr(std::move(other.ptr)) {}

        BigStorage& operator=(const BigStorage& other) {
            if (this != &other) {
                ptr = std::make_unique<BigRationalType>(*other.ptr);
            }
            return *this;
        }
        BigStorage& operator=(BigStorage&& other) noexcept {
            if (this != &other) {
                ptr = std::move(other.ptr);
            }
            return *this;
        }

        bool is_zero() const {
            assert(ptr);
            return ptr->backend().num().size() == 0;
        }
        bool is_negative() const {
            assert(ptr);
            return ptr->backend().num().sign();
        }
        bool is_positive() const {
            assert(ptr);
            const auto& n = ptr->backend().num();
            return (n.size() > 0) && !n.sign();
        }
        bool is_one() const {
            assert(ptr);
            const auto& n = ptr->backend().num();
            const auto& d = ptr->backend().denom();
            auto check_one = [](const auto& val) {
                return val.size() == 1 && val.limbs()[0] == 1 && !val.sign();
                };
            return check_one(n) && check_one(d);
        }

        [[nodiscard]] dumb_int numerator() const {
            assert(ptr);
            return boost::multiprecision::numerator(*ptr);
        }
        [[nodiscard]] dumb_int denominator() const {
            assert(ptr);
            return boost::multiprecision::denominator(*ptr);
        }

        [[nodiscard]] bool is_normalized() const noexcept { return true; }
        void normalize() noexcept {}
    };

    // ============================================================================
    // Value – ручной tagged union, now with explicit `small_reduced` flag.
    // ============================================================================
    enum class ValueType : uint8_t { Small, Big, Lazy };

    union ValueStorage {
        SmallStorage small;
        BigStorage big;
        int lazy;
        ValueStorage() {}
        ~ValueStorage() {}
    };

    struct Value {
        ValueType tag;
        bool small_reduced;        // only valid when tag == Small
        ValueStorage storage;

        // Constructors
        Value() : tag(ValueType::Small), small_reduced(true) {
            new (&storage.small) SmallStorage();
        }
        Value(const SmallStorage& s, bool reduced = false) : tag(ValueType::Small), small_reduced(reduced) {
            new (&storage.small) SmallStorage(s);
        }
        Value(SmallStorage&& s, bool reduced = false) : tag(ValueType::Small), small_reduced(reduced) {
            new (&storage.small) SmallStorage(std::move(s));
        }
        Value(const BigStorage& b) : tag(ValueType::Big), small_reduced(false) {
            new (&storage.big) BigStorage(b);
        }
        Value(BigStorage&& b) : tag(ValueType::Big), small_reduced(false) {
            new (&storage.big) BigStorage(std::move(b));
        }
        explicit Value(int idx) : tag(ValueType::Lazy), small_reduced(false) {
            storage.lazy = idx;
        }

        // Copy
        Value(const Value& other) : tag(other.tag), small_reduced(other.small_reduced) {
            switch (tag) {
            case ValueType::Small: new (&storage.small) SmallStorage(other.storage.small); break;
            case ValueType::Big:   new (&storage.big)   BigStorage(other.storage.big);     break;
            case ValueType::Lazy:  storage.lazy = other.storage.lazy; break;
            }
        }

        // Move
        Value(Value&& other) noexcept : tag(other.tag), small_reduced(other.small_reduced) {
            switch (tag) {
            case ValueType::Small: new (&storage.small) SmallStorage(std::move(other.storage.small)); break;
            case ValueType::Big:   new (&storage.big)   BigStorage(std::move(other.storage.big));     break;
            case ValueType::Lazy:  storage.lazy = other.storage.lazy; break;
            }
            // Leave other in a valid but unspecified state (small_reduced stays)
        }

        ~Value() {
            switch (tag) {
            case ValueType::Small: storage.small.~SmallStorage(); break;
            case ValueType::Big:   storage.big.~BigStorage();     break;
            case ValueType::Lazy:  break;
            }
        }

        Value& operator=(const Value& other) {
            if (this == &other) return *this;
            this->~Value();
            new (this) Value(other);
            return *this;
        }
        Value& operator=(Value&& other) noexcept {
            if (this == &other) return *this;
            this->~Value();
            new (this) Value(std::move(other));
            return *this;
        }

        // Convenience helpers
        bool is_normalized() const {
            if (tag != ValueType::Small) return true;
            return small_reduced;
        }
        void normalize() {
            if (tag == ValueType::Small && !small_reduced) {
                storage.small.normalize(small_reduced);
            }
        }
        absl::uint128 denominator() const {
            if (tag != ValueType::Small) return 1; // For Big/Lazy, denominator is not defined this way
            return storage.small.den;
        }
    };

    static_assert(sizeof(Value) == 40, "Value size must be 40 bytes (fits in cache line, stores BigStorage by pointer, allows additional payload)");

    // ============================================================================
    // normalize_to_dumb_int – adapted to use the flag from Value
    // ============================================================================
    inline std::pair<dumb_int, dumb_int> normalize_to_dumb_int(const Value& v) {
        if (v.tag == ValueType::Small) {
            SmallStorage s = v.storage.small;
            if (!v.small_reduced) {
                bool dummy = false;
                s.normalize(dummy);
            }
            return { to_dumb_int(s.num), to_dumb_int(s.den) };
        }
        else if (v.tag == ValueType::Big) {
            const auto& b = v.storage.big;
            return { b.numerator(), b.denominator() };
        }
        else {
            throw std::invalid_argument("normalize_to_dumb_int: cannot normalize lazy index");
        }
    }

    // ============================================================================
    // Optimized comparisons – cross multiplication, now using raw denominator.
    // ============================================================================
    inline bool operator==(const Value& a, const Value& b) {
        if (a.tag == ValueType::Small && b.tag == ValueType::Small) {
            const auto& sa = a.storage.small;
            const auto& sb = b.storage.small;
            absl::int128 left_num = sa.num;
            absl::uint128 left_den = sa.den;
            absl::int128 right_num = sb.num;
            absl::uint128 right_den = sb.den;

            // If one is not normalized, we need to normalize? Actually cross multiplication works regardless,
            // because we compare a.num * b.den vs b.num * a.den. The flag doesn't affect equality.
            // But we must be careful about overflow.
            if (would_overflow_mul(left_num, static_cast<absl::int128>(right_den)) ||
                would_overflow_mul(right_num, static_cast<absl::int128>(left_den))) {
                dumb_int l = to_dumb_int(left_num) * to_dumb_int(right_den);
                dumb_int r = to_dumb_int(right_num) * to_dumb_int(left_den);
                return l == r;
            }
            return left_num * static_cast<absl::int128>(right_den) ==
                right_num * static_cast<absl::int128>(left_den);
        }
        auto [anum, aden] = normalize_to_dumb_int(a);
        auto [bnum, bden] = normalize_to_dumb_int(b);
        return anum * bden == bnum * aden;
    }

    inline bool operator<(const Value& a, const Value& b) {
        if (a.tag == ValueType::Small && b.tag == ValueType::Small) {
            const auto& sa = a.storage.small;
            const auto& sb = b.storage.small;
            absl::int128 left_num = sa.num;
            absl::uint128 left_den = sa.den;
            absl::int128 right_num = sb.num;
            absl::uint128 right_den = sb.den;
            if (would_overflow_mul(left_num, static_cast<absl::int128>(right_den)) ||
                would_overflow_mul(right_num, static_cast<absl::int128>(left_den))) {
                dumb_int l = to_dumb_int(left_num) * to_dumb_int(right_den);
                dumb_int r = to_dumb_int(right_num) * to_dumb_int(left_den);
                return l < r;
            }
            return left_num * static_cast<absl::int128>(right_den) <
                right_num * static_cast<absl::int128>(left_den);
        }
        auto [anum, aden] = normalize_to_dumb_int(a);
        auto [bnum, bden] = normalize_to_dumb_int(b);
        return anum * bden < bnum * aden;
    }

    inline bool operator>(const Value& a, const Value& b) { return b < a; }
    inline bool operator<=(const Value& a, const Value& b) { return !(b < a); }
    inline bool operator>=(const Value& a, const Value& b) { return !(a < b); }

    // ============================================================================
    // String conversion (debug only)
    // ============================================================================
    inline std::string to_string(const SmallStorage& s) {
        // We need a normalized copy for string conversion.
        SmallStorage norm = s;
        bool red = false;
        norm.normalize(red);
        if (norm.den == 1) return int128_to_string(norm.num);
        return int128_to_string(norm.num) + "/" + uint128_to_string(norm.den);
    }

    inline std::string to_string(const BigStorage& b) {
        if (b.denominator() == 1) return b.numerator().str();
        return b.numerator().str() + "/" + b.denominator().str();
    }

    inline std::string to_string(const Value& v) {
        if (v.tag == ValueType::Small)
            return to_string(v.storage.small);
        else if (v.tag == ValueType::Big)
            return to_string(v.storage.big);
        else
            return "lazy(" + std::to_string(v.storage.lazy) + ")";
    }

} // namespace delta::internal