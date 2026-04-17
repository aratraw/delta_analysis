// storage.h
#pragma once

#include "rational_fwd.h"
#include "utils.h"

#include <absl/numeric/int128.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/number.hpp>
#include <boost/multiprecision/rational_adaptor.hpp>
#include <absl/hash/hash.h>
#include <cassert>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace delta::internal {

    // ============================================================================
    // ARCHITECTURAL NOTE: WHY WE USE std::unique_ptr FOR BigStorage
    // ============================================================================
    // The Value type must fit into a single cache line (64 bytes) for optimal
    // performance. After moving the `reduced` flag from SmallStorage into Value,
    // sizeof(Value) = 40 bytes, which is well within the cache line.
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
    // stored in the Value structure. Definition must come before try_reduce_to_small.
    // ============================================================================
    struct SmallStorage {
        absl::int128 num;
        absl::uint128 den;   // Always the actual denominator (never has a flag bit)

        constexpr SmallStorage() noexcept : num(0), den(1) {}
        constexpr explicit SmallStorage(absl::int128 n) noexcept : num(n), den(1) {}
        constexpr SmallStorage(absl::int128 n, absl::uint128 d) noexcept : num(n), den(d) {}

        // Fast predicates (no normalisation)
        constexpr bool is_zero() const noexcept { return num == 0; }
        constexpr bool is_one() const noexcept { return num == den; }
        constexpr bool is_positive() const noexcept { return num > 0; }
        constexpr bool is_negative() const noexcept { return num < 0; }

        // Normalize in‑place. The caller must provide the reduced flag.
        // Optimisation for powers of two.
        void normalize(bool& reduced) {
            if (reduced) return;
            if (den == 0) throw std::domain_error("SmallStorage: denominator cannot be zero");
            if (num == 0) {
                den = 1;
                reduced = true;
                return;
            }

            absl::uint128 abs_num = (num < 0) ? static_cast<absl::uint128>(-num) : static_cast<absl::uint128>(num);
            absl::uint128 g;

            // Оптимизация для знаменателей – степеней двойки
            if (is_power_of_two(den)) {
                int shift_den = ctz128(den);
                int shift_num = ctz128(abs_num);
                int shift = std::min(shift_den, shift_num);
                if (shift) {
                    abs_num >>= shift;
                    den >>= shift;
                    // !!! КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: обновляем числитель с учётом знака
                    num = (num < 0) ? -static_cast<absl::int128>(abs_num) : static_cast<absl::int128>(abs_num);
                }
                g = 1; // дальнейшее сокращение не требуется
            }
            else {
                g = binary_gcd(abs_num, den);
                if (g > 1) {
                    abs_num /= g;
                    den /= g;
                    num = (num < 0) ? -static_cast<absl::int128>(abs_num) : static_cast<absl::int128>(abs_num);
                }
            }

            reduced = true;
        }
    };

    // ============================================================================
    // try_reduce_to_small – attempt to reduce a (num, den) pair and fit into SmallStorage
    // ============================================================================
    inline std::optional<SmallStorage> try_reduce_to_small(const dumb_int& num, const dumb_int& den) {
        if (den == 0) throw std::domain_error("denominator zero");
        dumb_int g = boost::multiprecision::gcd(num, den);
        dumb_int num2 = num / g;
        dumb_int den2 = den / g;
        if (den2 < 0) { den2 = -den2; num2 = -num2; }
        if (fits_in_int128(num2) && fits_in_uint128(den2)) {
            return SmallStorage(dumb_int_to_int128(num2), dumb_int_to_uint128(den2));
        }
        return std::nullopt;
    }

    // ============================================================================
    // BigRationalType – unchanged
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
    // Value – manual tagged union, now with explicit `small_reduced` flag.
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
            if (tag != ValueType::Small) return 1;
            return storage.small.den;
        }

        // Fast predicates
        bool is_zero() const {
            if (tag == ValueType::Small) return storage.small.is_zero();
            if (tag == ValueType::Big) return storage.big.is_zero();
            return false;
        }
        bool is_one() const {
            if (tag == ValueType::Small) return storage.small.is_one();
            if (tag == ValueType::Big) return storage.big.is_one();
            return false;
        }
        bool is_positive() const {
            if (tag == ValueType::Small) return storage.small.is_positive();
            if (tag == ValueType::Big) return storage.big.is_positive();
            return false;
        }
        bool is_negative() const {
            if (tag == ValueType::Small) return storage.small.is_negative();
            if (tag == ValueType::Big) return storage.big.is_negative();
            return false;
        }
    };

    static_assert(sizeof(Value) == 40, "Value size must be 40 bytes");

    // ============================================================================
    // normalize_to_dumb_int – NO normalisation for SmallStorage (only raw conversion)
    // ============================================================================
    inline std::pair<dumb_int, dumb_int> normalize_to_dumb_int(const Value& v) {
        if (v.tag == ValueType::Small) {
            const SmallStorage& s = v.storage.small;
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
    // Optimised comparisons – cross multiplication, now using raw denominator.
    // No normalisation is performed.
    // ============================================================================
    inline bool operator==(const Value& a, const Value& b) {
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
    // String conversion (debug only). NOT TO BE USED IN ANY CONVERSION/CALCULATION
    // ============================================================================
    inline std::string to_string(const SmallStorage& s) {
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
    // ----------------------------------------------------------------------------
// Хэширование для SmallStorage (нормализует перед хэшированием)
// ----------------------------------------------------------------------------
    template <typename H>
    H AbslHashValue(H h, const SmallStorage& s) {
        SmallStorage norm = s;
        bool red = false;
        norm.normalize(red);
        return H::combine(std::move(h), norm.num, norm.den);
    }

    // ----------------------------------------------------------------------------
    // Хэширование для BigStorage
    // ----------------------------------------------------------------------------
    template <typename H>
    H AbslHashValue(H h, const BigStorage& b) {
        return H::combine(std::move(h), b.numerator(), b.denominator());
    }

    // ----------------------------------------------------------------------------
    // Хэширование для Value
    // ----------------------------------------------------------------------------
    template <typename H>
    H AbslHashValue(H h, const Value& v) {
        switch (v.tag) {
        case ValueType::Small:
            return AbslHashValue(std::move(h), v.storage.small);
        case ValueType::Big:
            return AbslHashValue(std::move(h), v.storage.big);
        case ValueType::Lazy:
            return H::combine(std::move(h), v.storage.lazy);
        }
        return H::combine(std::move(h), 0);
    }
} // namespace delta::internal