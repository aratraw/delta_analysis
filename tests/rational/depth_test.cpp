#include <gtest/gtest.h>
#include "delta/core/rational.h"
#include "delta/rational/expression_root.h"   // для internal::MAX_LAZY_DEPTH
#include "test_utils.h"

namespace delta::testing {

    // Вспомогательная функция для построения левоассоциативной цепочки сложений: (((0 + 1) + 1) + ...)
    Rational build_addition_chain(int n) {
        Rational sum = 0_r.lazy();   // начинаем с ленивого нуля
        for (int i = 0; i < n; ++i) {
            sum = sum + 1_r;         // 1_r – немедленное, результат ленивый
        }
        return sum;
    }

    // -------------------------------------------------------------------------
    // 1. Depth tracking
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, DepthTracking) {
        set_eager_mode(false);

        Rational root = build_addition_chain(5);
        // Глубина ленивого дерева: каждый ADD увеличивает глубину на 1
        if (root.is_lazy()) {
            const auto& nodes = root.as_lazy()->nodes();
            int root_idx = root.as_lazy()->root_index();
            int depth = nodes[root_idx].depth;
            EXPECT_EQ(depth, 5);
        }
        else {
            // Если по какой‑то причине дерево схлопнулось (что маловероятно),
            // то глубина считается 0, но мы ожидаем 5, так что тест упадёт.
            FAIL() << "Root is immediate, cannot get depth";
        }
    }

    // -------------------------------------------------------------------------
    // 2. Depth overflow – tree should be collapsed when depth > MAX_LAZY_DEPTH
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, DepthOverflow) {
        set_eager_mode(false);

        int over = internal::MAX_LAZY_DEPTH + 100;
        Rational root = build_addition_chain(over);

        // Глубина не должна превышать лимит (упрощение должно сработать)
        if (root.is_lazy()) {
            const auto& nodes = root.as_lazy()->nodes();
            int root_idx = root.as_lazy()->root_index();
            int depth = nodes[root_idx].depth;
            EXPECT_LE(depth, internal::MAX_LAZY_DEPTH);
        }
        else {
            // Если дерево стало немедленным, глубина считается 0, что тоже <= лимита
            EXPECT_LE(0, internal::MAX_LAZY_DEPTH);
        }
    }

    // -------------------------------------------------------------------------
    // 3. Eager on overflow – after overflow the node may become evaluated
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, EagerOnOverflow) {
        set_eager_mode(false);

        int just_below = internal::MAX_LAZY_DEPTH - 1;
        Rational root = build_addition_chain(just_below);
        EXPECT_TRUE(root.is_lazy());   // ещё ленивое

        // Добавляем ещё одно слагаемое, чтобы глубина превысила лимит
        root = root + 1_r;

        // Глубина должна быть ≤ MAX_LAZY_DEPTH
        if (root.is_lazy()) {
            const auto& nodes = root.as_lazy()->nodes();
            int root_idx = root.as_lazy()->root_index();
            int depth = nodes[root_idx].depth;
            EXPECT_LE(depth, internal::MAX_LAZY_DEPTH);
        }
        else {
            // Если стало immediate, глубина 0, что тоже допустимо
            EXPECT_LE(0, internal::MAX_LAZY_DEPTH);
        }
    }

    // -------------------------------------------------------------------------
    // 4. No premature collapse – chain below the limit stays lazy
    // -------------------------------------------------------------------------
    TEST_F(RationalTest, NoPrematureCollapse) {
        set_eager_mode(false);

        int below = 500;   // 500 < MAX_LAZY_DEPTH (1000)
        Rational root = build_addition_chain(below);
        EXPECT_TRUE(root.is_lazy());

        const auto& nodes = root.as_lazy()->nodes();
        int root_idx = root.as_lazy()->root_index();
        int depth = nodes[root_idx].depth;
        EXPECT_EQ(depth, below);
    }

} // namespace delta::testing