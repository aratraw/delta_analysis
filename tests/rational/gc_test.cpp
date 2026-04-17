// tests/rational/gc_test.cpp
#pragma once
#include <gtest/gtest.h>
#include <vector>
#include "delta/core/rational.h"
#include "delta/rational/node_pool.h"
#include "test_utils.h"
#include "lazy_rational_test_fixture.h"   // добавляем фикстуру

namespace delta::testing {

    // Наследуем от LazyRationalTestFixture, чтобы получить доступ к методам интроспекции
    class GarbageCollectionTest : public delta::test::LazyRationalTestFixture {
    protected:
        size_t occupied_slots() const {
            size_t cnt = 0;
            for (size_t i = 0; i < internal::pool.nodes.size(); ++i) {
                const auto& node = internal::pool.nodes[i];
                if (node.op == internal::LazyOp::SUM) {
                    if (node.children && !node.children->empty()) ++cnt;
                }
                else {
                    if (!(node.value_idx == -1 && node.child0 == -1 && node.child1 == -1))
                        ++cnt;
                }
            }
            return cnt;
        }

        void reset_pool_with_size(size_t new_size) {
            internal::reset_pool();
            internal::set_pool_max_size(new_size);
        }
    };

    // -------------------------------------------------------------------------
    // 1. Проверка, что GC запускается при заполнении пула
    // -------------------------------------------------------------------------
    TEST_F(GarbageCollectionTest, PoolSizeLimit) {
        reset_pool_with_size(100);
        LazyRational sum;
        for (int i = 0; i < 150; ++i) {
            sum += Rational(1);
        }
        // Пул ограничен, GC должен был сработать
        EXPECT_LE(internal::pool.nodes.size(), 100);
        Rational result = sum.eval();
        EXPECT_EQ(result, 150_r);
    }

    // -------------------------------------------------------------------------
    // 2. Сохранение корней (корневые узлы превращаются в константы)
    // -------------------------------------------------------------------------
    TEST_F(GarbageCollectionTest, RootPreservation) {
        reset_pool_with_size(200);

        LazyRational root1 = Rational(1, 2).as_lazy();
        LazyRational root2 = delta::lazy_sqrt(Rational(2).as_lazy());
        LazyRational root3 = root1.clone() + root2.clone();
        root1.simplify_inplace();
        root2.simplify_inplace();
        root3.simplify_inplace();

        // Используем метод фикстуры для доступа к clean_index_
        int idx1 = clean_index(root1);
        int idx2 = clean_index(root2);
        int idx3 = clean_index(root3);

        for (int i = 0; i < 300; ++i) {
            LazyRational tmp = Rational(i).as_lazy() + Rational(i + 1).as_lazy();
            tmp.simplify_inplace();
        }

        EXPECT_EQ(root1.eval(), Rational(1, 2));
        EXPECT_RATIONAL_NEAR(root2.eval(), delta::sqrt(2_r), default_eps());
        EXPECT_RATIONAL_NEAR(root3.eval(), (Rational(1, 2) + delta::sqrt(2_r)), default_eps());

        const auto& node1 = internal::pool.nodes[idx1];
        const auto& node2 = internal::pool.nodes[idx2];
        const auto& node3 = internal::pool.nodes[idx3];
        EXPECT_EQ(node1.op, internal::LazyOp::CONST);
        EXPECT_EQ(node2.op, internal::LazyOp::CONST);
        EXPECT_EQ(node3.op, internal::LazyOp::CONST);
    }

    // -------------------------------------------------------------------------
    // 3. Инвариантность индексов корней после GC
    // -------------------------------------------------------------------------
    TEST_F(GarbageCollectionTest, IndexInvariance) {
        reset_pool_with_size(150);

        LazyRational a = Rational(1, 3).as_lazy();
        LazyRational b = delta::lazy_exp(Rational(1).as_lazy());
        a.simplify_inplace();
        b.simplify_inplace();
        int idx_a = clean_index(a);
        int idx_b = clean_index(b);

        for (int i = 0; i < 200; ++i) {
            LazyRational tmp = Rational(i).as_lazy() * Rational(i + 1).as_lazy();
            tmp.simplify_inplace();
        }

        EXPECT_EQ(clean_index(a), idx_a);
        EXPECT_EQ(clean_index(b), idx_b);
        EXPECT_EQ(a.eval(), Rational(1, 3));
        EXPECT_RATIONAL_NEAR(b.eval(), delta::exp(1_r), default_eps());
    }

    // -------------------------------------------------------------------------
    // 4. Принудительный вызов GC
    // -------------------------------------------------------------------------
    TEST_F(GarbageCollectionTest, ForceGC) {
        reset_pool_with_size(100);

        LazyRational r1 = Rational(2, 3).as_lazy();
        LazyRational r2 = r1.clone() * r1.clone();
        r1.simplify_inplace();
        r2.simplify_inplace();
        int idx1 = clean_index(r1);
        int idx2 = clean_index(r2);

        for (int i = 0; i < 50; ++i) {
            LazyRational tmp = Rational(i).as_lazy();
            tmp.simplify_inplace();
        }

        size_t old_next = internal::pool.next_free_index;
        size_t old_occupied = occupied_slots();

        internal::force_garbage_collect();

        EXPECT_LE(internal::pool.next_free_index, old_next);
        EXPECT_LE(occupied_slots(), old_occupied);

        EXPECT_EQ(r1.eval(), Rational(2, 3));
        EXPECT_EQ(r2.eval(), Rational(4, 9));
        EXPECT_EQ(internal::pool.nodes[idx1].op, internal::LazyOp::CONST);
        EXPECT_EQ(internal::pool.nodes[idx2].op, internal::LazyOp::CONST);
    }

    // -------------------------------------------------------------------------
    // 5. Управление счётчиком ссылок (refcount)
    // -------------------------------------------------------------------------
    TEST_F(GarbageCollectionTest, RefcountManagement) {
        reset_pool_with_size(1000);

        LazyRational a = Rational(5).as_lazy();
        a.simplify_inplace();
        int idx = clean_index(a);
        EXPECT_EQ(refcount(idx), 1);    // refcount унаследован от фикстуры

        LazyRational b = a.clone();
        EXPECT_EQ(refcount(idx), 2);

        LazyRational c = std::move(a);
        EXPECT_EQ(refcount(idx), 2);

        LazyRational d = Rational(0).as_lazy();
        d = b.clone();
        EXPECT_EQ(refcount(idx), 3);

        {
            LazyRational e = b.clone();
            EXPECT_EQ(refcount(idx), 4);
        }
        EXPECT_EQ(refcount(idx), 3);
    }

    // -------------------------------------------------------------------------
    // 6. Плотная упаковка после GC
    // -------------------------------------------------------------------------
    TEST_F(GarbageCollectionTest, CompactnessAfterGC) {
        reset_pool_with_size(200);

        std::vector<LazyRational> roots;
        for (int i = 0; i < 30; ++i) {
            roots.push_back(Rational(i).as_lazy());
            roots.back().simplify_inplace();
        }
        std::vector<int> indices;
        for (const auto& r : roots) indices.push_back(clean_index(r));

        for (int i = 0; i < 150; ++i) {
            LazyRational tmp = Rational(i + 100).as_lazy() + Rational(i + 101).as_lazy();
            tmp.simplify_inplace();
        }

        internal::force_garbage_collect();

        size_t nfi = internal::pool.next_free_index;
        for (size_t i = 0; i < internal::pool.nodes.size(); ++i) {
            bool occupied = false;
            const auto& node = internal::pool.nodes[i];
            if (node.op == internal::LazyOp::SUM) {
                occupied = node.children && !node.children->empty();
            }
            else {
                occupied = !(node.value_idx == -1 && node.child0 == -1 && node.child1 == -1);
            }
            if (occupied) {
                EXPECT_LT(i, nfi);
            }
        }
        for (int idx : indices) {
            EXPECT_LT(idx, static_cast<int>(nfi));
            EXPECT_EQ(internal::pool.nodes[idx].op, internal::LazyOp::CONST);
        }
    }

    // -------------------------------------------------------------------------
    // 7. Исчерпание пула корнями (исключение)
    // -------------------------------------------------------------------------
    TEST_F(GarbageCollectionTest, ExhaustedByRoots) {
        reset_pool_with_size(10);
        std::vector<LazyRational> roots;
        for (int i = 0; i < 10; ++i) {
            roots.push_back(Rational(i).as_lazy());
            roots.back().simplify_inplace();
        }
        EXPECT_EQ(internal::pool.next_free_index, 10);
        EXPECT_THROW({
            LazyRational extra = Rational(42).as_lazy();
            extra.simplify_inplace();
            }, std::runtime_error);
    }

    // -------------------------------------------------------------------------
    // 8. GC при отсутствии корней
    // -------------------------------------------------------------------------
    TEST_F(GarbageCollectionTest, EmptyPoolGC) {
        reset_pool_with_size(100);
        for (int i = 0; i < 150; ++i) {
            LazyRational tmp = Rational(i).as_lazy();
            tmp.simplify_inplace();   // но tmp уничтожается
        }
        internal::force_garbage_collect();
        EXPECT_EQ(internal::pool.next_free_index, 0);
        for (size_t i = 0; i < internal::pool.nodes.size(); ++i) {
            const auto& node = internal::pool.nodes[i];
            bool occupied = false;
            if (node.op == internal::LazyOp::SUM) {
                occupied = node.children && !node.children->empty();
            }
            else {
                occupied = !(node.value_idx == -1 && node.child0 == -1 && node.child1 == -1);
            }
            EXPECT_FALSE(occupied) << "Slot " << i << " not empty";
        }
    }

} // namespace delta::testing