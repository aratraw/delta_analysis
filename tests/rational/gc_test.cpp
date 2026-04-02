// tests/rational/gc_test.cpp
#include <gtest/gtest.h>
#include <vector>
#include "delta/core/rational.h"
#include "delta/rational/node_pool.h"
#include "test_utils.h"

namespace delta::testing {

    class GarbageCollectionTest : public RationalTest {
    protected:
        size_t occupied_slots() const {
            size_t cnt = 0;
            for (size_t i = 0; i < internal::pool.nodes.size(); ++i) {
                const auto& node = internal::pool.nodes[i];
                if (!(node.value_idx == -1 && node.child0 == -1 && node.child1 == -1))
                    ++cnt;
            }
            return cnt;
        }

        int refcount(int idx) const {
            return internal::pool.refcount[idx];
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
        set_eager_mode(false);
        EXPECT_EQ(internal::pool.max_size, 100);

        Rational sum = 0_r.lazy();
        for (int i = 0; i < 150; ++i) {
            sum = sum + 1_r;
        }
        EXPECT_EQ(internal::pool.nodes.size(), 100);
        EXPECT_LT(internal::pool.next_free_index, 100);

        Rational result = sum.eval();
        EXPECT_EQ(result, 150_r);
    }

    // -------------------------------------------------------------------------
    // 2. Сохранение корней (корневые узлы превращаются в константы)
    // -------------------------------------------------------------------------
    TEST_F(GarbageCollectionTest, RootPreservation) {
        reset_pool_with_size(200);
        set_eager_mode(false);

        Rational root1 = "1/2"_r.lazy();
        Rational root2 = delta::sqrt(2_r);
        Rational root3 = root1 + root2;
        int idx1 = root1.root_index();
        int idx2 = root2.root_index();
        int idx3 = root3.root_index();

        for (int i = 0; i < 300; ++i) {
            Rational tmp = Rational(i).lazy() + Rational(i + 1).lazy();
        }

        EXPECT_EQ(root1.eval(), "1/2"_r);
        EXPECT_RATIONAL_NEAR(root2.eval(), delta::sqrt(2_r).eval(), default_eps());
        EXPECT_RATIONAL_NEAR(root3.eval(), (Rational("1/2"_r) + delta::sqrt(2_r)).eval(), default_eps());

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
        set_eager_mode(false);

        Rational a = "1/3"_r.lazy();
        Rational b = delta::exp(1_r);
        int idx_a = a.root_index();
        int idx_b = b.root_index();

        for (int i = 0; i < 200; ++i) {
            Rational tmp = Rational(i).lazy() * Rational(i + 1).lazy();
        }

        EXPECT_EQ(a.root_index(), idx_a);
        EXPECT_EQ(b.root_index(), idx_b);
        EXPECT_EQ(a.eval(), "1/3"_r);
        EXPECT_RATIONAL_NEAR(b.eval(), delta::exp(1_r).eval(), default_eps());
    }

    // -------------------------------------------------------------------------
    // 4. Принудительный вызов GC
    // -------------------------------------------------------------------------
    TEST_F(GarbageCollectionTest, ForceGC) {
        reset_pool_with_size(100);
        set_eager_mode(false);

        Rational r1 = "2/3"_r.lazy();
        Rational r2 = r1 * r1;
        int idx1 = r1.root_index();
        int idx2 = r2.root_index();

        for (int i = 0; i < 50; ++i) {
            Rational tmp = Rational(i).lazy();
        }

        size_t old_next = internal::pool.next_free_index;
        size_t old_occupied = occupied_slots();

        internal::force_garbage_collect();

        EXPECT_LE(internal::pool.next_free_index, old_next);
        EXPECT_LE(occupied_slots(), old_occupied);

        EXPECT_EQ(r1.eval(), "2/3"_r);
        EXPECT_EQ(r2.eval(), "4/9"_r);
        EXPECT_EQ(internal::pool.nodes[idx1].op, internal::LazyOp::CONST);
        EXPECT_EQ(internal::pool.nodes[idx2].op, internal::LazyOp::CONST);
    }

    // -------------------------------------------------------------------------
    // 5. Управление счётчиком ссылок (refcount)
    // -------------------------------------------------------------------------
    TEST_F(GarbageCollectionTest, RefcountManagement) {
        reset_pool_with_size(1000);
        set_eager_mode(false);

        // Создание узла
        Rational a = "5"_r.lazy();
        int idx = a.root_index();
        EXPECT_EQ(refcount(idx), 1);

        // Копирование увеличивает счётчик
        Rational b = a;
        EXPECT_EQ(refcount(idx), 2);

        // Перемещение не меняет счётчик (передача владения)
        Rational c = std::move(a);
        EXPECT_EQ(refcount(idx), 2);
        EXPECT_FALSE(a.is_lazy());

        // Присваивание копированием: левая часть была immediate
        Rational d = 0_r;
        d = b;
        EXPECT_EQ(refcount(idx), 3);  // новая ссылка

        // Локальная копия
        {
            Rational e = b;
            EXPECT_EQ(refcount(idx), 4);
        }
        EXPECT_EQ(refcount(idx), 3);

        // Переключение с другого узла
        Rational f = "10"_r.lazy();
        int idx_f = f.root_index();
        EXPECT_EQ(refcount(idx_f), 1);

        f = b;  // f переключается с idx_f на idx
        EXPECT_EQ(refcount(idx_f), 0);
        EXPECT_EQ(refcount(idx), 4);

        // GC должен удалить мёртвый узел idx_f
        internal::force_garbage_collect();
        const auto& node = internal::pool.nodes[idx_f];
        bool is_empty = (node.value_idx == -1 && node.child0 == -1 && node.child1 == -1);
        EXPECT_TRUE(is_empty);
    }

    // -------------------------------------------------------------------------
    // 6. Плотная упаковка после GC (все занятые слоты в начале)
    // -------------------------------------------------------------------------
    TEST_F(GarbageCollectionTest, CompactnessAfterGC) {
        reset_pool_with_size(200);
        set_eager_mode(false);

        std::vector<Rational> roots;
        for (int i = 0; i < 30; ++i) {
            roots.push_back(Rational(i).lazy());
        }
        std::vector<int> indices;
        for (const auto& r : roots) indices.push_back(r.root_index());

        for (int i = 0; i < 150; ++i) {
            Rational tmp = Rational(i + 100).lazy() + Rational(i + 101).lazy();
        }

        internal::force_garbage_collect();

        size_t nfi = internal::pool.next_free_index;
        for (size_t i = 0; i < internal::pool.nodes.size(); ++i) {
            bool occupied = !(internal::pool.nodes[i].value_idx == -1 &&
                internal::pool.nodes[i].child0 == -1 &&
                internal::pool.nodes[i].child1 == -1);
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
    // 7. Метод immediate() после GC
    // -------------------------------------------------------------------------
    TEST_F(GarbageCollectionTest, ImmediateAfterGC) {
        reset_pool_with_size(100);
        set_eager_mode(false);

        Rational a = delta::pi();
        int idx = a.root_index();
        for (int i = 0; i < 150; ++i) {
            Rational tmp = Rational(i).lazy();
        }
        EXPECT_EQ(internal::pool.nodes[idx].op, internal::LazyOp::CONST);

        Rational b = a.immediate();
        EXPECT_FALSE(b.is_lazy());
        EXPECT_EQ(b, a.eval());

        Rational c = b.immediate();
        EXPECT_FALSE(c.is_lazy());
        EXPECT_EQ(c, b);
    }

    // -------------------------------------------------------------------------
    // 8. Стресс-тест с несколькими GC
    // -------------------------------------------------------------------------
    TEST_F(GarbageCollectionTest, MultipleGCStress) {
        reset_pool_with_size(500);
        set_eager_mode(false);

        const int NUM_ROOTS = 100;
        std::vector<Rational> roots;
        for (int i = 0; i < NUM_ROOTS; ++i) {
            roots.push_back(Rational(i + 1).lazy());
        }

        for (int cycle = 0; cycle < 5; ++cycle) {
            for (int i = 0; i < 600; ++i) {
                Rational tmp = Rational(i).lazy() * Rational(i + 1).lazy();
            }
            for (int j = 0; j < NUM_ROOTS; ++j) {
                EXPECT_EQ(roots[j].eval(), Rational(j + 1));
            }
        }
    }

    // -------------------------------------------------------------------------
    // 9. GC при отсутствии корней (пул должен полностью очиститься)
    // -------------------------------------------------------------------------
    TEST_F(GarbageCollectionTest, EmptyPoolGC) {
        reset_pool_with_size(100);
        set_eager_mode(false);

        // Создаём временные узлы, но не сохраняем корни
        for (int i = 0; i < 150; ++i) {
            Rational tmp = Rational(i).lazy();
        }
        // После 100 выделений next_free_index == 100, затем GC сбрасывает его в 0,
        // и следующие 50 выделений заполняют слоты 0..49, next_free_index = 50.
        EXPECT_EQ(internal::pool.next_free_index, 50);

        // Принудительная сборка мусора должна очистить все мёртвые узлы
        internal::force_garbage_collect();

        // После GC все слоты должны быть пустыми
        EXPECT_EQ(internal::pool.next_free_index, 0);
        for (size_t i = 0; i < internal::pool.nodes.size(); ++i) {
            const auto& node = internal::pool.nodes[i];
            EXPECT_TRUE(node.value_idx == -1 && node.child0 == -1 && node.child1 == -1)
                << "Slot " << i << " not empty";
        }
    }

    // -------------------------------------------------------------------------
    // 10. Исчерпание пула корнями (исключение)
    // -------------------------------------------------------------------------
    TEST_F(GarbageCollectionTest, ExhaustedByRoots) {
        reset_pool_with_size(10);
        set_eager_mode(false);

        std::vector<Rational> roots;
        // Заполняем все слоты корнями (создаём 10 разных констант)
        for (int i = 0; i < 10; ++i) {
            roots.push_back(Rational(i).lazy());
        }
        // Пул полностью занят корнями, next_free_index == max_size == 10
        EXPECT_EQ(internal::pool.next_free_index, 10);

        // Попытка создать ещё один ленивый узел должна бросить исключение
        EXPECT_THROW({
            Rational extra = 42_r.lazy();
            }, std::runtime_error);
    }

    // -------------------------------------------------------------------------
    // 11. Самоприсваивание не должно менять refcount
    // -------------------------------------------------------------------------
    TEST_F(GarbageCollectionTest, SelfAssignmentNoRefcountChange) {
        reset_pool_with_size(100);
        set_eager_mode(false);

        Rational a = "5"_r.lazy();
        int idx = a.root_index();
        EXPECT_EQ(refcount(idx), 1);

        a = a;  // самоприсваивание
        EXPECT_EQ(refcount(idx), 1);  // не должно измениться
    }

    // -------------------------------------------------------------------------
    // 12. immediate() на сложном ленивом дереве
    // -------------------------------------------------------------------------
    TEST_F(GarbageCollectionTest, ImmediateOnComplexTree) {
        set_eager_mode(false);
        Rational expr = delta::sqrt(2_r) + delta::exp(1_r);
        Rational imm = expr.immediate();
        EXPECT_FALSE(imm.is_lazy());
        EXPECT_EQ(imm, expr.eval());
    }

} // namespace delta::testing