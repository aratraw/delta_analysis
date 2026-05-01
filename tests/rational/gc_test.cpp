// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

//gc_test.cpp
#pragma once
#include <gtest/gtest.h>
#include <vector>
#include "delta/core/rational.h"
#include "delta/rational/node_pool.h"
#include "test_utils.h"
#include "lazy_rational_test_fixture.h"

namespace delta::testing {

    class GarbageCollectionTest : public delta::testing::LazyRationalTestFixture {
    protected:
        void SetUp() override {
            // Очищаем пул и приводим max_size к значению по умолчанию
            internal::reset_pool();
            // Убеждаемся, что max_size установлен в DEFAULT_POOL_MAX_SIZE
            // reset_pool уже создаёт пул с DEFAULT_POOL_MAX_SIZE, поэтому дополнительный вызов не нужен
        }

        void TearDown() override {
            // После каждого теста возвращаем пул в чистое состояние,
            // чтобы не влиять на другие тесты (хотя SetUp сделает это перед следующим)
            internal::reset_pool();
        }

        size_t occupied_slots() const {
            size_t cnt = 0;
            for (size_t i = 0; i < internal::pool.nodes.size(); ++i) {
                const auto& node = internal::pool.nodes[i];
                if (node.op == internal::LazyOp::SUM || node.op == internal::LazyOp::PRODUCT) {
                    if (!node.leaf_values.empty() || !node.children.empty()) ++cnt;
                }
                else if (node.op == internal::LazyOp::CONST) {
                    if (node.value_idx != -1) ++cnt;
                }
                else {
                    // Для унарных/бинарных операций: если есть дети или eps – узел занят
                    if (!node.children.empty() || node.eps_idx != -1) ++cnt;
                }
            }
            return cnt;
        }

        void reset_pool_with_size(size_t new_size) {
            internal::reset_pool();          // сброс к состоянию по умолчанию
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
        EXPECT_LE(internal::pool.nodes.size(), 100);
        Rational result = sum.eval();
        EXPECT_EQ(result, 150_r);
    }
    // -------------------------------------------------------------------------
    // 2. Проверка, что индексы корней сохраняются, всё корректно работает и всё такое прочее. 
    // -------------------------------------------------------------------------
    TEST_F(GarbageCollectionTest, RootPreservation) {
        Rational eps = "1/1000000000000000000000000000000"_r;
        set_precision(eps);
        reset_pool_with_size(200);

        LazyRational root1 = Rational(1, 2).as_lazy(); // 1/2
        LazyRational root2 = delta::lazy_sqrt(Rational(2).as_lazy()); // sqrt(2)
        LazyRational root3 = root1.clone() + root2.clone(); // 1/2 + sqrt(2)

        root1.simplify_inplace(); // Ожидание: чистое дерево с одним узлом CONST
        EXPECT_TRUE(is_clean(root1));
        const auto& root1_node = internal::pool.nodes[clean_index(root1)];
        EXPECT_EQ(root1_node.op, internal::LazyOp::CONST);

        root2.simplify_inplace(); // Ожидание: чистое дерево с одним узлом SQRT
        EXPECT_TRUE(is_clean(root2));
        const auto& root2_node = internal::pool.nodes[clean_index(root2)];
        EXPECT_EQ(root2_node.op, internal::LazyOp::SQRT);

        root3.simplify_inplace(); // Ожидание: чистое дерево SUM({children:SQRT(2), leaf_values:1/2})
        EXPECT_TRUE(is_clean(root3));
        const auto& root3_node = internal::pool.nodes[clean_index(root3)];
        EXPECT_EQ(root3_node.op, internal::LazyOp::SUM);
        // Проверяем, что children содержит SQRT (индекс root2)
        bool found_sqrt = false;
        for (int32_t child : root3_node.children) {
            if (internal::pool.nodes[child].op == internal::LazyOp::SQRT) {
                found_sqrt = true;
                break;
            }
        }
        EXPECT_TRUE(found_sqrt) << "SUM.children should contain SQRT node";
        // Проверяем, что leaf_values содержит 1/2
        bool found_half = false;
        for (const auto& leaf : root3_node.leaf_values) {
            if (leaf == Rational(1, 2).value()) {
                found_half = true;
                break;
            }
        }
        EXPECT_TRUE(found_half) << "SUM.leaf_values should contain 1/2";

        // ожидание: root1, root2, root3 - зарегистрировались в реестре чистых переменных
        EXPECT_TRUE(internal::g_clean_rationals.find(&root1) != internal::g_clean_rationals.end());
        EXPECT_TRUE(internal::g_clean_rationals.find(&root2) != internal::g_clean_rationals.end());
        EXPECT_TRUE(internal::g_clean_rationals.find(&root3) != internal::g_clean_rationals.end());

        int idx1 = clean_index(root1);
        int idx2 = clean_index(root2);
        int idx3 = clean_index(root3);

        // Запоминаем размер реестра до цикла (должен быть 3)
        size_t initial_registry_size = internal::g_clean_rationals.size();
        EXPECT_EQ(initial_registry_size, 3);

        // Намеренно объявляем переменную tmp внутри цикла, таким образом 300 раз будут вызваны конструктор и деструктор для tmp. 
        // Заметка: Это хорошо для тестового сценария - проверка неоптимального пути - 
        // но как production-performance код это было бы ужасное решение, хаха.
        for (int i = 0; i < 300; ++i) {
            LazyRational tmp = Rational(i).as_lazy() + Rational(i + 1).as_lazy();
            tmp.simplify_inplace(); // ожидание: tmp добавился в реестр чистых переменных.
            EXPECT_TRUE(internal::g_clean_rationals.find(&tmp) != internal::g_clean_rationals.end());
            // ожидание: в конце итерации tmp будет уничтожен и дерегистрирован
        }

        // После цикла реестр должен содержать только root1, root2, root3
        EXPECT_EQ(internal::g_clean_rationals.size(), initial_registry_size);

        // Ожидание: размер пула выставлен как 200, итераций 300 => посреди цикла пул должен вызвать сборку мусора.
        // Проверяем, что пул не превышает max_size (GC срабатывал и чистил)
        EXPECT_LE(internal::pool.next_free_index, internal::pool.max_size);

        // Проверяем типы корней. ВАЖНО: после GC внутри цикла корни МОГЛИ быть заменены на CONST,
        // но это корректное поведение (GC превращает любые корни в константы).
        const auto& node1 = internal::pool.nodes[idx1];
        const auto& node2 = internal::pool.nodes[idx2];
        const auto& node3 = internal::pool.nodes[idx3];

        // root1 всегда CONST (изначально был константой)
        EXPECT_EQ(node1.op, internal::LazyOp::CONST);
        // root2 мог остаться SQRT или быть заменён на CONST после GC
        EXPECT_TRUE(node2.op == internal::LazyOp::CONST || node2.op == internal::LazyOp::SQRT);
        // root3 мог остаться SUM или быть заменён на CONST после GC
        EXPECT_TRUE(node3.op == internal::LazyOp::CONST || node3.op == internal::LazyOp::SUM);

        // Ожидание: после сборки мусора максимальный размер пула остался 200
        EXPECT_EQ(internal::pool.max_size, 200);

        EXPECT_EQ(root1.eval(), Rational(1, 2));
        EXPECT_RATIONAL_NEAR(root2.eval(), delta::sqrt(2_r), default_eps());
        EXPECT_RATIONAL_NEAR(root3.eval(), (Rational(1, 2) + delta::sqrt(2_r)), default_eps());
    }
    //версия для дебага с выводами в консоль.
    TEST_F(GarbageCollectionTest, RootPreservationVerbose) {
        //just comment out the GTEST_SKIP() if need be.
        GTEST_SKIP() << "Same as GarbageCollectionTest.RootPreservation. "
            << "Left for potential verbose debug reference implementation";
        std::cout << "\n=== Starting RootPreservationVerbose ===" << std::endl;

        Rational eps = "1/1000000000000000000000000000000"_r;
        set_precision(eps);
        reset_pool_with_size(200);
        std::cout << "Initial pool after reset_pool_with_size(200):" << std::endl;
        print_pool("pool");

        LazyRational root1 = Rational(1, 2).as_lazy(); // 1/2
        LazyRational root2 = delta::lazy_sqrt(Rational(2).as_lazy()); // sqrt(2)
        LazyRational root3 = root1.clone() + root2.clone(); // 1/2 + sqrt(2)

        std::cout << "\nAfter creating roots (before simplify):" << std::endl;
        print_lazy(root1, "root1 (dirty)");
        print_lazy(root2, "root2 (dirty)");
        print_lazy(root3, "root3 (dirty)");

        root1.simplify_inplace(); // Ожидание: чистое дерево с одним узлом CONST
        EXPECT_TRUE(is_clean(root1));
        const auto& root1_node = internal::pool.nodes[clean_index(root1)];
        EXPECT_EQ(root1_node.op, internal::LazyOp::CONST);
        std::cout << "\nAfter root1.simplify_inplace (should be CONST):" << std::endl;
        print_lazy(root1, "root1");

        root2.simplify_inplace(); // Ожидание: чистое дерево с одним узлом SQRT
        EXPECT_TRUE(is_clean(root2));
        const auto& root2_node = internal::pool.nodes[clean_index(root2)];
        EXPECT_EQ(root2_node.op, internal::LazyOp::SQRT);
        std::cout << "\nAfter root2.simplify_inplace (should be SQRT):" << std::endl;
        print_lazy(root2, "root2");

        root3.simplify_inplace(); // Ожидание: чистое дерево SUM({children:SQRT(2), leaf_values:1/2})
        EXPECT_TRUE(is_clean(root3));
        const auto& root3_node = internal::pool.nodes[clean_index(root3)];
        EXPECT_EQ(root3_node.op, internal::LazyOp::SUM);
        // Проверяем, что children содержит SQRT (индекс root2)
        bool found_sqrt = false;
        for (int32_t child : root3_node.children) {
            if (internal::pool.nodes[child].op == internal::LazyOp::SQRT) {
                found_sqrt = true;
                break;
            }
        }
        EXPECT_TRUE(found_sqrt) << "SUM.children should contain SQRT node";
        // Проверяем, что leaf_values содержит 1/2
        bool found_half = false;
        for (const auto& leaf : root3_node.leaf_values) {
            if (leaf == Rational(1, 2).value()) {
                found_half = true;
                break;
            }
        }
        EXPECT_TRUE(found_half) << "SUM.leaf_values should contain 1/2";
        std::cout << "\nAfter root3.simplify_inplace (should be SUM with SQRT child and 1/2 leaf):" << std::endl;
        print_lazy(root3, "root3");

        // ожидание: root1, root2, root3 - зарегистрировались в реестре чистых переменных
        EXPECT_TRUE(internal::g_clean_rationals.find(&root1) != internal::g_clean_rationals.end());
        EXPECT_TRUE(internal::g_clean_rationals.find(&root2) != internal::g_clean_rationals.end());
        EXPECT_TRUE(internal::g_clean_rationals.find(&root3) != internal::g_clean_rationals.end());
        std::cout << "\nAfter simplify, clean registry:" << std::endl;
        print_clean_registry();

        int idx1 = clean_index(root1);
        int idx2 = clean_index(root2);
        int idx3 = clean_index(root3);
        std::cout << "\nIndices: idx1=" << idx1 << " idx2=" << idx2 << " idx3=" << idx3 << std::endl;

        // Запоминаем размер реестра до цикла (должен быть 3)
        size_t initial_registry_size = internal::g_clean_rationals.size();
        EXPECT_EQ(initial_registry_size, 3);

        std::cout << "\n--- Creating 300 temporary expressions (tmp inside loop) ---" << std::endl;
        for (int i = 0; i < 300; ++i) {
            LazyRational tmp = Rational(i).as_lazy() + Rational(i + 1).as_lazy();
            tmp.simplify_inplace(); // ожидание: tmp добавился в реестр чистых переменных.
            EXPECT_TRUE(internal::g_clean_rationals.find(&tmp) != internal::g_clean_rationals.end());
            // ожидание: в конце итерации tmp будет уничтожен и дерегистрирован
            if (i % 100 == 0) {
                std::cout << "i=" << i << ", pool.nodes.size()=" << internal::pool.nodes.size()
                    << " next_free_index=" << internal::pool.next_free_index
                    << " max_size=" << internal::pool.max_size << std::endl;
            }
        }

        std::cout << "\n--- After loop ---" << std::endl;
        print_pool("pool after loop");
        print_clean_registry();

        // После цикла реестр должен содержать только root1, root2, root3
        EXPECT_EQ(internal::g_clean_rationals.size(), initial_registry_size);

        // Ожидание: размер пула не превышает max_size (GC срабатывал внутри цикла)
        EXPECT_LE(internal::pool.next_free_index, internal::pool.max_size)
            << "Pool size should not exceed max_size after GC";

        // Проверяем типы корней. ВАЖНО: после GC внутри цикла корни МОГЛИ быть заменены на CONST,
        // но это корректное поведение (GC превращает любые корни в константы).
        const auto& node1 = internal::pool.nodes[idx1];
        const auto& node2 = internal::pool.nodes[idx2];
        const auto& node3 = internal::pool.nodes[idx3];

        std::cout << "\n--- Checking node types at indices ---" << std::endl;
        std::cout << "node1.op = " << static_cast<int>(node1.op)
            << " (expect CONST=" << static_cast<int>(internal::LazyOp::CONST) << ")" << std::endl;
        std::cout << "node2.op = " << static_cast<int>(node2.op)
            << " (expect CONST or SQRT depending on GC)" << std::endl;
        std::cout << "node3.op = " << static_cast<int>(node3.op)
            << " (expect CONST or SUM depending on GC)" << std::endl;

        // root1 всегда CONST (изначально был константой)
        EXPECT_EQ(node1.op, internal::LazyOp::CONST);
        // root2 мог остаться SQRT или быть заменён на CONST после GC
        EXPECT_TRUE(node2.op == internal::LazyOp::CONST || node2.op == internal::LazyOp::SQRT);
        // root3 мог остаться SUM или быть заменён на CONST после GC
        EXPECT_TRUE(node3.op == internal::LazyOp::CONST || node3.op == internal::LazyOp::SUM);

        // Ожидание: после сборки мусора максимальный размер пула остался 200
        EXPECT_EQ(internal::pool.max_size, 200);

        std::cout << "\n--- Checking roots eval ---" << std::endl;
        Rational val1 = root1.eval();
        Rational val2 = root2.eval();
        Rational val3 = root3.eval();
        std::cout << "root1.eval() = " << val1.to_string() << std::endl;
        std::cout << "root2.eval() = " << val2.to_string() << std::endl;
        std::cout << "root3.eval() = " << val3.to_string() << std::endl;

        EXPECT_EQ(root1.eval(), Rational(1, 2));
        EXPECT_RATIONAL_NEAR(root2.eval(), delta::sqrt(2_r), default_eps());
        EXPECT_RATIONAL_NEAR(root3.eval(), (Rational(1, 2) + delta::sqrt(2_r)), default_eps());

        std::cout << "\n=== RootPreservationVerbose PASS ===" << std::endl;
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
        EXPECT_EQ(refcount(idx), 1);

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
            if (node.op == internal::LazyOp::SUM || node.op == internal::LazyOp::PRODUCT) {
                occupied = !node.leaf_values.empty() || !node.children.empty();
            }
            else if (node.op == internal::LazyOp::CONST) {
                occupied = node.value_idx != -1;
            }
            else {
                // children удалено – теперь все дети в children
                occupied = !node.children.empty() || node.eps_idx != -1;
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
            tmp.simplify_inplace();
        }
        internal::force_garbage_collect();
        EXPECT_EQ(internal::pool.next_free_index, 0);
        for (size_t i = 0; i < internal::pool.nodes.size(); ++i) {
            const auto& node = internal::pool.nodes[i];
            bool occupied = false;
            if (node.op == internal::LazyOp::SUM || node.op == internal::LazyOp::PRODUCT) {
                occupied = !node.leaf_values.empty() || !node.children.empty();
            }
            else if (node.op == internal::LazyOp::CONST) {
                occupied = node.value_idx != -1;
            }
            else {
                // children удалено – теперь все дети в children
                occupied = !node.children.empty() || node.eps_idx != -1;
            }
            EXPECT_FALSE(occupied) << "Slot " << i << " not empty";
        }
    }

} // namespace delta::testing