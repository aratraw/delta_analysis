// (c) 2026 Timofey Ishimtsev.
// Licensed under PolyForm Small Business License 1.0.0

// tests/rational/gc_test.cpp
// ============================================================================
// GARBAGE COLLECTION TESTS FOR THE GLOBAL NODE POOL
// ============================================================================
//
// This file tests the behaviour of the internal garbage collector (GC) in the
// global node pool used by LazyRational. The GC is triggered automatically
// when the pool is nearly full, and also can be forced manually.
//
// Verified properties:
//   - GC runs when the number of allocated nodes exceeds the threshold.
//   - Root indices remain valid after GC (the indices are preserved).
//   - Clean objects are correctly registered / unregistered in the registry.
//   - Reference counting works.
//   - After GC the pool may become more compact; empty slots are reclaimed.
//   - Exception is thrown when the pool is exhausted by roots.
//   - GC can handle an empty pool (no roots) correctly.
//
// All tests use a custom maximum pool size to force GC earlier and examine its
// behaviour. The global default epsilon is kept at its default value unless
// overridden.
// ============================================================================

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
            // Clear the pool and reset max_size to the default value
            internal::reset_pool();
            // Ensure that max_size is set to DEFAULT_POOL_MAX_SIZE; reset_pool already does that.
        }

        void TearDown() override {
            // After each test, return the pool to a clean state
            // so that other tests are not affected (SetUp will do it anyway before the next test)
            internal::reset_pool();
        }

        // Count the number of occupied slots in the pool
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
                    // For unary/binary ops: if there are children or an epsilon index, the node is occupied
                    if (!node.children.empty() || node.eps_idx != -1) ++cnt;
                }
            }
            return cnt;
        }

        void reset_pool_with_size(size_t new_size) {
            internal::reset_pool();          // reset to default state
            internal::set_pool_max_size(new_size);
        }
    };

    // -------------------------------------------------------------------------
    // 1. Check that garbage collection is triggered when the pool gets full
    // -------------------------------------------------------------------------
    /**
     * @test PoolSizeLimit
     * @brief Verifies that the garbage collector runs automatically when the pool
     *        reaches its capacity and that the number of nodes stays within the limit.
     */
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
    // 2. Check that root indices are preserved and everything works correctly
    // -------------------------------------------------------------------------
    /**
     * @test RootPreservation
     * @brief Tests that after many temporary allocations and forced GC, the root
     *        indices remain valid and the roots evaluate to their original values.
     *        Also verifies clean object registry behaviour.
     */
    TEST_F(GarbageCollectionTest, RootPreservation) {
        Rational eps = "1/1000000000000000000000000000000"_r;
        set_precision(eps);
        reset_pool_with_size(200);

        LazyRational root1 = Rational(1, 2).as_lazy(); // 1/2
        LazyRational root2 = delta::lazy_sqrt(Rational(2).as_lazy()); // sqrt(2)
        LazyRational root3 = root1.clone() + root2.clone(); // 1/2 + sqrt(2)

        root1.simplify_inplace(); // Expected: clean tree with a single CONST node
        EXPECT_TRUE(is_clean(root1));
        const auto& root1_node = internal::pool.nodes[clean_index(root1)];
        EXPECT_EQ(root1_node.op, internal::LazyOp::CONST);

        root2.simplify_inplace(); // Expected: clean tree with a single SQRT node
        EXPECT_TRUE(is_clean(root2));
        const auto& root2_node = internal::pool.nodes[clean_index(root2)];
        EXPECT_EQ(root2_node.op, internal::LazyOp::SQRT);

        root3.simplify_inplace(); // Expected: clean SUM node (children: SQRT(2), leaf_values: 1/2)
        EXPECT_TRUE(is_clean(root3));
        const auto& root3_node = internal::pool.nodes[clean_index(root3)];
        EXPECT_EQ(root3_node.op, internal::LazyOp::SUM);
        // Check that children contain SQRT (index of root2)
        bool found_sqrt = false;
        for (int32_t child : root3_node.children) {
            if (internal::pool.nodes[child].op == internal::LazyOp::SQRT) {
                found_sqrt = true;
                break;
            }
        }
        EXPECT_TRUE(found_sqrt) << "SUM.children should contain SQRT node";
        // Check that leaf_values contain 1/2
        bool found_half = false;
        for (const auto& leaf : root3_node.leaf_values) {
            if (leaf == Rational(1, 2).value()) {
                found_half = true;
                break;
            }
        }
        EXPECT_TRUE(found_half) << "SUM.leaf_values should contain 1/2";

        // Expect: root1, root2, root3 are registered in the clean object registry
        EXPECT_TRUE(internal::g_clean_rationals.find(&root1) != internal::g_clean_rationals.end());
        EXPECT_TRUE(internal::g_clean_rationals.find(&root2) != internal::g_clean_rationals.end());
        EXPECT_TRUE(internal::g_clean_rationals.find(&root3) != internal::g_clean_rationals.end());

        int idx1 = clean_index(root1);
        int idx2 = clean_index(root2);
        int idx3 = clean_index(root3);

        // Remember the registry size before the loop (should be 3)
        size_t initial_registry_size = internal::g_clean_rationals.size();
        EXPECT_EQ(initial_registry_size, 3);

        // Deliberately declare tmp inside the loop; thus 300 times the constructor and destructor for tmp will be called.
        // Note: This is good for a test scenario (checking a non‑optimal path),
        // but as production‑performance code it would be terrible, haha.
        for (int i = 0; i < 300; ++i) {
            LazyRational tmp = Rational(i).as_lazy() + Rational(i + 1).as_lazy();
            tmp.simplify_inplace(); // Expected: tmp added to the clean object registry
            EXPECT_TRUE(internal::g_clean_rationals.find(&tmp) != internal::g_clean_rationals.end());
            // Expected: at the end of the iteration tmp will be destroyed and unregistered
        }

        // After the loop, the registry should contain only root1, root2, root3
        EXPECT_EQ(internal::g_clean_rationals.size(), initial_registry_size);

        // Expect: pool size is set to 200, 300 iterations => somewhere in the middle the pool must have called GC.
        // Verify that the pool does not exceed max_size (GC worked)
        EXPECT_LE(internal::pool.next_free_index, internal::pool.max_size);

        // Check the root node types. IMPORTANT: after GC inside the loop, roots MAY have been replaced by CONST,
        // but that is correct behaviour (GC turns any root into a constant).
        const auto& node1 = internal::pool.nodes[idx1];
        const auto& node2 = internal::pool.nodes[idx2];
        const auto& node3 = internal::pool.nodes[idx3];

        // root1 is always CONST (originally a constant)
        EXPECT_EQ(node1.op, internal::LazyOp::CONST);
        // root2 could remain SQRT or be replaced by CONST after GC
        EXPECT_TRUE(node2.op == internal::LazyOp::CONST || node2.op == internal::LazyOp::SQRT);
        // root3 could remain SUM or be replaced by CONST after GC
        EXPECT_TRUE(node3.op == internal::LazyOp::CONST || node3.op == internal::LazyOp::SUM);

        // Expect: after garbage collection the maximum pool size remains 200
        EXPECT_EQ(internal::pool.max_size, 200);

        EXPECT_EQ(root1.eval(), Rational(1, 2));
        EXPECT_RATIONAL_NEAR(root2.eval(), delta::sqrt(2_r), default_eps());
        EXPECT_RATIONAL_NEAR(root3.eval(), (Rational(1, 2) + delta::sqrt(2_r)), default_eps());
    }

    // -------------------------------------------------------------------------
    // Verbose version (legacy for debugging, skipped by default)
    // -------------------------------------------------------------------------
    /**
     * @test RootPreservationVerbose
     * @brief Same as GarbageCollectionTest.RootPreservation but prints
     *        detailed information to stdout. Skipped in normal runs;
     *        kept as a reference for manual debugging capabilities
     */
    TEST_F(GarbageCollectionTest, RootPreservationVerbose) {
        // just comment out the GTEST_SKIP() if need be.
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

        root1.simplify_inplace(); // Expected: clean tree with a single CONST node
        EXPECT_TRUE(is_clean(root1));
        const auto& root1_node = internal::pool.nodes[clean_index(root1)];
        EXPECT_EQ(root1_node.op, internal::LazyOp::CONST);
        std::cout << "\nAfter root1.simplify_inplace (should be CONST):" << std::endl;
        print_lazy(root1, "root1");

        root2.simplify_inplace(); // Expected: clean tree with a single SQRT node
        EXPECT_TRUE(is_clean(root2));
        const auto& root2_node = internal::pool.nodes[clean_index(root2)];
        EXPECT_EQ(root2_node.op, internal::LazyOp::SQRT);
        std::cout << "\nAfter root2.simplify_inplace (should be SQRT):" << std::endl;
        print_lazy(root2, "root2");

        root3.simplify_inplace(); // Expected: clean SUM node (children: SQRT(2), leaf_values: 1/2)
        EXPECT_TRUE(is_clean(root3));
        const auto& root3_node = internal::pool.nodes[clean_index(root3)];
        EXPECT_EQ(root3_node.op, internal::LazyOp::SUM);
        // Check that children contain SQRT (index of root2)
        bool found_sqrt = false;
        for (int32_t child : root3_node.children) {
            if (internal::pool.nodes[child].op == internal::LazyOp::SQRT) {
                found_sqrt = true;
                break;
            }
        }
        EXPECT_TRUE(found_sqrt) << "SUM.children should contain SQRT node";
        // Check that leaf_values contain 1/2
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

        // Expect: root1, root2, root3 are registered in the clean object registry
        EXPECT_TRUE(internal::g_clean_rationals.find(&root1) != internal::g_clean_rationals.end());
        EXPECT_TRUE(internal::g_clean_rationals.find(&root2) != internal::g_clean_rationals.end());
        EXPECT_TRUE(internal::g_clean_rationals.find(&root3) != internal::g_clean_rationals.end());
        std::cout << "\nAfter simplify, clean registry:" << std::endl;
        print_clean_registry();

        int idx1 = clean_index(root1);
        int idx2 = clean_index(root2);
        int idx3 = clean_index(root3);
        std::cout << "\nIndices: idx1=" << idx1 << " idx2=" << idx2 << " idx3=" << idx3 << std::endl;

        // Remember the registry size before the loop (should be 3)
        size_t initial_registry_size = internal::g_clean_rationals.size();
        EXPECT_EQ(initial_registry_size, 3);

        std::cout << "\n--- Creating 300 temporary expressions (tmp inside loop) ---" << std::endl;
        for (int i = 0; i < 300; ++i) {
            LazyRational tmp = Rational(i).as_lazy() + Rational(i + 1).as_lazy();
            tmp.simplify_inplace(); // Expected: tmp added to the clean object registry
            EXPECT_TRUE(internal::g_clean_rationals.find(&tmp) != internal::g_clean_rationals.end());
            // Expected: at the end of the iteration tmp will be destroyed and unregistered
            if (i % 100 == 0) {
                std::cout << "i=" << i << ", pool.nodes.size()=" << internal::pool.nodes.size()
                    << " next_free_index=" << internal::pool.next_free_index
                    << " max_size=" << internal::pool.max_size << std::endl;
            }
        }

        std::cout << "\n--- After loop ---" << std::endl;
        print_pool("pool after loop");
        print_clean_registry();

        // After the loop, the registry should contain only root1, root2, root3
        EXPECT_EQ(internal::g_clean_rationals.size(), initial_registry_size);

        // Expect: pool size does not exceed max_size (GC was triggered inside the loop)
        EXPECT_LE(internal::pool.next_free_index, internal::pool.max_size)
            << "Pool size should not exceed max_size after GC";

        // Check the root node types. IMPORTANT: after GC inside the loop, roots MUST be replaced by CONST,
        // that is correct behaviour (GC turns any root into a constant).
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

        // root1 is always CONST (originally a constant)
        EXPECT_EQ(node1.op, internal::LazyOp::CONST);
        // root2 could remain SQRT or be replaced by CONST after GC
        EXPECT_TRUE(node2.op == internal::LazyOp::CONST || node2.op == internal::LazyOp::SQRT);
        // root3 could remain SUM or be replaced by CONST after GC
        EXPECT_TRUE(node3.op == internal::LazyOp::CONST || node3.op == internal::LazyOp::SUM);

        // Expect: after garbage collection the maximum pool size remains 200
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
    // 3. Invariance of root indices after GC
    // -------------------------------------------------------------------------
    /**
     * @test IndexInvariance
     * @brief Checks that the clean indices of permanent roots do not change
     *        after a large number of temporary allocations and GC.
     */
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
    // 4. Forced GC
    // -------------------------------------------------------------------------
    /**
     * @test ForceGC
     * @brief Tests manual invocation of force_garbage_collect() and verifies
     *        that the pool size shrinks and occupied slots decrease.
     */
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
    // 5. Reference counting management
    // -------------------------------------------------------------------------
    /**
     * @test RefcountManagement
     * @brief Verifies that increment_ref and decrement_ref work correctly
     *        and that the reference count reflects the number of live owners.
     */
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
    // 6. Compactness after GC
    // -------------------------------------------------------------------------
    /**
     * @test CompactnessAfterGC
     * @brief After forced GC, all occupied slots should be below next_free_index
     *        and the pool should be compact.
     */
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
    // 7. Pool exhaustion by roots (exception)
    // -------------------------------------------------------------------------
    /**
     * @test ExhaustedByRoots
     * @brief Checks that an exception is thrown when the maximum number of
     *        root nodes is reached and no GC can free space.
     */
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
    // 8. GC with no roots
    // -------------------------------------------------------------------------
    /**
     * @test EmptyPoolGC
     * @brief When there are no clean objects (roots), garbage collection
     *        resets the pool to an empty state.
     */
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
                occupied = !node.children.empty() || node.eps_idx != -1;
            }
            EXPECT_FALSE(occupied) << "Slot " << i << " not empty";
        }
    }

} // namespace delta::testing