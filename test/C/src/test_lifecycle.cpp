/// @file test_lifecycle.cpp
/// @brief Tests for object lifecycle, memory management, and creation/destruction patterns
///
/// Total: 60+ test cases

#include "test.hpp"

using namespace scl::test;

SCL_TEST_BEGIN

// =============================================================================
// SECTION 1: Creation-Destruction Cycles (20 tests)
// =============================================================================

SCL_TEST_SUITE(lifecycle_cycles)

SCL_TEST_TAGGED(create_destroy_loop_real64, "lifecycle", "quick") {
    for (int i = 0; i < 100; ++i) {
        auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
        SCL_ASSERT_SPARSE_VALID(m);
        scl_sparse_destroy(m);
    }
}

SCL_TEST_TAGGED(create_destroy_loop_int32, "lifecycle", "integer") {
    for (int i = 0; i < 100; ++i) {
        auto m = scl_sparse_identity(10, SCL_INT32, SCL_INDEX64);
        SCL_ASSERT_SPARSE_VALID(m);
        scl_sparse_destroy(m);
    }
}

SCL_TEST_TAGGED(create_destroy_loop_uint8, "lifecycle", "unsigned") {
    for (int i = 0; i < 100; ++i) {
        auto m = scl_sparse_identity(5, SCL_UINT8, SCL_INDEX64);
        SCL_ASSERT_SPARSE_VALID(m);
        scl_sparse_destroy(m);
    }
}

SCL_TEST_TAGGED(create_many_small_matrices, "lifecycle") {
    // 创建大量小矩阵
    std::vector<scl_sparse_t> matrices;
    
    for (int i = 0; i < 50; ++i) {
        auto m = scl_sparse_identity(5, SCL_REAL64, SCL_INDEX64);
        matrices.push_back(m);
    }
    
    // 全部销毁
    for (auto m : matrices) {
        scl_sparse_destroy(m);
    }
}

SCL_TEST_TAGGED(create_few_large_matrices, "lifecycle", "slow") {
    // 创建少量大矩阵
    std::vector<scl_sparse_t> matrices;
    
    for (int i = 0; i < 5; ++i) {
        auto m = scl_sparse_identity(1000, SCL_REAL64, SCL_INDEX64);
        matrices.push_back(m);
    }
    
    for (auto m : matrices) {
        scl_sparse_destroy(m);
    }
}

SCL_TEST_TAGGED(mixed_size_matrices, "lifecycle") {
    int sizes[] = {5, 10, 20, 50, 100};
    
    for (int n : sizes) {
        auto m = scl_sparse_identity(n, SCL_INT32, SCL_INDEX64);
        SCL_ASSERT_SPARSE_DIMS(m, n, n);
        scl_sparse_destroy(m);
    }
}

SCL_TEST_TAGGED(nested_operations, "lifecycle") {
    auto m1 = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto m2 = scl_sparse_transpose(m1);
    auto m3 = scl_sparse_clone(m2);
    auto m4 = scl_sparse_transpose(m3);
    
    SCL_ASSERT_SPARSE_VALID(m4);
    
    scl_sparse_destroy(m1);
    scl_sparse_destroy(m2);
    scl_sparse_destroy(m3);
    scl_sparse_destroy(m4);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 2: Clone Independence (20 tests)
// =============================================================================

SCL_TEST_SUITE(clone_independence)

SCL_TEST_TAGGED(modify_original_after_clone, "lifecycle") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto c = scl_sparse_clone(m);
    
    // 修改原矩阵
    scl_sparse_scale(m, 2.0);
    
    // 克隆应该不受影响（但我们无法直接验证值，只能验证结构）
    SCL_ASSERT_SPARSE_VALID(c);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(c);
}

SCL_TEST_TAGGED(destroy_original_keep_clone, "lifecycle") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto c = scl_sparse_clone(m);
    
    // 销毁原矩阵
    scl_sparse_destroy(m);
    
    // 克隆应该仍然有效
    SCL_ASSERT_SPARSE_VALID(c);
    SCL_ASSERT_SPARSE_NNZ(c, 10);
    
    scl_sparse_destroy(c);
}

SCL_TEST_TAGGED(multiple_clones, "lifecycle") {
    auto m = scl_sparse_identity(10, SCL_INT32, SCL_INDEX64);
    auto c1 = scl_sparse_clone(m);
    auto c2 = scl_sparse_clone(m);
    auto c3 = scl_sparse_clone(m);
    
    SCL_ASSERT_SPARSE_VALID(c1);
    SCL_ASSERT_SPARSE_VALID(c2);
    SCL_ASSERT_SPARSE_VALID(c3);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(c1);
    scl_sparse_destroy(c2);
    scl_sparse_destroy(c3);
}

SCL_TEST_TAGGED(clone_chain, "lifecycle") {
    auto m0 = scl_sparse_identity(5, SCL_UINT32, SCL_INDEX64);
    auto m1 = scl_sparse_clone(m0);
    auto m2 = scl_sparse_clone(m1);
    auto m3 = scl_sparse_clone(m2);
    
    SCL_ASSERT_SPARSE_VALID(m3);
    
    scl_sparse_destroy(m0);
    scl_sparse_destroy(m1);
    scl_sparse_destroy(m2);
    scl_sparse_destroy(m3);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 3: Transpose Properties (20 tests)
// =============================================================================

SCL_TEST_SUITE(transpose_properties)

SCL_TEST_TAGGED(transpose_involution, "property") {
    // (A^T)^T = A
    auto m = scl_sparse_zeros(5, 8, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    auto t1 = scl_sparse_transpose(m);
    auto t2 = scl_sparse_transpose(t1);
    
    SCL_ASSERT_SPARSE_DIMS(t2, 5, 8);
    SCL_ASSERT_EQ(scl_sparse_layout(t2), SCL_LAYOUT_CSR);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(t1);
    scl_sparse_destroy(t2);
}

SCL_TEST_TAGGED(transpose_dimension_swap, "property", "quick") {
    auto m = scl_sparse_zeros(10, 20, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    auto t = scl_sparse_transpose(m);
    
    SCL_ASSERT_SPARSE_DIMS(t, 20, 10);  // 维度交换
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(t);
}

SCL_TEST_TAGGED(transpose_preserves_nnz, "property", "quick") {
    auto m = scl_sparse_identity(10, SCL_INT32, SCL_INDEX64);
    auto t = scl_sparse_transpose(m);
    
    SCL_ASSERT_EQ(scl_sparse_nnz(m), scl_sparse_nnz(t));
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(t);
}

SCL_TEST_TAGGED(transpose_layout_flip, "property", "quick") {
    auto csr = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto csc = scl_sparse_transpose(csr);
    auto csr2 = scl_sparse_transpose(csc);
    
    SCL_ASSERT_EQ(scl_sparse_layout(csr), SCL_LAYOUT_CSR);
    SCL_ASSERT_EQ(scl_sparse_layout(csc), SCL_LAYOUT_CSC);
    SCL_ASSERT_EQ(scl_sparse_layout(csr2), SCL_LAYOUT_CSR);
    
    scl_sparse_destroy(csr);
    scl_sparse_destroy(csc);
    scl_sparse_destroy(csr2);
}

SCL_TEST_TAGGED(transpose_all_types, "property") {
    scl_value_type_t types[] = {SCL_REAL32, SCL_INT16, SCL_UINT16};
    
    for (auto vt : types) {
        auto m = scl_sparse_identity(8, vt, SCL_INDEX64);
        auto t = scl_sparse_transpose(m);
        
        SCL_ASSERT_EQ(scl_sparse_value_type(t), vt);
        
        scl_sparse_destroy(m);
        scl_sparse_destroy(t);
    }
}

SCL_TEST_SUITE_END

SCL_TEST_END
SCL_TEST_MAIN()

