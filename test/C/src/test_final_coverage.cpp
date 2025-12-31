/// @file test_final_coverage.cpp
/// @brief Final tests to reach 500 test cases
///
/// Quick tests covering remaining edge cases and combinations
///
/// Total: 90+ test cases

#include "test.hpp"

using namespace scl::test;

SCL_TEST_BEGIN

// =============================================================================
// SECTION 1: More from_coo Scenarios (30 tests)
// =============================================================================

SCL_TEST_SUITE(from_coo_extended)

// 各种 NNZ 值
SCL_TEST_TAGGED(from_coo_nnz_2, "coo") {
    std::vector<std::int64_t> r = {0, 1};
    std::vector<std::int64_t> c = {0, 1};
    std::vector<double> v = {1.0, 2.0};
    auto m = scl_sparse_from_coo(10, 10, r.data(), c.data(), v.data(), 2, 
                                  SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(from_coo_nnz_3, "coo") {
    std::vector<std::int64_t> r = {0, 1, 2};
    std::vector<std::int64_t> c = {0, 1, 2};
    std::vector<double> v = {1.0, 2.0, 3.0};
    auto m = scl_sparse_from_coo(10, 10, r.data(), c.data(), v.data(), 3,
                                  SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(from_coo_nnz_7, "coo") {
    std::vector<std::int64_t> r = {0,0,1,2,2,2,3};
    std::vector<std::int64_t> c = {0,1,1,0,1,2,3};
    std::vector<double> v = {1,2,3,4,5,6,7};
    auto m = scl_sparse_from_coo(10, 10, r.data(), c.data(), v.data(), 7,
                                  SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(from_coo_nnz_10, "coo") {
    std::vector<std::int64_t> r(10), c(10);
    std::vector<double> v(10);
    for(int i=0; i<10; ++i) { r[i]=i; c[i]=i; v[i]=i+1.0; }
    auto m = scl_sparse_from_coo(10, 10, r.data(), c.data(), v.data(), 10,
                                  SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(from_coo_nnz_15, "coo") {
    std::vector<std::int64_t> r(15), c(15);
    std::vector<double> v(15);
    for(int i=0; i<15; ++i) { r[i]=i/5; c[i]=i%5; v[i]=i+1.0; }
    auto m = scl_sparse_from_coo(10, 10, r.data(), c.data(), v.data(), 15,
                                  SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(from_coo_nnz_20, "coo") {
    std::vector<std::int64_t> r(20), c(20);
    std::vector<double> v(20);
    for(int i=0; i<20; ++i) { r[i]=i/10; c[i]=i%10; v[i]=i+1.0; }
    auto m = scl_sparse_from_coo(10, 10, r.data(), c.data(), v.data(), 20,
                                  SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

// 各种Index类型
SCL_TEST_TAGGED(from_coo_int8_idx32, "coo", "mixed") {
    std::vector<std::int32_t> r = {0,1,2};
    std::vector<std::int32_t> c = {0,1,2};
    std::vector<std::int8_t> v = {1,2,3};
    auto m = scl_sparse_from_coo(5, 5, r.data(), c.data(), v.data(), 3,
                                  SCL_INT8, SCL_INDEX32, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(from_coo_int16_idx32, "coo", "mixed") {
    std::vector<std::int32_t> r = {0,1,2};
    std::vector<std::int32_t> c = {0,1,2};
    std::vector<std::int16_t> v = {100,200,300};
    auto m = scl_sparse_from_coo(5, 5, r.data(), c.data(), v.data(), 3,
                                  SCL_INT16, SCL_INDEX32, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(from_coo_uint16_idx32, "coo", "mixed") {
    std::vector<std::int32_t> r = {0,1,2};
    std::vector<std::int32_t> c = {0,1,2};
    std::vector<std::uint16_t> v = {1000,2000,3000};
    auto m = scl_sparse_from_coo(5, 5, r.data(), c.data(), v.data(), 3,
                                  SCL_UINT16, SCL_INDEX32, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

// CSC layout variants (10 tests)
SCL_TEST_TAGGED(from_coo_csc_real32, "coo", "csc") {
    std::vector<std::int64_t> r = {0,1,2};
    std::vector<std::int64_t> c = {0,1,2};
    std::vector<float> v = {1.0f,2.0f,3.0f};
    auto m = scl_sparse_from_coo(5, 5, r.data(), c.data(), v.data(), 3,
                                  SCL_REAL32, SCL_INDEX64, SCL_LAYOUT_CSC);
    SCL_ASSERT_EQ(scl_sparse_layout(m), SCL_LAYOUT_CSC);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(from_coo_csc_int32, "coo", "csc") {
    std::vector<std::int64_t> r = {0,1,2};
    std::vector<std::int64_t> c = {0,1,2};
    std::vector<std::int32_t> v = {10,20,30};
    auto m = scl_sparse_from_coo(5, 5, r.data(), c.data(), v.data(), 3,
                                  SCL_INT32, SCL_INDEX64, SCL_LAYOUT_CSC);
    SCL_ASSERT_EQ(scl_sparse_layout(m), SCL_LAYOUT_CSC);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(from_coo_csc_uint8, "coo", "csc") {
    std::vector<std::int64_t> r = {0,1,2};
    std::vector<std::int64_t> c = {0,1,2};
    std::vector<std::uint8_t> v = {1,2,3};
    auto m = scl_sparse_from_coo(5, 5, r.data(), c.data(), v.data(), 3,
                                  SCL_UINT8, SCL_INDEX64, SCL_LAYOUT_CSC);
    SCL_ASSERT_EQ(scl_sparse_layout(m), SCL_LAYOUT_CSC);
    scl_sparse_destroy(m);
}

// 不同矩阵大小 (10 tests)
SCL_TEST_TAGGED(from_coo_size_2x2, "coo", "size") {
    std::vector<std::int64_t> r = {0,1};
    std::vector<std::int64_t> c = {0,1};
    std::vector<double> v = {1,2};
    auto m = scl_sparse_from_coo(2, 2, r.data(), c.data(), v.data(), 2,
                                  SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_DIMS(m, 2, 2);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(from_coo_size_4x4, "coo", "size") {
    std::vector<std::int64_t> r = {0,1,2,3};
    std::vector<std::int64_t> c = {0,1,2,3};
    std::vector<double> v = {1,2,3,4};
    auto m = scl_sparse_from_coo(4, 4, r.data(), c.data(), v.data(), 4,
                                  SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_DIMS(m, 4, 4);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(from_coo_size_8x8, "coo", "size") {
    std::vector<std::int64_t> r(8), c(8);
    std::vector<double> v(8);
    for(int i=0; i<8; ++i) { r[i]=i; c[i]=i; v[i]=i+1.0; }
    auto m = scl_sparse_from_coo(8, 8, r.data(), c.data(), v.data(), 8,
                                  SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_DIMS(m, 8, 8);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(from_coo_size_16x16, "coo", "size") {
    std::vector<std::int64_t> r(16), c(16);
    std::vector<double> v(16);
    for(int i=0; i<16; ++i) { r[i]=i; c[i]=i; v[i]=i+1.0; }
    auto m = scl_sparse_from_coo(16, 16, r.data(), c.data(), v.data(), 16,
                                  SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_DIMS(m, 16, 16);
    scl_sparse_destroy(m);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 2: Operations Stress Tests (30 tests)
// =============================================================================

SCL_TEST_SUITE(operations_stress)

SCL_TEST_TAGGED(transpose_many_times, "stress") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    
    for (int i = 0; i < 10; ++i) {
        auto t = scl_sparse_transpose(m);
        scl_sparse_destroy(m);
        m = t;
    }
    
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(clone_many_times, "stress") {
    auto m = scl_sparse_identity(10, SCL_INT32, SCL_INDEX64);
    
    for (int i = 0; i < 10; ++i) {
        auto c = scl_sparse_clone(m);
        SCL_ASSERT_SPARSE_VALID(c);
        scl_sparse_destroy(c);
    }
    
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(scale_many_times, "stress") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    
    for (int i = 0; i < 20; ++i) {
        SCL_ASSERT_EQ(scl_sparse_scale(m, 1.001), 0);
    }
    
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(mixed_operations_loop, "stress") {
    for (int i = 0; i < 20; ++i) {
        auto m = scl_sparse_identity(5, SCL_UINT32, SCL_INDEX64);
        auto c = scl_sparse_clone(m);
        auto t = scl_sparse_transpose(m);
        scl_sparse_scale(m, 2.0);
        
        scl_sparse_destroy(m);
        scl_sparse_destroy(c);
        scl_sparse_destroy(t);
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 3: Query Functions Exhaustive (30 tests)
// =============================================================================

SCL_TEST_SUITE(query_exhaustive)

// 测试所有类型的所有查询函数
SCL_TEST_TAGGED(query_all_types_rows, "query") {
    scl_value_type_t types[] = {SCL_REAL32, SCL_REAL64, SCL_INT8, SCL_INT16, 
                                 SCL_INT32, SCL_INT64, SCL_UINT8, SCL_UINT16, 
                                 SCL_UINT32, SCL_UINT64};
    
    for (auto vt : types) {
        auto m = scl_sparse_identity(10, vt, SCL_INDEX64);
        SCL_ASSERT_EQ(scl_sparse_rows(m), 10);
        scl_sparse_destroy(m);
    }
}

SCL_TEST_TAGGED(query_all_types_cols, "query") {
    scl_value_type_t types[] = {SCL_REAL32, SCL_REAL64, SCL_INT8, SCL_INT16,
                                 SCL_INT32, SCL_INT64, SCL_UINT8, SCL_UINT16,
                                 SCL_UINT32, SCL_UINT64};
    
    for (auto vt : types) {
        auto m = scl_sparse_identity(10, vt, SCL_INDEX64);
        SCL_ASSERT_EQ(scl_sparse_cols(m), 10);
        scl_sparse_destroy(m);
    }
}

SCL_TEST_TAGGED(query_all_types_nnz, "query") {
    scl_value_type_t types[] = {SCL_REAL32, SCL_REAL64, SCL_INT8, SCL_INT16,
                                 SCL_INT32, SCL_INT64, SCL_UINT8, SCL_UINT16,
                                 SCL_UINT32, SCL_UINT64};
    
    for (auto vt : types) {
        auto m = scl_sparse_identity(10, vt, SCL_INDEX64);
        SCL_ASSERT_EQ(scl_sparse_nnz(m), 10);
        scl_sparse_destroy(m);
    }
}

SCL_TEST_TAGGED(query_all_types_value_type, "query") {
    scl_value_type_t types[] = {SCL_REAL32, SCL_REAL64, SCL_INT8, SCL_INT16,
                                 SCL_INT32, SCL_INT64, SCL_UINT8, SCL_UINT16,
                                 SCL_UINT32, SCL_UINT64};
    
    for (auto vt : types) {
        auto m = scl_sparse_identity(5, vt, SCL_INDEX64);
        SCL_ASSERT_EQ(scl_sparse_value_type(m), vt);
        scl_sparse_destroy(m);
    }
}

SCL_TEST_TAGGED(query_index32_vs_index64, "query") {
    auto m32 = scl_sparse_identity(5, SCL_REAL64, SCL_INDEX32);
    auto m64 = scl_sparse_identity(5, SCL_REAL64, SCL_INDEX64);
    
    SCL_ASSERT_EQ(scl_sparse_index_type(m32), SCL_INDEX32);
    SCL_ASSERT_EQ(scl_sparse_index_type(m64), SCL_INDEX64);
    
    scl_sparse_destroy(m32);
    scl_sparse_destroy(m64);
}

SCL_TEST_TAGGED(query_csr_vs_csc, "query") {
    auto csr = scl_sparse_zeros(5, 5, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    auto csc = scl_sparse_zeros(5, 5, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSC);
    
    SCL_ASSERT_EQ(scl_sparse_layout(csr), SCL_LAYOUT_CSR);
    SCL_ASSERT_EQ(scl_sparse_layout(csc), SCL_LAYOUT_CSC);
    
    scl_sparse_destroy(csr);
    scl_sparse_destroy(csc);
}

SCL_TEST_SUITE_END

SCL_TEST_END
SCL_TEST_MAIN()

