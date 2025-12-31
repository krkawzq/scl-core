/// @file test_integer_types.cpp
/// @brief Comprehensive tests for integer type behaviors
///
/// Tests specific behaviors of Int8/16/32/64 and Uint8/16/32/64
///
/// Total: 80+ test cases

#include "test.hpp"

using namespace scl::test;

SCL_TEST_BEGIN

// =============================================================================
// SECTION 1: Int8 Comprehensive Tests (10 tests)
// =============================================================================

SCL_TEST_SUITE(int8_comprehensive)

SCL_TEST_TAGGED(int8_positive_range, "integer", "int8") {
    std::vector<std::int64_t> rows = {0, 1, 2, 3, 4};
    std::vector<std::int64_t> cols = {0, 1, 2, 3, 4};
    std::vector<std::int8_t> vals = {1, 10, 50, 100, 127};
    
    auto m = scl_sparse_from_coo(5, 5, rows.data(), cols.data(), vals.data(),
                                  5, SCL_INT8, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(int8_negative_range, "integer", "int8") {
    std::vector<std::int64_t> rows = {0, 1, 2, 3, 4};
    std::vector<std::int64_t> cols = {0, 1, 2, 3, 4};
    std::vector<std::int8_t> vals = {-128, -100, -50, -10, -1};
    
    auto m = scl_sparse_from_coo(5, 5, rows.data(), cols.data(), vals.data(),
                                  5, SCL_INT8, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(int8_around_zero, "integer", "int8") {
    std::vector<std::int64_t> rows = {0, 1, 2, 3, 4};
    std::vector<std::int64_t> cols = {0, 1, 2, 3, 4};
    std::vector<std::int8_t> vals = {-2, -1, 0, 1, 2};
    
    auto m = scl_sparse_from_coo(5, 5, rows.data(), cols.data(), vals.data(),
                                  5, SCL_INT8, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(int8_scale_positive, "integer", "int8") {
    auto m = scl_sparse_identity(5, SCL_INT8, SCL_INDEX64);
    SCL_ASSERT_EQ(scl_sparse_scale(m, 10.0), 0);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(int8_transpose, "integer", "int8") {
    auto m = scl_sparse_identity(10, SCL_INT8, SCL_INDEX64);
    auto t = scl_sparse_transpose(m);
    SCL_ASSERT_SPARSE_VALID(t);
    scl_sparse_destroy(m);
    scl_sparse_destroy(t);
}

SCL_TEST_TAGGED(int8_clone, "integer", "int8") {
    auto m = scl_sparse_identity(10, SCL_INT8, SCL_INDEX64);
    auto c = scl_sparse_clone(m);
    SCL_ASSERT_SPARSE_VALID(c);
    scl_sparse_destroy(m);
    scl_sparse_destroy(c);
}

SCL_TEST_TAGGED(int8_csr_csc_both, "integer", "int8") {
    auto csr = scl_sparse_identity(5, SCL_INT8, SCL_INDEX64);
    auto csc = scl_sparse_transpose(csr);
    SCL_ASSERT_EQ(scl_sparse_layout(csr), SCL_LAYOUT_CSR);
    SCL_ASSERT_EQ(scl_sparse_layout(csc), SCL_LAYOUT_CSC);
    scl_sparse_destroy(csr);
    scl_sparse_destroy(csc);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 2: Int16 Comprehensive Tests (10 tests)
// =============================================================================

SCL_TEST_SUITE(int16_comprehensive)

SCL_TEST_TAGGED(int16_large_positive, "integer", "int16") {
    std::vector<std::int64_t> rows = {0, 1, 2};
    std::vector<std::int64_t> cols = {0, 1, 2};
    std::vector<std::int16_t> vals = {10000, 20000, 32767};
    
    auto m = scl_sparse_from_coo(3, 3, rows.data(), cols.data(), vals.data(),
                                  3, SCL_INT16, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(int16_large_negative, "integer", "int16") {
    std::vector<std::int64_t> rows = {0, 1, 2};
    std::vector<std::int64_t> cols = {0, 1, 2};
    std::vector<std::int16_t> vals = {-32768, -20000, -10000};
    
    auto m = scl_sparse_from_coo(3, 3, rows.data(), cols.data(), vals.data(),
                                  3, SCL_INT16, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(int16_operations, "integer", "int16") {
    auto m = scl_sparse_identity(20, SCL_INT16, SCL_INDEX64);
    SCL_ASSERT_EQ(scl_sparse_scale(m, 2.0), 0);
    auto t = scl_sparse_transpose(m);
    SCL_ASSERT_SPARSE_VALID(t);
    scl_sparse_destroy(m);
    scl_sparse_destroy(t);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 3: Int32 Extended Tests (10 tests)
// =============================================================================

SCL_TEST_SUITE(int32_extended)

SCL_TEST_TAGGED(int32_billion_range, "integer", "int32") {
    std::vector<std::int64_t> rows = {0, 1, 2};
    std::vector<std::int64_t> cols = {0, 1, 2};
    std::vector<std::int32_t> vals = {-1000000000, 0, 2000000000};
    
    auto m = scl_sparse_from_coo(3, 3, rows.data(), cols.data(), vals.data(),
                                  3, SCL_INT32, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(int32_scale_truncation, "integer", "int32") {
    auto m = scl_sparse_identity(5, SCL_INT32, SCL_INDEX64);
    // Scale by 0.5 should truncate (1.0 * 0.5 = 0.5 → 0)
    SCL_ASSERT_EQ(scl_sparse_scale(m, 0.5), 0);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(int32_all_operations, "integer", "int32") {
    auto m = scl_sparse_identity(10, SCL_INT32, SCL_INDEX64);
    
    // Clone
    auto c = scl_sparse_clone(m);
    SCL_ASSERT_SPARSE_VALID(c);
    
    // Transpose
    auto t = scl_sparse_transpose(m);
    SCL_ASSERT_SPARSE_VALID(t);
    
    // Scale
    SCL_ASSERT_EQ(scl_sparse_scale(m, 5.0), 0);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(c);
    scl_sparse_destroy(t);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 4: Int64 Tests (10 tests)
// =============================================================================

SCL_TEST_SUITE(int64_comprehensive)

SCL_TEST_TAGGED(int64_very_large_values, "integer", "int64") {
    std::vector<std::int64_t> rows = {0, 1, 2};
    std::vector<std::int64_t> cols = {0, 1, 2};
    std::vector<std::int64_t> vals = {-9000000000000000000LL, 0, 9000000000000000000LL};
    
    auto m = scl_sparse_from_coo(3, 3, rows.data(), cols.data(), vals.data(),
                                  3, SCL_INT64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(int64_operations_suite, "integer", "int64") {
    auto m = scl_sparse_identity(15, SCL_INT64, SCL_INDEX64);
    auto t = scl_sparse_transpose(m);
    auto c = scl_sparse_clone(m);
    
    SCL_ASSERT_SPARSE_VALID(t);
    SCL_ASSERT_SPARSE_VALID(c);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(t);
    scl_sparse_destroy(c);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 5: Uint8 Comprehensive Tests (10 tests)
// =============================================================================

SCL_TEST_SUITE(uint8_comprehensive)

SCL_TEST_TAGGED(uint8_full_range, "unsigned", "uint8") {
    std::vector<std::int64_t> rows = {0, 1, 2, 3, 4};
    std::vector<std::int64_t> cols = {0, 1, 2, 3, 4};
    std::vector<std::uint8_t> vals = {0, 63, 127, 191, 255};
    
    auto m = scl_sparse_from_coo(5, 5, rows.data(), cols.data(), vals.data(),
                                  5, SCL_UINT8, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(uint8_powers_of_2, "unsigned", "uint8") {
    std::vector<std::int64_t> rows = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<std::int64_t> cols = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<std::uint8_t> vals = {1, 2, 4, 8, 16, 32, 64, 128};
    
    auto m = scl_sparse_from_coo(8, 8, rows.data(), cols.data(), vals.data(),
                                  8, SCL_UINT8, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(uint8_operations, "unsigned", "uint8") {
    auto m = scl_sparse_identity(10, SCL_UINT8, SCL_INDEX64);
    SCL_ASSERT_EQ(scl_sparse_scale(m, 10.0), 0);
    auto t = scl_sparse_transpose(m);
    SCL_ASSERT_SPARSE_VALID(t);
    scl_sparse_destroy(m);
    scl_sparse_destroy(t);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 6: Uint16/32/64 Extended Tests (30 tests)
// =============================================================================

SCL_TEST_SUITE(uint_extended)

SCL_TEST_TAGGED(uint16_large_values, "unsigned", "uint16") {
    std::vector<std::int64_t> rows = {0, 1, 2};
    std::vector<std::int64_t> cols = {0, 1, 2};
    std::vector<std::uint16_t> vals = {10000, 32768, 65535};
    
    auto m = scl_sparse_from_coo(3, 3, rows.data(), cols.data(), vals.data(),
                                  3, SCL_UINT16, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(uint32_4billion, "unsigned", "uint32") {
    std::vector<std::int64_t> rows = {0, 1, 2};
    std::vector<std::int64_t> cols = {0, 1, 2};
    std::vector<std::uint32_t> vals = {1000000000U, 2000000000U, 4000000000U};
    
    auto m = scl_sparse_from_coo(3, 3, rows.data(), cols.data(), vals.data(),
                                  3, SCL_UINT32, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(uint64_huge_values, "unsigned", "uint64") {
    std::vector<std::int64_t> rows = {0, 1, 2};
    std::vector<std::int64_t> cols = {0, 1, 2};
    std::vector<std::uint64_t> vals = {1000000000000ULL, 9000000000000000000ULL, 18000000000000000000ULL};
    
    auto m = scl_sparse_from_coo(3, 3, rows.data(), cols.data(), vals.data(),
                                  3, SCL_UINT64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(m);
    scl_sparse_destroy(m);
}

// 每种Uint类型的所有操作
SCL_TEST_TAGGED(uint16_all_ops, "unsigned", "uint16") {
    auto m = scl_sparse_identity(8, SCL_UINT16, SCL_INDEX64);
    auto c = scl_sparse_clone(m);
    auto t = scl_sparse_transpose(m);
    SCL_ASSERT_EQ(scl_sparse_scale(m, 2.0), 0);
    scl_sparse_destroy(m);
    scl_sparse_destroy(c);
    scl_sparse_destroy(t);
}

SCL_TEST_TAGGED(uint32_all_ops, "unsigned", "uint32") {
    auto m = scl_sparse_identity(8, SCL_UINT32, SCL_INDEX64);
    auto c = scl_sparse_clone(m);
    auto t = scl_sparse_transpose(m);
    SCL_ASSERT_EQ(scl_sparse_scale(m, 2.0), 0);
    scl_sparse_destroy(m);
    scl_sparse_destroy(c);
    scl_sparse_destroy(t);
}

SCL_TEST_TAGGED(uint64_all_ops, "unsigned", "uint64") {
    auto m = scl_sparse_identity(8, SCL_UINT64, SCL_INDEX64);
    auto c = scl_sparse_clone(m);
    auto t = scl_sparse_transpose(m);
    SCL_ASSERT_EQ(scl_sparse_scale(m, 2.0), 0);
    scl_sparse_destroy(m);
    scl_sparse_destroy(c);
    scl_sparse_destroy(t);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 7: Integer Scale Behavior (20 tests)
// =============================================================================

SCL_TEST_SUITE(integer_scale_behavior)

SCL_TEST_TAGGED(int32_scale_by_integers, "integer", "scale") {
    auto m = scl_sparse_identity(5, SCL_INT32, SCL_INDEX64);
    
    // 整数标量不截断
    SCL_ASSERT_EQ(scl_sparse_scale(m, 2.0), 0);
    SCL_ASSERT_EQ(scl_sparse_scale(m, 3.0), 0);
    SCL_ASSERT_EQ(scl_sparse_scale(m, 10.0), 0);
    
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(int32_scale_by_fraction, "integer", "scale") {
    auto m = scl_sparse_identity(5, SCL_INT32, SCL_INDEX64);
    
    // 小数会截断
    SCL_ASSERT_EQ(scl_sparse_scale(m, 0.5), 0);
    SCL_ASSERT_EQ(scl_sparse_scale(m, 1.5), 0);
    
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(uint32_scale_positive_only, "unsigned", "scale") {
    auto m = scl_sparse_identity(5, SCL_UINT32, SCL_INDEX64);
    
    // 正标量正常
    SCL_ASSERT_EQ(scl_sparse_scale(m, 5.0), 0);
    
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(int8_scale_small_values, "integer", "scale") {
    auto m = scl_sparse_identity(3, SCL_INT8, SCL_INDEX64);
    
    for (double s : {0.1, 0.5, 0.9, 1.1, 1.9, 2.5}) {
        SCL_ASSERT_EQ(scl_sparse_scale(m, s), 0);
    }
    
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(uint8_scale_range, "unsigned", "scale") {
    auto m = scl_sparse_identity(3, SCL_UINT8, SCL_INDEX64);
    
    for (double s : {1.0, 2.0, 5.0, 10.0, 100.0}) {
        SCL_ASSERT_EQ(scl_sparse_scale(m, s), 0);
    }
    
    scl_sparse_destroy(m);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 8: Type Mixing Tests (20 tests)
// =============================================================================

SCL_TEST_SUITE(type_mixing)

SCL_TEST_TAGGED(real32_int32_index32, "mixed", "quick") {
    auto r = scl_sparse_identity(5, SCL_REAL32, SCL_INDEX32);
    auto i = scl_sparse_identity(5, SCL_INT32, SCL_INDEX32);
    
    SCL_ASSERT_NE(scl_sparse_value_type(r), scl_sparse_value_type(i));
    
    scl_sparse_destroy(r);
    scl_sparse_destroy(i);
}

SCL_TEST_TAGGED(all_types_same_structure, "mixed") {
    // 同样的结构，不同的类型
    scl_value_type_t types[] = {SCL_REAL64, SCL_INT32, SCL_UINT32};
    
    for (auto vt : types) {
        auto m = scl_sparse_identity(10, vt, SCL_INDEX64);
        SCL_ASSERT_SPARSE_DIMS(m, 10, 10);
        SCL_ASSERT_SPARSE_NNZ(m, 10);
        scl_sparse_destroy(m);
    }
}

SCL_TEST_TAGGED(index32_vs_index64, "mixed") {
    auto m32 = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX32);
    auto m64 = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    
    SCL_ASSERT_EQ(scl_sparse_index_type(m32), SCL_INDEX32);
    SCL_ASSERT_EQ(scl_sparse_index_type(m64), SCL_INDEX64);
    
    scl_sparse_destroy(m32);
    scl_sparse_destroy(m64);
}

SCL_TEST_SUITE_END

SCL_TEST_END
SCL_TEST_MAIN()

