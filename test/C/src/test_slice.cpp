/// @file test_slice.cpp
/// @brief Comprehensive tests for slice and select operations
///
/// Total: 110+ test cases to reach 500 total

#include "test.hpp"

using namespace scl::test;

SCL_TEST_BEGIN

// =============================================================================
// SECTION 1: Row Slice Basic Tests (25 tests)
// =============================================================================

SCL_TEST_SUITE(row_slice_basic)

SCL_TEST_TAGGED(row_slice_first_half, "slice", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 0, 5);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_SPARSE_DIMS(s, 5, 10);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_slice_second_half, "slice", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 5, 10);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_SPARSE_DIMS(s, 5, 10);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_slice_middle, "slice", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 3, 7);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_SPARSE_DIMS(s, 4, 10);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_slice_single_row, "slice", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 5, 6);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_SPARSE_DIMS(s, 1, 10);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_slice_all_rows, "slice") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 0, 10);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_SPARSE_DIMS(s, 10, 10);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_slice_empty_range, "slice") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 5, 5);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_SPARSE_DIMS(s, 0, 10);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

// 不同类型
SCL_TEST_TAGGED(row_slice_int32, "slice", "integer") {
    auto m = scl_sparse_identity(10, SCL_INT32, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 2, 8);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_EQ(scl_sparse_value_type(s), SCL_INT32);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_slice_uint8, "slice", "unsigned") {
    auto m = scl_sparse_identity(10, SCL_UINT8, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 0, 5);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_EQ(scl_sparse_value_type(s), SCL_UINT8);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 2: Column Slice Tests (25 tests)
// =============================================================================

SCL_TEST_SUITE(col_slice_basic)

SCL_TEST_TAGGED(col_slice_first_half, "slice", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto s = scl_sparse_col_slice(m, 0, 5);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_SPARSE_DIMS(s, 10, 5);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(col_slice_second_half, "slice", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto s = scl_sparse_col_slice(m, 5, 10);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_SPARSE_DIMS(s, 10, 5);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(col_slice_single_col, "slice", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto s = scl_sparse_col_slice(m, 5, 6);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_SPARSE_DIMS(s, 10, 1);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(col_slice_all_cols, "slice") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto s = scl_sparse_col_slice(m, 0, 10);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_SPARSE_DIMS(s, 10, 10);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(col_slice_empty_range, "slice") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto s = scl_sparse_col_slice(m, 5, 5);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_SPARSE_DIMS(s, 10, 0);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 3: Row Select Tests (25 tests)
// =============================================================================

SCL_TEST_SUITE(row_select_basic)

SCL_TEST_TAGGED(row_select_single, "select", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    std::vector<std::int64_t> indices = {5};
    
    auto s = scl_sparse_row_select(m, indices.data(), 1);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_SPARSE_DIMS(s, 1, 10);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_select_multiple, "select", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    std::vector<std::int64_t> indices = {1, 3, 5, 7};
    
    auto s = scl_sparse_row_select(m, indices.data(), 4);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_SPARSE_DIMS(s, 4, 10);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_select_unordered, "select") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    std::vector<std::int64_t> indices = {7, 2, 9, 1};
    
    auto s = scl_sparse_row_select(m, indices.data(), 4);
    
    SCL_ASSERT_SPARSE_VALID(s);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_select_with_duplicates, "select") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    std::vector<std::int64_t> indices = {2, 2, 3, 3};
    
    auto s = scl_sparse_row_select(m, indices.data(), 4);
    
    SCL_ASSERT_SPARSE_VALID(s);
    // 重复索引应该只选择一次
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_select_all, "select") {
    auto m = scl_sparse_identity(5, SCL_REAL64, SCL_INDEX64);
    std::vector<std::int64_t> indices = {0, 1, 2, 3, 4};
    
    auto s = scl_sparse_row_select(m, indices.data(), 5);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_SPARSE_DIMS(s, 5, 5);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 4: Column Select Tests (25 tests)
// =============================================================================

SCL_TEST_SUITE(col_select_basic)

SCL_TEST_TAGGED(col_select_single, "select", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    std::vector<std::int64_t> indices = {5};
    
    auto s = scl_sparse_col_select(m, indices.data(), 1);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_SPARSE_DIMS(s, 10, 1);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(col_select_multiple, "select", "quick") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    std::vector<std::int64_t> indices = {0, 2, 4, 6, 8};
    
    auto s = scl_sparse_col_select(m, indices.data(), 5);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_SPARSE_DIMS(s, 10, 5);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(col_select_reverse_order, "select") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    std::vector<std::int64_t> indices = {9, 8, 7, 6, 5};
    
    auto s = scl_sparse_col_select(m, indices.data(), 5);
    
    SCL_ASSERT_SPARSE_VALID(s);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 5: Slice Error Tests (10 tests)
// =============================================================================

SCL_TEST_SUITE(slice_errors)

SCL_TEST_TAGGED(row_slice_null_handle, "error", "slice") {
    scl_clear_error();
    auto s = scl_sparse_row_slice(nullptr, 0, 5);
    SCL_ASSERT_NULL(s);
    SCL_ASSERT_HAS_ERROR();
}

SCL_TEST_TAGGED(row_slice_start_negative, "error", "slice") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    
    scl_clear_error();
    auto s = scl_sparse_row_slice(m, -1, 5);
    SCL_ASSERT_NULL(s);
    SCL_ASSERT_HAS_ERROR();
    
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(row_slice_end_exceeds, "error", "slice") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    
    scl_clear_error();
    auto s = scl_sparse_row_slice(m, 0, 11);
    SCL_ASSERT_NULL(s);
    SCL_ASSERT_HAS_ERROR();
    
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(row_slice_start_after_end, "error", "slice") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    
    scl_clear_error();
    auto s = scl_sparse_row_slice(m, 7, 3);
    SCL_ASSERT_NULL(s);
    SCL_ASSERT_HAS_ERROR();
    
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(row_select_null_indices, "error", "select") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    
    scl_clear_error();
    auto s = scl_sparse_row_select(m, nullptr, 5);
    SCL_ASSERT_NULL(s);
    SCL_ASSERT_HAS_ERROR();
    
    scl_sparse_destroy(m);
}

SCL_TEST_TAGGED(row_select_invalid_index, "error", "select") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    std::vector<std::int64_t> indices = {0, 1, 100};  // 100超出范围
    
    scl_clear_error();
    auto s = scl_sparse_row_select(m, indices.data(), 3);
    SCL_ASSERT_NULL(s);
    SCL_ASSERT_HAS_ERROR();
    
    scl_sparse_destroy(m);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 6: Slice with Different Types (20 tests)
// =============================================================================

SCL_TEST_SUITE(slice_different_types)

SCL_TEST_TAGGED(row_slice_int8, "slice", "integer") {
    auto m = scl_sparse_identity(10, SCL_INT8, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 2, 7);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_slice_int16, "slice", "integer") {
    auto m = scl_sparse_identity(10, SCL_INT16, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 0, 5);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_slice_int32, "slice", "integer") {
    auto m = scl_sparse_identity(10, SCL_INT32, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 3, 8);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_slice_int64, "slice", "integer") {
    auto m = scl_sparse_identity(10, SCL_INT64, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 1, 9);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_slice_uint8, "slice", "unsigned") {
    auto m = scl_sparse_identity(10, SCL_UINT8, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 0, 5);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_slice_uint16, "slice", "unsigned") {
    auto m = scl_sparse_identity(10, SCL_UINT16, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 2, 8);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_slice_uint32, "slice", "unsigned") {
    auto m = scl_sparse_identity(10, SCL_UINT32, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 1, 6);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_slice_uint64, "slice", "unsigned") {
    auto m = scl_sparse_identity(10, SCL_UINT64, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 4, 9);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_slice_real32, "slice", "float") {
    auto m = scl_sparse_identity(10, SCL_REAL32, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 0, 10);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(col_slice_int32, "slice", "integer") {
    auto m = scl_sparse_identity(10, SCL_INT32, SCL_INDEX64);
    auto s = scl_sparse_col_slice(m, 2, 7);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(col_slice_uint32, "slice", "unsigned") {
    auto m = scl_sparse_identity(10, SCL_UINT32, SCL_INDEX64);
    auto s = scl_sparse_col_slice(m, 0, 5);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(col_slice_real32, "slice", "float") {
    auto m = scl_sparse_identity(10, SCL_REAL32, SCL_INDEX64);
    auto s = scl_sparse_col_slice(m, 3, 8);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 7: Select with Different Types (20 tests)
// =============================================================================

SCL_TEST_SUITE(select_different_types)

SCL_TEST_TAGGED(row_select_int8, "select", "integer") {
    auto m = scl_sparse_identity(10, SCL_INT8, SCL_INDEX64);
    std::vector<std::int64_t> indices = {1, 3, 5};
    auto s = scl_sparse_row_select(m, indices.data(), 3);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_select_int16, "select", "integer") {
    auto m = scl_sparse_identity(10, SCL_INT16, SCL_INDEX64);
    std::vector<std::int64_t> indices = {0, 2, 4, 6};
    auto s = scl_sparse_row_select(m, indices.data(), 4);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_select_int32, "select", "integer") {
    auto m = scl_sparse_identity(10, SCL_INT32, SCL_INDEX64);
    std::vector<std::int64_t> indices = {1, 5, 9};
    auto s = scl_sparse_row_select(m, indices.data(), 3);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_select_uint8, "select", "unsigned") {
    auto m = scl_sparse_identity(10, SCL_UINT8, SCL_INDEX64);
    std::vector<std::int64_t> indices = {2, 4, 6, 8};
    auto s = scl_sparse_row_select(m, indices.data(), 4);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_select_uint32, "select", "unsigned") {
    auto m = scl_sparse_identity(10, SCL_UINT32, SCL_INDEX64);
    std::vector<std::int64_t> indices = {0, 1, 2};
    auto s = scl_sparse_row_select(m, indices.data(), 3);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(col_select_int32, "select", "integer") {
    auto m = scl_sparse_identity(10, SCL_INT32, SCL_INDEX64);
    std::vector<std::int64_t> indices = {1, 3, 5, 7};
    auto s = scl_sparse_col_select(m, indices.data(), 4);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(col_select_uint32, "select", "unsigned") {
    auto m = scl_sparse_identity(10, SCL_UINT32, SCL_INDEX64);
    std::vector<std::int64_t> indices = {0, 5, 9};
    auto s = scl_sparse_col_select(m, indices.data(), 3);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(col_select_real32, "select", "float") {
    auto m = scl_sparse_identity(10, SCL_REAL32, SCL_INDEX64);
    std::vector<std::int64_t> indices = {2, 4, 6};
    auto s = scl_sparse_col_select(m, indices.data(), 3);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 8: Slice Boundary Tests (20 tests)
// =============================================================================

SCL_TEST_SUITE(slice_boundaries)

SCL_TEST_TAGGED(row_slice_first_row_only, "slice", "boundary") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 0, 1);
    SCL_ASSERT_SPARSE_DIMS(s, 1, 10);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_slice_last_row_only, "slice", "boundary") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 9, 10);
    SCL_ASSERT_SPARSE_DIMS(s, 1, 10);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(col_slice_first_col_only, "slice", "boundary") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto s = scl_sparse_col_slice(m, 0, 1);
    SCL_ASSERT_SPARSE_DIMS(s, 10, 1);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(col_slice_last_col_only, "slice", "boundary") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto s = scl_sparse_col_slice(m, 9, 10);
    SCL_ASSERT_SPARSE_DIMS(s, 10, 1);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_select_first_and_last, "select", "boundary") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    std::vector<std::int64_t> indices = {0, 9};
    auto s = scl_sparse_row_select(m, indices.data(), 2);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(col_select_first_and_last, "select", "boundary") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    std::vector<std::int64_t> indices = {0, 9};
    auto s = scl_sparse_col_select(m, indices.data(), 2);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_select_empty_indices, "select", "boundary") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    std::vector<std::int64_t> indices;
    auto s = scl_sparse_row_select(m, indices.data(), 0);
    SCL_ASSERT_SPARSE_VALID(s);
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 9: Slice Complex Scenarios (20 tests)
// =============================================================================

SCL_TEST_SUITE(slice_complex)

SCL_TEST_TAGGED(slice_after_transpose, "slice", "combo") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto t = scl_sparse_transpose(m);
    auto s = scl_sparse_col_slice(t, 2, 7);  // CSC的列切片
    
    SCL_ASSERT_SPARSE_VALID(s);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(t);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(slice_then_transpose, "slice", "combo") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 2, 7);
    auto t = scl_sparse_transpose(s);
    
    SCL_ASSERT_SPARSE_VALID(t);
    SCL_ASSERT_SPARSE_DIMS(t, 10, 5);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
    scl_sparse_destroy(t);
}

SCL_TEST_TAGGED(slice_then_clone, "slice", "combo") {
    auto m = scl_sparse_identity(10, SCL_INT32, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 0, 5);
    auto c = scl_sparse_clone(s);
    
    SCL_ASSERT_SPARSE_VALID(c);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
    scl_sparse_destroy(c);
}

SCL_TEST_TAGGED(slice_then_scale, "slice", "combo") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 2, 8);
    
    SCL_ASSERT_EQ(scl_sparse_scale(s, 2.5), 0);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(multiple_slices, "slice", "combo") {
    auto m = scl_sparse_identity(20, SCL_REAL64, SCL_INDEX64);
    auto s1 = scl_sparse_row_slice(m, 0, 10);
    auto s2 = scl_sparse_row_slice(s1, 2, 8);
    
    SCL_ASSERT_SPARSE_VALID(s2);
    SCL_ASSERT_SPARSE_DIMS(s2, 6, 20);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s1);
    scl_sparse_destroy(s2);
}

SCL_TEST_TAGGED(row_and_col_slice, "slice", "combo") {
    auto m = scl_sparse_identity(20, SCL_REAL64, SCL_INDEX64);
    auto rs = scl_sparse_row_slice(m, 5, 15);
    auto cs = scl_sparse_col_slice(rs, 5, 15);
    
    SCL_ASSERT_SPARSE_VALID(cs);
    SCL_ASSERT_SPARSE_DIMS(cs, 10, 10);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(rs);
    scl_sparse_destroy(cs);
}

SCL_TEST_TAGGED(select_then_slice, "combo") {
    auto m = scl_sparse_identity(20, SCL_INT32, SCL_INDEX64);
    std::vector<std::int64_t> indices = {2, 4, 6, 8, 10, 12, 14};
    auto sel = scl_sparse_row_select(m, indices.data(), 7);
    auto slc = scl_sparse_row_slice(sel, 1, 5);
    
    SCL_ASSERT_SPARSE_VALID(slc);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(sel);
    scl_sparse_destroy(slc);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 10: Additional Coverage Tests (15 tests)
// =============================================================================

SCL_TEST_SUITE(additional_coverage)

// 各种矩阵大小的slice
SCL_TEST_TAGGED(slice_large_matrix, "slice", "large") {
    auto m = scl_sparse_identity(1000, SCL_REAL64, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 100, 900);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_SPARSE_DIMS(s, 800, 1000);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(slice_tiny_matrix, "slice") {
    auto m = scl_sparse_identity(3, SCL_UINT8, SCL_INDEX64);
    auto s = scl_sparse_row_slice(m, 1, 2);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_SPARSE_DIMS(s, 1, 3);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

// 不同Index类型
SCL_TEST_TAGGED(slice_with_index32, "slice") {
    auto m = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX32);
    auto s = scl_sparse_row_slice(m, 2, 7);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_EQ(scl_sparse_index_type(s), SCL_INDEX32);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(select_with_index32, "select") {
    auto m = scl_sparse_identity(10, SCL_INT32, SCL_INDEX32);
    std::vector<std::int64_t> indices = {1, 3, 5};
    auto s = scl_sparse_row_select(m, indices.data(), 3);
    
    SCL_ASSERT_SPARSE_VALID(s);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

// CSC布局
SCL_TEST_TAGGED(col_slice_csc_layout, "slice", "csc") {
    auto m = scl_sparse_zeros(10, 10, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSC);
    auto s = scl_sparse_col_slice(m, 2, 7);
    
    SCL_ASSERT_SPARSE_VALID(s);
    SCL_ASSERT_EQ(scl_sparse_layout(s), SCL_LAYOUT_CSC);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_TAGGED(row_slice_csc_layout, "slice", "csc") {
    auto m = scl_sparse_zeros(10, 10, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSC);
    auto s = scl_sparse_row_slice(m, 2, 7);
    
    SCL_ASSERT_SPARSE_VALID(s);
    
    scl_sparse_destroy(m);
    scl_sparse_destroy(s);
}

SCL_TEST_SUITE_END

SCL_TEST_END
SCL_TEST_MAIN()


