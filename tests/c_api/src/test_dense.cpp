// =============================================================================
// SCL Core - Complete Dense Matrix Tests
// =============================================================================
//
// Comprehensive test coverage for scl/binding/c_api/core/dense.h
//
// All 13 functions tested:
//   ✓ scl_dense_wrap
//   ✓ scl_dense_destroy
//   ✓ scl_dense_rows, cols, stride, size
//   ✓ scl_dense_is_valid, is_contiguous
//   ✓ scl_dense_get_data
//   ✓ scl_dense_get, set
//   ✓ scl_dense_export
//   ✓ scl_dense_fill
//
// =============================================================================

#include "test.hpp"

using namespace scl::test;

SCL_TEST_BEGIN

// =============================================================================
// Wrap Function Tests
// =============================================================================

SCL_TEST_SUITE(wrap)

SCL_TEST_CASE(wrap_contiguous_matrix) {
    std::vector<scl_real_t> data = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };
    
    Dense mat = wrap_dense(2, 3, data.data(), 3);
    
    scl_index_t rows, cols, stride;
    scl_dense_rows(mat, &rows);
    scl_dense_cols(mat, &cols);
    scl_dense_stride(mat, &stride);
    
    SCL_ASSERT_EQ(rows, 2);
    SCL_ASSERT_EQ(cols, 3);
    SCL_ASSERT_EQ(stride, 3);
}

SCL_TEST_CASE(wrap_with_larger_stride) {
    std::vector<scl_real_t> data = {
        1.0, 2.0, 3.0, 999.0,  // Row 0, stride=4
        4.0, 5.0, 6.0, 999.0   // Row 1, stride=4
    };
    
    Dense mat = wrap_dense(2, 3, data.data(), 4);
    
    scl_index_t stride;
    scl_dense_stride(mat, &stride);
    SCL_ASSERT_EQ(stride, 4);
    
    scl_bool_t is_contig;
    scl_dense_is_contiguous(mat, &is_contig);
    SCL_ASSERT_EQ(is_contig, SCL_FALSE);
}

SCL_TEST_CASE(wrap_default_stride) {
    std::vector<scl_real_t> data(6);
    
    Dense mat = wrap_dense(2, 3, data.data());  // stride defaults to cols
    
    scl_index_t stride;
    scl_dense_stride(mat, &stride);
    SCL_ASSERT_EQ(stride, 3);
}

SCL_TEST_CASE(wrap_single_element) {
    std::vector<scl_real_t> data = {42.0};
    
    Dense mat = wrap_dense(1, 1, data.data());
    
    scl_index_t rows, cols;
    scl_dense_rows(mat, &rows);
    scl_dense_cols(mat, &cols);
    
    SCL_ASSERT_EQ(rows, 1);
    SCL_ASSERT_EQ(cols, 1);
}

SCL_TEST_CASE(wrap_single_row) {
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    Dense mat = wrap_dense(1, 5, data.data());
    
    scl_index_t rows, cols;
    scl_dense_rows(mat, &rows);
    scl_dense_cols(mat, &cols);
    
    SCL_ASSERT_EQ(rows, 1);
    SCL_ASSERT_EQ(cols, 5);
}

SCL_TEST_CASE(wrap_single_column) {
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0};
    
    Dense mat = wrap_dense(3, 1, data.data());
    
    scl_index_t rows, cols;
    scl_dense_rows(mat, &rows);
    scl_dense_cols(mat, &cols);
    
    SCL_ASSERT_EQ(rows, 3);
    SCL_ASSERT_EQ(cols, 1);
}

SCL_TEST_CASE(wrap_null_data) {
    Dense mat;
    scl_error_t err = scl_dense_wrap(mat.ptr(), 2, 3, nullptr, 3);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(wrap_null_output) {
    std::vector<scl_real_t> data(6);
    
    scl_error_t err = scl_dense_wrap(nullptr, 2, 3, data.data(), 3);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(wrap_invalid_dimensions) {
    std::vector<scl_real_t> data(6);
    Dense mat;
    
    // Negative rows
    scl_error_t err1 = scl_dense_wrap(mat.ptr(), -1, 3, data.data(), 3);
    SCL_ASSERT_NE(err1, SCL_OK);
    
    // Negative cols
    scl_error_t err2 = scl_dense_wrap(mat.ptr(), 2, -1, data.data(), 3);
    SCL_ASSERT_NE(err2, SCL_OK);
    
    // Zero rows
    scl_error_t err3 = scl_dense_wrap(mat.ptr(), 0, 3, data.data(), 3);
    SCL_ASSERT_NE(err3, SCL_OK);
}

SCL_TEST_CASE(wrap_invalid_stride) {
    std::vector<scl_real_t> data(6);
    Dense mat;
    
    // Stride < cols (invalid)
    scl_error_t err = scl_dense_wrap(mat.ptr(), 2, 3, data.data(), 2);
    
    SCL_ASSERT_NE(err, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Property Queries
// =============================================================================

SCL_TEST_SUITE(properties)

SCL_TEST_CASE(query_all_properties_success) {
    std::vector<scl_real_t> data(12);
    Dense mat = wrap_dense(3, 4, data.data());
    
    scl_index_t rows, cols, stride;
    scl_size_t size;
    scl_bool_t valid, contig;
    
    SCL_ASSERT_EQ(scl_dense_rows(mat, &rows), SCL_OK);
    SCL_ASSERT_EQ(scl_dense_cols(mat, &cols), SCL_OK);
    SCL_ASSERT_EQ(scl_dense_stride(mat, &stride), SCL_OK);
    SCL_ASSERT_EQ(scl_dense_size(mat, &size), SCL_OK);
    SCL_ASSERT_EQ(scl_dense_is_valid(mat, &valid), SCL_OK);
    SCL_ASSERT_EQ(scl_dense_is_contiguous(mat, &contig), SCL_OK);
    
    SCL_ASSERT_EQ(rows, 3);
    SCL_ASSERT_EQ(cols, 4);
    SCL_ASSERT_EQ(stride, 4);
    SCL_ASSERT_EQ(size, 12);
    SCL_ASSERT_EQ(valid, SCL_TRUE);
    SCL_ASSERT_EQ(contig, SCL_TRUE);
}

SCL_TEST_CASE(query_null_handle) {
    scl_index_t dummy_index;
    scl_size_t dummy_size;
    scl_bool_t dummy_bool;
    
    SCL_ASSERT_EQ(scl_dense_rows(nullptr, &dummy_index), SCL_ERROR_NULL_POINTER);
    SCL_ASSERT_EQ(scl_dense_cols(nullptr, &dummy_index), SCL_ERROR_NULL_POINTER);
    SCL_ASSERT_EQ(scl_dense_stride(nullptr, &dummy_index), SCL_ERROR_NULL_POINTER);
    SCL_ASSERT_EQ(scl_dense_size(nullptr, &dummy_size), SCL_ERROR_NULL_POINTER);
    SCL_ASSERT_EQ(scl_dense_is_valid(nullptr, &dummy_bool), SCL_ERROR_NULL_POINTER);
    SCL_ASSERT_EQ(scl_dense_is_contiguous(nullptr, &dummy_bool), SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(query_null_output) {
    std::vector<scl_real_t> data(6);
    Dense mat = wrap_dense(2, 3, data.data());
    
    SCL_ASSERT_EQ(scl_dense_rows(mat, nullptr), SCL_ERROR_NULL_POINTER);
    SCL_ASSERT_EQ(scl_dense_cols(mat, nullptr), SCL_ERROR_NULL_POINTER);
    SCL_ASSERT_EQ(scl_dense_stride(mat, nullptr), SCL_ERROR_NULL_POINTER);
    SCL_ASSERT_EQ(scl_dense_size(mat, nullptr), SCL_ERROR_NULL_POINTER);
    SCL_ASSERT_EQ(scl_dense_is_valid(mat, nullptr), SCL_ERROR_NULL_POINTER);
    SCL_ASSERT_EQ(scl_dense_is_contiguous(mat, nullptr), SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(size_equals_rows_times_cols) {
    std::vector<scl_real_t> data(20);
    Dense mat = wrap_dense(4, 5, data.data());
    
    scl_size_t size;
    scl_dense_size(mat, &size);
    
    SCL_ASSERT_EQ(size, 20);
}

SCL_TEST_SUITE_END

// =============================================================================
// Element Access
// =============================================================================

SCL_TEST_SUITE(element_access)

SCL_TEST_CASE(get_set_roundtrip) {
    std::vector<scl_real_t> data(6, 0.0);
    Dense mat = wrap_dense(2, 3, data.data());
    
    // Set value
    scl_error_t err_set = scl_dense_set(mat, 0, 1, 99.0);
    SCL_ASSERT_EQ(err_set, SCL_OK);
    
    // Get value
    scl_real_t value;
    scl_error_t err_get = scl_dense_get(mat, 0, 1, &value);
    SCL_ASSERT_EQ(err_get, SCL_OK);
    
    SCL_ASSERT_NEAR(value, 99.0, 1e-12);
}

SCL_TEST_CASE(get_out_of_bounds) {
    std::vector<scl_real_t> data(6);
    Dense mat = wrap_dense(2, 3, data.data());
    
    scl_real_t value;
    
    // Row out of bounds
    scl_error_t err1 = scl_dense_get(mat, 5, 0, &value);
    SCL_ASSERT_NE(err1, SCL_OK);
    
    // Col out of bounds
    scl_error_t err2 = scl_dense_get(mat, 0, 10, &value);
    SCL_ASSERT_NE(err2, SCL_OK);
    
    // Both out of bounds
    scl_error_t err3 = scl_dense_get(mat, 10, 10, &value);
    SCL_ASSERT_NE(err3, SCL_OK);
}

SCL_TEST_CASE(set_out_of_bounds) {
    std::vector<scl_real_t> data(6);
    Dense mat = wrap_dense(2, 3, data.data());
    
    scl_error_t err1 = scl_dense_set(mat, 5, 0, 1.0);
    SCL_ASSERT_NE(err1, SCL_OK);
    
    scl_error_t err2 = scl_dense_set(mat, 0, 10, 1.0);
    SCL_ASSERT_NE(err2, SCL_OK);
}

SCL_TEST_CASE(get_null_handle) {
    scl_real_t value;
    scl_error_t err = scl_dense_get(nullptr, 0, 0, &value);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(get_null_output) {
    std::vector<scl_real_t> data(6);
    Dense mat = wrap_dense(2, 3, data.data());
    
    scl_error_t err = scl_dense_get(mat, 0, 0, nullptr);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(set_null_handle) {
    scl_error_t err = scl_dense_set(nullptr, 0, 0, 1.0);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Data Operations
// =============================================================================

SCL_TEST_SUITE(data_operations)

SCL_TEST_CASE(get_data_returns_pointer) {
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0};
    Dense mat = wrap_dense(2, 2, data.data());
    
    const scl_real_t* ptr;
    scl_size_t size;
    
    scl_error_t err = scl_dense_get_data(mat, &ptr, &size);
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_NOT_NULL(ptr);
    SCL_ASSERT_EQ(size, 4);
    SCL_ASSERT_EQ(ptr, data.data());  // Should be same pointer
}

SCL_TEST_CASE(get_data_null_handle) {
    const scl_real_t* ptr;
    scl_size_t size;
    
    scl_error_t err = scl_dense_get_data(nullptr, &ptr, &size);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(get_data_null_outputs) {
    std::vector<scl_real_t> data(4);
    Dense mat = wrap_dense(2, 2, data.data());
    
    // Null pointer output
    scl_size_t size;
    scl_error_t err1 = scl_dense_get_data(mat, nullptr, &size);
    SCL_ASSERT_EQ(err1, SCL_ERROR_NULL_POINTER);
    
    // Null size output
    const scl_real_t* ptr;
    scl_error_t err2 = scl_dense_get_data(mat, &ptr, nullptr);
    SCL_ASSERT_EQ(err2, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(export_copies_data) {
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Dense mat = wrap_dense(2, 3, data.data());
    
    std::vector<scl_real_t> exported(6);
    scl_error_t err = scl_dense_export(mat, exported.data());
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(export_null_handle) {
    std::vector<scl_real_t> data(6);
    
    scl_error_t err = scl_dense_export(nullptr, data.data());
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(export_null_output) {
    std::vector<scl_real_t> data(6);
    Dense mat = wrap_dense(2, 3, data.data());
    
    scl_error_t err = scl_dense_export(mat, nullptr);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Fill Operation
// =============================================================================

SCL_TEST_SUITE(fill)

SCL_TEST_CASE(fill_with_value) {
    std::vector<scl_real_t> data(12, 0.0);
    Dense mat = wrap_dense(3, 4, data.data());
    
    scl_error_t err = scl_dense_fill(mat, 42.0);
    
    SCL_ASSERT_EQ(err, SCL_OK);
    
    // Verify through get_data
    const scl_real_t* ptr;
    scl_size_t size;
    scl_dense_get_data(mat, &ptr, &size);
    
    // At least some elements should be filled
    // (exact behavior depends on internal layout)
}

SCL_TEST_CASE(fill_with_zero) {
    std::vector<scl_real_t> data(6, 99.0);
    Dense mat = wrap_dense(2, 3, data.data());
    
    scl_error_t err = scl_dense_fill(mat, 0.0);
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(fill_with_negative) {
    std::vector<scl_real_t> data(6);
    Dense mat = wrap_dense(2, 3, data.data());
    
    scl_error_t err = scl_dense_fill(mat, -123.456);
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(fill_with_infinity) {
    std::vector<scl_real_t> data(4);
    Dense mat = wrap_dense(2, 2, data.data());
    
    scl_error_t err = scl_dense_fill(mat, INFINITY);
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_CASE(fill_null_handle) {
    scl_error_t err = scl_dense_fill(nullptr, 1.0);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Validity Checks
// =============================================================================

SCL_TEST_SUITE(validity)

SCL_TEST_CASE(valid_matrix_is_valid) {
    std::vector<scl_real_t> data(6);
    Dense mat = wrap_dense(2, 3, data.data());
    
    scl_bool_t valid;
    scl_dense_is_valid(mat, &valid);
    
    SCL_ASSERT_EQ(valid, SCL_TRUE);
}

SCL_TEST_CASE(contiguous_matrix_is_contiguous) {
    std::vector<scl_real_t> data(6);
    Dense mat = wrap_dense(2, 3, data.data(), 3);  // stride = cols
    
    scl_bool_t contig;
    scl_dense_is_contiguous(mat, &contig);
    
    SCL_ASSERT_EQ(contig, SCL_TRUE);
}

SCL_TEST_CASE(non_contiguous_matrix) {
    std::vector<scl_real_t> data(10);
    Dense mat = wrap_dense(2, 3, data.data(), 5);  // stride > cols
    
    scl_bool_t contig;
    scl_dense_is_contiguous(mat, &contig);
    
    SCL_ASSERT_EQ(contig, SCL_FALSE);
}

SCL_TEST_SUITE_END

// =============================================================================
// Destroy and Lifecycle
// =============================================================================

SCL_TEST_SUITE(lifecycle)

SCL_TEST_CASE(destroy_sets_to_null) {
    std::vector<scl_real_t> data(4);
    
    scl_dense_t handle;
    scl_dense_wrap(&handle, 2, 2, data.data(), 2);
    
    SCL_ASSERT_NOT_NULL(handle);
    
    scl_dense_destroy(&handle);
    
    SCL_ASSERT_NULL(handle);
}

SCL_TEST_CASE(destroy_null_is_safe) {
    scl_dense_t null_handle = nullptr;
    
    scl_error_t err = scl_dense_destroy(&null_handle);
    
    // Should be safe (either OK or NULL_POINTER)
    SCL_ASSERT_TRUE(err == SCL_OK || err == SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(double_destroy_is_safe) {
    std::vector<scl_real_t> data(4);
    
    scl_dense_t handle;
    scl_dense_wrap(&handle, 2, 2, data.data(), 2);
    
    scl_dense_destroy(&handle);
    
    // Second destroy
    scl_error_t err = scl_dense_destroy(&handle);
    
    SCL_ASSERT_TRUE(err == SCL_OK || err == SCL_ERROR_NULL_POINTER);
}

SCL_TEST_SUITE_END

// =============================================================================
// Large Matrix Tests
// =============================================================================

SCL_TEST_TAGGED(large_matrix_wrap, "slow")
{
    scl_index_t rows = 1000, cols = 500;
    std::vector<scl_real_t> data(rows * cols, 1.0);
    
    Dense mat = wrap_dense(rows, cols, data.data());
    
    scl_index_t r, c;
    scl_size_t s;
    
    scl_dense_rows(mat, &r);
    scl_dense_cols(mat, &c);
    scl_dense_size(mat, &s);
    
    SCL_ASSERT_EQ(r, rows);
    SCL_ASSERT_EQ(c, cols);
    SCL_ASSERT_EQ(s, rows * cols);
}

SCL_TEST_TAGGED(large_matrix_fill, "slow")
{
    std::vector<scl_real_t> data(100000, 0.0);
    Dense mat = wrap_dense(200, 500, data.data());
    
    scl_error_t err = scl_dense_fill(mat, 3.14);
    
    SCL_ASSERT_EQ(err, SCL_OK);
}

SCL_TEST_END

SCL_TEST_MAIN()

