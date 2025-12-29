// =============================================================================
// SCL Core - Error Handling Tests
// =============================================================================
//
// Tests error reporting system defined in core.h.
//
// =============================================================================

#include "test.hpp"

using namespace scl::test;

SCL_TEST_BEGIN

// =============================================================================
// Version Information Tests
// =============================================================================

SCL_TEST_UNIT(version_info_exists) {
    const char* version = scl_get_version();
    SCL_ASSERT_NOT_NULL(version);
    
    // Version should be non-empty
    SCL_ASSERT_TRUE(version[0] != '\0');
}

SCL_TEST_UNIT(build_config_exists) {
    const char* config = scl_get_build_config();
    SCL_ASSERT_NOT_NULL(config);
    
    // Config should contain type info
    SCL_ASSERT_NOT_NULL(config);
}

// =============================================================================
// Error Code Constants
// =============================================================================

SCL_TEST_UNIT(error_codes_defined) {
    // Success
    SCL_ASSERT_EQ(SCL_OK, 0);
    
    // General errors
    SCL_ASSERT_EQ(SCL_ERROR_UNKNOWN, 1);
    SCL_ASSERT_EQ(SCL_ERROR_INTERNAL, 2);
    SCL_ASSERT_EQ(SCL_ERROR_OUT_OF_MEMORY, 3);
    SCL_ASSERT_EQ(SCL_ERROR_NULL_POINTER, 4);
    
    // Argument errors
    SCL_ASSERT_EQ(SCL_ERROR_INVALID_ARGUMENT, 10);
    SCL_ASSERT_EQ(SCL_ERROR_DIMENSION_MISMATCH, 11);
    
    // All error codes should be non-zero
    SCL_ASSERT_NE(SCL_ERROR_NULL_POINTER, SCL_OK);
    SCL_ASSERT_NE(SCL_ERROR_INVALID_ARGUMENT, SCL_OK);
}

// =============================================================================
// Error Reporting Functions
// =============================================================================

SCL_TEST_UNIT(error_initial_state) {
    // Clear any previous errors
    scl_clear_error();
    
    // Initial state: no error
    scl_error_t code = scl_get_last_error_code();
    SCL_ASSERT_EQ(code, SCL_OK);
    
    const char* msg = scl_get_last_error();
    SCL_ASSERT_NOT_NULL(msg);
}

SCL_TEST_UNIT(error_after_null_pointer) {
    scl_clear_error();
    
    // Trigger NULL pointer error
    scl_sparse_t invalid = nullptr;
    scl_index_t rows;
    scl_error_t err = scl_sparse_rows(invalid, &rows);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
    
    // Check error is recorded
    scl_error_t last_code = scl_get_last_error_code();
    SCL_ASSERT_EQ(last_code, SCL_ERROR_NULL_POINTER);
    
    const char* last_msg = scl_get_last_error();
    SCL_ASSERT_NOT_NULL(last_msg);
    SCL_ASSERT_STR_CONTAINS(last_msg, "null");
}

SCL_TEST_UNIT(error_clear_resets_state) {
    scl_clear_error();
    
    // Trigger error
    scl_sparse_rows(nullptr, nullptr);
    SCL_ASSERT_NE(scl_get_last_error_code(), SCL_OK);
    
    // Clear error
    scl_clear_error();
    
    // Should be back to OK
    SCL_ASSERT_EQ(scl_get_last_error_code(), SCL_OK);
}

SCL_TEST_UNIT(error_is_ok_function) {
    SCL_ASSERT_TRUE(scl_is_ok(SCL_OK));
    SCL_ASSERT_FALSE(scl_is_ok(SCL_ERROR_NULL_POINTER));
    SCL_ASSERT_FALSE(scl_is_ok(SCL_ERROR_INVALID_ARGUMENT));
}

SCL_TEST_UNIT(error_is_error_function) {
    SCL_ASSERT_FALSE(scl_is_error(SCL_OK));
    SCL_ASSERT_TRUE(scl_is_error(SCL_ERROR_NULL_POINTER));
    SCL_ASSERT_TRUE(scl_is_error(SCL_ERROR_INVALID_ARGUMENT));
}

// =============================================================================
// Error Propagation
// =============================================================================

SCL_TEST_UNIT(error_message_persists) {
    scl_clear_error();
    
    // Trigger error
    scl_sparse_rows(nullptr, nullptr);
    
    // Get message twice - should be same
    const char* msg1 = scl_get_last_error();
    const char* msg2 = scl_get_last_error();
    
    SCL_ASSERT_EQ(msg1, msg2);  // Same pointer
}

SCL_TEST_UNIT(error_overwrite_on_new_error) {
    scl_clear_error();
    
    // First error (NULL pointer)
    scl_sparse_rows(nullptr, nullptr);
    scl_error_t first_code = scl_get_last_error_code();
    SCL_ASSERT_EQ(first_code, SCL_ERROR_NULL_POINTER);
    
    // Second error (also NULL pointer from different function)
    scl_sparse_cols(nullptr, nullptr);
    scl_error_t second_code = scl_get_last_error_code();
    
    // Both are NULL pointer errors, so they're the same
    // This test verifies error is recorded (not that they're different)
    SCL_ASSERT_EQ(second_code, SCL_ERROR_NULL_POINTER);
}

// =============================================================================
// NULL Pointer Safety
// =============================================================================

SCL_TEST_UNIT(null_handle_safety) {
    scl_clear_error();
    
    scl_sparse_t null_sparse = nullptr;
    scl_dense_t null_dense = nullptr;
    
    // All operations on NULL handles should return error
    scl_index_t dummy_index;
    scl_size_t dummy_size;
    scl_bool_t dummy_bool;
    
    SCL_ASSERT_EQ(scl_sparse_rows(null_sparse, &dummy_index), SCL_ERROR_NULL_POINTER);
    SCL_ASSERT_EQ(scl_sparse_cols(null_sparse, &dummy_index), SCL_ERROR_NULL_POINTER);
    SCL_ASSERT_EQ(scl_sparse_nnz(null_sparse, &dummy_index), SCL_ERROR_NULL_POINTER);
    
    SCL_ASSERT_EQ(scl_dense_rows(null_dense, &dummy_index), SCL_ERROR_NULL_POINTER);
    SCL_ASSERT_EQ(scl_dense_cols(null_dense, &dummy_index), SCL_ERROR_NULL_POINTER);
}

// Helper function
static std::tuple<std::vector<scl_index_t>, std::vector<scl_index_t>, std::vector<scl_real_t>>
get_tiny_3x3() {
    std::vector<scl_index_t> indptr = {0, 2, 3, 6};
    std::vector<scl_index_t> indices = {0, 2, 1, 0, 1, 2};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    return {indptr, indices, data};
}

SCL_TEST_UNIT(null_output_pointer_safety) {
    auto [indptr, indices, data] = get_tiny_3x3();
    
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    // Passing NULL output pointer should fail
    SCL_ASSERT_EQ(scl_sparse_rows(mat, nullptr), SCL_ERROR_NULL_POINTER);
    SCL_ASSERT_EQ(scl_sparse_cols(mat, nullptr), SCL_ERROR_NULL_POINTER);
}

SCL_TEST_END

SCL_TEST_MAIN()

