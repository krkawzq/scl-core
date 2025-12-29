// =============================================================================
// SCL Core - Comprehensive core.h Tests
// =============================================================================
//
// Complete test coverage for scl/binding/c_api/core/core.h
//
// Functions tested:
//   - scl_get_version()
//   - scl_get_build_config()
//   - scl_get_last_error()
//   - scl_get_last_error_code()
//   - scl_clear_error()
//   - scl_is_ok()
//   - scl_is_error()
//
// =============================================================================

#include "test.hpp"

using namespace scl::test;

SCL_TEST_BEGIN

// =============================================================================
// Version Information Tests
// =============================================================================

SCL_TEST_SUITE(version_info)

SCL_TEST_CASE(get_version_returns_valid_string) {
    const char* version = scl_get_version();
    
    SCL_ASSERT_NOT_NULL(version);
    SCL_ASSERT_TRUE(version[0] != '\0');
    
    // Should contain at least one digit
    bool has_digit = false;
    for (const char* p = version; *p; ++p) {
        if (*p >= '0' && *p <= '9') {
            has_digit = true;
            break;
        }
    }
    SCL_ASSERT_TRUE(has_digit);
}

SCL_TEST_CASE(get_version_is_stable) {
    // Multiple calls should return same pointer (static storage)
    const char* v1 = scl_get_version();
    const char* v2 = scl_get_version();
    
    SCL_ASSERT_EQ(v1, v2);
}

SCL_TEST_CASE(get_build_config_returns_valid_string) {
    const char* config = scl_get_build_config();
    
    SCL_ASSERT_NOT_NULL(config);
    SCL_ASSERT_TRUE(config[0] != '\0');
}

SCL_TEST_CASE(get_build_config_contains_type_info) {
    const char* config = scl_get_build_config();
    
    // Should mention float type
    SCL_ASSERT_TRUE(
        std::strstr(config, "float") != nullptr ||
        std::strstr(config, "f32") != nullptr ||
        std::strstr(config, "f64") != nullptr
    );
}

SCL_TEST_CASE(get_build_config_is_stable) {
    const char* c1 = scl_get_build_config();
    const char* c2 = scl_get_build_config();
    
    SCL_ASSERT_EQ(c1, c2);
}

SCL_TEST_SUITE_END

// =============================================================================
// Error Code Constants Tests
// =============================================================================

SCL_TEST_SUITE(error_codes)

SCL_TEST_CASE(success_code_is_zero) {
    SCL_ASSERT_EQ(SCL_OK, 0);
}

SCL_TEST_CASE(all_error_codes_are_nonzero) {
    // General errors
    SCL_ASSERT_NE(SCL_ERROR_UNKNOWN, 0);
    SCL_ASSERT_NE(SCL_ERROR_INTERNAL, 0);
    SCL_ASSERT_NE(SCL_ERROR_OUT_OF_MEMORY, 0);
    SCL_ASSERT_NE(SCL_ERROR_NULL_POINTER, 0);
    
    // Argument errors
    SCL_ASSERT_NE(SCL_ERROR_INVALID_ARGUMENT, 0);
    SCL_ASSERT_NE(SCL_ERROR_DIMENSION_MISMATCH, 0);
    SCL_ASSERT_NE(SCL_ERROR_DOMAIN_ERROR, 0);
    SCL_ASSERT_NE(SCL_ERROR_RANGE_ERROR, 0);
    SCL_ASSERT_NE(SCL_ERROR_INDEX_OUT_OF_BOUNDS, 0);
    
    // Type errors
    SCL_ASSERT_NE(SCL_ERROR_TYPE_ERROR, 0);
    SCL_ASSERT_NE(SCL_ERROR_TYPE_MISMATCH, 0);
    
    // I/O errors
    SCL_ASSERT_NE(SCL_ERROR_IO_ERROR, 0);
    SCL_ASSERT_NE(SCL_ERROR_FILE_NOT_FOUND, 0);
    SCL_ASSERT_NE(SCL_ERROR_PERMISSION_DENIED, 0);
    SCL_ASSERT_NE(SCL_ERROR_READ_ERROR, 0);
    SCL_ASSERT_NE(SCL_ERROR_WRITE_ERROR, 0);
    SCL_ASSERT_NE(SCL_ERROR_UNREGISTERED_POINTER, 0);
    SCL_ASSERT_NE(SCL_ERROR_BUFFER_NOT_FOUND, 0);
    
    // Feature errors
    SCL_ASSERT_NE(SCL_ERROR_NOT_IMPLEMENTED, 0);
    SCL_ASSERT_NE(SCL_ERROR_FEATURE_UNAVAILABLE, 0);
    
    // Numerical errors
    SCL_ASSERT_NE(SCL_ERROR_NUMERICAL_ERROR, 0);
    SCL_ASSERT_NE(SCL_ERROR_DIVISION_BY_ZERO, 0);
    SCL_ASSERT_NE(SCL_ERROR_OVERFLOW, 0);
    SCL_ASSERT_NE(SCL_ERROR_UNDERFLOW, 0);
    SCL_ASSERT_NE(SCL_ERROR_CONVERGENCE_ERROR, 0);
}

SCL_TEST_CASE(error_codes_have_expected_values) {
    // Verify documented values
    SCL_ASSERT_EQ(SCL_OK, 0);
    SCL_ASSERT_EQ(SCL_ERROR_UNKNOWN, 1);
    SCL_ASSERT_EQ(SCL_ERROR_INTERNAL, 2);
    SCL_ASSERT_EQ(SCL_ERROR_OUT_OF_MEMORY, 3);
    SCL_ASSERT_EQ(SCL_ERROR_NULL_POINTER, 4);
    
    SCL_ASSERT_EQ(SCL_ERROR_INVALID_ARGUMENT, 10);
    SCL_ASSERT_EQ(SCL_ERROR_DIMENSION_MISMATCH, 11);
    
    SCL_ASSERT_EQ(SCL_ERROR_TYPE_ERROR, 20);
    SCL_ASSERT_EQ(SCL_ERROR_IO_ERROR, 30);
    SCL_ASSERT_EQ(SCL_ERROR_NOT_IMPLEMENTED, 40);
    SCL_ASSERT_EQ(SCL_ERROR_NUMERICAL_ERROR, 50);
}

SCL_TEST_CASE(error_codes_are_distinct) {
    // Check some key codes are unique
    SCL_ASSERT_NE(SCL_ERROR_NULL_POINTER, SCL_ERROR_INVALID_ARGUMENT);
    SCL_ASSERT_NE(SCL_ERROR_NULL_POINTER, SCL_ERROR_OUT_OF_MEMORY);
    SCL_ASSERT_NE(SCL_ERROR_INVALID_ARGUMENT, SCL_ERROR_DIMENSION_MISMATCH);
}

SCL_TEST_SUITE_END

// =============================================================================
// Error State Functions Tests
// =============================================================================

SCL_TEST_SUITE(error_state)

SCL_TEST_CASE(initial_state_is_no_error) {
    scl_clear_error();
    
    scl_error_t code = scl_get_last_error_code();
    SCL_ASSERT_EQ(code, SCL_OK);
}

SCL_TEST_CASE(get_last_error_returns_valid_pointer) {
    scl_clear_error();
    
    const char* msg = scl_get_last_error();
    SCL_ASSERT_NOT_NULL(msg);
}

SCL_TEST_CASE(get_last_error_returns_nonnull_on_no_error) {
    scl_clear_error();
    
    const char* msg = scl_get_last_error();
    SCL_ASSERT_NOT_NULL(msg);
    SCL_ASSERT_TRUE(msg[0] != '\0');
}

SCL_TEST_CASE(clear_error_resets_to_ok) {
    // Set an error
    scl_sparse_rows(nullptr, nullptr);
    SCL_ASSERT_NE(scl_get_last_error_code(), SCL_OK);
    
    // Clear
    scl_clear_error();
    
    // Should be OK now
    SCL_ASSERT_EQ(scl_get_last_error_code(), SCL_OK);
}

SCL_TEST_CASE(clear_error_is_idempotent) {
    scl_clear_error();
    scl_error_t code1 = scl_get_last_error_code();
    
    scl_clear_error();
    scl_error_t code2 = scl_get_last_error_code();
    
    SCL_ASSERT_EQ(code1, code2);
    SCL_ASSERT_EQ(code1, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Error Recording Tests
// =============================================================================

SCL_TEST_SUITE(error_recording)

SCL_TEST_CASE(error_is_recorded_after_null_handle) {
    scl_clear_error();
    
    scl_index_t dummy;
    scl_error_t err = scl_sparse_rows(nullptr, &dummy);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
    SCL_ASSERT_EQ(scl_get_last_error_code(), SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(error_is_recorded_after_null_output) {
    scl_clear_error();
    
    // Create valid matrix
    std::vector<scl_index_t> indptr = {0, 2, 3, 6};
    std::vector<scl_index_t> indices = {0, 2, 1, 0, 1, 2};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    // Pass NULL output pointer
    scl_error_t err = scl_sparse_rows(mat, nullptr);
    
    SCL_ASSERT_EQ(err, SCL_ERROR_NULL_POINTER);
    SCL_ASSERT_EQ(scl_get_last_error_code(), SCL_ERROR_NULL_POINTER);
}

SCL_TEST_CASE(error_message_contains_context) {
    scl_clear_error();
    
    scl_sparse_rows(nullptr, nullptr);
    
    const char* msg = scl_get_last_error();
    SCL_ASSERT_NOT_NULL(msg);
    
    // Message should contain "null" or "NULL"
    std::string msg_str(msg);
    std::transform(msg_str.begin(), msg_str.end(), msg_str.begin(), ::tolower);
    SCL_ASSERT_TRUE(msg_str.find("null") != std::string::npos);
}

SCL_TEST_CASE(error_persists_across_successful_operations) {
    scl_clear_error();
    
    // Trigger error
    scl_sparse_rows(nullptr, nullptr);
    scl_error_t error_code = scl_get_last_error_code();
    SCL_ASSERT_NE(error_code, SCL_OK);
    
    // Successful operation
    std::vector<scl_index_t> indptr = {0, 2, 3, 6};
    std::vector<scl_index_t> indices = {0, 2, 1, 0, 1, 2};
    std::vector<scl_real_t> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    Sparse mat = make_sparse_csr(3, 3, 6, indptr.data(), indices.data(), data.data());
    
    // Error should be cleared after successful operation
    scl_error_t new_code = scl_get_last_error_code();
    SCL_ASSERT_EQ(new_code, SCL_OK);
}

SCL_TEST_SUITE_END

// =============================================================================
// Error Query Functions Tests
// =============================================================================

SCL_TEST_SUITE(error_query)

SCL_TEST_CASE(is_ok_with_success_code) {
    SCL_ASSERT_TRUE(scl_is_ok(SCL_OK));
    SCL_ASSERT_EQ(scl_is_ok(SCL_OK), SCL_TRUE);
}

SCL_TEST_CASE(is_ok_with_error_codes) {
    SCL_ASSERT_FALSE(scl_is_ok(SCL_ERROR_NULL_POINTER));
    SCL_ASSERT_FALSE(scl_is_ok(SCL_ERROR_INVALID_ARGUMENT));
    SCL_ASSERT_FALSE(scl_is_ok(SCL_ERROR_OUT_OF_MEMORY));
    SCL_ASSERT_FALSE(scl_is_ok(SCL_ERROR_INTERNAL));
    
    SCL_ASSERT_EQ(scl_is_ok(SCL_ERROR_NULL_POINTER), SCL_FALSE);
}

SCL_TEST_CASE(is_error_with_success_code) {
    SCL_ASSERT_FALSE(scl_is_error(SCL_OK));
    SCL_ASSERT_EQ(scl_is_error(SCL_OK), SCL_FALSE);
}

SCL_TEST_CASE(is_error_with_error_codes) {
    SCL_ASSERT_TRUE(scl_is_error(SCL_ERROR_NULL_POINTER));
    SCL_ASSERT_TRUE(scl_is_error(SCL_ERROR_INVALID_ARGUMENT));
    SCL_ASSERT_TRUE(scl_is_error(SCL_ERROR_OUT_OF_MEMORY));
    
    SCL_ASSERT_EQ(scl_is_error(SCL_ERROR_NULL_POINTER), SCL_TRUE);
}

SCL_TEST_CASE(is_ok_and_is_error_are_inverse) {
    // For SCL_OK
    SCL_ASSERT_TRUE(scl_is_ok(SCL_OK));
    SCL_ASSERT_FALSE(scl_is_error(SCL_OK));
    
    // For errors
    SCL_ASSERT_FALSE(scl_is_ok(SCL_ERROR_NULL_POINTER));
    SCL_ASSERT_TRUE(scl_is_error(SCL_ERROR_NULL_POINTER));
    
    SCL_ASSERT_FALSE(scl_is_ok(SCL_ERROR_INVALID_ARGUMENT));
    SCL_ASSERT_TRUE(scl_is_error(SCL_ERROR_INVALID_ARGUMENT));
}

SCL_TEST_SUITE_END

// =============================================================================
// Error Propagation Tests
// =============================================================================

SCL_TEST_SUITE(error_propagation)

SCL_TEST_CASE(error_message_pointer_is_stable) {
    scl_clear_error();
    
    // Trigger error
    scl_sparse_rows(nullptr, nullptr);
    
    const char* msg1 = scl_get_last_error();
    const char* msg2 = scl_get_last_error();
    const char* msg3 = scl_get_last_error();
    
    // Should return same pointer
    SCL_ASSERT_EQ(msg1, msg2);
    SCL_ASSERT_EQ(msg2, msg3);
}

SCL_TEST_CASE(successful_operation_clears_error) {
    scl_clear_error();
    
    // Trigger error
    scl_sparse_rows(nullptr, nullptr);
    SCL_ASSERT_NE(scl_get_last_error_code(), SCL_OK);
    
    // Successful operation should clear error
    std::vector<scl_index_t> indptr = {0, 1};
    std::vector<scl_index_t> indices = {0};
    std::vector<scl_real_t> data = {1.0};
    
    Sparse mat;
    scl_error_t err = scl_sparse_create(
        mat.ptr(), 1, 1, 1,
        indptr.data(), indices.data(), data.data(),
        SCL_TRUE
    );
    
    SCL_ASSERT_EQ(err, SCL_OK);
    SCL_ASSERT_EQ(scl_get_last_error_code(), SCL_OK);
}

SCL_TEST_CASE(error_persists_until_cleared_or_overwritten) {
    scl_clear_error();
    
    // First error
    scl_sparse_rows(nullptr, nullptr);
    scl_error_t first_code = scl_get_last_error_code();
    
    // Query multiple times
    scl_error_t code2 = scl_get_last_error_code();
    scl_error_t code3 = scl_get_last_error_code();
    
    SCL_ASSERT_EQ(first_code, code2);
    SCL_ASSERT_EQ(code2, code3);
}

SCL_TEST_SUITE_END

// =============================================================================
// Thread-Local Error Storage Tests
// =============================================================================

SCL_TEST_SUITE(thread_local_errors)

SCL_TEST_CASE(error_clear_affects_current_thread_only) {
    scl_clear_error();
    
    // Set error
    scl_sparse_rows(nullptr, nullptr);
    SCL_ASSERT_NE(scl_get_last_error_code(), SCL_OK);
    
    // Clear in same thread
    scl_clear_error();
    SCL_ASSERT_EQ(scl_get_last_error_code(), SCL_OK);
}

SCL_TEST_CASE(multiple_clears_are_safe) {
    for (int i = 0; i < 100; ++i) {
        scl_clear_error();
        SCL_ASSERT_EQ(scl_get_last_error_code(), SCL_OK);
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// Edge Cases and Boundary Tests
// =============================================================================

SCL_TEST_SUITE(edge_cases)

SCL_TEST_CASE(get_error_without_any_operation) {
    // Fresh start, no operations called yet
    scl_clear_error();
    
    scl_error_t code = scl_get_last_error_code();
    const char* msg = scl_get_last_error();
    
    SCL_ASSERT_EQ(code, SCL_OK);
    SCL_ASSERT_NOT_NULL(msg);
}

SCL_TEST_CASE(is_ok_with_arbitrary_values) {
    // Test boundary values
    SCL_ASSERT_TRUE(scl_is_ok(0));
    SCL_ASSERT_FALSE(scl_is_ok(1));
    SCL_ASSERT_FALSE(scl_is_ok(-1));
    SCL_ASSERT_FALSE(scl_is_ok(999));
}

SCL_TEST_CASE(is_error_with_arbitrary_values) {
    SCL_ASSERT_FALSE(scl_is_error(0));
    SCL_ASSERT_TRUE(scl_is_error(1));
    SCL_ASSERT_TRUE(scl_is_error(-1));
    SCL_ASSERT_TRUE(scl_is_error(999));
}

SCL_TEST_CASE(error_message_never_null) {
    scl_clear_error();
    
    // In OK state
    SCL_ASSERT_NOT_NULL(scl_get_last_error());
    
    // After error
    scl_sparse_rows(nullptr, nullptr);
    SCL_ASSERT_NOT_NULL(scl_get_last_error());
    
    // After clear
    scl_clear_error();
    SCL_ASSERT_NOT_NULL(scl_get_last_error());
}

SCL_TEST_CASE(rapid_error_generation) {
    scl_clear_error();
    
    // Generate errors rapidly
    for (int i = 0; i < 1000; ++i) {
        scl_sparse_rows(nullptr, nullptr);
        scl_error_t code = scl_get_last_error_code();
        SCL_ASSERT_EQ(code, SCL_ERROR_NULL_POINTER);
    }
}

SCL_TEST_CASE(alternating_error_and_clear) {
    for (int i = 0; i < 100; ++i) {
        scl_clear_error();
        SCL_ASSERT_EQ(scl_get_last_error_code(), SCL_OK);
        
        scl_sparse_rows(nullptr, nullptr);
        SCL_ASSERT_EQ(scl_get_last_error_code(), SCL_ERROR_NULL_POINTER);
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// Type Definitions Tests
// =============================================================================

SCL_TEST_SUITE(type_definitions)

SCL_TEST_CASE(bool_type_values) {
    SCL_ASSERT_EQ(SCL_TRUE, 1);
    SCL_ASSERT_EQ(SCL_FALSE, 0);
    SCL_ASSERT_NE(SCL_TRUE, SCL_FALSE);
}

SCL_TEST_CASE(index_type_size) {
    // scl_index_t should be at least 32 bits
    SCL_ASSERT_GE(sizeof(scl_index_t), 4);
}

SCL_TEST_CASE(real_type_size) {
    // scl_real_t should be at least 32 bits
    SCL_ASSERT_GE(sizeof(scl_real_t), 4);
}

SCL_TEST_CASE(size_type_matches_platform) {
    SCL_ASSERT_EQ(sizeof(scl_size_t), sizeof(size_t));
}

SCL_TEST_CASE(error_type_is_int32) {
    SCL_ASSERT_EQ(sizeof(scl_error_t), 4);
}

SCL_TEST_SUITE_END

// =============================================================================
// API Version Tests
// =============================================================================

SCL_TEST_SUITE(api_version)

SCL_TEST_CASE(version_macros_defined) {
    // Just verify they compile
    int major = SCL_C_API_VERSION_MAJOR;
    int minor = SCL_C_API_VERSION_MINOR;
    int patch = SCL_C_API_VERSION_PATCH;
    
    SCL_ASSERT_GE(major, 0);
    SCL_ASSERT_GE(minor, 0);
    SCL_ASSERT_GE(patch, 0);
}

SCL_TEST_CASE(version_is_reasonable) {
    SCL_ASSERT_GE(SCL_C_API_VERSION_MAJOR, 0);
    SCL_ASSERT_LE(SCL_C_API_VERSION_MAJOR, 10);
    
    SCL_ASSERT_GE(SCL_C_API_VERSION_MINOR, 0);
    SCL_ASSERT_LE(SCL_C_API_VERSION_MINOR, 100);
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

