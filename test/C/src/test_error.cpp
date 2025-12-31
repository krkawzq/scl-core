// =============================================================================
// SCL Core v0.5 - Error Handling C-API Tests
// =============================================================================
//
// Tests for scl/api/core/error.h
//
// =============================================================================

#include "test.hpp"

using namespace scl::test;

SCL_TEST_BEGIN

// =============================================================================
// Version and Library Information
// =============================================================================

SCL_TEST_SUITE(version_info)

SCL_TEST_CASE(get_version_returns_valid_string) {
    const char* version = scl_get_version();
    
    SCL_ASSERT_NOT_NULL(version);
    SCL_ASSERT_TRUE(version[0] != '\0');
    
    printf("    Library version: %s\n", version);
}

SCL_TEST_CASE(get_version_components_match) {
    std::int32_t major, minor, patch;
    scl_get_version_components(&major, &minor, &patch);
    
    SCL_ASSERT_GE(major, 0);
    SCL_ASSERT_GE(minor, 0);
    SCL_ASSERT_GE(patch, 0);
    
    printf("    Version: %d.%d.%d\n", major, minor, patch);
}

SCL_TEST_SUITE_END

// =============================================================================
// Error Code Functions
// =============================================================================

SCL_TEST_SUITE(error_codes)

SCL_TEST_CASE(error_code_name_returns_valid_strings) {
    const char* name = scl_error_code_name(scl_error_success());
    SCL_ASSERT_NOT_NULL(name);
    
    name = scl_error_code_name(scl_error_unknown());
    SCL_ASSERT_NOT_NULL(name);
    
    name = scl_error_code_name(scl_error_out_of_memory());
    SCL_ASSERT_NOT_NULL(name);
}

SCL_TEST_CASE(success_code_is_zero) {
    SCL_ASSERT_EQ(scl_error_success(), 0);
}

SCL_TEST_CASE(error_codes_are_nonzero) {
    SCL_ASSERT_NE(scl_error_unknown(), 0);
    SCL_ASSERT_NE(scl_error_out_of_memory(), 0);
    SCL_ASSERT_NE(scl_error_null_pointer(), 0);
    SCL_ASSERT_NE(scl_error_invalid_argument(), 0);
}

SCL_TEST_CASE(is_success_works_correctly) {
    SCL_ASSERT_TRUE(scl_is_success(scl_error_success()));
    SCL_ASSERT_FALSE(scl_is_success(scl_error_unknown()));
    SCL_ASSERT_FALSE(scl_is_success(scl_error_out_of_memory()));
}

SCL_TEST_CASE(is_error_works_correctly) {
    SCL_ASSERT_FALSE(scl_is_error(scl_error_success()));
    SCL_ASSERT_TRUE(scl_is_error(scl_error_unknown()));
    SCL_ASSERT_TRUE(scl_is_error(scl_error_out_of_memory()));
}

SCL_TEST_SUITE_END

// =============================================================================
// Thread-Local Error State
// =============================================================================

SCL_TEST_SUITE(error_state)

SCL_TEST_CASE(initial_state_is_no_error) {
    scl_clear_error();
    
    std::int32_t code = scl_get_last_error();
    SCL_ASSERT_EQ(code, scl_error_success());
}

SCL_TEST_CASE(get_error_message_returns_valid_pointer) {
    scl_clear_error();
    
    const char* msg = scl_get_error_message();
    SCL_ASSERT_NOT_NULL(msg);
}

SCL_TEST_CASE(clear_error_resets_state) {
    // Trigger an error by passing null handle
    auto rows = scl_sparse_rows(nullptr);
    SCL_ASSERT_EQ(rows, 0);
    
    // Error should be set
    std::int32_t code = scl_get_last_error();
    SCL_ASSERT_NE(code, scl_error_success());
    
    // Clear error
    scl_clear_error();
    
    // Should be success now
    code = scl_get_last_error();
    SCL_ASSERT_EQ(code, scl_error_success());
}

SCL_TEST_CASE(has_error_works_correctly) {
    scl_clear_error();
    SCL_ASSERT_FALSE(scl_has_error());
    
    // Trigger error
    scl_sparse_rows(nullptr);
    SCL_ASSERT_TRUE(scl_has_error());
    
    // Clear
    scl_clear_error();
    SCL_ASSERT_FALSE(scl_has_error());
}

SCL_TEST_CASE(set_error_works) {
    scl_clear_error();
    
    scl_set_error(scl_error_invalid_argument(), "Test error message");
    
    SCL_ASSERT_EQ(scl_get_last_error(), scl_error_invalid_argument());
    
    const char* msg = scl_get_error_message();
    SCL_ASSERT_TRUE(std::strstr(msg, "Test error") != nullptr);
}

SCL_TEST_SUITE_END

// =============================================================================
// Error Information Functions
// =============================================================================

SCL_TEST_SUITE(error_info)

SCL_TEST_CASE(get_error_info_formats_correctly) {
    scl_clear_error();
    scl_set_error(scl_error_null_pointer(), "Test null pointer");
    
    char buffer[256];
    std::size_t written = scl_get_error_info(buffer, sizeof(buffer));
    
    SCL_ASSERT_GT(written, 0);
    SCL_ASSERT_LT(written, sizeof(buffer));
    SCL_ASSERT_TRUE(std::strstr(buffer, "Test null pointer") != nullptr);
}

SCL_TEST_CASE(copy_error_message_works) {
    scl_clear_error();
    scl_set_error(scl_error_invalid_argument(), "Test message for copy");
    
    char buffer[256];
    std::size_t copied = scl_copy_error_message(buffer, sizeof(buffer));
    
    SCL_ASSERT_GT(copied, 0);
    SCL_ASSERT_TRUE(std::strstr(buffer, "Test message") != nullptr);
}

SCL_TEST_SUITE_END

// =============================================================================
// Error Code Range Queries
// =============================================================================

SCL_TEST_SUITE(error_ranges)

SCL_TEST_CASE(memory_error_range_is_valid) {
    std::int32_t min = scl_error_memory_min();
    std::int32_t max = scl_error_memory_max();
    
    SCL_ASSERT_LT(min, max);
    SCL_ASSERT_GE(scl_error_out_of_memory(), min);
    SCL_ASSERT_LE(scl_error_out_of_memory(), max);
}

SCL_TEST_CASE(dimension_error_range_is_valid) {
    std::int32_t min = scl_error_dimension_min();
    std::int32_t max = scl_error_dimension_max();
    
    SCL_ASSERT_LT(min, max);
    SCL_ASSERT_GE(scl_error_dimension_mismatch(), min);
    SCL_ASSERT_LE(scl_error_dimension_mismatch(), max);
}

SCL_TEST_CASE(type_error_range_is_valid) {
    std::int32_t min = scl_error_type_min();
    std::int32_t max = scl_error_type_max();
    
    SCL_ASSERT_LT(min, max);
    SCL_ASSERT_GE(scl_error_type_mismatch(), min);
    SCL_ASSERT_LE(scl_error_type_mismatch(), max);
}

SCL_TEST_SUITE_END

// =============================================================================
// Error Propagation
// =============================================================================

SCL_TEST_SUITE(error_propagation)

SCL_TEST_CASE(error_persists_across_queries) {
    scl_clear_error();
    
    // Trigger error
    scl_sparse_rows(nullptr);
    std::int32_t code1 = scl_get_last_error();
    
    // Query multiple times
    std::int32_t code2 = scl_get_last_error();
    std::int32_t code3 = scl_get_last_error();
    
    SCL_ASSERT_EQ(code1, code2);
    SCL_ASSERT_EQ(code2, code3);
    SCL_ASSERT_NE(code1, scl_error_success());
}

SCL_TEST_CASE(multiple_clears_are_safe) {
    for (int i = 0; i < 10; ++i) {
        scl_clear_error();
        SCL_ASSERT_EQ(scl_get_last_error(), scl_error_success());
    }
}

SCL_TEST_SUITE_END

SCL_TEST_END

SCL_TEST_MAIN()

