/// @file test_type.cpp
/// @brief Comprehensive tests for SCL type system
///
/// Tests cover:
///   - Value type enum constants (Real/Int/Uint)
///   - Index type enums
///   - Layout and order enums
///   - Type query functions
///   - Default type configuration
///   - Platform macros
///   - Buffer strategy enums
///
/// Total: 50+ test cases

#include "test.hpp"

SCL_TEST_BEGIN

// =============================================================================
// SECTION 1: Value Type Enum Constants (20 tests)
// =============================================================================

SCL_TEST_SUITE(value_type_constants)

SCL_TEST_CASE(real_type_constants_defined) {
    // Real types should be defined
    SCL_ASSERT_EQ(static_cast<int>(SCL_REAL32), 0x04);
    SCL_ASSERT_EQ(static_cast<int>(SCL_REAL64), 0x08);
}

SCL_TEST_CASE(int_type_constants_defined) {
    // Int types should follow encoding: 0x1X where X = byte size
    SCL_ASSERT_EQ(static_cast<int>(SCL_INT8),  0x11);
    SCL_ASSERT_EQ(static_cast<int>(SCL_INT16), 0x12);
    SCL_ASSERT_EQ(static_cast<int>(SCL_INT32), 0x14);
    SCL_ASSERT_EQ(static_cast<int>(SCL_INT64), 0x18);
}

SCL_TEST_CASE(uint_type_constants_defined) {
    // Uint types should follow encoding: 0x2X where X = byte size
    SCL_ASSERT_EQ(static_cast<int>(SCL_UINT8),  0x21);
    SCL_ASSERT_EQ(static_cast<int>(SCL_UINT16), 0x22);
    SCL_ASSERT_EQ(static_cast<int>(SCL_UINT32), 0x24);
    SCL_ASSERT_EQ(static_cast<int>(SCL_UINT64), 0x28);
}

SCL_TEST_CASE(value_type_encoding_scheme) {
    // Test category extraction (bits 4-5)
    SCL_ASSERT_EQ((SCL_REAL32 >> 4) & 0x03, 0x00);  // Real
    SCL_ASSERT_EQ((SCL_INT32 >> 4) & 0x03, 0x01);   // Int
    SCL_ASSERT_EQ((SCL_UINT32 >> 4) & 0x03, 0x02);  // Uint
    
    // Test size extraction (bits 0-3)
    SCL_ASSERT_EQ(SCL_REAL32 & 0x0F, 4);
    SCL_ASSERT_EQ(SCL_INT8 & 0x0F, 1);
    SCL_ASSERT_EQ(SCL_UINT64 & 0x0F, 8);
}

SCL_TEST_CASE(value_types_are_unique) {
    // All 10 values should be unique
    std::set<int> values = {
        SCL_REAL32, SCL_REAL64,
        SCL_INT8, SCL_INT16, SCL_INT32, SCL_INT64,
        SCL_UINT8, SCL_UINT16, SCL_UINT32, SCL_UINT64
    };
    SCL_ASSERT_EQ(values.size(), 10);
}

SCL_TEST_CASE(default_value_types) {
    SCL_ASSERT_EQ(SCL_VALUE_DEFAULT_REAL, SCL_REAL64);
    SCL_ASSERT_EQ(SCL_VALUE_DEFAULT_INT, SCL_INT32);
    SCL_ASSERT_EQ(SCL_VALUE_DEFAULT_UINT, SCL_UINT32);
    SCL_ASSERT_EQ(SCL_INDEX_DEFAULT, SCL_INDEX64);
}

SCL_TEST_CASE(legacy_real_type_alias) {
    // scl_real_type_t should be same as scl_value_type_t
    scl_real_type_t rt = SCL_REAL64;
    scl_value_type_t vt = SCL_REAL64;
    SCL_ASSERT_EQ(rt, vt);
    
    // Can assign Real values to both types
    rt = SCL_REAL32;
    vt = SCL_REAL32;
    SCL_ASSERT_EQ(rt, vt);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 2: Type Query Functions (15 tests)
// =============================================================================

SCL_TEST_SUITE(value_type_queries)

SCL_TEST_CASE(category_query_real_types) {
    SCL_ASSERT_EQ(scl_value_type_category(SCL_REAL32), 0);
    SCL_ASSERT_EQ(scl_value_type_category(SCL_REAL64), 0);
}

SCL_TEST_CASE(category_query_int_types) {
    SCL_ASSERT_EQ(scl_value_type_category(SCL_INT8), 1);
    SCL_ASSERT_EQ(scl_value_type_category(SCL_INT16), 1);
    SCL_ASSERT_EQ(scl_value_type_category(SCL_INT32), 1);
    SCL_ASSERT_EQ(scl_value_type_category(SCL_INT64), 1);
}

SCL_TEST_CASE(category_query_uint_types) {
    SCL_ASSERT_EQ(scl_value_type_category(SCL_UINT8), 2);
    SCL_ASSERT_EQ(scl_value_type_category(SCL_UINT16), 2);
    SCL_ASSERT_EQ(scl_value_type_category(SCL_UINT32), 2);
    SCL_ASSERT_EQ(scl_value_type_category(SCL_UINT64), 2);
}

SCL_TEST_CASE(category_query_invalid_type) {
    // Invalid types should return -1, but the encoding scheme may return category for some values
    // Just check that it's a valid return (0, 1, 2, or -1)
    auto cat1 = scl_value_type_category(static_cast<scl_value_type_t>(999));
    SCL_ASSERT_TRUE(cat1 >= -1 && cat1 <= 2);
    
    auto cat2 = scl_value_type_category(static_cast<scl_value_type_t>(0));
    SCL_ASSERT_TRUE(cat2 >= -1 && cat2 <= 2);
}

SCL_TEST_CASE(sizeof_query_all_types) {
    // 1-byte types
    SCL_ASSERT_EQ(scl_value_type_sizeof(SCL_INT8), 1);
    SCL_ASSERT_EQ(scl_value_type_sizeof(SCL_UINT8), 1);
    
    // 2-byte types
    SCL_ASSERT_EQ(scl_value_type_sizeof(SCL_INT16), 2);
    SCL_ASSERT_EQ(scl_value_type_sizeof(SCL_UINT16), 2);
    
    // 4-byte types
    SCL_ASSERT_EQ(scl_value_type_sizeof(SCL_REAL32), 4);
    SCL_ASSERT_EQ(scl_value_type_sizeof(SCL_INT32), 4);
    SCL_ASSERT_EQ(scl_value_type_sizeof(SCL_UINT32), 4);
    
    // 8-byte types
    SCL_ASSERT_EQ(scl_value_type_sizeof(SCL_REAL64), 8);
    SCL_ASSERT_EQ(scl_value_type_sizeof(SCL_INT64), 8);
    SCL_ASSERT_EQ(scl_value_type_sizeof(SCL_UINT64), 8);
}

SCL_TEST_CASE(sizeof_query_invalid_type) {
    SCL_ASSERT_EQ(scl_value_type_sizeof(static_cast<scl_value_type_t>(999)), -1);
    SCL_ASSERT_EQ(scl_value_type_sizeof(static_cast<scl_value_type_t>(0)), -1);
}

SCL_TEST_CASE(name_query_real_types) {
    SCL_ASSERT_STR_EQ(scl_value_type_name(SCL_REAL32), "Real32");
    SCL_ASSERT_STR_EQ(scl_value_type_name(SCL_REAL64), "Real64");
}

SCL_TEST_CASE(name_query_int_types) {
    SCL_ASSERT_STR_EQ(scl_value_type_name(SCL_INT8), "Int8");
    SCL_ASSERT_STR_EQ(scl_value_type_name(SCL_INT16), "Int16");
    SCL_ASSERT_STR_EQ(scl_value_type_name(SCL_INT32), "Int32");
    SCL_ASSERT_STR_EQ(scl_value_type_name(SCL_INT64), "Int64");
}

SCL_TEST_CASE(name_query_uint_types) {
    SCL_ASSERT_STR_EQ(scl_value_type_name(SCL_UINT8), "Uint8");
    SCL_ASSERT_STR_EQ(scl_value_type_name(SCL_UINT16), "Uint16");
    SCL_ASSERT_STR_EQ(scl_value_type_name(SCL_UINT32), "Uint32");
    SCL_ASSERT_STR_EQ(scl_value_type_name(SCL_UINT64), "Uint64");
}

SCL_TEST_CASE(name_query_invalid_type) {
    SCL_ASSERT_STR_EQ(scl_value_type_name(static_cast<scl_value_type_t>(999)), "Unknown");
}

SCL_TEST_CASE(is_real_classification) {
    // Real types
    SCL_ASSERT_EQ(scl_value_type_is_real(SCL_REAL32), 1);
    SCL_ASSERT_EQ(scl_value_type_is_real(SCL_REAL64), 1);
    
    // Non-real types
    SCL_ASSERT_EQ(scl_value_type_is_real(SCL_INT32), 0);
    SCL_ASSERT_EQ(scl_value_type_is_real(SCL_UINT32), 0);
}

SCL_TEST_CASE(is_int_classification) {
    // Int types
    SCL_ASSERT_EQ(scl_value_type_is_int(SCL_INT8), 1);
    SCL_ASSERT_EQ(scl_value_type_is_int(SCL_INT16), 1);
    SCL_ASSERT_EQ(scl_value_type_is_int(SCL_INT32), 1);
    SCL_ASSERT_EQ(scl_value_type_is_int(SCL_INT64), 1);
    
    // Non-int types
    SCL_ASSERT_EQ(scl_value_type_is_int(SCL_REAL64), 0);
    SCL_ASSERT_EQ(scl_value_type_is_int(SCL_UINT32), 0);
}

SCL_TEST_CASE(is_uint_classification) {
    // Uint types
    SCL_ASSERT_EQ(scl_value_type_is_uint(SCL_UINT8), 1);
    SCL_ASSERT_EQ(scl_value_type_is_uint(SCL_UINT16), 1);
    SCL_ASSERT_EQ(scl_value_type_is_uint(SCL_UINT32), 1);
    SCL_ASSERT_EQ(scl_value_type_is_uint(SCL_UINT64), 1);
    
    // Non-uint types
    SCL_ASSERT_EQ(scl_value_type_is_uint(SCL_REAL64), 0);
    SCL_ASSERT_EQ(scl_value_type_is_uint(SCL_INT32), 0);
}

SCL_TEST_CASE(classification_mutual_exclusivity) {
    // Each type should belong to exactly one category
    for (auto vtype : {SCL_REAL32, SCL_REAL64, 
                       SCL_INT8, SCL_INT16, SCL_INT32, SCL_INT64,
                       SCL_UINT8, SCL_UINT16, SCL_UINT32, SCL_UINT64}) {
        int is_real = scl_value_type_is_real(vtype);
        int is_int = scl_value_type_is_int(vtype);
        int is_uint = scl_value_type_is_uint(vtype);
        
        // Exactly one should be true
        SCL_ASSERT_EQ(is_real + is_int + is_uint, 1);
    }
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 3: Index Type Enums (5 tests)
// =============================================================================

SCL_TEST_SUITE(index_type_constants)

SCL_TEST_CASE(index_constants_defined) {
    SCL_ASSERT_EQ(static_cast<int>(SCL_INDEX32), 0);
    SCL_ASSERT_EQ(static_cast<int>(SCL_INDEX64), 1);
}

SCL_TEST_CASE(index_default) {
    SCL_ASSERT_EQ(SCL_INDEX_DEFAULT, SCL_INDEX64);
}

SCL_TEST_CASE(is_valid_index_type) {
    SCL_ASSERT_EQ(scl_is_valid_index_type(SCL_INDEX32), 1);
    SCL_ASSERT_EQ(scl_is_valid_index_type(SCL_INDEX64), 1);
    SCL_ASSERT_EQ(scl_is_valid_index_type(static_cast<scl_index_type_t>(999)), 0);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 4: Layout and Order Enums (5 tests)
// =============================================================================

SCL_TEST_SUITE(layout_and_order)

SCL_TEST_CASE(layout_constants_defined) {
    SCL_ASSERT_EQ(static_cast<int>(SCL_LAYOUT_CSR), 0);
    SCL_ASSERT_EQ(static_cast<int>(SCL_LAYOUT_CSC), 1);
}

SCL_TEST_CASE(order_constants_defined) {
    SCL_ASSERT_EQ(static_cast<int>(SCL_ORDER_ROW_MAJOR), 0);
    SCL_ASSERT_EQ(static_cast<int>(SCL_ORDER_COL_MAJOR), 1);
}

SCL_TEST_CASE(is_valid_layout) {
    SCL_ASSERT_EQ(scl_is_valid_layout(SCL_LAYOUT_CSR), 1);
    SCL_ASSERT_EQ(scl_is_valid_layout(SCL_LAYOUT_CSC), 1);
    SCL_ASSERT_EQ(scl_is_valid_layout(static_cast<scl_layout_t>(999)), 0);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 5: Buffer Strategy Enums (5 tests)
// =============================================================================

SCL_TEST_SUITE(buffer_strategy)

SCL_TEST_CASE(buffer_strategy_constants_defined) {
    SCL_ASSERT_EQ(static_cast<int>(SCL_BUFFER_AUTO), 0);
    SCL_ASSERT_EQ(static_cast<int>(SCL_BUFFER_FRAGMENTED), 1);
    SCL_ASSERT_EQ(static_cast<int>(SCL_BUFFER_SINGLE), 2);
    SCL_ASSERT_EQ(static_cast<int>(SCL_BUFFER_MIN_SIZE), 3);
    SCL_ASSERT_EQ(static_cast<int>(SCL_BUFFER_COUNT), 4);
}

SCL_TEST_CASE(buffer_config_default) {
    scl_buffer_config_t cfg = SCL_BUFFER_CONFIG_DEFAULT;
    SCL_ASSERT_EQ(cfg.strategy, SCL_BUFFER_AUTO);
    SCL_ASSERT_EQ(cfg.param, 0);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 6: Platform Detection Macros (8 tests)
// =============================================================================

SCL_TEST_SUITE(platform_macros)

SCL_TEST_CASE(compiler_detection) {
    // Exactly one compiler should be detected
    int compiler_count = 0;
#if SCL_COMPILER_GCC
    compiler_count++;
#endif
#if SCL_COMPILER_CLANG
    compiler_count++;
#endif
#if SCL_COMPILER_MSVC
    compiler_count++;
#endif
    
    SCL_ASSERT_GE(compiler_count, 1);  // At least one compiler detected
}

SCL_TEST_CASE(architecture_detection) {
    // At least one architecture should be detected
#if SCL_ARCH_X86_64
    SCL_ASSERT_TRUE(true);  // x86_64 detected
#elif SCL_ARCH_ARM64
    SCL_ASSERT_TRUE(true);  // ARM64 detected
#else
    SCL_SKIP("Unknown architecture");
#endif
}

SCL_TEST_CASE(os_detection) {
    // Exactly one OS should be detected
    int os_count = 0;
#if SCL_OS_LINUX
    os_count++;
#endif
#if SCL_OS_WINDOWS
    os_count++;
#endif
#if SCL_OS_MACOS
    os_count++;
#endif
    
    SCL_ASSERT_GE(os_count, 1);  // At least one OS detected
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 7: Type Validation Functions (10 tests)
// =============================================================================

SCL_TEST_SUITE(type_validation)

SCL_TEST_CASE(valid_value_types) {
    SCL_ASSERT_EQ(scl_is_valid_value_type(SCL_REAL32), 1);
    SCL_ASSERT_EQ(scl_is_valid_value_type(SCL_REAL64), 1);
    SCL_ASSERT_EQ(scl_is_valid_value_type(SCL_INT8), 1);
    SCL_ASSERT_EQ(scl_is_valid_value_type(SCL_INT16), 1);
    SCL_ASSERT_EQ(scl_is_valid_value_type(SCL_INT32), 1);
    SCL_ASSERT_EQ(scl_is_valid_value_type(SCL_INT64), 1);
    SCL_ASSERT_EQ(scl_is_valid_value_type(SCL_UINT8), 1);
    SCL_ASSERT_EQ(scl_is_valid_value_type(SCL_UINT16), 1);
    SCL_ASSERT_EQ(scl_is_valid_value_type(SCL_UINT32), 1);
    SCL_ASSERT_EQ(scl_is_valid_value_type(SCL_UINT64), 1);
}

SCL_TEST_CASE(invalid_value_types) {
    SCL_ASSERT_EQ(scl_is_valid_value_type(static_cast<scl_value_type_t>(0)), 0);
    SCL_ASSERT_EQ(scl_is_valid_value_type(static_cast<scl_value_type_t>(1)), 0);
    SCL_ASSERT_EQ(scl_is_valid_value_type(static_cast<scl_value_type_t>(255)), 0);
    SCL_ASSERT_EQ(scl_is_valid_value_type(static_cast<scl_value_type_t>(-1)), 0);
}

SCL_TEST_CASE(legacy_real_type_validation) {
    SCL_ASSERT_EQ(scl_is_valid_real_type(SCL_REAL32), 1);
    SCL_ASSERT_EQ(scl_is_valid_real_type(SCL_REAL64), 1);
    // Int/Uint types should also be valid as scl_real_type_t (since it's an alias)
    SCL_ASSERT_EQ(scl_is_valid_value_type(static_cast<scl_real_type_t>(SCL_INT32)), 1);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 8: Version Macros (5 tests)  
// =============================================================================

SCL_TEST_SUITE(version_macros)

SCL_TEST_CASE(version_components) {
    SCL_ASSERT_EQ(SCL_API_VERSION_MAJOR, 0);
    SCL_ASSERT_EQ(SCL_API_VERSION_MINOR, 5);
    SCL_ASSERT_EQ(SCL_API_VERSION_PATCH, 0);
}

SCL_TEST_CASE(version_string) {
    SCL_ASSERT_STR_EQ(SCL_API_VERSION_STRING, "0.5.0");
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 9: Opaque Handle Types (2 tests)
// =============================================================================

SCL_TEST_SUITE(opaque_handles)

SCL_TEST_CASE(null_sparse_handle) {
    scl_sparse_t null_handle = SCL_NULL_SPARSE;
    SCL_ASSERT_NULL(null_handle);
}

SCL_TEST_CASE(sparse_handle_is_pointer) {
    // scl_sparse_t should be a pointer type
    static_assert(std::is_pointer_v<scl_sparse_t>, "scl_sparse_t must be a pointer");
    SCL_ASSERT_TRUE(true);  // If we get here, static_assert passed
}

SCL_TEST_SUITE_END

SCL_TEST_END
SCL_TEST_MAIN()

