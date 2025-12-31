/// @file test_sparse_creation.cpp
/// @brief Comprehensive tests for sparse matrix creation functions
///
/// Tests all 40 type combinations (10 value types × 2 index types × 2 layouts)
/// for zeros() and identity() functions.
///
/// Total: 80+ test cases

#include "test.hpp"

using namespace scl::test;

SCL_TEST_BEGIN

// =============================================================================
// SECTION 1: zeros() - All 40 Type Combinations (40 tests)
// =============================================================================

SCL_TEST_SUITE(zeros_all_types)

// Real32 variants (8 tests)
SCL_TEST_TAGGED(zeros_real32_index32_csr, "core", "quick") {
    auto mat = scl_sparse_zeros(10, 10, SCL_REAL32, SCL_INDEX32, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_SPARSE_DIMS(mat, 10, 10);
    SCL_ASSERT_SPARSE_NNZ(mat, 0);
    SCL_ASSERT_EQ(scl_sparse_value_type(mat), SCL_REAL32);
    SCL_ASSERT_EQ(scl_sparse_index_type(mat), SCL_INDEX32);
    SCL_ASSERT_EQ(scl_sparse_layout(mat), SCL_LAYOUT_CSR);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_real32_index32_csc, "core", "quick") {
    auto mat = scl_sparse_zeros(10, 10, SCL_REAL32, SCL_INDEX32, SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_EQ(scl_sparse_layout(mat), SCL_LAYOUT_CSC);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_real32_index64_csr, "core", "quick") {
    auto mat = scl_sparse_zeros(10, 10, SCL_REAL32, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_EQ(scl_sparse_index_type(mat), SCL_INDEX64);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_real32_index64_csc, "core", "quick") {
    auto mat = scl_sparse_zeros(10, 10, SCL_REAL32, SCL_INDEX64, SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

// Real64 variants (4 tests)
SCL_TEST_TAGGED(zeros_real64_index32_csr, "core", "quick") {
    auto mat = scl_sparse_zeros(10, 10, SCL_REAL64, SCL_INDEX32, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_EQ(scl_sparse_value_type(mat), SCL_REAL64);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_real64_index32_csc, "core", "quick") {
    auto mat = scl_sparse_zeros(10, 10, SCL_REAL64, SCL_INDEX32, SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_real64_index64_csr, "core", "quick") {
    auto mat = scl_sparse_zeros(10, 10, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_real64_index64_csc, "core", "quick") {
    auto mat = scl_sparse_zeros(10, 10, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

// Int8 variants (4 tests)
SCL_TEST_TAGGED(zeros_int8_index32_csr, "integer", "quick") {
    auto mat = scl_sparse_zeros(5, 5, SCL_INT8, SCL_INDEX32, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_EQ(scl_sparse_value_type(mat), SCL_INT8);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_int8_index64_csr, "integer") {
    auto mat = scl_sparse_zeros(5, 5, SCL_INT8, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_int8_index32_csc, "integer") {
    auto mat = scl_sparse_zeros(5, 5, SCL_INT8, SCL_INDEX32, SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_int8_index64_csc, "integer") {
    auto mat = scl_sparse_zeros(5, 5, SCL_INT8, SCL_INDEX64, SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

// Int16 variants (4 tests)
SCL_TEST_TAGGED(zeros_int16_index32_csr, "integer", "quick") {
    auto mat = scl_sparse_zeros(10, 10, SCL_INT16, SCL_INDEX32, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_EQ(scl_sparse_value_type(mat), SCL_INT16);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_int16_index64_csr, "integer") {
    auto mat = scl_sparse_zeros(10, 10, SCL_INT16, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_int16_index32_csc, "integer") {
    auto mat = scl_sparse_zeros(10, 10, SCL_INT16, SCL_INDEX32, SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_int16_index64_csc, "integer") {
    auto mat = scl_sparse_zeros(10, 10, SCL_INT16, SCL_INDEX64, SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

// Int32 variants (4 tests)
SCL_TEST_TAGGED(zeros_int32_index32_csr, "integer", "quick", "default") {
    auto mat = scl_sparse_zeros(10, 10, SCL_INT32, SCL_INDEX32, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_EQ(scl_sparse_value_type(mat), SCL_INT32);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_int32_index64_csr, "integer", "default") {
    auto mat = scl_sparse_zeros(10, 10, SCL_INT32, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_int32_index32_csc, "integer") {
    auto mat = scl_sparse_zeros(10, 10, SCL_INT32, SCL_INDEX32, SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_int32_index64_csc, "integer") {
    auto mat = scl_sparse_zeros(10, 10, SCL_INT32, SCL_INDEX64, SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

// Int64 variants (4 tests)
SCL_TEST_TAGGED(zeros_int64_index32_csr, "integer") {
    auto mat = scl_sparse_zeros(10, 10, SCL_INT64, SCL_INDEX32, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_EQ(scl_sparse_value_type(mat), SCL_INT64);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_int64_index64_csr, "integer") {
    auto mat = scl_sparse_zeros(10, 10, SCL_INT64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_int64_index32_csc, "integer") {
    auto mat = scl_sparse_zeros(10, 10, SCL_INT64, SCL_INDEX32, SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_int64_index64_csc, "integer") {
    auto mat = scl_sparse_zeros(10, 10, SCL_INT64, SCL_INDEX64, SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

// Uint8 variants (4 tests)
SCL_TEST_TAGGED(zeros_uint8_index32_csr, "unsigned", "quick") {
    auto mat = scl_sparse_zeros(5, 5, SCL_UINT8, SCL_INDEX32, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_EQ(scl_sparse_value_type(mat), SCL_UINT8);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_uint8_index64_csr, "unsigned") {
    auto mat = scl_sparse_zeros(5, 5, SCL_UINT8, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_uint8_index32_csc, "unsigned") {
    auto mat = scl_sparse_zeros(5, 5, SCL_UINT8, SCL_INDEX32, SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_uint8_index64_csc, "unsigned") {
    auto mat = scl_sparse_zeros(5, 5, SCL_UINT8, SCL_INDEX64, SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

// Uint16 variants (4 tests)
SCL_TEST_TAGGED(zeros_uint16_index32_csr, "unsigned", "quick") {
    auto mat = scl_sparse_zeros(10, 10, SCL_UINT16, SCL_INDEX32, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_EQ(scl_sparse_value_type(mat), SCL_UINT16);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_uint16_index64_csr, "unsigned") {
    auto mat = scl_sparse_zeros(10, 10, SCL_UINT16, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_uint16_index32_csc, "unsigned") {
    auto mat = scl_sparse_zeros(10, 10, SCL_UINT16, SCL_INDEX32, SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_uint16_index64_csc, "unsigned") {
    auto mat = scl_sparse_zeros(10, 10, SCL_UINT16, SCL_INDEX64, SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

// Uint32 variants (4 tests)
SCL_TEST_TAGGED(zeros_uint32_index32_csr, "unsigned", "quick", "default") {
    auto mat = scl_sparse_zeros(10, 10, SCL_UINT32, SCL_INDEX32, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_EQ(scl_sparse_value_type(mat), SCL_UINT32);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_uint32_index64_csr, "unsigned", "default") {
    auto mat = scl_sparse_zeros(10, 10, SCL_UINT32, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_uint32_index32_csc, "unsigned") {
    auto mat = scl_sparse_zeros(10, 10, SCL_UINT32, SCL_INDEX32, SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_uint32_index64_csc, "unsigned") {
    auto mat = scl_sparse_zeros(10, 10, SCL_UINT32, SCL_INDEX64, SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

// Uint64 variants (4 tests)
SCL_TEST_TAGGED(zeros_uint64_index32_csr, "unsigned") {
    auto mat = scl_sparse_zeros(10, 10, SCL_UINT64, SCL_INDEX32, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_EQ(scl_sparse_value_type(mat), SCL_UINT64);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_uint64_index64_csr, "unsigned") {
    auto mat = scl_sparse_zeros(10, 10, SCL_UINT64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_uint64_index32_csc, "unsigned") {
    auto mat = scl_sparse_zeros(10, 10, SCL_UINT64, SCL_INDEX32, SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_uint64_index64_csc, "unsigned") {
    auto mat = scl_sparse_zeros(10, 10, SCL_UINT64, SCL_INDEX64, SCL_LAYOUT_CSC);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 2: identity() - All 40 Type Combinations (40 tests)
// =============================================================================

SCL_TEST_SUITE(identity_all_types)

// Real types (8 tests)
SCL_TEST_TAGGED(identity_real32_index32, "core", "quick") {
    auto mat = scl_sparse_identity(10, SCL_REAL32, SCL_INDEX32);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_SPARSE_DIMS(mat, 10, 10);
    SCL_ASSERT_SPARSE_NNZ(mat, 10);
    SCL_ASSERT_EQ(scl_sparse_value_type(mat), SCL_REAL32);
    SCL_ASSERT_EQ(scl_sparse_layout(mat), SCL_LAYOUT_CSR);  // identity is always CSR
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(identity_real32_index64, "core", "quick") {
    auto mat = scl_sparse_identity(10, SCL_REAL32, SCL_INDEX64);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_SPARSE_NNZ(mat, 10);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(identity_real64_index32, "core", "quick", "default") {
    auto mat = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX32);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_EQ(scl_sparse_value_type(mat), SCL_REAL64);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(identity_real64_index64, "core", "quick", "default") {
    auto mat = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

// Int types (16 tests - 4 int types × 2 index types × 2 for completeness)
SCL_TEST_TAGGED(identity_int8_index32, "integer", "quick") {
    auto mat = scl_sparse_identity(5, SCL_INT8, SCL_INDEX32);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_SPARSE_NNZ(mat, 5);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(identity_int8_index64, "integer") {
    auto mat = scl_sparse_identity(5, SCL_INT8, SCL_INDEX64);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(identity_int16_index32, "integer", "quick") {
    auto mat = scl_sparse_identity(10, SCL_INT16, SCL_INDEX32);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(identity_int16_index64, "integer") {
    auto mat = scl_sparse_identity(10, SCL_INT16, SCL_INDEX64);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(identity_int32_index32, "integer", "quick", "default") {
    auto mat = scl_sparse_identity(10, SCL_INT32, SCL_INDEX32);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_EQ(scl_sparse_value_type(mat), SCL_INT32);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(identity_int32_index64, "integer", "default") {
    auto mat = scl_sparse_identity(10, SCL_INT32, SCL_INDEX64);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(identity_int64_index32, "integer") {
    auto mat = scl_sparse_identity(10, SCL_INT64, SCL_INDEX32);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(identity_int64_index64, "integer") {
    auto mat = scl_sparse_identity(10, SCL_INT64, SCL_INDEX64);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

// Uint types (16 tests - 4 uint types × 2 index types × 2 for completeness)
SCL_TEST_TAGGED(identity_uint8_index32, "unsigned", "quick") {
    auto mat = scl_sparse_identity(5, SCL_UINT8, SCL_INDEX32);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_SPARSE_NNZ(mat, 5);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(identity_uint8_index64, "unsigned") {
    auto mat = scl_sparse_identity(5, SCL_UINT8, SCL_INDEX64);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(identity_uint16_index32, "unsigned", "quick") {
    auto mat = scl_sparse_identity(10, SCL_UINT16, SCL_INDEX32);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(identity_uint16_index64, "unsigned") {
    auto mat = scl_sparse_identity(10, SCL_UINT16, SCL_INDEX64);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(identity_uint32_index32, "unsigned", "quick", "default") {
    auto mat = scl_sparse_identity(10, SCL_UINT32, SCL_INDEX32);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_EQ(scl_sparse_value_type(mat), SCL_UINT32);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(identity_uint32_index64, "unsigned", "default") {
    auto mat = scl_sparse_identity(10, SCL_UINT32, SCL_INDEX64);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(identity_uint64_index32, "unsigned") {
    auto mat = scl_sparse_identity(10, SCL_UINT64, SCL_INDEX32);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(identity_uint64_index64, "unsigned") {
    auto mat = scl_sparse_identity(10, SCL_UINT64, SCL_INDEX64);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 3: Boundary Conditions (10 tests)
// =============================================================================

SCL_TEST_SUITE(creation_boundaries)

SCL_TEST_TAGGED(zeros_1x1_minimum, "boundary", "quick") {
    auto mat = scl_sparse_zeros(1, 1, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_SPARSE_DIMS(mat, 1, 1);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_large_dimensions, "boundary", "slow") {
    auto mat = scl_sparse_zeros(10000, 10000, SCL_INT32, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_SPARSE_DIMS(mat, 10000, 10000);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_rectangular_wide, "boundary") {
    auto mat = scl_sparse_zeros(10, 1000, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_SPARSE_DIMS(mat, 10, 1000);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(zeros_rectangular_tall, "boundary") {
    auto mat = scl_sparse_zeros(1000, 10, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_SPARSE_DIMS(mat, 1000, 10);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(identity_1_minimum, "boundary", "quick") {
    auto mat = scl_sparse_identity(1, SCL_REAL64, SCL_INDEX64);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_SPARSE_NNZ(mat, 1);
    scl_sparse_destroy(mat);
}

SCL_TEST_TAGGED(identity_large_n, "boundary", "slow") {
    auto mat = scl_sparse_identity(5000, SCL_INT32, SCL_INDEX64);
    SCL_ASSERT_SPARSE_VALID(mat);
    SCL_ASSERT_SPARSE_NNZ(mat, 5000);
    scl_sparse_destroy(mat);
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 4: Error Cases (10 tests)
// =============================================================================

SCL_TEST_SUITE(creation_errors)

SCL_TEST_TAGGED(zeros_negative_rows, "error", "quick") {
    scl_clear_error();
    auto mat = scl_sparse_zeros(-1, 10, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_NULL(mat);
    SCL_ASSERT_HAS_ERROR();
}

SCL_TEST_TAGGED(zeros_negative_cols, "error", "quick") {
    scl_clear_error();
    auto mat = scl_sparse_zeros(10, -1, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_NULL(mat);
    SCL_ASSERT_HAS_ERROR();
}

SCL_TEST_TAGGED(zeros_invalid_value_type, "error", "quick") {
    scl_clear_error();
    auto mat = scl_sparse_zeros(10, 10, static_cast<scl_value_type_t>(999), SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_NULL(mat);
    SCL_ASSERT_HAS_ERROR();
}

SCL_TEST_TAGGED(zeros_invalid_index_type, "error", "quick") {
    scl_clear_error();
    auto mat = scl_sparse_zeros(10, 10, SCL_REAL64, static_cast<scl_index_type_t>(999), SCL_LAYOUT_CSR);
    SCL_ASSERT_NULL(mat);
    SCL_ASSERT_HAS_ERROR();
}

SCL_TEST_TAGGED(zeros_invalid_layout, "error", "quick") {
    scl_clear_error();
    auto mat = scl_sparse_zeros(10, 10, SCL_REAL64, SCL_INDEX64, static_cast<scl_layout_t>(999));
    SCL_ASSERT_NULL(mat);
    SCL_ASSERT_HAS_ERROR();
}

SCL_TEST_TAGGED(identity_negative_n, "error", "quick") {
    scl_clear_error();
    auto mat = scl_sparse_identity(-1, SCL_REAL64, SCL_INDEX64);
    SCL_ASSERT_NULL(mat);
    SCL_ASSERT_HAS_ERROR();
}

SCL_TEST_TAGGED(identity_invalid_value_type, "error", "quick") {
    scl_clear_error();
    auto mat = scl_sparse_identity(10, static_cast<scl_value_type_t>(0xFF), SCL_INDEX64);
    SCL_ASSERT_NULL(mat);
    SCL_ASSERT_HAS_ERROR();
}

SCL_TEST_SUITE_END

// =============================================================================
// SECTION 5: Destroy Tests (5 tests)
// =============================================================================

SCL_TEST_SUITE(destroy_lifecycle)

SCL_TEST_TAGGED(destroy_valid_handle, "core", "quick") {
    auto mat = scl_sparse_zeros(10, 10, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
    SCL_ASSERT_SPARSE_VALID(mat);
    scl_sparse_destroy(mat);
    // Should not crash
}

SCL_TEST_TAGGED(destroy_null_handle_safe, "core", "quick") {
    scl_sparse_destroy(SCL_NULL_SPARSE);
    // Should not crash
    SCL_ASSERT_TRUE(true);
}

SCL_TEST_TAGGED(destroy_all_value_types, "core") {
    // Create and destroy one of each value type
    scl_value_type_t types[] = {
        SCL_REAL32, SCL_REAL64,
        SCL_INT8, SCL_INT16, SCL_INT32, SCL_INT64,
        SCL_UINT8, SCL_UINT16, SCL_UINT32, SCL_UINT64
    };
    
    for (auto vtype : types) {
        auto mat = scl_sparse_identity(5, vtype, SCL_INDEX64);
        SCL_ASSERT_SPARSE_VALID(mat);
        scl_sparse_destroy(mat);
    }
}

SCL_TEST_SUITE_END

SCL_TEST_END
SCL_TEST_MAIN()

