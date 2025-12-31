#pragma once

// =============================================================================
// SCL C-API Test Framework - Main Header
// =============================================================================
//
// Comprehensive testing framework for SCL C-API with:
//   - Advanced test registration and execution (core.hpp)
//   - RAII handle wrappers (guard.hpp)
//   - Random data generators (data.hpp)
//   - Numerical precision comparisons (precision.hpp)
//   - Eigen reference implementations (oracle.hpp)
//   - BLAS reference (blas.hpp - optional)
//
// Usage:
//   #include "test.hpp"
//   
//   SCL_TEST_BEGIN
//   
//   SCL_TEST_UNIT(my_test) {
//       auto mat = scl_sparse_identity(10, SCL_REAL64, SCL_INDEX64, SCL_LAYOUT_CSR);
//       SCL_ASSERT_NOT_NULL(mat);
//       SCL_ASSERT_EQ(scl_sparse_rows(mat), 10);
//       scl_sparse_destroy(mat);
//   }
//   
//   SCL_TEST_END
//   SCL_TEST_MAIN()
//
// =============================================================================

// Core test framework
#include "core.hpp"

// RAII guards for automatic cleanup
#include "guard.hpp"

// Random data generation
#include "data.hpp"

// Numerical precision comparison
#include "precision.hpp"

// Eigen reference implementations
#include "oracle.hpp"

// BLAS reference (optional, included if available)
#ifdef SCL_TEST_USE_BLAS
#include "blas.hpp"
#endif

// =============================================================================
// Helper Utilities
// =============================================================================

namespace scl::test {

/// Convert scl_real_type_t to string
inline const char* real_type_to_string(scl_real_type_t type) {
    switch (type) {
        case SCL_REAL32: return "REAL32";
        case SCL_REAL64: return "REAL64";
        default: return "UNKNOWN";
    }
}

/// Convert scl_index_type_t to string
inline const char* index_type_to_string(scl_index_type_t type) {
    switch (type) {
        case SCL_INDEX32: return "INDEX32";
        case SCL_INDEX64: return "INDEX64";
        default: return "UNKNOWN";
    }
}

/// Convert scl_layout_t to string
inline const char* layout_to_string(scl_layout_t layout) {
    switch (layout) {
        case SCL_LAYOUT_CSR: return "CSR";
        case SCL_LAYOUT_CSC: return "CSC";
        default: return "UNKNOWN";
    }
}

/// Format sparse matrix info as string
inline std::string sparse_info(scl_sparse_t mat) {
    if (!mat) return "null";
    
    std::ostringstream oss;
    oss << "[" << scl_sparse_rows(mat) << "x" << scl_sparse_cols(mat) 
        << ", nnz=" << scl_sparse_nnz(mat)
        << ", " << real_type_to_string(scl_sparse_real_type(mat))
        << ", " << index_type_to_string(scl_sparse_index_type(mat))
        << ", " << layout_to_string(scl_sparse_layout(mat))
        << "]";
    return oss.str();
}

/// Check if error occurred and get message
inline std::string get_last_error() {
    if (!scl_has_error()) {
        return "";
    }
    
    char buffer[512];
    std::size_t length = scl_get_error_info(buffer, sizeof(buffer));
    return std::string(buffer, std::min(length, sizeof(buffer) - 1));
}

/// Clear error before test
inline void clear_test_error() {
    scl_clear_error();
}

} // namespace scl::test

// =============================================================================
// Additional Assertion Macros for C-API
// =============================================================================

/// Assert no SCL error occurred
#define SCL_ASSERT_NO_ERROR() \
    do { \
        if (scl_has_error()) { \
            auto err_msg = ::scl::test::get_last_error(); \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Expected no error, but got: " + err_msg); \
        } \
    } while (0)

/// Assert SCL error occurred
#define SCL_ASSERT_HAS_ERROR() \
    do { \
        if (!scl_has_error()) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Expected error but none occurred"); \
        } \
    } while (0)

/// Assert specific error code
#define SCL_ASSERT_ERROR_CODE(expected_code) \
    do { \
        std::int32_t code = scl_get_error_code(); \
        if (code != static_cast<std::int32_t>(expected_code)) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Wrong error code", \
                std::to_string(static_cast<int>(expected_code)), \
                std::to_string(code)); \
        } \
    } while (0)

/// Assert sparse matrix is not null
#define SCL_ASSERT_SPARSE_VALID(mat) \
    do { \
        if (!(mat)) { \
            auto err_msg = ::scl::test::get_last_error(); \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Expected valid sparse matrix, got null. Error: " + err_msg); \
        } \
    } while (0)

/// Assert sparse matrix dimensions
#define SCL_ASSERT_SPARSE_DIMS(mat, expected_rows, expected_cols) \
    do { \
        auto _r = scl_sparse_rows(mat); \
        auto _c = scl_sparse_cols(mat); \
        if (_r != (expected_rows) || _c != (expected_cols)) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Wrong sparse matrix dimensions", \
                std::to_string(expected_rows) + "x" + std::to_string(expected_cols), \
                std::to_string(_r) + "x" + std::to_string(_c)); \
        } \
    } while (0)

/// Assert sparse matrix NNZ
#define SCL_ASSERT_SPARSE_NNZ(mat, expected_nnz) \
    do { \
        auto _nnz = scl_sparse_nnz(mat); \
        if (_nnz != (expected_nnz)) { \
            throw ::scl::test::TestException(__FILE__, __LINE__, \
                "Wrong sparse matrix NNZ", \
                std::to_string(expected_nnz), \
                std::to_string(_nnz)); \
        } \
    } while (0)

