#pragma once

// =============================================================================
// FILE: scl/binding/c_api/core/core.h
// BRIEF: C ABI for scl-core high-performance operator library
// =============================================================================
//
// DESIGN PRINCIPLES:
//   - Stable C ABI for Python FFI and cross-language bindings
//   - Type-safe opaque handles (no raw pointers exposed)
//   - Thread-safe error reporting via thread-local storage
//   - Zero-overhead when no errors occur
//   - Compatible with C99 and C++11+
//
// ABI STABILITY GUARANTEE:
//   - Error codes are stable across versions
//   - Handle types remain opaque
//   - Function signatures will not change within major version
//   - New functions may be added but existing ones won't be removed
//
// MEMORY MODEL:
//   - All memory lifecycle managed by internal Registry
//   - Handles must be destroyed via scl_*_destroy()
//   - View/slice operations share memory (zero-copy)
//   - Thread-safe reference counting for shared data
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// =============================================================================
// Version Information
// =============================================================================

#define SCL_C_API_VERSION_MAJOR 1
#define SCL_C_API_VERSION_MINOR 0
#define SCL_C_API_VERSION_PATCH 0

// =============================================================================
// Export Macro (Platform-Specific DLL/SO Symbol Visibility)
// =============================================================================

#if defined(_MSC_VER)
    #define SCL_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
    #define SCL_EXPORT __attribute__((visibility("default")))
#else
    #define SCL_EXPORT
#endif

// Get runtime version string (e.g., "1.0.0-beta")
const char* scl_get_version(void);

// Get build configuration (e.g., "float64+int64+avx2")
const char* scl_get_build_config(void);

// =============================================================================
// Opaque Handle Types (Type-Safe, ABI-Stable)
// =============================================================================

// Forward declarations for type safety
// These are deliberately incomplete types to prevent direct member access
typedef struct scl_sparse_matrix scl_sparse_matrix;
typedef struct scl_dense_matrix scl_dense_matrix;

// Handle types (non-null pointers to opaque structs)
// NULL is never a valid handle - all operations check for NULL and return error
typedef scl_sparse_matrix* scl_sparse_t;
typedef scl_dense_matrix* scl_dense_t;

// =============================================================================
// Basic Value Types (Must Match C++ scl::Real and scl::Index)
// =============================================================================

// Real type: controlled by compile-time macros
// Default: double (float64) for numerical stability
#if defined(SCL_USE_FLOAT32)
typedef float scl_real_t;
#define SCL_REAL_TYPE_NAME "float32"
#elif defined(SCL_USE_FLOAT64)
typedef double scl_real_t;
#define SCL_REAL_TYPE_NAME "float64"
#elif defined(SCL_USE_FLOAT16)
typedef _Float16 scl_real_t;
#define SCL_REAL_TYPE_NAME "float16"
#else
// Default to float64 if not specified
typedef double scl_real_t;
#define SCL_REAL_TYPE_NAME "float64"
#endif

// Index type: controlled by compile-time macros
// Default: int64_t for large-scale applications
#if defined(SCL_USE_INT16)
typedef int16_t scl_index_t;
#define SCL_INDEX_TYPE_NAME "int16"
#elif defined(SCL_USE_INT32)
typedef int32_t scl_index_t;
#define SCL_INDEX_TYPE_NAME "int32"
#elif defined(SCL_USE_INT64)
typedef int64_t scl_index_t;
#define SCL_INDEX_TYPE_NAME "int64"
#else
// Default to int64 if not specified
typedef int64_t scl_index_t;
#define SCL_INDEX_TYPE_NAME "int64"
#endif

// Size type: always matches platform size_t
typedef size_t scl_size_t;

// Boolean type for C (int is standard for C API)
typedef int scl_bool_t;
#define SCL_TRUE 1
#define SCL_FALSE 0

// =============================================================================
// Error Handling (Thread-Safe, Zero-Overhead When No Errors)
// =============================================================================

// Error codes (stable across versions, matches scl::ErrorCode)
typedef int32_t scl_error_t;

// Success
#define SCL_OK 0

// General errors (1-9)
#define SCL_ERROR_UNKNOWN 1
#define SCL_ERROR_INTERNAL 2
#define SCL_ERROR_OUT_OF_MEMORY 3
#define SCL_ERROR_NULL_POINTER 4

// Argument errors (10-19)
#define SCL_ERROR_INVALID_ARGUMENT 10
#define SCL_ERROR_DIMENSION_MISMATCH 11
#define SCL_ERROR_DOMAIN_ERROR 12
#define SCL_ERROR_RANGE_ERROR 13
#define SCL_ERROR_INDEX_OUT_OF_BOUNDS 14

// Type errors (20-29)
#define SCL_ERROR_TYPE_ERROR 20
#define SCL_ERROR_TYPE_MISMATCH 21

// I/O errors (30-39)
#define SCL_ERROR_IO_ERROR 30
#define SCL_ERROR_FILE_NOT_FOUND 31
#define SCL_ERROR_PERMISSION_DENIED 32
#define SCL_ERROR_READ_ERROR 33
#define SCL_ERROR_WRITE_ERROR 34
#define SCL_ERROR_UNREGISTERED_POINTER 35
#define SCL_ERROR_BUFFER_NOT_FOUND 36

// Feature errors (40-49)
#define SCL_ERROR_NOT_IMPLEMENTED 40
#define SCL_ERROR_FEATURE_UNAVAILABLE 41

// Numerical errors (50-59)
#define SCL_ERROR_NUMERICAL_ERROR 50
#define SCL_ERROR_DIVISION_BY_ZERO 51
#define SCL_ERROR_OVERFLOW 52
#define SCL_ERROR_UNDERFLOW 53
#define SCL_ERROR_CONVERGENCE_ERROR 54

// Get human-readable error message for last error (thread-local)
// Returns: pointer to static thread-local buffer (valid until next error or clear)
// Returns "No error" if no error has occurred
const char* scl_get_last_error(void);

// Get last error code (thread-local)
// Returns: SCL_OK if no error, otherwise the error code
scl_error_t scl_get_last_error_code(void);

// Clear last error (thread-local)
// After this call, scl_get_last_error() returns "No error"
void scl_clear_error(void);

// Check if an error code represents success
// Returns: SCL_TRUE if code == SCL_OK, otherwise SCL_FALSE
scl_bool_t scl_is_ok(scl_error_t code);

// Check if an error code represents an error
// Returns: SCL_TRUE if code != SCL_OK, otherwise SCL_FALSE
scl_bool_t scl_is_error(scl_error_t code);

#ifdef __cplusplus
}
#endif
