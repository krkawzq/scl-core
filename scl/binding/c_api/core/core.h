#pragma once

// =============================================================================
// FILE: scl/binding/c_api/core/core.h
// BRIEF: Core C API types and error handling
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// =============================================================================
// Opaque Handle Types (Type-Safe Pointers)
// =============================================================================

// Forward declarations for type safety
typedef struct scl_sparse_matrix  scl_sparse_matrix;
typedef struct scl_dense_matrix   scl_dense_matrix;

// Handle types (pointers to opaque structs)
typedef scl_sparse_matrix* scl_sparse_t;
typedef scl_dense_matrix*  scl_dense_t;

// =============================================================================
// Basic Value Types
// =============================================================================

#if defined(SCL_USE_FLOAT32)
    typedef float scl_real_t;
#elif defined(SCL_USE_FLOAT64)
    typedef double scl_real_t;
#elif defined(SCL_USE_FLOAT16)
    typedef _Float16 scl_real_t;
#else
    typedef float scl_real_t;  // Default
#endif

#if defined(SCL_USE_INT16)
    typedef int16_t scl_index_t;
#elif defined(SCL_USE_INT32)
    typedef int32_t scl_index_t;
#elif defined(SCL_USE_INT64)
    typedef int64_t scl_index_t;
#else
    typedef int64_t scl_index_t;  // Default
#endif

typedef size_t scl_size_t;

// =============================================================================
// Error Handling
// =============================================================================

typedef int32_t scl_error_t;

#define SCL_OK                      0
#define SCL_ERROR_UNKNOWN           1
#define SCL_ERROR_INTERNAL          2
#define SCL_ERROR_NULL_POINTER      10
#define SCL_ERROR_INVALID_ARGUMENT  11
#define SCL_ERROR_DIMENSION_MISMATCH 12
#define SCL_ERROR_OUT_OF_MEMORY     20
#define SCL_ERROR_TYPE_MISMATCH     30
#define SCL_ERROR_NOT_IMPLEMENTED   40

// Get human-readable error message for last error
const char* scl_get_last_error(void);

// Clear last error
void scl_clear_error(void);

#ifdef __cplusplus
}
#endif
