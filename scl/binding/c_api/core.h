#pragma once

// =============================================================================
// FILE: scl/binding/c_api/core.h
// BRIEF: C API core types and error handling
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// =============================================================================
// Basic Types
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
    typedef int32_t scl_index_t;  // Default
#endif

typedef size_t scl_size_t;

// =============================================================================
// Error Handling
// =============================================================================

typedef int32_t scl_error_t;

#define SCL_ERROR_OK 0
#define SCL_ERROR_UNKNOWN 1
#define SCL_ERROR_INTERNAL_ERROR 2
#define SCL_ERROR_INVALID_ARGUMENT 10
#define SCL_ERROR_DIMENSION_MISMATCH 11
#define SCL_ERROR_DOMAIN_ERROR 12
#define SCL_ERROR_TYPE_ERROR 20
#define SCL_ERROR_IO_ERROR 30
#define SCL_ERROR_UNREGISTERED_POINTER 35
#define SCL_ERROR_NOT_IMPLEMENTED 40

// =============================================================================
// Sparse Matrix Handle
// =============================================================================

typedef void* scl_sparse_matrix_t;

#ifdef __cplusplus
}
#endif
