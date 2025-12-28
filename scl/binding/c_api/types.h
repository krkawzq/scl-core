#pragma once

// =============================================================================
// FILE: scl/binding/c_api/types.h
// BRIEF: C API type definitions
// =============================================================================

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Basic Types
// =============================================================================

// Real type (float/double/_Float16 based on SCL_PRECISION)
#if SCL_PRECISION == 0
    typedef float scl_real_t;
#elif SCL_PRECISION == 1
    typedef double scl_real_t;
#elif SCL_PRECISION == 2
    typedef _Float16 scl_real_t;
#else
    typedef float scl_real_t;  // Default
#endif

// Index type (int16_t/int32_t/int64_t based on SCL_INDEX_PRECISION)
#if SCL_INDEX_PRECISION == 0
    typedef int16_t scl_index_t;
#elif SCL_INDEX_PRECISION == 1
    typedef int32_t scl_index_t;
#elif SCL_INDEX_PRECISION == 2
    typedef int64_t scl_index_t;
#else
    typedef int32_t scl_index_t;  // Default
#endif

typedef size_t scl_size_t;

// =============================================================================
// Error Codes
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
// Sparse Matrix Handle (opaque pointer)
// =============================================================================

typedef void* scl_sparse_matrix_t;

#ifdef __cplusplus
}
#endif
