#pragma once

// =============================================================================
// FILE: scl/binding/c_api/core_types.h
// BRIEF: Core C API type definitions
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

// Forward declarations
typedef struct scl_sparse_matrix_t scl_sparse_matrix_t;

// Basic types matching C++ types
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

// Error codes
typedef enum {
    SCL_ERROR_OK = 0,
    SCL_ERROR_UNKNOWN = 1,
    SCL_ERROR_INTERNAL = 2,
    SCL_ERROR_INVALID_ARGUMENT = 10,
    SCL_ERROR_DIMENSION_MISMATCH = 11,
    SCL_ERROR_DOMAIN = 12,
    SCL_ERROR_TYPE = 20,
    SCL_ERROR_IO = 30,
    SCL_ERROR_UNREGISTERED_POINTER = 35,
    SCL_ERROR_NOT_IMPLEMENTED = 40
} scl_error_t;

#ifdef __cplusplus
}
#endif
