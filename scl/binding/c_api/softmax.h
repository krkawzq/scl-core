#pragma once

// =============================================================================
// FILE: scl/binding/c_api/softmax/softmax.h
// BRIEF: C API for softmax operations
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Dense Array Softmax
// =============================================================================

scl_error_t scl_softmax_array(
    scl_real_t* values,
    scl_size_t n,
    scl_real_t temperature
);

scl_error_t scl_log_softmax_array(
    scl_real_t* values,
    scl_size_t n,
    scl_real_t temperature
);

// =============================================================================
// Sparse Matrix Softmax
// =============================================================================

scl_error_t scl_softmax_matrix(
    scl_sparse_t* matrix,
    scl_real_t temperature
);

scl_error_t scl_log_softmax_matrix(
    scl_sparse_t* matrix,
    scl_real_t temperature
);

#ifdef __cplusplus
}
#endif
