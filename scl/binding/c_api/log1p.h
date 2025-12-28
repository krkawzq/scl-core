#pragma once

// =============================================================================
// FILE: scl/binding/c_api/log1p.h
// BRIEF: C API for logarithmic transforms
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "scl/binding/c_api/core_types.h"

// Apply log1p transform in-place to sparse matrix
scl_error_t scl_log1p_inplace(scl_sparse_matrix_t* matrix);

// Apply log2p1 transform in-place to sparse matrix
scl_error_t scl_log2p1_inplace(scl_sparse_matrix_t* matrix);

// Apply expm1 transform in-place to sparse matrix
scl_error_t scl_expm1_inplace(scl_sparse_matrix_t* matrix);

#ifdef __cplusplus
}
#endif
