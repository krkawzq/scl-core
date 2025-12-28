#pragma once

// =============================================================================
// FILE: scl/binding/c_api/slice/slice.h
// BRIEF: C API for sparse matrix slicing operations
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Slice Primary Dimension (Rows for CSR, Cols for CSC)
// =============================================================================

scl_error_t scl_slice_inspect_primary(
    scl_sparse_t matrix,
    const scl_index_t* keep_indices,
    scl_size_t n_keep,
    scl_index_t* out_nnz
);

scl_error_t scl_slice_primary(
    scl_sparse_t matrix,
    const scl_index_t* keep_indices,
    scl_size_t n_keep,
    scl_sparse_t* out_matrix
);

// =============================================================================
// Filter Secondary Dimension (Cols for CSR, Rows for CSC)
// =============================================================================

scl_error_t scl_slice_inspect_filter_secondary(
    scl_sparse_t matrix,
    const uint8_t* mask,
    scl_size_t secondary_dim,
    scl_index_t* out_nnz
);

scl_error_t scl_slice_filter_secondary(
    scl_sparse_t matrix,
    const uint8_t* mask,
    scl_size_t secondary_dim,
    scl_sparse_t* out_matrix
);

#ifdef __cplusplus
}
#endif
