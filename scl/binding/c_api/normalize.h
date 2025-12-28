#pragma once

// =============================================================================
// FILE: scl/binding/c_api/normalize.h
// BRIEF: C API for normalization operations
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "scl/binding/c_api/core_types.h"

// Compute row sums
scl_error_t scl_normalize_compute_row_sums(
    scl_sparse_matrix_t matrix,
    scl_real_t* output,
    scl_size_t n_rows
);

// Scale primary dimension
scl_error_t scl_normalize_scale_primary(
    scl_sparse_matrix_t matrix,
    const scl_real_t* scales,
    scl_size_t n_rows
);

// Primary sums with mask
scl_error_t scl_normalize_primary_sums_masked(
    scl_sparse_matrix_t matrix,
    const uint8_t* mask,
    scl_real_t* output,
    scl_size_t n_rows
);

// Detect highly expressed genes
scl_error_t scl_normalize_detect_highly_expressed(
    scl_sparse_matrix_t matrix,
    const scl_real_t* row_sums,
    scl_real_t max_fraction,
    uint8_t* out_mask,
    scl_size_t n_cols
);

#ifdef __cplusplus
}
#endif
