#pragma once

// =============================================================================
// FILE: scl/binding/c_api/normalize/normalize.h
// BRIEF: C API for normalization operations
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Row Sums Computation
// =============================================================================

scl_error_t scl_norm_compute_row_sums(
    scl_sparse_t matrix,
    scl_real_t* output                    // Output [n_rows]
);

// =============================================================================
// Scale Primary Dimension
// =============================================================================

scl_error_t scl_norm_scale_primary(
    scl_sparse_t matrix,                  // Modified in-place
    const scl_real_t* scales             // [n_rows] or [n_cols] depending on format
);

// =============================================================================
// Masked Sums
// =============================================================================

scl_error_t scl_norm_primary_sums_masked(
    scl_sparse_t matrix,
    const unsigned char* mask,           // [secondary_dim] mask array
    scl_real_t* output                    // Output [primary_dim]
);

// =============================================================================
// Highly Expressed Detection
// =============================================================================

scl_error_t scl_norm_detect_highly_expressed(
    scl_sparse_t matrix,
    const scl_real_t* row_sums,          // [n_rows] precomputed row sums
    scl_real_t max_fraction,
    unsigned char* out_mask              // Output [n_cols] mask
);

#ifdef __cplusplus
}
#endif
