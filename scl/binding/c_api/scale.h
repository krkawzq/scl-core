#pragma once

// =============================================================================
// FILE: scl/binding/c_api/scale/scale.h
// BRIEF: C API for scaling operations
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Standardization
// =============================================================================

// Standardize matrix: (x - mean) / std, with optional clipping
// Modifies matrix in-place
scl_error_t scl_scale_standardize(
    scl_sparse_t matrix,                // Matrix to standardize (modified in-place)
    const scl_real_t* means,            // Mean values [primary_dim]
    const scl_real_t* stds,             // Standard deviations [primary_dim]
    scl_size_t primary_dim,
    scl_real_t max_value,                // Max value for clipping (0 = no clipping)
    int zero_center                      // 1 = subtract mean, 0 = only scale
);

// =============================================================================
// Row/Column Scaling
// =============================================================================

// Scale each row/column by a factor
scl_error_t scl_scale_rows(
    scl_sparse_t matrix,                // Matrix to scale (modified in-place)
    const scl_real_t* scales,           // Scale factors [primary_dim]
    scl_size_t primary_dim
);

// Shift each row/column by an offset
scl_error_t scl_scale_shift_rows(
    scl_sparse_t matrix,                // Matrix to shift (modified in-place)
    const scl_real_t* offsets,         // Offset values [primary_dim]
    scl_size_t primary_dim
);

#ifdef __cplusplus
}
#endif
