#pragma once

// =============================================================================
// FILE: scl/binding/c_api/feature.h
// BRIEF: C API for Feature Statistics
// =============================================================================

#include "scl/binding/c_api/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Feature Statistics
// =============================================================================

// Compute standard moments (mean and variance) for each row/column
// matrix: Sparse matrix handle (CSR format)
// out_means: Output means [primary_dim]
// out_vars: Output variances [primary_dim]
// ddof: Delta degrees of freedom (0 for population, 1 for sample)
// Returns: Error code
scl_error_t scl_standard_moments(
    scl_sparse_matrix_t matrix,
    scl_real_t* out_means,
    scl_real_t* out_vars,
    int ddof
);

// Compute clipped moments (mean and variance) with clipping values
// matrix: Sparse matrix handle (CSR format)
// clip_vals: Clipping values for each row/column [primary_dim]
// out_means: Output means [primary_dim]
// out_vars: Output variances [primary_dim]
// Returns: Error code
scl_error_t scl_clipped_moments(
    scl_sparse_matrix_t matrix,
    const scl_real_t* clip_vals,
    scl_real_t* out_means,
    scl_real_t* out_vars
);

// Compute detection rate (fraction of non-zero values) for each row/column
// matrix: Sparse matrix handle (CSR format)
// out_rates: Output detection rates [primary_dim]
// Returns: Error code
scl_error_t scl_detection_rate(
    scl_sparse_matrix_t matrix,
    scl_real_t* out_rates
);

// Compute dispersion (variance/mean) from means and variances
// means: Input means [n]
// vars: Input variances [n]
// out_dispersion: Output dispersion values [n]
// n: Number of features
// Returns: Error code
scl_error_t scl_dispersion(
    const scl_real_t* means,
    const scl_real_t* vars,
    scl_real_t* out_dispersion,
    scl_size_t n
);

#ifdef __cplusplus
}
#endif
