#pragma once

// =============================================================================
// FILE: scl/binding/c_api/correlation/correlation.h
// BRIEF: C API for Pearson correlation computation
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Statistics Computation
// =============================================================================

scl_error_t scl_corr_compute_stats(
    scl_sparse_t matrix,
    scl_real_t* out_means,                // Output [n_rows]
    scl_real_t* out_inv_stds              // Output [n_rows]
);

// =============================================================================
// Pearson Correlation Matrix
// =============================================================================

scl_error_t scl_corr_pearson(
    scl_sparse_t matrix,
    const scl_real_t* means,              // [n_rows] precomputed means
    const scl_real_t* inv_stds,           // [n_rows] precomputed inverse stds
    scl_real_t* output                    // Output [n_rows * n_rows]
);

scl_error_t scl_corr_pearson_auto(
    scl_sparse_t matrix,
    scl_real_t* output                    // Output [n_rows * n_rows]
);

#ifdef __cplusplus
}
#endif
