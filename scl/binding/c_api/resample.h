#pragma once

// =============================================================================
// FILE: scl/binding/c_api/resample/resample.h
// BRIEF: C API for resampling operations
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Downsampling
// =============================================================================

// Downsample matrix to target sum per row/column (modifies matrix in-place)
scl_error_t scl_resample_downsample(
    scl_sparse_t matrix,                // Matrix to downsample (modified in-place)
    scl_real_t target_sum,              // Target sum per row/column
    uint64_t seed                       // Random seed
);

// Downsample with variable target counts per row/column
scl_error_t scl_resample_downsample_variable(
    scl_sparse_t matrix,
    const scl_real_t* target_counts,    // Target counts [primary_dim]
    scl_size_t primary_dim,
    uint64_t seed
);

// =============================================================================
// Resampling
// =============================================================================

// Binomial resampling: each value is resampled from Binomial(count, p)
scl_error_t scl_resample_binomial(
    scl_sparse_t matrix,                // Matrix to resample (modified in-place)
    scl_real_t p,                       // Success probability
    uint64_t seed
);

// Poisson resampling: each value is resampled from Poisson(count * lambda)
scl_error_t scl_resample_poisson(
    scl_sparse_t matrix,
    scl_real_t lambda,                  // Poisson rate multiplier
    uint64_t seed
);

#ifdef __cplusplus
}
#endif
