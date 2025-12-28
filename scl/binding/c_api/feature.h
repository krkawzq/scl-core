#pragma once

// =============================================================================
// FILE: scl/binding/c_api/feature/feature.h
// BRIEF: C API for feature statistics operations
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Standard Moments (Mean and Variance)
// =============================================================================

scl_error_t scl_feature_standard_moments(
    scl_sparse_t matrix,
    scl_real_t* out_means,
    scl_real_t* out_vars,
    scl_size_t n_features,
    int ddof
);

// =============================================================================
// Clipped Moments
// =============================================================================

scl_error_t scl_feature_clipped_moments(
    scl_sparse_t matrix,
    const scl_real_t* clip_vals,
    scl_real_t* out_means,
    scl_real_t* out_vars,
    scl_size_t n_features
);

// =============================================================================
// Detection Rate
// =============================================================================

scl_error_t scl_feature_detection_rate(
    scl_sparse_t matrix,
    scl_real_t* out_rates,
    scl_size_t n_features
);

// =============================================================================
// Dispersion
// =============================================================================

scl_error_t scl_feature_dispersion(
    const scl_real_t* means,
    const scl_real_t* vars,
    scl_real_t* out_dispersion,
    scl_size_t n_features
);

#ifdef __cplusplus
}
#endif
