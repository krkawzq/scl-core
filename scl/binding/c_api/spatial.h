#pragma once

// =============================================================================
// FILE: scl/binding/c_api/spatial/spatial.h
// BRIEF: C API for spatial statistics
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Weight Sum
// =============================================================================

scl_error_t scl_spatial_weight_sum(
    scl_sparse_t graph,                    // Spatial graph
    scl_real_t* weight_sum                 // Output: total weight sum
);

// =============================================================================
// Moran's I Statistic
// =============================================================================

scl_error_t scl_spatial_morans_i(
    scl_sparse_t graph,                    // Spatial graph [n_cells, n_cells]
    scl_sparse_t features,                 // Feature matrix [n_features, n_cells]
    scl_real_t* output                     // Output [n_features]
);

// =============================================================================
// Geary's C Statistic
// =============================================================================

scl_error_t scl_spatial_gearys_c(
    scl_sparse_t graph,                    // Spatial graph [n_cells, n_cells]
    scl_sparse_t features,                 // Feature matrix [n_features, n_cells]
    scl_real_t* output                     // Output [n_features]
);

#ifdef __cplusplus
}
#endif
