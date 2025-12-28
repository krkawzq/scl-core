#pragma once

// =============================================================================
// FILE: scl/binding/c_api/projection/projection.h
// BRIEF: C API for random projection dimensionality reduction
// =============================================================================

#include "scl/binding/c_api/core/core.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Projection Types
// =============================================================================

typedef enum {
    SCL_PROJECTION_GAUSSIAN = 0,
    SCL_PROJECTION_ACHLIOPTAS = 1,
    SCL_PROJECTION_SPARSE = 2,
    SCL_PROJECTION_COUNTSKETCH = 3,
    SCL_PROJECTION_FEATURE_HASH = 4
} scl_projection_type_t;

// =============================================================================
// Projection Functions
// =============================================================================

// Project sparse matrix to lower dimension
// matrix: Input sparse matrix (CSR or CSC)
// output_dim: Target output dimension
// output: Output dense matrix [rows * output_dim] (row-major)
// type: Projection type
// seed: Random seed
scl_error_t scl_projection_project(
    scl_sparse_t matrix,
    scl_size_t output_dim,
    scl_real_t* output,
    scl_projection_type_t type,
    uint64_t seed
);

// Auto-select best projection method based on dimensions
scl_error_t scl_projection_project_auto(
    scl_sparse_t matrix,
    scl_size_t output_dim,
    scl_real_t* output,
    uint64_t seed
);

// Gaussian projection (on-the-fly)
scl_error_t scl_projection_gaussian(
    scl_sparse_t matrix,
    scl_size_t output_dim,
    scl_real_t* output,
    uint64_t seed
);

// Achlioptas sparse projection
scl_error_t scl_projection_achlioptas(
    scl_sparse_t matrix,
    scl_size_t output_dim,
    scl_real_t* output,
    uint64_t seed
);

// Sparse projection with specified density
scl_error_t scl_projection_sparse(
    scl_sparse_t matrix,
    scl_size_t output_dim,
    scl_real_t* output,
    scl_real_t density,
    uint64_t seed
);

// CountSketch projection
scl_error_t scl_projection_countsketch(
    scl_sparse_t matrix,
    scl_size_t output_dim,
    scl_real_t* output,
    uint64_t seed
);

// Feature hash projection
scl_error_t scl_projection_feature_hash(
    scl_sparse_t matrix,
    scl_size_t output_dim,
    scl_real_t* output,
    scl_size_t n_hashes,
    uint64_t seed
);

// Compute optimal JL dimension
scl_size_t scl_projection_jl_dimension(
    scl_size_t n_samples,
    scl_real_t epsilon
);

#ifdef __cplusplus
}
#endif
