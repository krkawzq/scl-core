#pragma once

// =============================================================================
// FILE: scl/binding/c_api/projection.h
// BRIEF: C API for random projection
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "scl/binding/c_api/core_types.h"

// Projection type
typedef enum {
    SCL_PROJECTION_GAUSSIAN = 0,
    SCL_PROJECTION_ACHLIOPTAS = 1,
    SCL_PROJECTION_SPARSE = 2,
    SCL_PROJECTION_COUNTSKETCH = 3,
    SCL_PROJECTION_FEATUREHASH = 4
} scl_projection_type_t;

// Project sparse matrix
scl_error_t scl_projection_project(
    scl_sparse_matrix_t input,
    scl_real_t* output,
    scl_index_t n_rows,
    scl_index_t n_input_dims,
    scl_index_t n_output_dims,
    scl_projection_type_t type,
    uint64_t seed
);

#ifdef __cplusplus
}
#endif
