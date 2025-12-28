#pragma once

// =============================================================================
// FILE: scl/binding/c_api/spatial_pattern.h
// BRIEF: C API for spatial pattern detection
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "scl/binding/c_api/core_types.h"

// Detect spatially variable genes
scl_error_t scl_spatial_pattern_detect_variable_genes(
    scl_sparse_matrix_t expression,
    const scl_real_t* coords,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_dims,
    scl_real_t* p_values,
    scl_real_t bandwidth
);

#ifdef __cplusplus
}
#endif
