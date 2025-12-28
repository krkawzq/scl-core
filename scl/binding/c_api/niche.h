#pragma once

// =============================================================================
// FILE: scl/binding/c_api/niche.h
// BRIEF: C API for cellular neighborhood analysis
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "scl/binding/c_api/core_types.h"

// Compute neighborhood composition
scl_error_t scl_niche_neighborhood_composition(
    scl_sparse_matrix_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    scl_index_t n_cells,
    scl_index_t n_cell_types,
    scl_real_t* composition_output
);

// Compute colocalization score
scl_error_t scl_niche_colocalization_score(
    scl_sparse_matrix_t spatial_neighbors,
    const scl_index_t* cell_type_labels,
    scl_index_t n_cells,
    scl_index_t n_cell_types,
    scl_real_t* colocalization_matrix
);

#ifdef __cplusplus
}
#endif
