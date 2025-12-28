#pragma once

// =============================================================================
// FILE: scl/binding/c_api/lineage/lineage.h
// BRIEF: C API for lineage tracing
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Lineage Coupling Matrix
// =============================================================================

scl_error_t scl_lineage_coupling(
    const scl_index_t* clone_ids,
    const scl_index_t* cell_types,
    scl_real_t* coupling_matrix,
    scl_size_t n_cells,
    scl_size_t n_clones,
    scl_size_t n_types
);

// =============================================================================
// Fate Bias Score
// =============================================================================

scl_error_t scl_lineage_fate_bias(
    const scl_index_t* clone_ids,
    const scl_index_t* cell_types,
    scl_real_t* bias_scores,
    scl_size_t n_cells,
    scl_size_t n_types
);

// =============================================================================
// Build Lineage Tree
// =============================================================================

scl_error_t scl_lineage_build_tree(
    const scl_index_t* clone_ids,
    const scl_real_t* pseudotime,
    scl_index_t* parent,
    scl_size_t n_cells,
    scl_size_t n_clones
);

#ifdef __cplusplus
}
#endif
