#pragma once

// =============================================================================
// FILE: scl/binding/c_api/comparison.h
// BRIEF: C API for group comparison and differential abundance analysis
// =============================================================================

#include "scl/binding/c_api/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Composition Analysis
// =============================================================================

scl_error_t scl_comparison_composition_analysis(
    const scl_index_t* cell_types,
    const scl_index_t* conditions,
    scl_size_t n_cells,
    scl_size_t n_types,
    scl_size_t n_conditions,
    scl_real_t* proportions,
    scl_real_t* p_values
);

// =============================================================================
// Abundance Test
// =============================================================================

scl_error_t scl_comparison_abundance_test(
    const scl_index_t* cluster_labels,
    const scl_index_t* conditions,
    scl_size_t n_cells,
    scl_real_t* fold_changes,
    scl_real_t* p_values,
    scl_size_t max_clusters
);

// =============================================================================
// Differential Abundance (DA)
// =============================================================================

scl_error_t scl_comparison_differential_abundance(
    const scl_index_t* cluster_labels,
    const scl_index_t* sample_ids,
    const scl_index_t* conditions,
    scl_size_t n_cells,
    scl_real_t* da_scores,
    scl_real_t* p_values,
    scl_size_t max_clusters
);

// =============================================================================
// Condition Response
// =============================================================================

scl_error_t scl_comparison_condition_response(
    scl_sparse_matrix_t expression,
    const scl_index_t* conditions,
    scl_size_t n_genes,
    scl_real_t* response_scores,
    scl_real_t* p_values
);

// =============================================================================
// Effect Size
// =============================================================================

scl_real_t scl_comparison_effect_size(
    const scl_real_t* group1,
    scl_size_t n1,
    const scl_real_t* group2,
    scl_size_t n2
);

scl_real_t scl_comparison_glass_delta(
    const scl_real_t* control,
    scl_size_t n_control,
    const scl_real_t* treatment,
    scl_size_t n_treatment
);

scl_real_t scl_comparison_hedges_g(
    const scl_real_t* group1,
    scl_size_t n1,
    const scl_real_t* group2,
    scl_size_t n2
);

#ifdef __cplusplus
}
#endif
