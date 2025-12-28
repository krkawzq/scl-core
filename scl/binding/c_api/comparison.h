#pragma once

// =============================================================================
// FILE: scl/binding/c_api/comparison/comparison.h
// BRIEF: C API for group comparison and differential abundance analysis
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Composition Analysis
// =============================================================================

scl_error_t scl_comp_composition_analysis(
    const scl_index_t* cell_types,         // [n_cells]
    const scl_index_t* conditions,        // [n_cells]
    scl_size_t n_cells,
    scl_size_t n_types,
    scl_size_t n_conditions,
    scl_real_t* proportions,              // Output [n_types * n_conditions]
    scl_real_t* p_values                 // Output [n_types]
);

// =============================================================================
// Abundance Test
// =============================================================================

scl_error_t scl_comp_abundance_test(
    const scl_index_t* cluster_labels,    // [n_cells]
    const scl_index_t* condition,        // [n_cells]
    scl_size_t n_cells,
    scl_real_t* fold_changes,             // Output [n_clusters]
    scl_real_t* p_values                  // Output [n_clusters]
);

// =============================================================================
// Differential Abundance
// =============================================================================

scl_error_t scl_comp_differential_abundance(
    const scl_index_t* cluster_labels,    // [n_cells]
    const scl_index_t* sample_ids,        // [n_cells]
    const scl_index_t* conditions,         // [n_cells]
    scl_size_t n_cells,
    scl_real_t* da_scores,                 // Output [n_clusters]
    scl_real_t* p_values                  // Output [n_clusters]
);

// =============================================================================
// Condition Response
// =============================================================================

scl_error_t scl_comp_condition_response(
    scl_sparse_t expression,
    const scl_index_t* conditions,         // [n_cells]
    scl_size_t n_genes,
    scl_real_t* response_scores,           // Output [n_genes]
    scl_real_t* p_values                   // Output [n_genes]
);

// =============================================================================
// Effect Size
// =============================================================================

scl_error_t scl_comp_effect_size(
    const scl_real_t* group1,              // [n1]
    scl_size_t n1,
    const scl_real_t* group2,              // [n2]
    scl_size_t n2,
    scl_real_t* effect_size                // Output: Cohen's d
);

scl_error_t scl_comp_glass_delta(
    const scl_real_t* control,             // [n_control]
    scl_size_t n_control,
    const scl_real_t* treatment,           // [n_treatment]
    scl_size_t n_treatment,
    scl_real_t* delta                      // Output: Glass's delta
);

scl_error_t scl_comp_hedges_g(
    const scl_real_t* group1,              // [n1]
    scl_size_t n1,
    const scl_real_t* group2,              // [n2]
    scl_size_t n2,
    scl_real_t* hedges_g                    // Output: Hedges' g
);

#ifdef __cplusplus
}
#endif
