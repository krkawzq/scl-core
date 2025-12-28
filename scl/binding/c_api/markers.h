#pragma once

// =============================================================================
// FILE: scl/binding/c_api/markers.h
// BRIEF: C API for marker gene selection and specificity scoring
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Group Mean Expression
// =============================================================================

scl_error_t scl_markers_group_mean_expression(
    scl_sparse_t expression,
    const scl_index_t* group_labels,
    scl_size_t n_cells,
    scl_index_t n_groups,
    scl_index_t n_genes,
    scl_real_t* mean_expr              // Output [n_genes * n_groups]
);

// =============================================================================
// Percent Expressed
// =============================================================================

scl_error_t scl_markers_percent_expressed(
    scl_sparse_t expression,
    const scl_index_t* group_labels,
    scl_size_t n_cells,
    scl_index_t n_groups,
    scl_index_t n_genes,
    scl_real_t* pct_expr,              // Output [n_genes * n_groups]
    scl_real_t threshold               // Expression threshold (default 0)
);

// =============================================================================
// Find Marker Genes
// =============================================================================

scl_error_t scl_markers_find_markers(
    scl_sparse_t expression,
    const scl_index_t* group_labels,
    scl_size_t n_cells,
    scl_index_t n_groups,
    scl_index_t n_genes,
    scl_index_t target_group,
    scl_index_t* marker_genes,          // Output [n_markers]
    scl_real_t* marker_scores,          // Output [n_markers]
    scl_size_t* n_markers,              // Output: number of markers found
    scl_real_t min_fold_change,         // Minimum fold change
    scl_real_t min_pct,                 // Minimum percent expressed
    scl_index_t n_top                   // Number of top markers to return
);

#ifdef __cplusplus
}
#endif
