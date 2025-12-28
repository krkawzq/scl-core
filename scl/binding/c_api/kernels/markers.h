#pragma once

// =============================================================================
// FILE: scl/binding/c_api/kernels/markers.h
// BRIEF: C API for marker gene selection
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "../core.h"

// =============================================================================
// Ranking Methods
// =============================================================================

typedef enum {
    SCL_MARKERS_RANKING_FOLD_CHANGE = 0,
    SCL_MARKERS_RANKING_EFFECT_SIZE = 1,
    SCL_MARKERS_RANKING_P_VALUE = 2,
    SCL_MARKERS_RANKING_COMBINED = 3
} scl_markers_ranking_method_t;

// =============================================================================
// Group Mean Expression
// =============================================================================

scl_error_t scl_markers_group_mean_expression(
    scl_sparse_matrix_t X,              // CSR or CSC sparse matrix
    const scl_index_t* group_labels,    // Group labels [n_cells]
    scl_index_t n_cells,                // Number of cells
    scl_index_t n_groups,               // Number of groups
    scl_index_t n_genes,                // Number of genes
    scl_real_t* mean_expr              // Output: mean expression [n_genes * n_groups]
);

// =============================================================================
// Percent Expressed
// =============================================================================

scl_error_t scl_markers_percent_expressed(
    scl_sparse_matrix_t X,              // CSR or CSC sparse matrix
    const scl_index_t* group_labels,    // Group labels [n_cells]
    scl_index_t n_cells,                // Number of cells
    scl_index_t n_groups,               // Number of groups
    scl_index_t n_genes,                // Number of genes
    scl_real_t* pct_expr,               // Output: percent expressed [n_genes * n_groups]
    scl_real_t threshold                // Expression threshold (default: 0.0)
);

// =============================================================================
// Log Fold Change
// =============================================================================

scl_error_t scl_markers_log_fold_change(
    scl_sparse_matrix_t X,              // CSR or CSC sparse matrix
    const scl_index_t* group_labels,    // Group labels [n_cells]
    scl_index_t n_cells,                // Number of cells
    scl_index_t n_groups,               // Number of groups
    scl_index_t target_group,          // Target group index
    scl_index_t n_genes,                // Number of genes
    scl_real_t* log_fc,                 // Output: log2 fold change [n_genes]
    scl_real_t pseudo_count             // Pseudo count (default: 1.0)
);

// =============================================================================
// One-vs-Rest Statistics
// =============================================================================

scl_error_t scl_markers_one_vs_rest_stats(
    scl_sparse_matrix_t X,              // CSR or CSC sparse matrix
    const scl_index_t* group_labels,    // Group labels [n_cells]
    scl_index_t n_cells,                // Number of cells
    scl_index_t n_groups,               // Number of groups
    scl_index_t target_group,          // Target group index
    scl_index_t n_genes,                // Number of genes
    scl_real_t* log_fc,                 // Output: log2 fold change [n_genes]
    scl_real_t* effect_size,            // Output: Cohen's d effect size [n_genes]
    scl_real_t* pct_in,                  // Output: percent in target [n_genes]
    scl_real_t* pct_out                 // Output: percent out of target [n_genes]
);

// =============================================================================
// Rank Genes Groups
// =============================================================================

scl_error_t scl_markers_rank_genes_groups(
    scl_sparse_matrix_t X,              // CSR or CSC sparse matrix
    const scl_index_t* group_labels,    // Group labels [n_cells]
    scl_index_t n_cells,                // Number of cells
    scl_index_t n_groups,               // Number of groups
    scl_index_t n_genes,                // Number of genes
    scl_markers_ranking_method_t method, // Ranking method
    scl_index_t* ranked_indices,        // Output: ranked gene indices [n_groups * n_genes]
    scl_real_t* ranked_scores           // Output: ranked scores [n_groups * n_genes]
);

// =============================================================================
// Tau Specificity
// =============================================================================

scl_error_t scl_markers_tau_specificity(
    const scl_real_t* group_means,      // Group means [n_genes * n_groups]
    scl_index_t n_genes,                // Number of genes
    scl_index_t n_groups,               // Number of groups
    scl_real_t* tau_scores              // Output: tau scores [n_genes]
);

// =============================================================================
// Gini Specificity
// =============================================================================

scl_error_t scl_markers_gini_specificity(
    const scl_real_t* group_means,      // Group means [n_genes * n_groups]
    scl_index_t n_genes,                // Number of genes
    scl_index_t n_groups,               // Number of groups
    scl_real_t* gini_scores             // Output: Gini scores [n_genes]
);

#ifdef __cplusplus
}
#endif
