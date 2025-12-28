#pragma once

// =============================================================================
// FILE: scl/binding/c_api/scoring/scoring.h
// BRIEF: C API for gene set scoring operations
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Scoring Methods
// =============================================================================

typedef enum {
    SCL_SCORING_MEAN = 0,
    SCL_SCORING_RANK_BASED = 1,
    SCL_SCORING_WEIGHTED = 2,
    SCL_SCORING_SEURAT_MODULE = 3,
    SCL_SCORING_ZSCORE = 4
} scl_scoring_method_t;

// =============================================================================
// Gene Set Scoring
// =============================================================================

scl_error_t scl_scoring_gene_set_score(
    scl_sparse_t expression,
    const scl_index_t* gene_set,
    scl_size_t n_genes_in_set,
    scl_real_t* scores,
    scl_size_t n_cells,
    scl_index_t n_genes,
    scl_scoring_method_t method,
    scl_real_t quantile
);

scl_error_t scl_scoring_mean_score(
    scl_sparse_t expression,
    const scl_index_t* gene_set,
    scl_size_t n_genes_in_set,
    scl_real_t* scores,
    scl_size_t n_cells,
    scl_index_t n_genes
);

scl_error_t scl_scoring_auc_score(
    scl_sparse_t expression,
    const scl_index_t* gene_set,
    scl_size_t n_genes_in_set,
    scl_real_t* scores,
    scl_size_t n_cells,
    scl_index_t n_genes,
    scl_real_t quantile
);

scl_error_t scl_scoring_module_score(
    scl_sparse_t expression,
    const scl_index_t* gene_set,
    scl_size_t n_genes_in_set,
    scl_real_t* scores,
    scl_size_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_control_per_gene,
    scl_index_t n_bins
);

scl_error_t scl_scoring_differential_score(
    scl_sparse_t expression,
    const scl_index_t* positive_genes,
    scl_size_t n_positive,
    const scl_index_t* negative_genes,
    scl_size_t n_negative,
    scl_real_t* scores,
    scl_size_t n_cells,
    scl_index_t n_genes
);

#ifdef __cplusplus
}
#endif
