#pragma once

// =============================================================================
// FILE: scl/binding/c_api/grn.h
// BRIEF: C API for gene regulatory network inference
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "scl/binding/c_api/core_types.h"

// GRN method types
typedef enum {
    SCL_GRN_CORRELATION = 0,
    SCL_GRN_PARTIAL_CORRELATION = 1,
    SCL_GRN_MUTUAL_INFORMATION = 2,
    SCL_GRN_GENIE3 = 3,
    SCL_GRN_COMBINED = 4
} scl_grn_method_t;

// Correlation network
scl_error_t scl_grn_correlation_network(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t threshold,
    scl_real_t* correlation_matrix,
    int use_spearman
);

// Sparse correlation network
scl_error_t scl_grn_correlation_network_sparse(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t threshold,
    scl_index_t* edge_row,
    scl_index_t* edge_col,
    scl_real_t* edge_weight,
    scl_index_t max_edges,
    scl_index_t* n_edges_out,
    int use_spearman
);

// Partial correlation network
scl_error_t scl_grn_partial_correlation_network(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t threshold,
    scl_real_t* partial_corr_matrix,
    scl_real_t regularization
);

// Mutual information network
scl_error_t scl_grn_mutual_information_network(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t threshold,
    scl_real_t* mi_matrix,
    scl_index_t n_bins
);

// GENIE3 importance scores
scl_error_t scl_grn_genie3_importance(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_tfs,
    const scl_index_t* tf_indices,
    scl_real_t* importance_matrix,
    scl_index_t n_trees,
    scl_index_t subsample
);

// Regulon activity
scl_error_t scl_grn_regulon_activity(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    const scl_real_t* grn_matrix,
    scl_index_t n_tfs,
    scl_real_t threshold,
    scl_real_t* regulon_activity
);

// Infer GRN
scl_error_t scl_grn_infer_grn(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_grn_method_t method,
    scl_real_t threshold,
    scl_real_t* grn_matrix
);

// TF activity from regulons
scl_error_t scl_grn_tf_activity_from_regulons(
    scl_sparse_matrix_t expression,
    const scl_index_t* regulon_tf,
    const scl_index_t* regulon_offsets,
    const scl_index_t* regulon_targets,
    const scl_real_t* grn_matrix,
    scl_index_t n_regulons,
    scl_index_t n_tfs,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* tf_activity
);

#ifdef __cplusplus
}
#endif
