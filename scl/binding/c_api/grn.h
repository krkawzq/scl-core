#pragma once

// =============================================================================
// FILE: scl/binding/c_api/grn/grn.h
// BRIEF: C API for Gene Regulatory Network inference
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// GRN Method Types
// =============================================================================

typedef enum {
    SCL_GRN_METHOD_CORRELATION = 0,
    SCL_GRN_METHOD_PARTIAL_CORRELATION = 1,
    SCL_GRN_METHOD_MUTUAL_INFORMATION = 2,
    SCL_GRN_METHOD_GENIE3 = 3,
    SCL_GRN_METHOD_COMBINED = 4
} scl_grn_method_t;

// =============================================================================
// Correlation Network
// =============================================================================

scl_error_t scl_grn_correlation_network(
    scl_sparse_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t threshold,
    scl_real_t* correlation_matrix,
    int use_spearman
);

// =============================================================================
// Sparse Correlation Network
// =============================================================================

scl_error_t scl_grn_correlation_network_sparse(
    scl_sparse_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t threshold,
    scl_index_t* edge_row,
    scl_index_t* edge_col,
    scl_real_t* edge_weight,
    scl_index_t max_edges,
    scl_index_t* out_n_edges,
    int use_spearman
);

// =============================================================================
// TF-Target Score
// =============================================================================

scl_error_t scl_grn_tf_target_score(
    scl_sparse_t expression,
    const scl_index_t* tf_genes,
    scl_index_t n_tfs,
    const scl_index_t* target_genes,
    scl_index_t n_targets,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* scores
);

// =============================================================================
// GRN Inference
// =============================================================================

scl_error_t scl_grn_infer(
    scl_sparse_t expression,
    const scl_index_t* tf_genes,
    scl_index_t n_tfs,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* grn_matrix,
    scl_grn_method_t method,
    scl_real_t threshold
);

#ifdef __cplusplus
}
#endif
