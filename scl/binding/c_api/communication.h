#pragma once

// =============================================================================
// FILE: scl/binding/c_api/communication/communication.h
// BRIEF: C API for cell-cell communication analysis
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Score Method Enumeration
// =============================================================================

typedef enum {
    SCL_COMM_SCORE_MEAN_PRODUCT = 0,
    SCL_COMM_SCORE_GEOMETRIC_MEAN = 1,
    SCL_COMM_SCORE_MIN_MEAN = 2,
    SCL_COMM_SCORE_PRODUCT = 3,
    SCL_COMM_SCORE_NATMI = 4
} scl_comm_score_method_t;

// =============================================================================
// L-R Score Matrix
// =============================================================================

scl_error_t scl_comm_lr_score_matrix(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    scl_index_t ligand_gene,
    scl_index_t receptor_gene,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* score_matrix,
    scl_comm_score_method_t method
);

// =============================================================================
// Batch L-R Scores
// =============================================================================

scl_error_t scl_comm_lr_score_batch(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t* ligand_genes,
    const scl_index_t* receptor_genes,
    scl_index_t n_pairs,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* scores,
    scl_comm_score_method_t method
);

// =============================================================================
// Permutation Test
// =============================================================================

scl_error_t scl_comm_lr_permutation_test(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    scl_index_t ligand_gene,
    scl_index_t receptor_gene,
    scl_index_t sender_type,
    scl_index_t receiver_type,
    scl_index_t n_cells,
    scl_index_t n_permutations,
    scl_real_t* observed_score,
    scl_real_t* p_value,
    scl_comm_score_method_t method,
    uint64_t seed
);

// =============================================================================
// Communication Probability
// =============================================================================

scl_error_t scl_comm_probability(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t* ligand_genes,
    const scl_index_t* receptor_genes,
    scl_index_t n_pairs,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* p_values,
    scl_real_t* scores,
    scl_index_t n_permutations,
    scl_comm_score_method_t method,
    uint64_t seed
);

// =============================================================================
// Filter Significant Interactions
// =============================================================================

scl_error_t scl_comm_filter_significant(
    const scl_real_t* p_values,
    scl_index_t n_pairs,
    scl_index_t n_types,
    scl_real_t p_threshold,
    scl_index_t* pair_indices,
    scl_index_t* sender_types,
    scl_index_t* receiver_types,
    scl_real_t* filtered_pvalues,
    scl_index_t max_results,
    scl_index_t* n_results
);

// =============================================================================
// Aggregate to Network
// =============================================================================

scl_error_t scl_comm_aggregate_to_network(
    const scl_real_t* scores,
    const scl_real_t* p_values,
    scl_index_t n_pairs,
    scl_index_t n_types,
    scl_real_t p_threshold,
    scl_real_t* network_weights,
    scl_index_t* network_counts
);

// =============================================================================
// Sender/Receiver Scores
// =============================================================================

scl_error_t scl_comm_sender_score(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t* ligand_genes,
    scl_index_t n_ligands,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* scores
);

scl_error_t scl_comm_receiver_score(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t* receptor_genes,
    scl_index_t n_receptors,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* scores
);

// =============================================================================
// Network Centrality
// =============================================================================

scl_error_t scl_comm_network_centrality(
    const scl_real_t* network_weights,
    scl_index_t n_types,
    scl_real_t* in_degree,
    scl_real_t* out_degree,
    scl_real_t* betweenness
);

// =============================================================================
// Spatial Communication Score
// =============================================================================

scl_error_t scl_comm_spatial_score(
    scl_sparse_t expression,
    scl_sparse_t spatial_graph,
    scl_index_t ligand_gene,
    scl_index_t receptor_gene,
    scl_index_t n_cells,
    scl_real_t* cell_scores
);

// =============================================================================
// Expression Specificity
// =============================================================================

scl_error_t scl_comm_expression_specificity(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    scl_index_t gene,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* specificity
);

// =============================================================================
// NATMI Edge Weight
// =============================================================================

scl_error_t scl_comm_natmi_edge_weight(
    scl_sparse_t expression,
    const scl_index_t* cell_type_labels,
    scl_index_t ligand_gene,
    scl_index_t receptor_gene,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* edge_weights
);

#ifdef __cplusplus
}
#endif
