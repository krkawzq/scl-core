#pragma once

// =============================================================================
// FILE: scl/binding/c_api/communication.h
// BRIEF: C API for cell-cell communication analysis
// =============================================================================

#include "scl/binding/c_api/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Communication Score Methods
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

scl_error_t scl_communication_lr_score_matrix(
    scl_sparse_matrix_t expression,
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

scl_error_t scl_communication_lr_score_batch(
    scl_sparse_matrix_t expression,
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
// Communication Probability
// =============================================================================

scl_error_t scl_communication_probability(
    scl_sparse_matrix_t expression,
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
// Sender/Receiver Scores
// =============================================================================

scl_error_t scl_communication_sender_score(
    scl_sparse_matrix_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t* ligand_genes,
    scl_index_t n_ligands,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* scores
);

scl_error_t scl_communication_receiver_score(
    scl_sparse_matrix_t expression,
    const scl_index_t* cell_type_labels,
    const scl_index_t* receptor_genes,
    scl_index_t n_receptors,
    scl_index_t n_cells,
    scl_index_t n_types,
    scl_real_t* scores
);

// =============================================================================
// Filter Significant Interactions
// =============================================================================

scl_index_t scl_communication_filter_significant(
    const scl_real_t* p_values,
    scl_index_t n_pairs,
    scl_index_t n_types,
    scl_real_t p_threshold,
    scl_index_t* pair_indices,
    scl_index_t* sender_types,
    scl_index_t* receiver_types,
    scl_real_t* filtered_pvalues,
    scl_index_t max_results
);

#ifdef __cplusplus
}
#endif
