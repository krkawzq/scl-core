#pragma once

// =============================================================================
// FILE: scl/binding/c_api/coexpression.h
// BRIEF: C API for co-expression module detection (WGCNA-style)
// =============================================================================

#include "scl/binding/c_api/types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Correlation Types
// =============================================================================

typedef enum {
    SCL_CORRELATION_PEARSON = 0,
    SCL_CORRELATION_SPEARMAN = 1,
    SCL_CORRELATION_BICOR = 2
} scl_correlation_type_t;

typedef enum {
    SCL_ADJACENCY_UNSIGNED = 0,
    SCL_ADJACENCY_SIGNED = 1,
    SCL_ADJACENCY_SIGNED_HYBRID = 2
} scl_adjacency_type_t;

// =============================================================================
// Correlation Matrix
// =============================================================================

scl_error_t scl_correlation_matrix(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* corr_matrix,
    scl_correlation_type_t corr_type
);

// =============================================================================
// WGCNA Adjacency Matrix
// =============================================================================

scl_error_t scl_wgcna_adjacency(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t power,
    scl_real_t* adjacency,
    scl_correlation_type_t corr_type,
    scl_adjacency_type_t adj_type
);

// =============================================================================
// Topological Overlap Matrix
// =============================================================================

scl_error_t scl_topological_overlap_matrix(
    const scl_real_t* adjacency,
    scl_index_t n_genes,
    scl_real_t* tom
);

scl_error_t scl_tom_dissimilarity(
    const scl_real_t* tom,
    scl_index_t n_genes,
    scl_real_t* dissim
);

// =============================================================================
// Module Detection
// =============================================================================

scl_error_t scl_detect_modules(
    const scl_real_t* dissim,
    scl_index_t n_genes,
    scl_index_t* module_labels,
    scl_index_t* n_modules,
    scl_index_t min_module_size,
    scl_real_t merge_cut_height
);

// =============================================================================
// Module Eigengene
// =============================================================================

scl_error_t scl_module_eigengene(
    scl_sparse_matrix_t expression,
    const scl_index_t* module_labels,
    scl_index_t module_id,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* eigengene
);

scl_error_t scl_all_module_eigengenes(
    scl_sparse_matrix_t expression,
    const scl_index_t* module_labels,
    scl_index_t n_modules,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* eigengenes
);

// =============================================================================
// Module-Trait Correlation
// =============================================================================

scl_error_t scl_module_trait_correlation(
    const scl_real_t* eigengenes,
    const scl_real_t* traits,
    scl_index_t n_samples,
    scl_index_t n_modules,
    scl_index_t n_traits,
    scl_real_t* correlations,
    scl_real_t* p_values
);

// =============================================================================
// Pick Soft Threshold
// =============================================================================

scl_error_t scl_pick_soft_threshold(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    const scl_real_t* powers_to_test,
    scl_index_t n_powers,
    scl_real_t* scale_free_fits,
    scl_real_t* mean_connectivity,
    scl_real_t* best_power,
    scl_correlation_type_t corr_type
);

// =============================================================================
// Blockwise Modules
// =============================================================================

scl_error_t scl_blockwise_modules(
    scl_sparse_matrix_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t block_size,
    scl_real_t power,
    scl_index_t min_module_size,
    scl_index_t* module_labels,
    scl_index_t* n_modules,
    scl_correlation_type_t corr_type
);

#ifdef __cplusplus
}
#endif
