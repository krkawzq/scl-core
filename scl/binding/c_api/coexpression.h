#pragma once

// =============================================================================
// FILE: scl/binding/c_api/coexpression/coexpression.h
// BRIEF: C API for co-expression module detection (WGCNA-style)
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Correlation Types
// =============================================================================

typedef enum {
    SCL_CORR_PEARSON = 0,
    SCL_CORR_SPEARMAN = 1,
    SCL_CORR_BICOR = 2
} scl_correlation_type_t;

typedef enum {
    SCL_ADJ_UNSIGNED = 0,
    SCL_ADJ_SIGNED = 1,
    SCL_ADJ_SIGNED_HYBRID = 2
} scl_adjacency_type_t;

// =============================================================================
// Correlation Matrix
// =============================================================================

scl_error_t scl_correlation_matrix(
    scl_sparse_t expression,           // Expression matrix (cells x genes, CSR)
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* corr_matrix,          // Output: correlation matrix [n_genes * n_genes]
    scl_correlation_type_t corr_type
);

// =============================================================================
// WGCNA Adjacency Matrix
// =============================================================================

scl_error_t scl_wgcna_adjacency(
    scl_sparse_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t power,                  // Soft threshold power
    scl_real_t* adjacency,              // Output: adjacency matrix [n_genes * n_genes]
    scl_correlation_type_t corr_type,
    scl_adjacency_type_t adj_type
);

// =============================================================================
// Topological Overlap Matrix (TOM)
// =============================================================================

scl_error_t scl_topological_overlap_matrix(
    const scl_real_t* adjacency,      // Adjacency matrix [n_genes * n_genes]
    scl_index_t n_genes,
    scl_real_t* tom                    // Output: TOM [n_genes * n_genes]
);

// =============================================================================
// TOM Dissimilarity
// =============================================================================

scl_error_t scl_tom_dissimilarity(
    const scl_real_t* tom,
    scl_index_t n_genes,
    scl_real_t* dissim                 // Output: dissimilarity [n_genes * n_genes]
);

// =============================================================================
// Module Detection
// =============================================================================

scl_error_t scl_detect_modules(
    const scl_real_t* dissim,          // Dissimilarity matrix [n_genes * n_genes]
    scl_index_t n_genes,
    scl_index_t* module_labels,        // Output: module assignment [n_genes]
    scl_index_t* n_modules,            // Output: number of modules
    scl_index_t min_module_size,        // Minimum module size (default 30)
    scl_real_t merge_cut_height        // Merge cut height (default 0.25)
);

// =============================================================================
// Module Eigengene
// =============================================================================

scl_error_t scl_module_eigengene(
    scl_sparse_t expression,
    const scl_index_t* module_labels,
    scl_index_t module_id,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* eigengene             // Output: eigengene [n_cells]
);

scl_error_t scl_all_module_eigengenes(
    scl_sparse_t expression,
    const scl_index_t* module_labels,
    scl_index_t n_modules,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* eigengenes            // Output: [n_cells * n_modules]
);

// =============================================================================
// Module-Trait Correlation
// =============================================================================

scl_error_t scl_module_trait_correlation(
    const scl_real_t* eigengenes,      // [n_samples * n_modules]
    const scl_real_t* traits,          // [n_samples * n_traits]
    scl_index_t n_samples,
    scl_index_t n_modules,
    scl_index_t n_traits,
    scl_real_t* correlations,          // Output: [n_modules * n_traits]
    scl_real_t* p_values               // Output: [n_modules * n_traits] (optional, can be NULL)
);

// =============================================================================
// Hub Gene Identification
// =============================================================================

scl_error_t scl_identify_hub_genes(
    const scl_real_t* adjacency,
    const scl_index_t* module_labels,
    scl_index_t module_id,
    scl_index_t n_genes,
    scl_index_t* hub_genes,            // Output: hub gene indices [max_hubs]
    scl_real_t* hub_scores,             // Output: hub scores [max_hubs]
    scl_index_t* n_hubs,                // Output: number of hub genes
    scl_index_t max_hubs
);

// =============================================================================
// Gene Module Membership
// =============================================================================

scl_error_t scl_gene_module_membership(
    scl_sparse_t expression,
    const scl_real_t* eigengenes,      // [n_cells * n_modules]
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_modules,
    scl_real_t* kme_matrix             // Output: kME matrix [n_genes * n_modules]
);

// =============================================================================
// Pick Soft Threshold
// =============================================================================

scl_error_t scl_pick_soft_threshold(
    scl_sparse_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    const scl_real_t* powers_to_test,   // Powers to test [n_powers]
    scl_index_t n_powers,
    scl_real_t* scale_free_fits,        // Output: R^2 values [n_powers]
    scl_real_t* mean_connectivity,      // Output: mean connectivity [n_powers]
    scl_real_t* best_power,              // Output: recommended power
    scl_correlation_type_t corr_type
);

// =============================================================================
// Blockwise Modules
// =============================================================================

scl_error_t scl_blockwise_modules(
    scl_sparse_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t block_size,             // Block size for processing
    scl_real_t power,
    scl_index_t min_module_size,
    scl_index_t* module_labels,         // Output: module assignment [n_genes]
    scl_index_t* n_modules,              // Output: number of modules
    scl_correlation_type_t corr_type
);

#ifdef __cplusplus
}
#endif
