#pragma once

// =============================================================================
// FILE: scl/binding/c_api/outlier/outlier.h
// BRIEF: C API for outlier and anomaly detection
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Isolation Score
// =============================================================================

scl_error_t scl_outlier_isolation_score(
    scl_sparse_t data,
    scl_real_t* scores                    // Output [n_cells]
);

// =============================================================================
// Local Outlier Factor (LOF)
// =============================================================================

scl_error_t scl_outlier_local_outlier_factor(
    scl_sparse_t data,
    scl_sparse_t neighbors,               // Neighbor graph
    scl_sparse_t distances,                // Distance matrix
    scl_real_t* lof_scores                 // Output [n_cells]
);

// =============================================================================
// Ambient RNA Detection
// =============================================================================

scl_error_t scl_outlier_ambient_detection(
    scl_sparse_t expression,
    scl_real_t* ambient_scores             // Output [n_cells]
);

// =============================================================================
// Empty Droplet Detection
// =============================================================================

scl_error_t scl_outlier_empty_drops(
    scl_sparse_t raw_counts,
    unsigned char* is_empty,              // Output [n_cells]: 1 if empty, 0 otherwise
    scl_real_t fdr_threshold
);

// =============================================================================
// Outlier Gene Detection
// =============================================================================

scl_error_t scl_outlier_outlier_genes(
    scl_sparse_t expression,
    scl_index_t* outlier_gene_indices,    // Output [max_outliers]
    scl_size_t max_outliers,
    scl_size_t* n_outliers,               // Output: actual number of outliers
    scl_real_t threshold
);

// =============================================================================
// Doublet Score
// =============================================================================

scl_error_t scl_outlier_doublet_score(
    scl_sparse_t expression,
    scl_sparse_t neighbors,
    scl_real_t* scores                    // Output [n_cells]
);

// =============================================================================
// Mitochondrial Outliers
// =============================================================================

scl_error_t scl_outlier_mitochondrial_outliers(
    scl_sparse_t expression,
    const scl_index_t* mito_genes,        // [n_mito_genes]
    scl_size_t n_mito_genes,
    scl_real_t* mito_fraction,            // Output [n_cells]
    unsigned char* is_outlier,             // Output [n_cells]
    scl_real_t threshold
);

// =============================================================================
// QC Filter
// =============================================================================

scl_error_t scl_outlier_qc_filter(
    scl_sparse_t expression,
    scl_real_t min_genes,
    scl_real_t max_genes,
    scl_real_t min_counts,
    scl_real_t max_counts,
    scl_real_t max_mito_fraction,
    const scl_index_t* mito_genes,        // [n_mito_genes]
    scl_size_t n_mito_genes,
    unsigned char* pass_qc                 // Output [n_cells]: 1 if pass, 0 otherwise
);

#ifdef __cplusplus
}
#endif
