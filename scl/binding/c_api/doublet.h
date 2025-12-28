#pragma once

// =============================================================================
// FILE: scl/binding/c_api/doublet/doublet.h
// BRIEF: C API for doublet detection operations
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// Doublet detection methods
typedef enum {
    SCL_DOUBLET_METHOD_SCRUBLET = 0,
    SCL_DOUBLET_METHOD_DOUBLETFINDER = 1,
    SCL_DOUBLET_METHOD_HYBRID = 2
} scl_doublet_method_t;

// Simulate doublets by averaging random cell pairs
scl_error_t scl_doublet_simulate_doublets(
    scl_sparse_t X,                   // Cell x gene matrix (CSR format required)
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_doublets,
    scl_real_t* doublet_profiles,     // Output [n_doublets x n_genes], row-major
    uint64_t seed
);

// Compute k-NN based doublet scores from simulated doublets
scl_error_t scl_doublet_compute_knn_scores(
    scl_sparse_t X,
    scl_index_t n_cells,
    scl_index_t n_genes,
    const scl_real_t* doublet_profiles,
    scl_index_t n_doublets,
    scl_index_t k_neighbors,
    scl_real_t* doublet_scores        // Output [n_cells]
);

// Compute k-NN based doublet scores from PCA embeddings
scl_error_t scl_doublet_compute_knn_scores_pca(
    const scl_real_t* cell_embeddings,    // [n_cells x n_dims], row-major
    scl_index_t n_cells,
    scl_index_t n_dims,
    const scl_real_t* doublet_embeddings, // [n_doublets x n_dims], row-major
    scl_index_t n_doublets,
    scl_index_t k_neighbors,
    scl_real_t* doublet_scores
);

// Scrublet-style doublet detection
scl_error_t scl_doublet_scrublet_scores(
    scl_sparse_t X,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* scores,               // Output [n_cells]
    scl_index_t n_simulated,          // 0 = auto (2x n_cells)
    scl_index_t k_neighbors,
    uint64_t seed
);

// DoubletFinder pANN score
scl_error_t scl_doublet_doubletfinder_pann(
    const scl_real_t* cell_embeddings,
    scl_index_t n_cells,
    scl_index_t n_dims,
    const scl_real_t* doublet_embeddings,
    scl_index_t n_doublets,
    scl_real_t pK,                    // Proportion of k to use
    scl_real_t* pann_scores           // Output [n_cells]
);

// Estimate threshold from score distribution
scl_error_t scl_doublet_estimate_threshold(
    const scl_real_t* scores,
    scl_size_t n_scores,
    scl_real_t expected_doublet_rate,
    scl_real_t* threshold_out
);

// Call doublets based on score threshold
scl_error_t scl_doublet_call_doublets(
    const scl_real_t* scores,
    scl_size_t n_scores,
    scl_real_t threshold,
    int* is_doublet,                  // Output boolean array [n_scores]
    scl_index_t* n_doublets_out
);

// Detect bimodal threshold from score distribution
scl_error_t scl_doublet_detect_bimodal_threshold(
    const scl_real_t* scores,
    scl_size_t n_scores,
    scl_index_t n_bins,
    scl_real_t* threshold_out
);

// Full doublet detection pipeline
scl_error_t scl_doublet_detect_doublets(
    scl_sparse_t X,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* scores,               // Output [n_cells]
    int* is_doublet,                  // Output [n_cells]
    scl_doublet_method_t method,
    scl_real_t expected_rate,
    scl_index_t k_neighbors,
    uint64_t seed,
    scl_index_t* n_doublets_out
);

// Get singlet indices from boolean array
scl_error_t scl_doublet_get_singlet_indices(
    const int* is_doublet,
    scl_size_t n_cells,
    scl_index_t* singlet_indices,     // Output indices
    scl_size_t max_indices,
    scl_index_t* n_singlets_out
);

// Compute doublet score statistics
scl_error_t scl_doublet_score_stats(
    const scl_real_t* scores,
    scl_size_t n_scores,
    scl_real_t* mean_out,
    scl_real_t* std_dev_out,
    scl_real_t* median_out
);

#ifdef __cplusplus
}
#endif
