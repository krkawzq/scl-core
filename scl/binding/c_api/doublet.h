#pragma once

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// FILE: scl/binding/c_api/doublet.h
// BRIEF: C API for doublet detection operations
// =============================================================================

// Simulate doublets by averaging random cell pairs
scl_error_t scl_doublet_simulate_doublets(
    scl_sparse_matrix_t X,           // Input matrix (CSR, cells x genes)
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_doublets,
    scl_real_t* doublet_profiles,    // Output [n_doublets * n_genes], row-major
    uint64_t seed
);

// Compute k-NN doublet scores
scl_error_t scl_doublet_compute_knn_scores(
    scl_sparse_matrix_t X,           // Input matrix (CSR, cells x genes)
    scl_index_t n_cells,
    scl_index_t n_genes,
    const scl_real_t* doublet_profiles,  // Simulated doublets [n_doublets * n_genes]
    scl_index_t n_doublets,
    scl_index_t k_neighbors,
    scl_real_t* doublet_scores       // Output [n_cells]
);

// Compute k-NN doublet scores on PCA-reduced data
scl_error_t scl_doublet_compute_knn_scores_pca(
    const scl_real_t* cell_embeddings,   // [n_cells * n_dims], row-major
    scl_index_t n_cells,
    scl_index_t n_dims,
    const scl_real_t* doublet_embeddings, // [n_doublets * n_dims], row-major
    scl_index_t n_doublets,
    scl_index_t k_neighbors,
    scl_real_t* doublet_scores       // Output [n_cells]
);

// Scrublet-style doublet detection
scl_error_t scl_doublet_scrublet_scores(
    scl_sparse_matrix_t X,           // Input matrix (CSR, cells x genes)
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* scores,              // Output [n_cells]
    scl_index_t n_simulated,         // 0 for auto (2x n_cells)
    scl_index_t k_neighbors,
    uint64_t seed
);

// DoubletFinder-style pANN score
scl_error_t scl_doublet_doubletfinder_pann(
    const scl_real_t* cell_embeddings,
    scl_index_t n_cells,
    scl_index_t n_dims,
    const scl_real_t* doublet_embeddings,
    scl_index_t n_doublets,
    scl_real_t pK,                   // Proportion of k to use
    scl_real_t* pann_scores          // Output [n_cells]
);

// Estimate threshold from score distribution
scl_real_t scl_doublet_estimate_threshold(
    const scl_real_t* scores,
    scl_size_t n_scores,
    scl_real_t expected_doublet_rate
);

// Call doublets based on score threshold
scl_index_t scl_doublet_call_doublets(
    const scl_real_t* scores,
    scl_size_t n_scores,
    scl_real_t threshold,
    int* is_doublet                  // Output [n_scores], 1 for doublet, 0 for singlet
);

// Detect bimodal threshold
scl_real_t scl_doublet_detect_bimodal_threshold(
    const scl_real_t* scores,
    scl_size_t n_scores,
    scl_index_t n_bins
);

// Expected number of doublets
scl_index_t scl_doublet_expected_doublets(
    scl_index_t n_cells,
    scl_real_t doublet_rate
);

// Estimate doublet rate from loading
scl_real_t scl_doublet_estimate_doublet_rate(
    scl_index_t n_cells_loaded,
    scl_real_t cells_per_droplet_mean
);

// Classify doublet types
scl_error_t scl_doublet_classify_doublet_types(
    const scl_index_t* cluster_labels,
    const int* is_doublet,
    scl_size_t n_cells,
    scl_index_t n_clusters,
    scl_index_t* doublet_type        // Output [n_cells], 0=singlet, 1=heterotypic, 2=homotypic
);

// Classify doublet types using k-NN graph
scl_error_t scl_doublet_classify_doublet_types_knn(
    scl_sparse_matrix_t knn_graph,   // k-NN graph (CSR)
    const scl_index_t* cluster_labels,
    const int* is_doublet,
    scl_size_t n_cells,
    scl_index_t n_clusters,
    scl_index_t* doublet_type        // Output [n_cells]
);

// Density-based doublet score
scl_error_t scl_doublet_density_doublet_score(
    scl_sparse_matrix_t knn_graph,   // k-NN graph (CSR)
    scl_real_t* density_scores,      // Output [n_cells]
    scl_size_t n_cells
);

// Variance-based doublet score
scl_error_t scl_doublet_variance_doublet_score(
    scl_sparse_matrix_t X,           // Input matrix (CSR, cells x genes)
    scl_index_t n_cells,
    scl_index_t n_genes,
    const scl_real_t* gene_means,    // Gene means [n_genes]
    scl_real_t* variance_scores      // Output [n_cells]
);

// Combined doublet score
scl_error_t scl_doublet_combined_doublet_score(
    const scl_real_t* knn_scores,
    const scl_real_t* density_scores,
    const scl_real_t* variance_scores,
    scl_size_t n_cells,
    scl_real_t knn_weight,
    scl_real_t density_weight,
    scl_real_t variance_weight,
    scl_real_t* combined_scores      // Output [n_cells]
);

// Full doublet detection pipeline
scl_index_t scl_doublet_detect_doublets(
    scl_sparse_matrix_t X,           // Input matrix (CSR, cells x genes)
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* scores,               // Output [n_cells]
    int* is_doublet,                  // Output [n_cells]
    scl_real_t expected_rate,
    scl_index_t k_neighbors,
    uint64_t seed
);

// Get singlet indices
scl_index_t scl_doublet_get_singlet_indices(
    const int* is_doublet,
    scl_size_t n_cells,
    scl_index_t* singlet_indices      // Output [n_singlets]
);

// Doublet score statistics
scl_error_t scl_doublet_doublet_score_stats(
    const scl_real_t* scores,
    scl_size_t n_scores,
    scl_real_t* mean,
    scl_real_t* std_dev,
    scl_real_t* median
);

// Multiplet rate (10x Genomics reference)
scl_real_t scl_doublet_multiplet_rate_10x(
    scl_index_t n_cells_recovered
);

// Cluster doublet enrichment
scl_error_t scl_doublet_cluster_doublet_enrichment(
    const scl_real_t* doublet_scores,
    const scl_index_t* cluster_labels,
    scl_size_t n_cells,
    scl_index_t n_clusters,
    scl_real_t* cluster_mean_scores,  // Output [n_clusters]
    scl_real_t* cluster_doublet_fraction  // Output [n_clusters]
);

#ifdef __cplusplus
}
#endif
