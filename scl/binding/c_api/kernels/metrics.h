#pragma once

// =============================================================================
// FILE: scl/binding/c_api/kernels/metrics.h
// BRIEF: C API for quality metrics
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#include "../core.h"

// =============================================================================
// Silhouette Score
// =============================================================================

scl_error_t scl_metrics_silhouette_score(
    scl_sparse_matrix_t distances,    // Distance matrix (CSR)
    const scl_index_t* labels,         // Cluster labels [n_cells]
    scl_index_t n_cells,              // Number of cells
    scl_real_t* score                 // Output: silhouette score
);

scl_error_t scl_metrics_silhouette_samples(
    scl_sparse_matrix_t distances,    // Distance matrix (CSR)
    const scl_index_t* labels,         // Cluster labels [n_cells]
    scl_index_t n_cells,              // Number of cells
    scl_real_t* scores                // Output: per-sample scores [n_cells]
);

// =============================================================================
// Adjusted Rand Index (ARI)
// =============================================================================

scl_error_t scl_metrics_adjusted_rand_index(
    const scl_index_t* labels1,       // First labeling [n]
    const scl_index_t* labels2,       // Second labeling [n]
    scl_index_t n,                    // Number of samples
    scl_real_t* ari                   // Output: ARI score
);

// =============================================================================
// Normalized Mutual Information (NMI)
// =============================================================================

scl_error_t scl_metrics_normalized_mutual_information(
    const scl_index_t* labels1,       // First labeling [n]
    const scl_index_t* labels2,       // Second labeling [n]
    scl_index_t n,                    // Number of samples
    scl_real_t* nmi                   // Output: NMI score
);

// =============================================================================
// Graph Connectivity
// =============================================================================

scl_error_t scl_metrics_graph_connectivity(
    scl_sparse_matrix_t adjacency,    // Adjacency matrix (CSR)
    const scl_index_t* labels,         // Cluster labels [n_cells]
    scl_index_t n_cells,              // Number of cells
    scl_real_t* connectivity           // Output: connectivity score
);

// =============================================================================
// Batch Entropy
// =============================================================================

scl_error_t scl_metrics_batch_entropy(
    scl_sparse_matrix_t neighbors,     // Neighbor graph (CSR, Index type)
    const scl_index_t* batch_labels,   // Batch labels [n_cells]
    scl_index_t n_cells,              // Number of cells
    scl_real_t* entropy_scores        // Output: entropy scores [n_cells]
);

// =============================================================================
// Local Inverse Simpson's Index (LISI)
// =============================================================================

scl_error_t scl_metrics_lisi(
    scl_sparse_matrix_t neighbors,     // Neighbor graph (CSR, Index type)
    const scl_index_t* labels,         // Labels [n_cells]
    scl_index_t n_cells,              // Number of cells
    scl_real_t* lisi_scores            // Output: LISI scores [n_cells]
);

// =============================================================================
// Additional Metrics
// =============================================================================

scl_error_t scl_metrics_fowlkes_mallows_index(
    const scl_index_t* labels1,       // First labeling [n]
    const scl_index_t* labels2,       // Second labeling [n]
    scl_index_t n,                    // Number of samples
    scl_real_t* fmi                   // Output: FMI score
);

scl_error_t scl_metrics_v_measure(
    const scl_index_t* labels_true,   // True labels [n]
    const scl_index_t* labels_pred,   // Predicted labels [n]
    scl_index_t n,                    // Number of samples
    scl_real_t beta,                  // Beta parameter
    scl_real_t* v_measure             // Output: V-measure score
);

scl_error_t scl_metrics_purity_score(
    const scl_index_t* labels_true,   // True labels [n]
    const scl_index_t* labels_pred,   // Predicted labels [n]
    scl_index_t n,                    // Number of samples
    scl_real_t* purity                // Output: purity score
);

#ifdef __cplusplus
}
#endif
