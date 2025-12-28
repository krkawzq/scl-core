#pragma once

// =============================================================================
// FILE: scl/binding/c_api/metrics.h
// BRIEF: C API for quality metrics for clustering and integration evaluation
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Silhouette Score
// =============================================================================

scl_error_t scl_metrics_silhouette_score(
    scl_sparse_t distances,            // Distance matrix
    const scl_index_t* labels,
    scl_size_t n_cells,
    scl_real_t* score                  // Output
);

// =============================================================================
// Silhouette Score per Sample
// =============================================================================

scl_error_t scl_metrics_silhouette_samples(
    scl_sparse_t distances,
    const scl_index_t* labels,
    scl_size_t n_cells,
    scl_real_t* scores                 // Output [n_cells]
);

// =============================================================================
// Adjusted Rand Index (ARI)
// =============================================================================

scl_error_t scl_metrics_adjusted_rand_index(
    const scl_index_t* labels1,
    const scl_index_t* labels2,
    scl_size_t n_cells,
    scl_real_t* ari                    // Output
);

// =============================================================================
// Normalized Mutual Information (NMI)
// =============================================================================

scl_error_t scl_metrics_normalized_mutual_information(
    const scl_index_t* labels1,
    const scl_index_t* labels2,
    scl_size_t n_cells,
    scl_real_t* nmi                    // Output
);

// =============================================================================
// Homogeneity, Completeness, V-Measure
// =============================================================================

scl_error_t scl_metrics_homogeneity_completeness_vmeasure(
    const scl_index_t* labels_true,
    const scl_index_t* labels_pred,
    scl_size_t n_cells,
    scl_real_t* homogeneity,           // Output
    scl_real_t* completeness,          // Output
    scl_real_t* v_measure              // Output
);

#ifdef __cplusplus
}
#endif
