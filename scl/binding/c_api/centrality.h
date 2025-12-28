#pragma once

// =============================================================================
// FILE: scl/binding/c_api/centrality/centrality.h
// BRIEF: C API for graph centrality measures
// =============================================================================

#include "scl/binding/c_api/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Degree Centrality
// =============================================================================

scl_error_t scl_degree_centrality(
    scl_sparse_t adjacency,           // Adjacency matrix (CSR)
    scl_real_t* centrality,           // Output [n_nodes]
    int normalize                     // 1 = normalize, 0 = raw counts
);

scl_error_t scl_weighted_degree_centrality(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    int normalize
);

// =============================================================================
// PageRank
// =============================================================================

scl_error_t scl_pagerank(
    scl_sparse_t adjacency,
    scl_real_t* scores,               // Output [n_nodes]
    scl_real_t damping,                // Damping factor (default 0.85)
    scl_index_t max_iter,              // Max iterations (default 100)
    scl_real_t tolerance                // Convergence tolerance (default 1e-6)
);

scl_error_t scl_personalized_pagerank(
    scl_sparse_t adjacency,
    const scl_index_t* seed_nodes,    // Seed node indices [n_seeds]
    scl_size_t n_seeds,
    scl_real_t* scores,
    scl_real_t damping,
    scl_index_t max_iter,
    scl_real_t tolerance
);

// =============================================================================
// HITS Algorithm
// =============================================================================

scl_error_t scl_hits(
    scl_sparse_t adjacency,
    scl_real_t* hub_scores,            // Output [n_nodes]
    scl_real_t* authority_scores,       // Output [n_nodes]
    scl_index_t max_iter,
    scl_real_t tolerance
);

// =============================================================================
// Eigenvector Centrality
// =============================================================================

scl_error_t scl_eigenvector_centrality(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    scl_index_t max_iter,
    scl_real_t tolerance
);

// =============================================================================
// Katz Centrality
// =============================================================================

scl_error_t scl_katz_centrality(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    scl_real_t alpha,                  // Attenuation factor (default 0.1)
    scl_real_t beta,                   // Constant term (default 1.0)
    scl_index_t max_iter,
    scl_real_t tolerance
);

// =============================================================================
// Closeness Centrality
// =============================================================================

scl_error_t scl_closeness_centrality(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    int normalize
);

// =============================================================================
// Betweenness Centrality
// =============================================================================

scl_error_t scl_betweenness_centrality(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    int normalize
);

scl_error_t scl_betweenness_centrality_sampled(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    scl_index_t n_samples,             // Number of source nodes to sample
    int normalize,
    uint64_t seed                       // Random seed
);

// =============================================================================
// Harmonic Centrality
// =============================================================================

scl_error_t scl_harmonic_centrality(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    int normalize
);

#ifdef __cplusplus
}
#endif
