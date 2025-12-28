#pragma once

// =============================================================================
// FILE: scl/binding/c_api/centrality.h
// BRIEF: C API for graph centrality measures
// =============================================================================

#include "scl/binding/c_api/types.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Degree Centrality
// =============================================================================

scl_error_t scl_degree_centrality(
    scl_sparse_matrix_t adjacency,
    scl_real_t* centrality,
    int normalize
);

scl_error_t scl_weighted_degree_centrality(
    scl_sparse_matrix_t adjacency,
    scl_real_t* centrality,
    int normalize
);

// =============================================================================
// PageRank
// =============================================================================

scl_error_t scl_pagerank(
    scl_sparse_matrix_t adjacency,
    scl_real_t* scores,
    scl_real_t damping,
    scl_index_t max_iter,
    scl_real_t tol
);

scl_error_t scl_personalized_pagerank(
    scl_sparse_matrix_t adjacency,
    const scl_index_t* seed_nodes,
    scl_size_t n_seeds,
    scl_real_t* scores,
    scl_real_t damping,
    scl_index_t max_iter,
    scl_real_t tol
);

// =============================================================================
// HITS Algorithm
// =============================================================================

scl_error_t scl_hits(
    scl_sparse_matrix_t adjacency,
    scl_real_t* hub_scores,
    scl_real_t* authority_scores,
    scl_index_t max_iter,
    scl_real_t tol
);

// =============================================================================
// Eigenvector Centrality
// =============================================================================

scl_error_t scl_eigenvector_centrality(
    scl_sparse_matrix_t adjacency,
    scl_real_t* centrality,
    scl_index_t max_iter,
    scl_real_t tol
);

// =============================================================================
// Katz Centrality
// =============================================================================

scl_error_t scl_katz_centrality(
    scl_sparse_matrix_t adjacency,
    scl_real_t* centrality,
    scl_real_t alpha,
    scl_real_t beta,
    scl_index_t max_iter,
    scl_real_t tol
);

// =============================================================================
// Closeness Centrality
// =============================================================================

scl_error_t scl_closeness_centrality(
    scl_sparse_matrix_t adjacency,
    scl_real_t* centrality,
    int normalize
);

// =============================================================================
// Betweenness Centrality
// =============================================================================

scl_error_t scl_betweenness_centrality(
    scl_sparse_matrix_t adjacency,
    scl_real_t* centrality,
    int normalize
);

scl_error_t scl_betweenness_centrality_sampled(
    scl_sparse_matrix_t adjacency,
    scl_real_t* centrality,
    scl_index_t n_samples,
    int normalize,
    uint64_t seed
);

// =============================================================================
// Harmonic Centrality
// =============================================================================

scl_error_t scl_harmonic_centrality(
    scl_sparse_matrix_t adjacency,
    scl_real_t* centrality,
    int normalize
);

#ifdef __cplusplus
}
#endif
