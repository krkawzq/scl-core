#pragma once

// =============================================================================
// FILE: scl/binding/c_api/centrality.h
// BRIEF: C API for graph centrality measures
// =============================================================================
//
// CENTRALITY MEASURES:
//   - Degree: Number of connections (unweighted/weighted)
//   - PageRank: Google's web page ranking algorithm
//   - HITS: Hub and authority scores
//   - Eigenvector: Principal eigenvector of adjacency matrix
//   - Katz: Weighted paths with attenuation
//   - Closeness: Average distance to all other nodes
//   - Betweenness: Number of shortest paths through node
//   - Harmonic: Harmonic mean of distances
//
// OPTIMIZATIONS:
//   - SIMD-accelerated vector operations
//   - Parallel power iteration algorithms
//   - Atomic accumulation for thread safety
//   - Cache-optimized BFS for shortest paths
//   - Sampled betweenness for large graphs
//
// CONVERGENCE:
//   - Iterative algorithms (PageRank, HITS, etc.) converge when:
//     L1 distance between iterations < tolerance
//   - Default: tolerance=1e-6, max_iter=100
//
// THREAD SAFETY:
//   - All operations are thread-safe
//   - Automatic parallelization for large graphs
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Degree Centrality
// =============================================================================

/// @brief Compute degree centrality (number of neighbors)
/// @param[in] adjacency Adjacency matrix (non-null)
/// @param[out] centrality Centrality scores [n_nodes] (non-null)
/// @param[in] n_nodes Number of nodes
/// @param[in] normalize SCL_TRUE to normalize by (n-1)
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_centrality_degree(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    scl_size_t n_nodes,
    scl_bool_t normalize
);

/// @brief Compute weighted degree centrality (sum of edge weights)
/// @param[in] adjacency Weighted adjacency matrix (non-null)
/// @param[out] centrality Centrality scores [n_nodes] (non-null)
/// @param[in] n_nodes Number of nodes
/// @param[in] normalize SCL_TRUE to normalize by max weight
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_centrality_weighted_degree(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    scl_size_t n_nodes,
    scl_bool_t normalize
);

// =============================================================================
// PageRank
// =============================================================================

/// @brief Compute PageRank centrality
/// @param[in] adjacency Adjacency matrix (non-null)
/// @param[out] scores PageRank scores [n_nodes] (non-null)
/// @param[in] n_nodes Number of nodes
/// @param[in] damping Damping factor [0, 1] (typical: 0.85)
/// @param[in] max_iter Maximum iterations
/// @param[in] tolerance Convergence tolerance
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_centrality_pagerank(
    scl_sparse_t adjacency,
    scl_real_t* scores,
    scl_size_t n_nodes,
    scl_real_t damping,
    scl_index_t max_iter,
    scl_real_t tolerance
);

/// @brief Compute personalized PageRank from seed nodes
/// @param[in] adjacency Adjacency matrix (non-null)
/// @param[in] seed_nodes Seed node indices [n_seeds] (non-null)
/// @param[in] n_seeds Number of seed nodes
/// @param[out] scores PageRank scores [n_nodes] (non-null)
/// @param[in] n_nodes Number of nodes
/// @param[in] damping Damping factor
/// @param[in] max_iter Maximum iterations
/// @param[in] tolerance Convergence tolerance
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_centrality_personalized_pagerank(
    scl_sparse_t adjacency,
    const scl_index_t* seed_nodes,
    scl_size_t n_seeds,
    scl_real_t* scores,
    scl_size_t n_nodes,
    scl_real_t damping,
    scl_index_t max_iter,
    scl_real_t tolerance
);

// =============================================================================
// HITS Algorithm
// =============================================================================

/// @brief Compute HITS hub and authority scores
/// @param[in] adjacency Adjacency matrix (non-null)
/// @param[out] hub_scores Hub scores [n_nodes] (non-null)
/// @param[out] authority_scores Authority scores [n_nodes] (non-null)
/// @param[in] n_nodes Number of nodes
/// @param[in] max_iter Maximum iterations
/// @param[in] tolerance Convergence tolerance
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_centrality_hits(
    scl_sparse_t adjacency,
    scl_real_t* hub_scores,
    scl_real_t* authority_scores,
    scl_size_t n_nodes,
    scl_index_t max_iter,
    scl_real_t tolerance
);

// =============================================================================
// Eigenvector Centrality
// =============================================================================

/// @brief Compute eigenvector centrality (principal eigenvector)
/// @param[in] adjacency Adjacency matrix (non-null)
/// @param[out] centrality Centrality scores [n_nodes] (non-null)
/// @param[in] n_nodes Number of nodes
/// @param[in] max_iter Maximum iterations
/// @param[in] tolerance Convergence tolerance
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_centrality_eigenvector(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    scl_size_t n_nodes,
    scl_index_t max_iter,
    scl_real_t tolerance
);

// =============================================================================
// Katz Centrality
// =============================================================================

/// @brief Compute Katz centrality (weighted path count)
/// @param[in] adjacency Adjacency matrix (non-null)
/// @param[out] centrality Centrality scores [n_nodes] (non-null)
/// @param[in] n_nodes Number of nodes
/// @param[in] alpha Attenuation factor (typical: 0.1)
/// @param[in] beta Base score (typical: 1.0)
/// @param[in] max_iter Maximum iterations
/// @param[in] tolerance Convergence tolerance
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_centrality_katz(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    scl_size_t n_nodes,
    scl_real_t alpha,
    scl_real_t beta,
    scl_index_t max_iter,
    scl_real_t tolerance
);

// =============================================================================
// Closeness Centrality
// =============================================================================

/// @brief Compute closeness centrality (inverse average distance)
/// @param[in] adjacency Adjacency matrix (non-null)
/// @param[out] centrality Centrality scores [n_nodes] (non-null)
/// @param[in] n_nodes Number of nodes
/// @param[in] normalize SCL_TRUE to normalize
/// @return SCL_OK on success, error code otherwise
/// @note Requires BFS from each node: O(n * (n + m))
scl_error_t scl_centrality_closeness(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    scl_size_t n_nodes,
    scl_bool_t normalize
);

// =============================================================================
// Betweenness Centrality
// =============================================================================

/// @brief Compute betweenness centrality (Brandes algorithm)
/// @param[in] adjacency Adjacency matrix (non-null)
/// @param[out] centrality Centrality scores [n_nodes] (non-null)
/// @param[in] n_nodes Number of nodes
/// @param[in] normalize SCL_TRUE to normalize
/// @return SCL_OK on success, error code otherwise
/// @note Exact algorithm: O(n * (n + m))
scl_error_t scl_centrality_betweenness(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    scl_size_t n_nodes,
    scl_bool_t normalize
);

/// @brief Compute approximate betweenness via sampling
/// @param[in] adjacency Adjacency matrix (non-null)
/// @param[out] centrality Centrality scores [n_nodes] (non-null)
/// @param[in] n_nodes Number of nodes
/// @param[in] n_samples Number of source nodes to sample
/// @param[in] normalize SCL_TRUE to normalize
/// @param[in] seed Random seed
/// @return SCL_OK on success, error code otherwise
/// @note Faster approximation: O(n_samples * (n + m))
scl_error_t scl_centrality_betweenness_sampled(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    scl_size_t n_nodes,
    scl_index_t n_samples,
    scl_bool_t normalize,
    uint64_t seed
);

// =============================================================================
// Harmonic Centrality
// =============================================================================

/// @brief Compute harmonic centrality (sum of inverse distances)
/// @param[in] adjacency Adjacency matrix (non-null)
/// @param[out] centrality Centrality scores [n_nodes] (non-null)
/// @param[in] n_nodes Number of nodes
/// @param[in] normalize SCL_TRUE to normalize
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_centrality_harmonic(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    scl_size_t n_nodes,
    scl_bool_t normalize
);

// =============================================================================
// Current Flow Betweenness (Approximate)
// =============================================================================

/// @brief Approximate current flow betweenness via random walks
/// @param[in] adjacency Adjacency matrix (non-null)
/// @param[out] centrality Centrality scores [n_nodes] (non-null)
/// @param[in] n_nodes Number of nodes
/// @param[in] n_walks Number of random walks
/// @param[in] walk_length Length of each walk
/// @param[in] seed Random seed
/// @return SCL_OK on success, error code otherwise
scl_error_t scl_centrality_current_flow_approx(
    scl_sparse_t adjacency,
    scl_real_t* centrality,
    scl_size_t n_nodes,
    scl_index_t n_walks,
    scl_index_t walk_length,
    uint64_t seed
);

#ifdef __cplusplus
}
#endif
