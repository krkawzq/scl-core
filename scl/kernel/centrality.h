// =============================================================================
// FILE: scl/kernel/centrality.h
// BRIEF: API reference for high-performance graph centrality measures
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::centrality {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_DAMPING = Real(0.85);
    constexpr Index DEFAULT_MAX_ITER = 100;
    constexpr Real DEFAULT_TOLERANCE = Real(1e-6);
    constexpr Real MIN_SCORE = Real(1e-15);
    constexpr Size PARALLEL_THRESHOLD = 256;
    constexpr Size PREFETCH_DISTANCE = 4;
    constexpr Size SIMD_THRESHOLD = 16;
}

// =============================================================================
// Centrality Measures
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: degree_centrality
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute degree centrality (sum of edge weights).
 *
 * PARAMETERS:
 *     adjacency  [in]  Adjacency matrix (CSR or CSC)
 *     centrality [out] Degree centrality scores [n_nodes]
 *     normalize  [in]  If true, normalize by maximum degree
 *
 * PRECONDITIONS:
 *     - centrality.len >= adjacency.primary_dim()
 *
 * POSTCONDITIONS:
 *     - centrality[i] contains degree of node i
 *     - If normalize=true, values are in [0, 1]
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over nodes
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void degree_centrality(
    const Sparse<T, IsCSR>& adjacency,      // Adjacency matrix
    Array<Real> centrality,                   // Output scores [n_nodes]
    bool normalize = true                     // Normalize by max
);

/* -----------------------------------------------------------------------------
 * FUNCTION: pagerank
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute PageRank centrality using power iteration.
 *
 * PARAMETERS:
 *     adjacency  [in]  Adjacency matrix (CSR or CSC)
 *     scores     [out] PageRank scores [n_nodes]
 *     damping    [in]  Damping factor (default 0.85)
 *     max_iter   [in]  Maximum iterations
 *     tol        [in]  Convergence tolerance
 *
 * PRECONDITIONS:
 *     - scores.len >= adjacency.primary_dim()
 *     - damping in (0, 1)
 *
 * POSTCONDITIONS:
 *     - scores contains PageRank values
 *     - Scores sum to 1.0
 *
 * ALGORITHM:
 *     Power iteration: scores = (1-d) * teleport + d * A^T * scores
 *
 * COMPLEXITY:
 *     Time:  O(max_iter * nnz)
 *     Space: O(n_nodes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized with atomic accumulation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void pagerank(
    const Sparse<T, IsCSR>& adjacency,      // Adjacency matrix
    Array<Real> scores,                      // Output PageRank scores [n_nodes]
    Real damping = config::DEFAULT_DAMPING,   // Damping factor
    Index max_iter = config::DEFAULT_MAX_ITER, // Max iterations
    Real tol = config::DEFAULT_TOLERANCE     // Convergence tolerance
);

/* -----------------------------------------------------------------------------
 * FUNCTION: personalized_pagerank
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute personalized PageRank with custom teleportation vector.
 *
 * PARAMETERS:
 *     adjacency      [in]  Adjacency matrix (CSR or CSC)
 *     personalization [in]  Teleportation probabilities [n_nodes]
 *     scores         [out] Personalized PageRank scores [n_nodes]
 *     damping        [in]  Damping factor
 *     max_iter       [in]  Maximum iterations
 *     tol            [in]  Convergence tolerance
 *
 * PRECONDITIONS:
 *     - personalization.len == adjacency.primary_dim()
 *     - personalization sums to 1.0
 *
 * POSTCONDITIONS:
 *     - scores contains personalized PageRank values
 *     - Scores sum to 1.0
 *
 * COMPLEXITY:
 *     Time:  O(max_iter * nnz)
 *     Space: O(n_nodes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized with atomic accumulation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void personalized_pagerank(
    const Sparse<T, IsCSR>& adjacency,      // Adjacency matrix
    Array<const Real> personalization,       // Teleportation vector [n_nodes]
    Array<Real> scores,                       // Output scores [n_nodes]
    Real damping = config::DEFAULT_DAMPING,   // Damping factor
    Index max_iter = config::DEFAULT_MAX_ITER, // Max iterations
    Real tol = config::DEFAULT_TOLERANCE     // Convergence tolerance
);

/* -----------------------------------------------------------------------------
 * FUNCTION: hits
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute HITS (Hyperlink-Induced Topic Search) hub and authority scores.
 *
 * PARAMETERS:
 *     adjacency        [in]  Adjacency matrix (CSR or CSC)
 *     hub_scores       [out] Hub scores [n_nodes]
 *     authority_scores [out] Authority scores [n_nodes]
 *     max_iter         [in]  Maximum iterations
 *     tol              [in]  Convergence tolerance
 *
 * PRECONDITIONS:
 *     - Both score arrays have length >= adjacency.primary_dim()
 *
 * POSTCONDITIONS:
 *     - hub_scores and authority_scores are normalized
 *     - auth[j] = sum_i(hub[i] * A[i,j])
 *     - hub[i] = sum_j(auth[j] * A[i,j])
 *
 * COMPLEXITY:
 *     Time:  O(max_iter * nnz)
 *     Space: O(n_nodes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized with atomic accumulation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void hits(
    const Sparse<T, IsCSR>& adjacency,      // Adjacency matrix
    Array<Real> hub_scores,                   // Output hub scores [n_nodes]
    Array<Real> authority_scores,             // Output authority scores [n_nodes]
    Index max_iter = config::DEFAULT_MAX_ITER, // Max iterations
    Real tol = config::DEFAULT_TOLERANCE     // Convergence tolerance
);

/* -----------------------------------------------------------------------------
 * FUNCTION: eigenvector_centrality
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute eigenvector centrality (dominant eigenvector).
 *
 * PARAMETERS:
 *     adjacency  [in]  Adjacency matrix (CSR or CSC)
 *     centrality [out] Eigenvector centrality [n_nodes]
 *     max_iter   [in]  Maximum iterations
 *     tol        [in]  Convergence tolerance
 *
 * PRECONDITIONS:
 *     - centrality.len >= adjacency.primary_dim()
 *
 * POSTCONDITIONS:
 *     - centrality contains dominant eigenvector
 *     - Vector is L2-normalized
 *
 * COMPLEXITY:
 *     Time:  O(max_iter * nnz)
 *     Space: O(n_nodes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized with atomic accumulation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void eigenvector_centrality(
    const Sparse<T, IsCSR>& adjacency,      // Adjacency matrix
    Array<Real> centrality,                   // Output scores [n_nodes]
    Index max_iter = config::DEFAULT_MAX_ITER, // Max iterations
    Real tol = config::DEFAULT_TOLERANCE     // Convergence tolerance
);

/* -----------------------------------------------------------------------------
 * FUNCTION: katz_centrality
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Katz centrality: centrality = alpha * A * centrality + beta.
 *
 * PARAMETERS:
 *     adjacency  [in]  Adjacency matrix (CSR or CSC)
 *     centrality [out] Katz centrality [n_nodes]
 *     alpha      [in]  Attenuation factor (default 0.1)
 *     beta       [in]  Constant term (default 1.0)
 *     max_iter   [in]  Maximum iterations
 *     tol        [in]  Convergence tolerance
 *
 * PRECONDITIONS:
 *     - centrality.len >= adjacency.primary_dim()
 *     - alpha < 1 / lambda_max
 *
 * POSTCONDITIONS:
 *     - centrality contains Katz scores
 *
 * COMPLEXITY:
 *     Time:  O(max_iter * nnz)
 *     Space: O(n_nodes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void katz_centrality(
    const Sparse<T, IsCSR>& adjacency,      // Adjacency matrix
    Array<Real> centrality,                   // Output scores [n_nodes]
    Real alpha = Real(0.1),                   // Attenuation factor
    Real beta = Real(1.0),                    // Constant term
    Index max_iter = config::DEFAULT_MAX_ITER, // Max iterations
    Real tol = config::DEFAULT_TOLERANCE     // Convergence tolerance
);

/* -----------------------------------------------------------------------------
 * FUNCTION: closeness_centrality
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute closeness centrality (inverse of average shortest path length).
 *
 * PARAMETERS:
 *     adjacency  [in]  Adjacency matrix (CSR or CSC)
 *     centrality [out] Closeness centrality [n_nodes]
 *     normalize  [in]  If true, normalize by (n-1)
 *
 * PRECONDITIONS:
 *     - centrality.len >= adjacency.primary_dim()
 *     - Graph must be connected (or component-wise)
 *
 * POSTCONDITIONS:
 *     - centrality[i] = (n-1) / sum(distances[i])
 *     - Isolated nodes have centrality 0
 *
 * COMPLEXITY:
 *     Time:  O(n_nodes * nnz) for BFS from each node
 *     Space: O(n_nodes) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over source nodes
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void closeness_centrality(
    const Sparse<T, IsCSR>& adjacency,      // Adjacency matrix
    Array<Real> centrality,                   // Output scores [n_nodes]
    bool normalize = true                     // Normalize by (n-1)
);

/* -----------------------------------------------------------------------------
 * FUNCTION: betweenness_centrality
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute betweenness centrality using Brandes algorithm.
 *
 * PARAMETERS:
 *     adjacency  [in]  Adjacency matrix (CSR or CSC)
 *     centrality [out] Betweenness centrality [n_nodes]
 *     normalize  [in]  If true, normalize by (n-1)*(n-2)/2
 *
 * PRECONDITIONS:
 *     - centrality.len >= adjacency.primary_dim()
 *
 * POSTCONDITIONS:
 *     - centrality[i] contains fraction of shortest paths passing through i
 *     - Normalized values in [0, 1]
 *
 * ALGORITHM:
 *     Brandes algorithm: BFS from each source, accumulate dependencies
 *
 * COMPLEXITY:
 *     Time:  O(n_nodes * nnz) for unweighted
 *     Space: O(n_nodes) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over source nodes
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void betweenness_centrality(
    const Sparse<T, IsCSR>& adjacency,      // Adjacency matrix
    Array<Real> centrality,                   // Output scores [n_nodes]
    bool normalize = true                     // Normalize
);

/* -----------------------------------------------------------------------------
 * FUNCTION: approximate_betweenness
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute approximate betweenness centrality using random sampling.
 *
 * PARAMETERS:
 *     adjacency  [in]  Adjacency matrix (CSR or CSC)
 *     centrality [out] Approximate betweenness [n_nodes]
 *     n_samples  [in]  Number of source nodes to sample
 *     normalize  [in]  If true, normalize
 *     seed       [in]  Random seed
 *
 * PRECONDITIONS:
 *     - centrality.len >= adjacency.primary_dim()
 *     - n_samples <= n_nodes
 *
 * POSTCONDITIONS:
 *     - centrality contains scaled approximation
 *
 * COMPLEXITY:
 *     Time:  O(n_samples * nnz)
 *     Space: O(n_nodes) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over samples
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void approximate_betweenness(
    const Sparse<T, IsCSR>& adjacency,      // Adjacency matrix
    Array<Real> centrality,                   // Output scores [n_nodes]
    Index n_samples,                          // Number of samples
    bool normalize = true,                    // Normalize
    uint64_t seed = 42                        // Random seed
);

/* -----------------------------------------------------------------------------
 * FUNCTION: harmonic_centrality
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute harmonic centrality (sum of inverse distances).
 *
 * PARAMETERS:
 *     adjacency  [in]  Adjacency matrix (CSR or CSC)
 *     centrality [out] Harmonic centrality [n_nodes]
 *     normalize  [in]  If true, normalize by (n-1)
 *
 * PRECONDITIONS:
 *     - centrality.len >= adjacency.primary_dim()
 *
 * POSTCONDITIONS:
 *     - centrality[i] = sum(1 / dist(i,j)) for j != i
 *     - Works for disconnected graphs (unlike closeness)
 *
 * COMPLEXITY:
 *     Time:  O(n_nodes * nnz)
 *     Space: O(n_nodes) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over source nodes
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void harmonic_centrality(
    const Sparse<T, IsCSR>& adjacency,      // Adjacency matrix
    Array<Real> centrality,                   // Output scores [n_nodes]
    bool normalize = true                     // Normalize
);

/* -----------------------------------------------------------------------------
 * FUNCTION: random_walk_centrality
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute centrality based on random walk visit frequencies.
 *
 * PARAMETERS:
 *     adjacency   [in]  Adjacency matrix (CSR or CSC)
 *     centrality  [out] Random walk centrality [n_nodes]
 *     n_walks     [in]  Number of random walks
 *     walk_length [in]  Length of each walk
 *     seed        [in]  Random seed
 *
 * PRECONDITIONS:
 *     - centrality.len >= adjacency.primary_dim()
 *
 * POSTCONDITIONS:
 *     - centrality[i] contains normalized visit frequency
 *     - Values sum to 1.0
 *
 * COMPLEXITY:
 *     Time:  O(n_walks * walk_length)
 *     Space: O(n_nodes) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over walks
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void random_walk_centrality(
    const Sparse<T, IsCSR>& adjacency,      // Adjacency matrix
    Array<Real> centrality,                   // Output scores [n_nodes]
    Index n_walks,                            // Number of walks
    Index walk_length,                        // Length of each walk
    uint64_t seed = 42                        // Random seed
);

} // namespace scl::kernel::centrality

