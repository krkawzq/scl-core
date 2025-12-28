// =============================================================================
// FILE: scl/kernel/metrics.h
// BRIEF: API reference for clustering and integration quality metrics
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::metrics {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Real LOG2_E = Real(1.4426950408889634);
    constexpr Size PARALLEL_THRESHOLD = 256;
}

// =============================================================================
// Silhouette Score
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: silhouette_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute the mean Silhouette Coefficient across all samples, measuring
 *     how similar each sample is to its own cluster compared to other clusters.
 *
 * PARAMETERS:
 *     distances [in]  Pairwise distance matrix (cells x cells, CSR)
 *     labels    [in]  Cluster assignments for each cell
 *
 * RETURNS:
 *     Mean silhouette score in range [-1, 1], where 1 indicates dense, well-
 *     separated clusters and -1 indicates incorrect clustering
 *
 * PRECONDITIONS:
 *     - distances.rows() == labels.len
 *     - At least 2 cells and 2 clusters for meaningful result
 *     - Distance values should be non-negative
 *
 * POSTCONDITIONS:
 *     - Returns 0 if fewer than 2 cells or clusters
 *     - Singleton clusters are excluded from computation
 *
 * ALGORITHM:
 *     For each cell i with cluster label c:
 *         1. Compute a(i) = mean distance to other cells in same cluster
 *         2. Compute b(i) = min of mean distances to cells in other clusters
 *         3. s(i) = (b(i) - a(i)) / max(a(i), b(i))
 *     Return mean of all s(i) values
 *
 * COMPLEXITY:
 *     Time:  O(n * nnz_per_row * n_clusters)
 *     Space: O(n_clusters) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Real silhouette_score(
    const Sparse<T, IsCSR>& distances,   // Pairwise distance matrix
    Array<const Index> labels             // Cluster labels [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: silhouette_samples
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Silhouette Coefficient for each individual sample.
 *
 * PARAMETERS:
 *     distances [in]  Pairwise distance matrix
 *     labels    [in]  Cluster assignments
 *     scores    [out] Per-sample silhouette scores
 *
 * PRECONDITIONS:
 *     - distances.rows() == labels.len == scores.len
 *     - At least 2 cells and 2 clusters
 *
 * POSTCONDITIONS:
 *     - scores[i] = silhouette for cell i, in [-1, 1]
 *     - Cells in singleton clusters have score 0
 *
 * COMPLEXITY:
 *     Time:  O(n * nnz_per_row * n_clusters)
 *     Space: O(n_clusters * n_threads) for thread-local buffers
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells with WorkspacePool
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void silhouette_samples(
    const Sparse<T, IsCSR>& distances,   // Pairwise distance matrix
    Array<const Index> labels,            // Cluster labels [n_cells]
    Array<Real> scores                    // Output scores [n_cells]
);

// =============================================================================
// Adjusted Rand Index (ARI)
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: adjusted_rand_index
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute the Adjusted Rand Index between two clusterings, measuring
 *     similarity adjusted for chance.
 *
 * PARAMETERS:
 *     labels1 [in]  First clustering assignment
 *     labels2 [in]  Second clustering assignment
 *
 * RETURNS:
 *     ARI score in range [-1, 1], where 1 indicates identical clusterings
 *     and 0 indicates random labeling
 *
 * PRECONDITIONS:
 *     - labels1.len == labels2.len
 *     - Labels must be non-negative integers
 *
 * POSTCONDITIONS:
 *     - Returns 1.0 for identical clusterings
 *     - Returns 0.0 on average for random clusterings
 *
 * ALGORITHM:
 *     1. Build contingency table n_ij
 *     2. Compute sum of C(n_ij, 2), C(a_i, 2), C(b_j, 2)
 *     3. ARI = (sum_nij - expected) / (mean - expected)
 *
 * COMPLEXITY:
 *     Time:  O(n + n_clusters1 * n_clusters2)
 *     Space: O(n_clusters1 * n_clusters2) for contingency table
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
Real adjusted_rand_index(
    Array<const Index> labels1,           // First clustering [n]
    Array<const Index> labels2            // Second clustering [n]
);

// =============================================================================
// Normalized Mutual Information (NMI)
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: normalized_mutual_information
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Normalized Mutual Information between two clusterings.
 *
 * PARAMETERS:
 *     labels1 [in]  First clustering assignment
 *     labels2 [in]  Second clustering assignment
 *
 * RETURNS:
 *     NMI score in range [0, 1], where 1 indicates perfect agreement
 *
 * PRECONDITIONS:
 *     - labels1.len == labels2.len
 *     - Labels must be non-negative integers
 *
 * POSTCONDITIONS:
 *     - Returns 1.0 for identical clusterings
 *     - Returns 0.0 for independent clusterings
 *
 * ALGORITHM:
 *     1. Build contingency table
 *     2. Compute entropies H(labels1), H(labels2)
 *     3. Compute mutual information MI
 *     4. NMI = 2 * MI / (H1 + H2)
 *
 * COMPLEXITY:
 *     Time:  O(n + n_clusters1 * n_clusters2)
 *     Space: O(n_clusters1 * n_clusters2)
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
Real normalized_mutual_information(
    Array<const Index> labels1,           // First clustering [n]
    Array<const Index> labels2            // Second clustering [n]
);

// =============================================================================
// Graph Connectivity
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: graph_connectivity
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Measure cluster connectivity as fraction of clusters that are fully
 *     connected (single component) in the graph.
 *
 * PARAMETERS:
 *     adjacency [in]  Cell neighborhood graph
 *     labels    [in]  Cluster assignments
 *
 * RETURNS:
 *     Fraction of clusters that are connected, in [0, 1]
 *
 * PRECONDITIONS:
 *     - adjacency.rows() == labels.len
 *     - Adjacency should be symmetric for undirected connectivity
 *
 * POSTCONDITIONS:
 *     - Returns 1.0 if all clusters are fully connected
 *     - Returns 0.0 if all clusters are fragmented
 *
 * ALGORITHM:
 *     1. BFS to find connected components within each cluster
 *     2. Count clusters with exactly one component
 *     3. Return ratio of connected clusters to total
 *
 * COMPLEXITY:
 *     Time:  O(n + nnz)
 *     Space: O(n) for component IDs and BFS queue
 *
 * THREAD SAFETY:
 *     Unsafe - sequential BFS
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Real graph_connectivity(
    const Sparse<T, IsCSR>& adjacency,   // Cell neighborhood graph
    Array<const Index> labels             // Cluster labels [n_cells]
);

// =============================================================================
// Batch Entropy
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: batch_entropy
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute normalized entropy of batch distribution in each cell's
 *     neighborhood, measuring batch mixing quality.
 *
 * PARAMETERS:
 *     neighbors       [in]  KNN graph (cells x cells, CSR)
 *     batch_labels    [in]  Batch assignment for each cell
 *     entropy_scores  [out] Per-cell normalized entropy
 *
 * PRECONDITIONS:
 *     - neighbors.rows() == batch_labels.len == entropy_scores.len
 *     - Batch labels must be non-negative integers
 *
 * POSTCONDITIONS:
 *     - entropy_scores[i] in [0, 1]
 *     - 1 indicates perfect batch mixing (uniform distribution)
 *     - 0 indicates single batch in neighborhood
 *
 * ALGORITHM:
 *     For each cell in parallel:
 *         1. Count batch occurrences in neighborhood (including self)
 *         2. Compute Shannon entropy of distribution
 *         3. Normalize by log(n_batches)
 *
 * COMPLEXITY:
 *     Time:  O(n * k) where k = average neighbors per cell
 *     Space: O(n_batches * n_threads) for thread-local counters
 *
 * THREAD SAFETY:
 *     Safe - parallelized with WorkspacePool
 * -------------------------------------------------------------------------- */
template <bool IsCSR>
void batch_entropy(
    const Sparse<Index, IsCSR>& neighbors,  // KNN graph
    Array<const Index> batch_labels,         // Batch labels [n_cells]
    Array<Real> entropy_scores               // Output entropy [n_cells]
);

// =============================================================================
// Local Inverse Simpson's Index (LISI)
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: lisi
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Local Inverse Simpson's Index for measuring label diversity
 *     in local neighborhoods.
 *
 * PARAMETERS:
 *     neighbors    [in]  KNN graph
 *     labels       [in]  Label assignments (batch or cell type)
 *     lisi_scores  [out] Per-cell LISI scores
 *
 * PRECONDITIONS:
 *     - neighbors.rows() == labels.len == lisi_scores.len
 *     - Labels must be non-negative integers
 *
 * POSTCONDITIONS:
 *     - lisi_scores[i] >= 1
 *     - LISI = 1 when all neighbors have same label
 *     - LISI approaches n_labels for uniform distribution
 *
 * ALGORITHM:
 *     For each cell in parallel:
 *         1. Count label occurrences in neighborhood
 *         2. Compute Simpson's index: sum(p_i^2)
 *         3. LISI = 1 / Simpson's index
 *
 * COMPLEXITY:
 *     Time:  O(n * k) where k = average neighbors
 *     Space: O(n_labels * n_threads) for thread-local counters
 *
 * THREAD SAFETY:
 *     Safe - parallelized with WorkspacePool
 * -------------------------------------------------------------------------- */
template <bool IsCSR>
void lisi(
    const Sparse<Index, IsCSR>& neighbors,  // KNN graph
    Array<const Index> labels,               // Labels [n_cells]
    Array<Real> lisi_scores                  // Output LISI [n_cells]
);

// =============================================================================
// Additional Metrics
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: fowlkes_mallows_index
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Fowlkes-Mallows Index measuring similarity between clusterings.
 *
 * PARAMETERS:
 *     labels1 [in]  First clustering
 *     labels2 [in]  Second clustering
 *
 * RETURNS:
 *     FMI in range [0, 1], geometric mean of precision and recall
 *
 * PRECONDITIONS:
 *     - labels1.len == labels2.len
 *
 * POSTCONDITIONS:
 *     - Returns 1.0 for identical clusterings
 *     - Returns 0.0 when no pairs agree
 *
 * COMPLEXITY:
 *     Time:  O(n + n_clusters1 * n_clusters2)
 *     Space: O(n_clusters1 * n_clusters2)
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
Real fowlkes_mallows_index(
    Array<const Index> labels1,           // First clustering [n]
    Array<const Index> labels2            // Second clustering [n]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: v_measure
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute V-measure, the harmonic mean of homogeneity and completeness.
 *
 * PARAMETERS:
 *     labels_true [in]  Ground truth labels
 *     labels_pred [in]  Predicted cluster labels
 *     beta        [in]  Weight for homogeneity vs completeness
 *
 * RETURNS:
 *     V-measure in range [0, 1]
 *
 * PRECONDITIONS:
 *     - labels_true.len == labels_pred.len
 *     - beta >= 0 (beta=1 gives equal weight)
 *
 * POSTCONDITIONS:
 *     - Returns 1.0 for perfect clustering
 *     - beta > 1 weights completeness more
 *     - beta < 1 weights homogeneity more
 *
 * COMPLEXITY:
 *     Time:  O(n + n_classes * n_clusters)
 *     Space: O(n_classes * n_clusters)
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
Real v_measure(
    Array<const Index> labels_true,       // True labels [n]
    Array<const Index> labels_pred,       // Predicted labels [n]
    Real beta = Real(1.0)                 // Homogeneity-completeness weight
);

/* -----------------------------------------------------------------------------
 * FUNCTION: homogeneity_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute homogeneity: each cluster contains only members of a single class.
 *
 * PARAMETERS:
 *     labels_true [in]  Ground truth labels
 *     labels_pred [in]  Predicted cluster labels
 *
 * RETURNS:
 *     Homogeneity in range [0, 1]
 *
 * COMPLEXITY:
 *     Time:  O(n + n_classes * n_clusters)
 *     Space: O(n_classes * n_clusters)
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
Real homogeneity_score(
    Array<const Index> labels_true,       // True labels [n]
    Array<const Index> labels_pred        // Predicted labels [n]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: completeness_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute completeness: all members of a class are assigned to same cluster.
 *
 * PARAMETERS:
 *     labels_true [in]  Ground truth labels
 *     labels_pred [in]  Predicted cluster labels
 *
 * RETURNS:
 *     Completeness in range [0, 1]
 *
 * COMPLEXITY:
 *     Time:  O(n + n_classes * n_clusters)
 *     Space: O(n_classes * n_clusters)
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
Real completeness_score(
    Array<const Index> labels_true,       // True labels [n]
    Array<const Index> labels_pred        // Predicted labels [n]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: purity_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute clustering purity as fraction of correctly assigned samples.
 *
 * PARAMETERS:
 *     labels_true [in]  Ground truth labels
 *     labels_pred [in]  Predicted cluster labels
 *
 * RETURNS:
 *     Purity in range [0, 1]
 *
 * PRECONDITIONS:
 *     - labels_true.len == labels_pred.len
 *
 * POSTCONDITIONS:
 *     - Returns fraction of samples in majority class per cluster
 *
 * COMPLEXITY:
 *     Time:  O(n + n_classes * n_clusters)
 *     Space: O(n_classes * n_clusters)
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
Real purity_score(
    Array<const Index> labels_true,       // True labels [n]
    Array<const Index> labels_pred        // Predicted labels [n]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: mean_lisi
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute mean LISI score across all cells.
 *
 * PARAMETERS:
 *     neighbors [in]  KNN graph
 *     labels    [in]  Label assignments
 *
 * RETURNS:
 *     Mean LISI score
 *
 * PRECONDITIONS:
 *     - neighbors.rows() == labels.len
 *
 * POSTCONDITIONS:
 *     - Returns mean of per-cell LISI scores
 *     - Uses SIMD-optimized summation
 *
 * COMPLEXITY:
 *     Time:  O(n * k)
 *     Space: O(n) for intermediate scores
 *
 * THREAD SAFETY:
 *     Safe - uses parallelized lisi internally
 * -------------------------------------------------------------------------- */
template <bool IsCSR>
Real mean_lisi(
    const Sparse<Index, IsCSR>& neighbors,  // KNN graph
    Array<const Index> labels                // Labels [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: mean_batch_entropy
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute mean batch entropy across all cells.
 *
 * PARAMETERS:
 *     neighbors    [in]  KNN graph
 *     batch_labels [in]  Batch assignments
 *
 * RETURNS:
 *     Mean normalized batch entropy
 *
 * PRECONDITIONS:
 *     - neighbors.rows() == batch_labels.len
 *
 * POSTCONDITIONS:
 *     - Returns mean of per-cell entropy scores
 *     - Uses SIMD-optimized summation
 *
 * COMPLEXITY:
 *     Time:  O(n * k)
 *     Space: O(n) for intermediate scores
 *
 * THREAD SAFETY:
 *     Safe - uses parallelized batch_entropy internally
 * -------------------------------------------------------------------------- */
template <bool IsCSR>
Real mean_batch_entropy(
    const Sparse<Index, IsCSR>& neighbors,  // KNN graph
    Array<const Index> batch_labels          // Batch labels [n_cells]
);

} // namespace scl::kernel::metrics
