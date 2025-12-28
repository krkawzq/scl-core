// =============================================================================
// FILE: scl/kernel/doublet.h
// BRIEF: API reference for doublet detection kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::doublet {

// =============================================================================
// Configuration Constants
// =============================================================================

namespace config {
    constexpr Real DEFAULT_DOUBLET_RATE = Real(0.06);
    constexpr Real DEFAULT_THRESHOLD = Real(0.5);
    constexpr Index DEFAULT_N_NEIGHBORS = 30;
    constexpr Index DEFAULT_N_SIMULATED = 0;  // 0 = auto (2x n_cells)
    constexpr Real MIN_SCORE = Real(1e-10);
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size PREFETCH_DISTANCE = 16;
}

// =============================================================================
// Enumerations
// =============================================================================

/* -----------------------------------------------------------------------------
 * ENUM: DoubletMethod
 * -----------------------------------------------------------------------------
 * VALUES:
 *     Scrublet       - Scrublet-style k-NN based detection
 *     DoubletFinder  - DoubletFinder pANN-style detection
 *     Hybrid         - Combined approach using multiple signals
 * -------------------------------------------------------------------------- */
enum class DoubletMethod {
    Scrublet,
    DoubletFinder,
    Hybrid
};

// =============================================================================
// Core Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: simulate_doublets
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Simulate synthetic doublets by averaging random cell pairs.
 *
 * PARAMETERS:
 *     X                [in]  CSR sparse matrix (n_cells x n_genes)
 *     n_cells          [in]  Number of cells in the matrix
 *     n_genes          [in]  Number of genes in the matrix
 *     n_doublets       [in]  Number of synthetic doublets to generate
 *     doublet_profiles [out] Output buffer (n_doublets x n_genes), row-major
 *     seed             [in]  Random seed for reproducibility
 *
 * PRECONDITIONS:
 *     - X must be CSR format (cells x genes)
 *     - doublet_profiles must be pre-allocated with n_doublets * n_genes elements
 *     - n_cells >= 2 for meaningful doublet simulation
 *
 * POSTCONDITIONS:
 *     - doublet_profiles[d * n_genes : (d+1) * n_genes] contains the average
 *       expression profile of two randomly selected cells
 *     - Each doublet profile is normalized by 0.5 (average of two cells)
 *
 * ALGORITHM:
 *     For each doublet d in parallel:
 *         1. Select two distinct random cells (cell1, cell2)
 *         2. For each gene expressed in cell1: profile[gene] += 0.5 * value
 *         3. For each gene expressed in cell2: profile[gene] += 0.5 * value
 *
 * COMPLEXITY:
 *     Time:  O(n_doublets * avg_nnz_per_cell)
 *     Space: O(n_doublets * n_genes) for output
 *
 * THREAD SAFETY:
 *     Safe - parallelized over doublets, each writes to independent memory
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void simulate_doublets(
    const Sparse<T, IsCSR>& X,      // CSR sparse matrix (cells x genes)
    Index n_cells,                   // Number of cells
    Index n_genes,                   // Number of genes
    Index n_doublets,                // Number of doublets to simulate
    Real* doublet_profiles,          // Output buffer [n_doublets x n_genes]
    uint64_t seed = 42               // Random seed
);

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_knn_doublet_scores
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute doublet scores by k-NN against observed and simulated cells.
 *
 * PARAMETERS:
 *     X                [in]  CSR sparse matrix (n_cells x n_genes)
 *     n_cells          [in]  Number of observed cells
 *     n_genes          [in]  Number of genes
 *     doublet_profiles [in]  Simulated doublet profiles (n_doublets x n_genes)
 *     n_doublets       [in]  Number of simulated doublets
 *     k_neighbors      [in]  Number of nearest neighbors to consider
 *     doublet_scores   [out] Output scores, one per cell
 *
 * PRECONDITIONS:
 *     - X must be CSR format
 *     - doublet_profiles must contain n_doublets profiles
 *     - doublet_scores.len >= n_cells
 *     - k_neighbors > 0 and <= n_cells + n_doublets
 *
 * POSTCONDITIONS:
 *     - doublet_scores[i] = fraction of k nearest neighbors that are simulated doublets
 *     - Scores range from 0 (no doublet neighbors) to 1 (all doublet neighbors)
 *
 * ALGORITHM:
 *     For each cell i in parallel:
 *         1. Convert cell i to dense vector
 *         2. Compute squared distances to all observed cells (excluding self)
 *         3. Compute squared distances to all simulated doublets
 *         4. Find k nearest neighbors using heap-based selection O(n log k)
 *         5. Count fraction of neighbors that are doublets
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * (n_cells + n_doublets) * n_genes)
 *     Space: O(n_threads * (n_genes + n_total + k_neighbors)) workspace
 *
 * THREAD SAFETY:
 *     Safe - uses WorkspacePool for thread-local buffers
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void compute_knn_doublet_scores(
    const Sparse<T, IsCSR>& X,       // CSR sparse matrix
    Index n_cells,                    // Number of observed cells
    Index n_genes,                    // Number of genes
    const Real* doublet_profiles,     // Simulated profiles [n_doublets x n_genes]
    Index n_doublets,                 // Number of simulated doublets
    Index k_neighbors,                // Number of neighbors for k-NN
    Array<Real> doublet_scores        // Output scores [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_knn_doublet_scores_pca
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute doublet scores on PCA-reduced embeddings (faster for large data).
 *
 * PARAMETERS:
 *     cell_embeddings    [in]  PCA embeddings of observed cells (n_cells x n_dims)
 *     n_cells            [in]  Number of observed cells
 *     n_dims             [in]  Embedding dimensionality
 *     doublet_embeddings [in]  PCA embeddings of simulated doublets (n_doublets x n_dims)
 *     n_doublets         [in]  Number of simulated doublets
 *     k_neighbors        [in]  Number of nearest neighbors
 *     doublet_scores     [out] Output scores [n_cells]
 *
 * PRECONDITIONS:
 *     - All embeddings must be row-major (samples x dims)
 *     - doublet_scores.len >= n_cells
 *
 * POSTCONDITIONS:
 *     - Same as compute_knn_doublet_scores but computed in PCA space
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * (n_cells + n_doublets) * n_dims)
 *     Space: O(n_threads * (n_total + k_neighbors)) workspace
 *
 * THREAD SAFETY:
 *     Safe - uses WorkspacePool for thread-local buffers
 * -------------------------------------------------------------------------- */
void compute_knn_doublet_scores_pca(
    const Real* cell_embeddings,      // Cell PCA embeddings [n_cells x n_dims]
    Index n_cells,                     // Number of cells
    Index n_dims,                      // Embedding dimensions
    const Real* doublet_embeddings,    // Doublet PCA embeddings [n_doublets x n_dims]
    Index n_doublets,                  // Number of doublets
    Index k_neighbors,                 // Number of neighbors
    Array<Real> doublet_scores         // Output scores [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: scrublet_scores
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Full Scrublet-style doublet detection pipeline.
 *
 * PARAMETERS:
 *     X           [in]  CSR sparse matrix (n_cells x n_genes)
 *     n_cells     [in]  Number of cells
 *     n_genes     [in]  Number of genes
 *     scores      [out] Output doublet scores [n_cells]
 *     n_simulated [in]  Number of simulated doublets (0 = auto: 2x n_cells)
 *     k_neighbors [in]  Number of neighbors for k-NN
 *     seed        [in]  Random seed
 *
 * PRECONDITIONS:
 *     - X must be CSR format
 *     - scores.len >= n_cells
 *
 * POSTCONDITIONS:
 *     - scores[i] contains the Scrublet doublet score for cell i
 *     - Higher scores indicate higher likelihood of being a doublet
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * (n_cells + n_simulated) * n_genes)
 *     Space: O(n_simulated * n_genes) for doublet profiles
 *
 * THREAD SAFETY:
 *     Safe - internally parallelized
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void scrublet_scores(
    const Sparse<T, IsCSR>& X,       // CSR sparse matrix
    Index n_cells,                    // Number of cells
    Index n_genes,                    // Number of genes
    Array<Real> scores,               // Output scores [n_cells]
    Index n_simulated = 0,            // 0 = auto (2x n_cells)
    Index k_neighbors = config::DEFAULT_N_NEIGHBORS,
    uint64_t seed = 42
);

/* -----------------------------------------------------------------------------
 * FUNCTION: doubletfinder_pann
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     DoubletFinder-style pANN (proportion of Artificial Nearest Neighbors) score.
 *
 * PARAMETERS:
 *     cell_embeddings    [in]  Cell PCA embeddings (n_cells x n_dims)
 *     n_cells            [in]  Number of cells
 *     n_dims             [in]  Embedding dimensions
 *     doublet_embeddings [in]  Doublet embeddings (n_doublets x n_dims)
 *     n_doublets         [in]  Number of doublets
 *     pK                 [in]  Proportion of cells+doublets to use as k
 *     pann_scores        [out] Output pANN scores [n_cells]
 *
 * PRECONDITIONS:
 *     - pK should be in range (0, 1]
 *     - pann_scores.len >= n_cells
 *
 * POSTCONDITIONS:
 *     - pann_scores[i] = proportion of neighbors that are artificial doublets
 *
 * THREAD SAFETY:
 *     Safe - internally parallelized
 * -------------------------------------------------------------------------- */
void doubletfinder_pann(
    const Real* cell_embeddings,      // Cell embeddings [n_cells x n_dims]
    Index n_cells,                     // Number of cells
    Index n_dims,                      // Dimensions
    const Real* doublet_embeddings,    // Doublet embeddings [n_doublets x n_dims]
    Index n_doublets,                  // Number of doublets
    Real pK,                           // Proportion of k
    Array<Real> pann_scores            // Output [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: estimate_threshold
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Estimate score threshold from expected doublet rate.
 *
 * PARAMETERS:
 *     scores               [in]  Doublet scores array
 *     expected_doublet_rate [in] Expected proportion of doublets (e.g., 0.06)
 *
 * RETURNS:
 *     Score threshold at (1 - expected_doublet_rate) percentile
 *
 * PRECONDITIONS:
 *     - scores.len > 0
 *     - expected_doublet_rate in (0, 1)
 *
 * POSTCONDITIONS:
 *     - Returns threshold such that approximately expected_doublet_rate fraction
 *       of cells would be called as doublets
 *
 * ALGORITHM:
 *     1. Copy scores to temporary buffer
 *     2. Sort using scl::sort::sort (SIMD-optimized, O(n log n))
 *     3. Return value at (1 - rate) percentile
 *
 * COMPLEXITY:
 *     Time:  O(n log n)
 *     Space: O(n) for sorted copy
 *
 * THREAD SAFETY:
 *     Safe - no shared mutable state
 * -------------------------------------------------------------------------- */
Real estimate_threshold(
    Array<const Real> scores,         // Doublet scores
    Real expected_doublet_rate        // Expected doublet proportion
);

/* -----------------------------------------------------------------------------
 * FUNCTION: call_doublets
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Call doublets based on score threshold.
 *
 * PARAMETERS:
 *     scores     [in]  Doublet scores array
 *     threshold  [in]  Score threshold for calling doublets
 *     is_doublet [out] Boolean array indicating doublet calls
 *
 * RETURNS:
 *     Number of cells called as doublets
 *
 * PRECONDITIONS:
 *     - is_doublet.len >= scores.len
 *
 * POSTCONDITIONS:
 *     - is_doublet[i] = true if scores[i] > threshold
 *     - Return value = count of doublets
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - no shared mutable state
 * -------------------------------------------------------------------------- */
Index call_doublets(
    Array<const Real> scores,         // Doublet scores
    Real threshold,                    // Score threshold
    Array<bool> is_doublet             // Output boolean array [n]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: detect_bimodal_threshold
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Detect threshold using bimodal distribution (histogram valley detection).
 *
 * PARAMETERS:
 *     scores [in]  Doublet scores array
 *     n_bins [in]  Number of histogram bins (default: 50)
 *
 * RETURNS:
 *     Estimated threshold at the valley between two peaks
 *
 * PRECONDITIONS:
 *     - scores.len >= 10 for meaningful histogram
 *
 * POSTCONDITIONS:
 *     - Returns threshold at local minimum between first and second peaks
 *     - Falls back to DEFAULT_THRESHOLD if no clear bimodal distribution
 *
 * ALGORITHM:
 *     1. Find score range using SIMD min/max
 *     2. Build histogram with n_bins
 *     3. Find first peak in lower half
 *     4. Find valley (local minimum) after first peak
 *     5. Return threshold at valley position
 *
 * COMPLEXITY:
 *     Time:  O(n + n_bins)
 *     Space: O(n_bins)
 *
 * THREAD SAFETY:
 *     Safe - no shared mutable state
 * -------------------------------------------------------------------------- */
Real detect_bimodal_threshold(
    Array<const Real> scores,         // Doublet scores
    Index n_bins = 50                  // Histogram bins
);

/* -----------------------------------------------------------------------------
 * FUNCTION: detect_doublets
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Full doublet detection pipeline (simulate, score, threshold, call).
 *
 * PARAMETERS:
 *     X             [in]  CSR sparse matrix (n_cells x n_genes)
 *     n_cells       [in]  Number of cells
 *     n_genes       [in]  Number of genes
 *     scores        [out] Doublet scores [n_cells]
 *     is_doublet    [out] Doublet calls [n_cells]
 *     method        [in]  Detection method
 *     expected_rate [in]  Expected doublet rate
 *     k_neighbors   [in]  Number of neighbors
 *     seed          [in]  Random seed
 *
 * RETURNS:
 *     Number of detected doublets
 *
 * PRECONDITIONS:
 *     - X must be CSR format
 *     - scores.len >= n_cells
 *     - is_doublet.len >= n_cells
 *
 * POSTCONDITIONS:
 *     - scores contains doublet scores
 *     - is_doublet contains boolean calls
 *     - Returns total doublet count
 *
 * THREAD SAFETY:
 *     Safe - internally parallelized
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
Index detect_doublets(
    const Sparse<T, IsCSR>& X,        // CSR sparse matrix
    Index n_cells,                     // Number of cells
    Index n_genes,                     // Number of genes
    Array<Real> scores,                // Output scores [n_cells]
    Array<bool> is_doublet,            // Output calls [n_cells]
    DoubletMethod method = DoubletMethod::Scrublet,
    Real expected_rate = config::DEFAULT_DOUBLET_RATE,
    Index k_neighbors = config::DEFAULT_N_NEIGHBORS,
    uint64_t seed = 42
);

// =============================================================================
// Auxiliary Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: doublet_score_stats
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute statistics (mean, std_dev, median) of doublet scores.
 *
 * PARAMETERS:
 *     scores  [in]  Doublet scores array
 *     mean    [out] Computed mean
 *     std_dev [out] Computed standard deviation
 *     median  [out] Computed median
 *
 * PRECONDITIONS:
 *     - All output pointers must be valid
 *
 * POSTCONDITIONS:
 *     - *mean = sum(scores) / n
 *     - *std_dev = sqrt(var(scores))
 *     - *median = middle value of sorted scores
 *
 * ALGORITHM:
 *     1. Mean: SIMD sum / n
 *     2. Variance: SIMD sum of squared deviations
 *     3. Median: scl::sort::sort + select middle
 *
 * COMPLEXITY:
 *     Time:  O(n log n) for median
 *     Space: O(n) for sorted copy
 *
 * THREAD SAFETY:
 *     Safe - no shared mutable state
 * -------------------------------------------------------------------------- */
void doublet_score_stats(
    Array<const Real> scores,         // Doublet scores
    Real* mean,                        // Output mean
    Real* std_dev,                     // Output standard deviation
    Real* median                       // Output median
);

/* -----------------------------------------------------------------------------
 * FUNCTION: combined_doublet_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Combine multiple score types into a single weighted score.
 *
 * PARAMETERS:
 *     knn_scores      [in]  k-NN based scores
 *     density_scores  [in]  Density-based scores
 *     variance_scores [in]  Variance-based scores
 *     combined_scores [out] Weighted combination
 *     knn_weight      [in]  Weight for k-NN scores (default: 0.6)
 *     density_weight  [in]  Weight for density scores (default: 0.2)
 *     variance_weight [in]  Weight for variance scores (default: 0.2)
 *
 * PRECONDITIONS:
 *     - All input arrays must have same length
 *     - combined_scores.len >= knn_scores.len
 *
 * POSTCONDITIONS:
 *     - combined_scores[i] = weighted average of normalized individual scores
 *     - Each score type is normalized by its maximum before combination
 *
 * ALGORITHM:
 *     1. Find max of each score type using SIMD
 *     2. Normalize and combine using SIMD FMA
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - no shared mutable state
 * -------------------------------------------------------------------------- */
void combined_doublet_score(
    Array<const Real> knn_scores,      // k-NN scores [n]
    Array<const Real> density_scores,  // Density scores [n]
    Array<const Real> variance_scores, // Variance scores [n]
    Array<Real> combined_scores,       // Output combined [n]
    Real knn_weight = Real(0.6),
    Real density_weight = Real(0.2),
    Real variance_weight = Real(0.2)
);

/* -----------------------------------------------------------------------------
 * FUNCTION: density_doublet_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute density-based doublet scores from k-NN graph.
 *
 * PARAMETERS:
 *     knn_graph      [in]  k-NN graph (sparse matrix with distances as values)
 *     density_scores [out] Output density scores
 *
 * PRECONDITIONS:
 *     - knn_graph values should be distances
 *     - density_scores.len >= knn_graph.primary_dim()
 *
 * POSTCONDITIONS:
 *     - density_scores[i] = 1 / avg_distance_to_neighbors
 *     - Higher values indicate denser local neighborhoods
 *
 * COMPLEXITY:
 *     Time:  O(n * k)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void density_doublet_score(
    const Sparse<T, IsCSR>& knn_graph, // k-NN graph with distances
    Array<Real> density_scores          // Output [n]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: variance_doublet_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute variance-based doublet scores (doublets often have higher variance).
 *
 * PARAMETERS:
 *     X               [in]  CSR sparse matrix (n_cells x n_genes)
 *     n_cells         [in]  Number of cells
 *     n_genes         [in]  Number of genes
 *     gene_means      [in]  Pre-computed gene means
 *     variance_scores [out] Output variance scores
 *
 * PRECONDITIONS:
 *     - X must be CSR format
 *     - gene_means.len >= n_genes
 *     - variance_scores.len >= n_cells
 *
 * POSTCONDITIONS:
 *     - variance_scores[i] = avg squared deviation from gene means for cell i
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1)
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void variance_doublet_score(
    const Sparse<T, IsCSR>& X,         // CSR sparse matrix
    Index n_cells,                      // Number of cells
    Index n_genes,                      // Number of genes
    Array<const Real> gene_means,       // Gene means [n_genes]
    Array<Real> variance_scores         // Output [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: classify_doublet_types_knn
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Classify doublets as heterotypic or homotypic based on neighbor clusters.
 *
 * PARAMETERS:
 *     knn_graph      [in]  k-NN graph (sparse)
 *     cluster_labels [in]  Cluster assignment for each cell
 *     is_doublet     [in]  Boolean doublet calls
 *     n_clusters     [in]  Total number of clusters
 *     doublet_type   [out] Classification: 0=singlet, 1=heterotypic, 2=homotypic
 *
 * PRECONDITIONS:
 *     - All arrays must have length >= knn_graph.primary_dim()
 *     - cluster_labels values in [0, n_clusters)
 *
 * POSTCONDITIONS:
 *     - doublet_type[i] = 0 if not a doublet
 *     - doublet_type[i] = 1 if heterotypic (neighbors from multiple clusters)
 *     - doublet_type[i] = 2 if homotypic (neighbors mostly from one cluster)
 *
 * ALGORITHM:
 *     For each doublet:
 *         1. Count neighbor cluster memberships
 *         2. Find top two cluster frequencies
 *         3. If second cluster > 20%, classify as heterotypic
 *
 * COMPLEXITY:
 *     Time:  O(n * k + n * n_clusters)
 *     Space: O(n_threads * n_clusters) workspace
 *
 * THREAD SAFETY:
 *     Safe - uses WorkspacePool
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void classify_doublet_types_knn(
    const Sparse<T, IsCSR>& knn_graph,  // k-NN graph
    Array<const Index> cluster_labels,   // Cluster labels [n]
    Array<const bool> is_doublet,        // Doublet calls [n]
    Index n_clusters,                     // Number of clusters
    Array<Index> doublet_type             // Output types [n]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: cluster_doublet_enrichment
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute doublet enrichment statistics per cluster.
 *
 * PARAMETERS:
 *     doublet_scores          [in]  Doublet scores for all cells
 *     cluster_labels          [in]  Cluster assignment for each cell
 *     n_clusters              [in]  Total number of clusters
 *     cluster_mean_scores     [out] Mean doublet score per cluster
 *     cluster_doublet_fraction [out] Fraction with score > global mean
 *
 * PRECONDITIONS:
 *     - cluster_mean_scores.len >= n_clusters
 *     - cluster_doublet_fraction.len >= n_clusters
 *
 * POSTCONDITIONS:
 *     - cluster_mean_scores[c] = average score for cells in cluster c
 *     - cluster_doublet_fraction[c] = fraction of cluster c above global mean
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(n_clusters)
 *
 * THREAD SAFETY:
 *     Safe - no shared mutable state
 * -------------------------------------------------------------------------- */
void cluster_doublet_enrichment(
    Array<const Real> doublet_scores,     // Scores [n]
    Array<const Index> cluster_labels,    // Labels [n]
    Index n_clusters,                      // Number of clusters
    Array<Real> cluster_mean_scores,       // Output means [n_clusters]
    Array<Real> cluster_doublet_fraction   // Output fractions [n_clusters]
);

// =============================================================================
// Utility Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: expected_doublets
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Calculate expected number of doublets given cell count and rate.
 *
 * PARAMETERS:
 *     n_cells      [in]  Number of cells
 *     doublet_rate [in]  Expected doublet rate (default: 0.06)
 *
 * RETURNS:
 *     Expected doublet count (rounded to nearest integer)
 * -------------------------------------------------------------------------- */
Index expected_doublets(
    Index n_cells,
    Real doublet_rate = config::DEFAULT_DOUBLET_RATE
);

/* -----------------------------------------------------------------------------
 * FUNCTION: estimate_doublet_rate
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Estimate doublet rate from loading parameters (Poisson model).
 *
 * PARAMETERS:
 *     n_cells_loaded         [in]  Number of cells loaded
 *     cells_per_droplet_mean [in]  Mean cells per droplet (default: 0.05)
 *
 * RETURNS:
 *     Estimated doublet rate based on loading density
 * -------------------------------------------------------------------------- */
Real estimate_doublet_rate(
    Index n_cells_loaded,
    Real cells_per_droplet_mean = Real(0.05)
);

/* -----------------------------------------------------------------------------
 * FUNCTION: multiplet_rate_10x
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Estimate multiplet rate using 10x Genomics reference curve.
 *
 * PARAMETERS:
 *     n_cells_recovered [in]  Number of cells recovered
 *
 * RETURNS:
 *     Estimated multiplet rate (approximately 0.8% per 1000 cells)
 * -------------------------------------------------------------------------- */
Real multiplet_rate_10x(Index n_cells_recovered);

/* -----------------------------------------------------------------------------
 * FUNCTION: get_singlet_indices
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Extract indices of cells that are not doublets.
 *
 * PARAMETERS:
 *     is_doublet      [in]  Boolean doublet calls
 *     singlet_indices [out] Output indices of singlets
 *
 * RETURNS:
 *     Number of singlets (indices written)
 *
 * PRECONDITIONS:
 *     - singlet_indices.len >= is_doublet.len
 *
 * POSTCONDITIONS:
 *     - singlet_indices[0..return_value] contains indices where is_doublet[i] = false
 * -------------------------------------------------------------------------- */
Index get_singlet_indices(
    Array<const bool> is_doublet,      // Doublet calls [n]
    Array<Index> singlet_indices        // Output indices [n]
);

} // namespace scl::kernel::doublet
