// =============================================================================
// FILE: scl/kernel/sampling.h
// BRIEF: API reference for sampling and downsampling kernels
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::sampling {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size DEFAULT_BINS = 64;
    constexpr Size MAX_ITERATIONS = 1000;
    constexpr Real CONVERGENCE_TOL = Real(1e-6);
    constexpr Size PARALLEL_THRESHOLD = 256;
}

// =============================================================================
// Geometric Sketching
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: geometric_sketching
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Sample cells using geometric sketching to preserve rare populations
 *     by ensuring uniform coverage of the data manifold.
 *
 * PARAMETERS:
 *     data             [in]  Expression matrix (cells x genes, CSR)
 *     target_size      [in]  Desired number of cells to select
 *     selected_indices [out] Indices of selected cells
 *     n_selected       [out] Actual number of cells selected
 *     seed             [in]  Random seed for reproducibility
 *
 * PRECONDITIONS:
 *     - selected_indices has capacity >= min(target_size, data.rows())
 *     - target_size > 0
 *
 * POSTCONDITIONS:
 *     - n_selected <= target_size
 *     - selected_indices[0..n_selected) contains selected cell indices
 *     - Cells are sampled uniformly from geometric grid buckets
 *
 * ALGORITHM:
 *     1. Compute data bounds for each feature
 *     2. Create grid with DEFAULT_BINS bins per dimension
 *     3. Assign each cell to a grid bucket via hash
 *     4. Sort cells by bucket using VQSort
 *     5. Sample proportionally from each bucket
 *
 * COMPLEXITY:
 *     Time:  O(n * d + n log n) where n = cells, d = features
 *     Space: O(n + d) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 *
 * NUMERICAL NOTES:
 *     Uses Xoshiro128+ PRNG for high-quality randomness
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void geometric_sketching(
    const Sparse<T, IsCSR>& data,        // Expression matrix [n_cells x n_genes]
    Size target_size,                     // Desired sample size
    Index* selected_indices,              // Output indices [target_size]
    Size& n_selected,                     // Output actual count
    uint64_t seed = 42                    // Random seed
);

// =============================================================================
// Density-Preserving Sampling
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: density_preserving
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Sample cells while preserving local density distribution by weighting
 *     samples inversely proportional to neighborhood density.
 *
 * PARAMETERS:
 *     data             [in]  Expression matrix
 *     neighbors        [in]  KNN graph (CSR)
 *     target_size      [in]  Desired number of cells
 *     selected_indices [out] Indices of selected cells
 *     n_selected       [out] Actual number selected
 *
 * PRECONDITIONS:
 *     - data.rows() == neighbors.rows()
 *     - selected_indices has capacity >= min(target_size, data.rows())
 *
 * POSTCONDITIONS:
 *     - Cells from sparse regions are more likely to be selected
 *     - Local density distribution is preserved in sample
 *
 * ALGORITHM:
 *     1. Compute local density for each cell from neighbor count
 *     2. Compute sampling weights as inverse density
 *     3. Normalize weights to sum to 1
 *     4. Use systematic sampling with normalized weights
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(n) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void density_preserving(
    const Sparse<T, IsCSR>& data,           // Expression matrix
    const Sparse<Index, IsCSR>& neighbors,  // KNN graph
    Size target_size,                        // Desired sample size
    Index* selected_indices,                 // Output indices
    Size& n_selected                         // Output count
);

// =============================================================================
// Landmark Selection
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: landmark_selection
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Select diverse landmark cells using KMeans++ initialization to ensure
 *     good coverage of the data space.
 *
 * PARAMETERS:
 *     data             [in]  Expression matrix
 *     n_landmarks      [in]  Number of landmarks to select
 *     landmark_indices [out] Indices of selected landmarks
 *     n_selected       [out] Actual number selected
 *     seed             [in]  Random seed
 *
 * PRECONDITIONS:
 *     - landmark_indices has capacity >= min(n_landmarks, data.rows())
 *
 * POSTCONDITIONS:
 *     - n_selected = min(n_landmarks, data.rows())
 *     - Landmarks are maximally spread in expression space
 *
 * ALGORITHM:
 *     KMeans++ initialization:
 *         1. Select first center uniformly at random
 *         2. For each subsequent center:
 *            a. Compute squared distance to nearest existing center
 *            b. Sample proportional to squared distance
 *         3. Repeat until n_landmarks selected
 *
 * COMPLEXITY:
 *     Time:  O(n_landmarks * n * d) for sparse distance computation
 *     Space: O(n) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential KMeans++
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void landmark_selection(
    const Sparse<T, IsCSR>& data,        // Expression matrix
    Size n_landmarks,                     // Number of landmarks
    Index* landmark_indices,              // Output indices
    Size& n_selected,                     // Output count
    uint64_t seed = 42                    // Random seed
);

// =============================================================================
// Representative Cells
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: representative_cells
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Select representative cells from each cluster, choosing cells closest
 *     to cluster centroids.
 *
 * PARAMETERS:
 *     data            [in]  Expression matrix
 *     cluster_labels  [in]  Cluster assignment for each cell
 *     per_cluster     [in]  Number of representatives per cluster
 *     representatives [out] Indices of representative cells
 *     n_selected      [out] Total representatives selected
 *     seed            [in]  Random seed (unused in current implementation)
 *
 * PRECONDITIONS:
 *     - data.rows() == cluster_labels.len
 *     - representatives has sufficient capacity
 *
 * POSTCONDITIONS:
 *     - n_selected = sum(min(per_cluster, cluster_size)) over clusters
 *     - Representatives are closest cells to each cluster centroid
 *
 * ALGORITHM:
 *     For each cluster:
 *         1. Compute centroid as mean of all cells in cluster
 *         2. Compute squared distance from each cell to centroid
 *         3. Use partial_sort to find closest per_cluster cells
 *         4. Add to representatives list
 *
 * COMPLEXITY:
 *     Time:  O(n * d + n_clusters * cluster_size * per_cluster)
 *     Space: O(n + d * n_clusters) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void representative_cells(
    const Sparse<T, IsCSR>& data,        // Expression matrix
    Array<const Index> cluster_labels,    // Cluster assignments [n_cells]
    Size per_cluster,                     // Representatives per cluster
    Index* representatives,               // Output indices
    Size& n_selected,                     // Output total count
    uint64_t seed = 42                    // Random seed
);

// =============================================================================
// Balanced Sampling
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: balanced_sampling
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Sample equal numbers from each group/label category.
 *
 * PARAMETERS:
 *     labels           [in]  Group labels for each element
 *     target_size      [in]  Total desired sample size
 *     selected_indices [out] Indices of selected elements
 *     n_selected       [out] Actual number selected
 *     seed             [in]  Random seed
 *
 * PRECONDITIONS:
 *     - selected_indices has capacity >= target_size
 *     - Labels are non-negative integers
 *
 * POSTCONDITIONS:
 *     - Each non-empty group contributes roughly target_size / n_groups samples
 *     - Remainder distributed to first groups
 *
 * ALGORITHM:
 *     1. Count elements per group
 *     2. Calculate per_group = target_size / n_non_empty_groups
 *     3. For each group: shuffle and take first min(per_group, group_size)
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(n) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
void balanced_sampling(
    Array<const Index> labels,           // Group labels [n]
    Size target_size,                     // Desired sample size
    Index* selected_indices,              // Output indices
    Size& n_selected,                     // Output count
    uint64_t seed = 42                    // Random seed
);

// =============================================================================
// Stratified Sampling
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: stratified_sampling
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Sample from strata defined by binning a continuous variable.
 *
 * PARAMETERS:
 *     values           [in]  Continuous values to stratify by
 *     n_strata         [in]  Number of strata to create
 *     target_size      [in]  Total desired sample size
 *     selected_indices [out] Indices of selected elements
 *     n_selected       [out] Actual number selected
 *     seed             [in]  Random seed
 *
 * PRECONDITIONS:
 *     - values.len > 0
 *     - n_strata > 0
 *
 * POSTCONDITIONS:
 *     - Elements are binned into n_strata equal-width strata
 *     - balanced_sampling is applied to strata labels
 *
 * ALGORITHM:
 *     1. Find min and max of values
 *     2. Create n_strata equal-width bins
 *     3. Assign each element to a stratum
 *     4. Use balanced_sampling with strata as labels
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(n) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
void stratified_sampling(
    Array<const Real> values,            // Values to stratify [n]
    Size n_strata,                        // Number of strata
    Size target_size,                     // Desired sample size
    Index* selected_indices,              // Output indices
    Size& n_selected,                     // Output count
    uint64_t seed = 42                    // Random seed
);

// =============================================================================
// Uniform Random Sampling
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: uniform_sampling
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Simple uniform random sampling without replacement.
 *
 * PARAMETERS:
 *     n                [in]  Total population size
 *     target_size      [in]  Desired sample size
 *     selected_indices [out] Indices of selected elements
 *     n_selected       [out] Actual number selected
 *     seed             [in]  Random seed
 *
 * PRECONDITIONS:
 *     - selected_indices has capacity >= min(target_size, n)
 *
 * POSTCONDITIONS:
 *     - n_selected = min(target_size, n)
 *     - Each element has equal probability of selection
 *
 * ALGORITHM:
 *     Fisher-Yates partial shuffle:
 *         1. Initialize indices [0, n)
 *         2. For i in [0, target_size):
 *            a. Swap indices[i] with random indices[j] where j >= i
 *            b. Output indices[i]
 *
 * COMPLEXITY:
 *     Time:  O(n) for initialization, O(target_size) for sampling
 *     Space: O(n) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
void uniform_sampling(
    Size n,                               // Population size
    Size target_size,                     // Desired sample size
    Index* selected_indices,              // Output indices
    Size& n_selected,                     // Output count
    uint64_t seed = 42                    // Random seed
);

// =============================================================================
// Importance Sampling
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: importance_sampling
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Sample elements with probability proportional to given weights
 *     (with replacement).
 *
 * PARAMETERS:
 *     weights          [in]  Sampling weights (non-negative)
 *     target_size      [in]  Number of samples to draw
 *     selected_indices [out] Indices of selected elements
 *     n_selected       [out] Actual number selected
 *     seed             [in]  Random seed
 *
 * PRECONDITIONS:
 *     - weights.len > 0
 *     - All weights >= 0
 *
 * POSTCONDITIONS:
 *     - n_selected = target_size
 *     - P(select i) proportional to weights[i]
 *     - Same element may appear multiple times (with replacement)
 *
 * ALGORITHM:
 *     1. Compute cumulative sum of normalized weights
 *     2. For each sample:
 *        a. Generate uniform random r in [0, 1)
 *        b. Binary search for smallest i where cumsum[i] >= r
 *
 * COMPLEXITY:
 *     Time:  O(n + target_size * log n)
 *     Space: O(n) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 *
 * NUMERICAL NOTES:
 *     Falls back to uniform_sampling if total weight < EPSILON
 * -------------------------------------------------------------------------- */
void importance_sampling(
    Array<const Real> weights,           // Sampling weights [n]
    Size target_size,                     // Number of samples
    Index* selected_indices,              // Output indices
    Size& n_selected,                     // Output count
    uint64_t seed = 42                    // Random seed
);

// =============================================================================
// Reservoir Sampling
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: reservoir_sampling
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Select k items uniformly at random from a stream of n items using
 *     reservoir sampling (Algorithm R).
 *
 * PARAMETERS:
 *     stream_size    [in]  Total number of items in stream
 *     reservoir_size [in]  Number of items to select
 *     reservoir      [out] Indices of selected items
 *     n_selected     [out] Actual number selected
 *     seed           [in]  Random seed
 *
 * PRECONDITIONS:
 *     - reservoir has capacity >= min(reservoir_size, stream_size)
 *
 * POSTCONDITIONS:
 *     - n_selected = min(reservoir_size, stream_size)
 *     - Each item has equal probability of being in reservoir
 *
 * ALGORITHM:
 *     Algorithm R (Vitter 1985):
 *         1. Fill reservoir with first k items
 *         2. For each subsequent item i:
 *            a. Generate random j in [0, i]
 *            b. If j < k: replace reservoir[j] with i
 *
 * COMPLEXITY:
 *     Time:  O(stream_size)
 *     Space: O(reservoir_size)
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 *
 * NUMERICAL NOTES:
 *     Useful for streaming scenarios where n is unknown or very large
 * -------------------------------------------------------------------------- */
void reservoir_sampling(
    Size stream_size,                     // Total stream size
    Size reservoir_size,                  // Reservoir capacity
    Index* reservoir,                     // Output reservoir
    Size& n_selected,                     // Output count
    uint64_t seed = 42                    // Random seed
);

} // namespace scl::kernel::sampling
