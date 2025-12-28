// =============================================================================
// FILE: scl/kernel/subpopulation.h
// BRIEF: API reference for subpopulation analysis and cluster refinement
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::subpopulation {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size MIN_CLUSTER_SIZE = 10;
    constexpr Size DEFAULT_K = 5;
    constexpr Size MAX_ITERATIONS = 100;
    constexpr Size DEFAULT_BOOTSTRAP = 100;
}

// =============================================================================
// Subpopulation Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: recursive_subclustering
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Perform recursive sub-clustering within clusters.
 *
 * PARAMETERS:
 *     expression    [in]  Expression matrix (cells x genes, CSR)
 *     cluster_labels [in]  Initial cluster labels [n_cells]
 *     n_cells       [in]  Number of cells
 *     subcluster_labels [out] Subcluster labels [n_cells]
 *     max_depth     [in]  Maximum recursion depth
 *     min_size      [in]  Minimum cluster size for splitting
 *
 * PRECONDITIONS:
 *     - subcluster_labels has capacity >= n_cells
 *
 * POSTCONDITIONS:
 *     - subcluster_labels contains refined subcluster assignments
 *
 * COMPLEXITY:
 *     Time:  O(max_depth * n_cells * log(n_cells))
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - recursive algorithm
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void recursive_subclustering(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    Array<const Index> cluster_labels,      // Initial cluster labels [n_cells]
    Index n_cells,                           // Number of cells
    Array<Index> subcluster_labels,           // Output subcluster labels [n_cells]
    Index max_depth = 3,                      // Maximum depth
    Size min_size = config::MIN_CLUSTER_SIZE  // Minimum size
);

/* -----------------------------------------------------------------------------
 * FUNCTION: cluster_stability
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Assess cluster stability using bootstrap resampling.
 *
 * PARAMETERS:
 *     expression    [in]  Expression matrix (cells x genes, CSR)
 *     cluster_labels [in]  Cluster labels [n_cells]
 *     n_cells       [in]  Number of cells
 *     stability_scores [out] Stability scores [n_clusters]
 *     n_bootstrap   [in]  Number of bootstrap iterations
 *     seed          [in]  Random seed
 *
 * PRECONDITIONS:
 *     - stability_scores has capacity >= n_clusters
 *
 * POSTCONDITIONS:
 *     - stability_scores[c] contains stability score for cluster c
 *
 * COMPLEXITY:
 *     Time:  O(n_bootstrap * n_cells * log(n_cells))
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over bootstrap iterations
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void cluster_stability(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    Array<const Index> cluster_labels,      // Cluster labels [n_cells]
    Index n_cells,                          // Number of cells
    Array<Real> stability_scores,            // Output stability [n_clusters]
    Size n_bootstrap = config::DEFAULT_BOOTSTRAP, // Bootstrap iterations
    uint64_t seed = 42                       // Random seed
);

} // namespace scl::kernel::subpopulation

