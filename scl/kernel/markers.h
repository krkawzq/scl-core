// =============================================================================
// FILE: scl/kernel/markers.h
// BRIEF: API reference for marker gene selection and specificity scoring
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::markers {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_MIN_FC = Real(1.5);
    constexpr Real DEFAULT_MIN_PCT = Real(0.1);
    constexpr Real DEFAULT_MAX_PVAL = Real(0.05);
    constexpr Real MIN_EXPR = Real(1e-9);
    constexpr Real PSEUDO_COUNT = Real(1.0);
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size SIMD_THRESHOLD = 32;
}

// =============================================================================
// Ranking Methods
// =============================================================================

enum class RankingMethod {
    FoldChange,
    EffectSize,
    PValue,
    Combined
};

// =============================================================================
// Marker Selection
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: find_markers
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Find marker genes for each cluster using differential expression.
 *
 * PARAMETERS:
 *     expression    [in]  Expression matrix (cells x genes, CSR)
 *     cluster_labels [in]  Cluster labels [n_cells]
 *     n_cells       [in]  Number of cells
 *     n_genes       [in]  Number of genes
 *     n_clusters    [in]  Number of clusters
 *     marker_genes  [out] Marker gene indices [n_clusters * max_markers]
 *     marker_scores [out] Marker scores [n_clusters * max_markers]
 *     max_markers   [in]  Maximum markers per cluster
 *     min_fc        [in]  Minimum fold change
 *     max_pval      [in]  Maximum p-value
 *     method        [in]  Ranking method
 *
 * PRECONDITIONS:
 *     - marker_genes has capacity >= n_clusters * max_markers
 *     - marker_scores has capacity >= n_clusters * max_markers
 *
 * POSTCONDITIONS:
 *     - marker_genes[c * max_markers + i] contains marker gene index
 *     - Returns number of markers found per cluster
 *
 * COMPLEXITY:
 *     Time:  O(n_clusters * n_genes * n_cells)
 *     Space: O(n_cells) auxiliary per cluster
 *
 * THREAD SAFETY:
 *     Safe - parallelized over clusters
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void find_markers(
    const Sparse<T, IsCSR>& expression,      // Expression matrix [n_cells x n_genes]
    Array<const Index> cluster_labels,      // Cluster labels [n_cells]
    Index n_cells,                          // Number of cells
    Index n_genes,                          // Number of genes
    Index n_clusters,                       // Number of clusters
    Index* marker_genes,                     // Output marker genes [n_clusters * max_markers]
    Real* marker_scores,                     // Output marker scores [n_clusters * max_markers]
    Index max_markers,                      // Maximum markers per cluster
    Real min_fc = config::DEFAULT_MIN_FC,  // Minimum fold change
    Real max_pval = config::DEFAULT_MAX_PVAL, // Maximum p-value
    RankingMethod method = RankingMethod::Combined // Ranking method
);

/* -----------------------------------------------------------------------------
 * FUNCTION: specificity_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute gene specificity score for a cluster.
 *
 * PARAMETERS:
 *     expression    [in]  Expression matrix (cells x genes, CSR)
 *     cluster_labels [in]  Cluster labels [n_cells]
 *     gene_index    [in]  Gene index
 *     target_cluster [in]  Target cluster ID
 *     n_cells       [in]  Number of cells
 *     specificity   [out] Specificity score
 *
 * PRECONDITIONS:
 *     - target_cluster is valid cluster ID
 *
 * POSTCONDITIONS:
 *     - specificity contains cluster-specific expression score
 *
 * COMPLEXITY:
 *     Time:  O(n_cells)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void specificity_score(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    Array<const Index> cluster_labels,       // Cluster labels [n_cells]
    Index gene_index,                        // Gene index
    Index target_cluster,                    // Target cluster
    Index n_cells,                           // Number of cells
    Real& specificity                         // Output specificity score
);

} // namespace scl::kernel::markers

