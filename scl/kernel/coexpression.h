// =============================================================================
// FILE: scl/kernel/coexpression.h
// BRIEF: API reference for high-performance co-expression module detection (WGCNA-style)
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::coexpression {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_SOFT_POWER = Real(6);
    constexpr Real EPSILON = Real(1e-15);
    constexpr Index DEFAULT_MIN_MODULE_SIZE = 30;
    constexpr Index DEFAULT_DEEP_SPLIT = 2;
    constexpr Real DEFAULT_MERGE_CUT_HEIGHT = Real(0.25);
    constexpr Index MAX_ITERATIONS = 100;
    constexpr Size PARALLEL_THRESHOLD = 64;
    constexpr Size SIMD_THRESHOLD = 16;
}

// =============================================================================
// Co-expression Types
// =============================================================================

enum class CorrelationType {
    Pearson,
    Spearman,
    Bicor
};

enum class AdjacencyType {
    Unsigned,
    Signed,
    SignedHybrid
};

// =============================================================================
// Correlation Computation
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: correlation_matrix
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute pairwise correlation matrix for genes.
 *
 * PARAMETERS:
 *     expression   [in]  Expression matrix (cells x genes, CSR)
 *     n_cells      [in]  Number of cells
 *     n_genes      [in]  Number of genes
 *     corr_matrix  [out] Correlation matrix [n_genes * n_genes] (upper triangular)
 *     corr_type    [in]  Correlation type (Pearson, Spearman, Bicor)
 *
 * PRECONDITIONS:
 *     - corr_matrix has capacity >= n_genes * n_genes
 *
 * POSTCONDITIONS:
 *     - corr_matrix[i * n_genes + j] contains correlation between gene i and j
 *     - Matrix is symmetric
 *
 * COMPLEXITY:
 *     Time:  O(n_genes^2 * n_cells)
 *     Space: O(n_genes * n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over gene pairs
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void correlation_matrix(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    Index n_cells,                          // Number of cells
    Index n_genes,                          // Number of genes
    Real* corr_matrix,                      // Output correlation matrix [n_genes^2]
    CorrelationType corr_type = CorrelationType::Pearson // Correlation type
);

/* -----------------------------------------------------------------------------
 * FUNCTION: adjacency_matrix
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Convert correlation matrix to adjacency matrix using soft power.
 *
 * PARAMETERS:
 *     corr_matrix  [in]  Correlation matrix [n_genes * n_genes]
 *     n_genes      [in]  Number of genes
 *     adjacency    [out] Adjacency matrix [n_genes * n_genes]
 *     power        [in]  Soft power (default 6)
 *     adj_type     [in]  Adjacency type (Unsigned, Signed, SignedHybrid)
 *
 * PRECONDITIONS:
 *     - adjacency has capacity >= n_genes * n_genes
 *
 * POSTCONDITIONS:
 *     - adjacency[i * n_genes + j] contains adjacency value
 *     - Values are in [0, 1] for unsigned, [-1, 1] for signed
 *
 * COMPLEXITY:
 *     Time:  O(n_genes^2)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over matrix elements
 * -------------------------------------------------------------------------- */
void adjacency_matrix(
    const Real* corr_matrix,                // Correlation matrix [n_genes^2]
    Index n_genes,                          // Number of genes
    Real* adjacency,                          // Output adjacency matrix [n_genes^2]
    Real power = config::DEFAULT_SOFT_POWER,  // Soft power
    AdjacencyType adj_type = AdjacencyType::Unsigned // Adjacency type
);

/* -----------------------------------------------------------------------------
 * FUNCTION: topological_overlap_matrix
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Topological Overlap Matrix (TOM) from adjacency matrix.
 *
 * PARAMETERS:
 *     adjacency    [in]  Adjacency matrix [n_genes * n_genes]
 *     n_genes      [in]  Number of genes
 *     tom          [out] TOM matrix [n_genes * n_genes]
 *
 * PRECONDITIONS:
 *     - tom has capacity >= n_genes * n_genes
 *
 * POSTCONDITIONS:
 *     - tom[i * n_genes + j] contains TOM value
 *     - TOM measures shared neighbors between genes
 *
 * COMPLEXITY:
 *     Time:  O(n_genes^3)
 *     Space: O(n_genes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over gene pairs
 * -------------------------------------------------------------------------- */
void topological_overlap_matrix(
    const Real* adjacency,                   // Adjacency matrix [n_genes^2]
    Index n_genes,                           // Number of genes
    Real* tom                                 // Output TOM matrix [n_genes^2]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: tom_dissimilarity
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Convert TOM matrix to dissimilarity matrix for clustering.
 *
 * PARAMETERS:
 *     tom      [in]  TOM matrix [n_genes * n_genes]
 *     n_genes  [in]  Number of genes
 *     dissim   [out] Dissimilarity matrix [n_genes * n_genes]
 *
 * PRECONDITIONS:
 *     - dissim has capacity >= n_genes * n_genes
 *
 * POSTCONDITIONS:
 *     - dissim[i * n_genes + j] = 1 - tom[i * n_genes + j]
 *
 * COMPLEXITY:
 *     Time:  O(n_genes^2)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over matrix elements
 * -------------------------------------------------------------------------- */
void tom_dissimilarity(
    const Real* tom,                         // TOM matrix [n_genes^2]
    Index n_genes,                           // Number of genes
    Real* dissim                              // Output dissimilarity [n_genes^2]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: hierarchical_clustering
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Perform hierarchical clustering on dissimilarity matrix.
 *
 * PARAMETERS:
 *     dissim       [in]  Dissimilarity matrix [n_genes * n_genes]
 *     n_genes      [in]  Number of genes
 *     merge_order  [out] Merge order [2 * (n_genes - 1)]
 *     merge_heights [out] Merge heights [n_genes - 1]
 *     dendrogram   [out] Optional dendrogram structure
 *
 * PRECONDITIONS:
 *     - merge_order has capacity >= 2 * (n_genes - 1)
 *     - merge_heights has capacity >= n_genes - 1
 *
 * POSTCONDITIONS:
 *     - merge_order contains pairs of merged clusters
 *     - merge_heights contains merge distances
 *
 * COMPLEXITY:
 *     Time:  O(n_genes^2 * log(n_genes))
 *     Space: O(n_genes^2) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential algorithm
 * -------------------------------------------------------------------------- */
void hierarchical_clustering(
    const Real* dissim,                      // Dissimilarity matrix [n_genes^2]
    Index n_genes,                           // Number of genes
    Index* merge_order,                      // Output merge order [2*(n_genes-1)]
    Real* merge_heights,                     // Output merge heights [n_genes-1]
    void* dendrogram = nullptr                // Optional dendrogram output
);

/* -----------------------------------------------------------------------------
 * FUNCTION: cut_tree
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Cut hierarchical clustering tree at specified height.
 *
 * PARAMETERS:
 *     merge_order     [in]  Merge order from hierarchical clustering [2*(n-1)]
 *     merge_heights   [in]  Merge heights [n-1]
 *     n_genes         [in]  Number of genes
 *     cut_height      [in]  Cut height threshold
 *     module_labels   [out] Module labels [n_genes]
 *
 * PRECONDITIONS:
 *     - module_labels has capacity >= n_genes
 *
 * POSTCONDITIONS:
 *     - module_labels[i] contains module ID for gene i
 *     - Returns number of modules
 *
 * COMPLEXITY:
 *     Time:  O(n_genes)
 *     Space: O(n_genes) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential algorithm
 * -------------------------------------------------------------------------- */
Index cut_tree(
    const Index* merge_order,                 // Merge order [2*(n_genes-1)]
    const Real* merge_heights,               // Merge heights [n_genes-1]
    Index n_genes,                           // Number of genes
    Real cut_height,                         // Cut height
    Index* module_labels                     // Output module labels [n_genes]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: detect_modules
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Detect co-expression modules from dissimilarity matrix.
 *
 * PARAMETERS:
 *     dissim           [in]  Dissimilarity matrix [n_genes * n_genes]
 *     n_genes          [in]  Number of genes
 *     module_labels    [out] Module labels [n_genes]
 *     min_module_size  [in]  Minimum module size
 *     merge_cut_height [in]  Merge cut height
 *
 * PRECONDITIONS:
 *     - module_labels has capacity >= n_genes
 *
 * POSTCONDITIONS:
 *     - module_labels[i] contains module ID for gene i
 *     - Returns number of modules detected
 *
 * COMPLEXITY:
 *     Time:  O(n_genes^2 * log(n_genes))
 *     Space: O(n_genes^2) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential clustering
 * -------------------------------------------------------------------------- */
Index detect_modules(
    const Real* dissim,                      // Dissimilarity matrix [n_genes^2]
    Index n_genes,                           // Number of genes
    Index* module_labels,                    // Output module labels [n_genes]
    Index min_module_size = config::DEFAULT_MIN_MODULE_SIZE, // Min module size
    Real merge_cut_height = config::DEFAULT_MERGE_CUT_HEIGHT // Merge threshold
);

/* -----------------------------------------------------------------------------
 * FUNCTION: module_eigengene
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute module eigengene (first principal component) for a module.
 *
 * PARAMETERS:
 *     expression    [in]  Expression matrix (cells x genes, CSR)
 *     module_labels [in]  Module labels [n_genes]
 *     module_id     [in]  Module ID to compute eigengene for
 *     n_cells       [in]  Number of cells
 *     n_genes       [in]  Number of genes
 *     eigengene     [out] Module eigengene [n_cells]
 *
 * PRECONDITIONS:
 *     - eigengene has capacity >= n_cells
 *
 * POSTCONDITIONS:
 *     - eigengene[i] contains first PC of module expression in cell i
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * n_module_genes)
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void module_eigengene(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    const Index* module_labels,              // Module labels [n_genes]
    Index module_id,                        // Module ID
    Index n_cells,                          // Number of cells
    Index n_genes,                          // Number of genes
    Array<Real> eigengene                    // Output eigengene [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: all_module_eigengenes
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute eigengenes for all modules.
 *
 * PARAMETERS:
 *     expression    [in]  Expression matrix (cells x genes, CSR)
 *     module_labels [in]  Module labels [n_genes]
 *     n_modules     [in]  Number of modules
 *     n_cells       [in]  Number of cells
 *     n_genes       [in]  Number of genes
 *     eigengenes    [out] Eigengene matrix [n_cells * n_modules]
 *
 * PRECONDITIONS:
 *     - eigengenes has capacity >= n_cells * n_modules
 *
 * POSTCONDITIONS:
 *     - eigengenes[i * n_modules + m] contains eigengene of module m in cell i
 *
 * COMPLEXITY:
 *     Time:  O(n_modules * n_cells * avg_module_size)
 *     Space: O(n_cells) auxiliary per module
 *
 * THREAD SAFETY:
 *     Safe - parallelized over modules
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void all_module_eigengenes(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    const Index* module_labels,              // Module labels [n_genes]
    Index n_modules,                         // Number of modules
    Index n_cells,                           // Number of cells
    Index n_genes,                           // Number of genes
    Real* eigengenes                         // Output eigengenes [n_cells * n_modules]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: module_trait_correlation
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute correlation between module eigengenes and traits.
 *
 * PARAMETERS:
 *     eigengenes   [in]  Module eigengenes [n_samples * n_modules]
 *     traits       [in]  Trait values [n_samples * n_traits]
 *     n_samples    [in]  Number of samples
 *     n_modules    [in]  Number of modules
 *     n_traits     [in]  Number of traits
 *     correlations [out] Correlation matrix [n_modules * n_traits]
 *     p_values     [out] Optional p-values [n_modules * n_traits]
 *
 * PRECONDITIONS:
 *     - correlations has capacity >= n_modules * n_traits
 *
 * POSTCONDITIONS:
 *     - correlations[m * n_traits + t] contains correlation
 *
 * COMPLEXITY:
 *     Time:  O(n_modules * n_traits * n_samples)
 *     Space: O(n_samples) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over module-trait pairs
 * -------------------------------------------------------------------------- */
void module_trait_correlation(
    const Real* eigengenes,                  // Module eigengenes [n_samples * n_modules]
    const Real* traits,                      // Trait values [n_samples * n_traits]
    Index n_samples,                         // Number of samples
    Index n_modules,                         // Number of modules
    Index n_traits,                          // Number of traits
    Real* correlations,                      // Output correlations [n_modules * n_traits]
    Real* p_values = nullptr                 // Optional p-values [n_modules * n_traits]
);

} // namespace scl::kernel::coexpression

