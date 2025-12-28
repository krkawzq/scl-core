// =============================================================================
// FILE: scl/kernel/annotation.h
// BRIEF: API reference for cell type annotation from reference
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::annotation {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_CONFIDENCE_THRESHOLD = Real(0.5);
    constexpr Real EPSILON = Real(1e-15);
    constexpr Index DEFAULT_K = 15;
    constexpr Real DEFAULT_NOVELTY_THRESHOLD = Real(0.3);
    constexpr Size PARALLEL_THRESHOLD = 500;
}

// =============================================================================
// Annotation Method Types
// =============================================================================

enum class AnnotationMethod {
    KNNVoting,      // K-nearest neighbor voting
    Correlation,    // Correlation with reference profiles
    MarkerScore,    // Marker gene scoring
    Weighted        // Weighted combination
};

enum class DistanceMetric {
    Cosine,
    Euclidean,
    Correlation,
    Manhattan
};

// =============================================================================
// Reference Mapping
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: count_cell_types
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Count number of distinct cell types in label array.
 *
 * PARAMETERS:
 *     labels [in]  Cell type labels [n]
 *     n      [in]  Number of cells
 *
 * PRECONDITIONS:
 *     - All label indices are non-negative
 *
 * POSTCONDITIONS:
 *     - Returns number of distinct types (max_label + 1)
 *
 * COMPLEXITY:
 *     Time:  O(n)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
Index count_cell_types(
    Array<const Index> labels,              // Cell type labels [n]
    Index n                                  // Number of cells
);

/* -----------------------------------------------------------------------------
 * FUNCTION: reference_mapping
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Annotate query cells using KNN voting from reference dataset.
 *
 * PARAMETERS:
 *     query_expression        [in]  Query expression matrix (cells x genes, CSR)
 *     reference_expression   [in]  Reference expression matrix (cells x genes, CSR)
 *     reference_labels       [in]  Reference cell type labels [n_ref]
 *     query_to_ref_neighbors [in]  KNN graph from query to reference (CSR)
 *     n_query                [in]  Number of query cells
 *     n_ref                  [in]  Number of reference cells
 *     n_types                [in]  Number of cell types
 *     query_labels           [out] Annotated query labels [n_query]
 *     confidence_scores      [out] Confidence scores [n_query]
 *
 * PRECONDITIONS:
 *     - query_labels has capacity >= n_query
 *     - confidence_scores has capacity >= n_query
 *     - KNN graph connects query cells to reference cells
 *
 * POSTCONDITIONS:
 *     - query_labels[i] contains assigned cell type for query cell i
 *     - confidence_scores[i] contains voting confidence
 *
 * COMPLEXITY:
 *     Time:  O(n_query * k * n_genes)
 *     Space: O(k * n_types) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over query cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void reference_mapping(
    const Sparse<T, IsCSR>& query_expression,        // Query expression [n_query x n_genes]
    const Sparse<T, IsCSR>& reference_expression,    // Reference expression [n_ref x n_genes]
    Array<const Index> reference_labels,             // Reference labels [n_ref]
    const Sparse<Index, IsCSR>& query_to_ref_neighbors, // KNN graph
    Index n_query,                                   // Number of query cells
    Index n_ref,                                     // Number of reference cells
    Index n_types,                                   // Number of cell types
    Array<Index> query_labels,                       // Output query labels [n_query]
    Array<Real> confidence_scores                     // Output confidence [n_query]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: correlation_assignment
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Assign cell types using correlation with reference profiles (SingleR-style).
 *
 * PARAMETERS:
 *     query_expression    [in]  Query expression matrix (cells x genes, CSR)
 *     reference_profiles  [in]  Reference type profiles (types x genes, CSR)
 *     n_query             [in]  Number of query cells
 *     n_types             [in]  Number of cell types
 *     n_genes             [in]  Number of genes
 *     assigned_labels     [out] Assigned labels [n_query]
 *     correlation_scores  [out] Best correlation scores [n_query]
 *     all_correlations    [out] Optional: all correlations [n_query * n_types]
 *
 * PRECONDITIONS:
 *     - assigned_labels has capacity >= n_query
 *     - correlation_scores has capacity >= n_query
 *
 * POSTCONDITIONS:
 *     - assigned_labels[i] contains type with highest correlation
 *     - correlation_scores[i] contains best correlation value
 *
 * COMPLEXITY:
 *     Time:  O(n_query * n_types * n_genes)
 *     Space: O(n_types) auxiliary per query cell
 *
 * THREAD SAFETY:
 *     Safe - parallelized over query cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void correlation_assignment(
    const Sparse<T, IsCSR>& query_expression,       // Query expression [n_query x n_genes]
    const Sparse<T, IsCSR>& reference_profiles,      // Reference profiles [n_types x n_genes]
    Index n_query,                                   // Number of query cells
    Index n_types,                                   // Number of cell types
    Index n_genes,                                   // Number of genes
    Array<Index> assigned_labels,                    // Output assigned labels [n_query]
    Array<Real> correlation_scores,                   // Output best correlations [n_query]
    Array<Real> all_correlations                     // Optional: all correlations [n_query * n_types]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: build_reference_profiles
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Build average expression profiles for each cell type in reference.
 *
 * PARAMETERS:
 *     reference_expression [in]  Reference expression matrix (cells x genes, CSR)
 *     reference_labels     [in]  Reference cell type labels [n_ref]
 *     n_ref                [in]  Number of reference cells
 *     n_types              [in]  Number of cell types
 *     n_genes              [in]  Number of genes
 *     profiles             [out] Type profiles [n_types * n_genes]
 *
 * PRECONDITIONS:
 *     - profiles has capacity >= n_types * n_genes
 *
 * POSTCONDITIONS:
 *     - profiles[t * n_genes + g] contains mean expression of type t for gene g
 *
 * COMPLEXITY:
 *     Time:  O(nnz_ref)
 *     Space: O(n_types) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over types
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void build_reference_profiles(
    const Sparse<T, IsCSR>& reference_expression,    // Reference expression [n_ref x n_genes]
    Array<const Index> reference_labels,             // Reference labels [n_ref]
    Index n_ref,                                     // Number of reference cells
    Index n_types,                                   // Number of cell types
    Index n_genes,                                   // Number of genes
    Real* profiles                                    // Output profiles [n_types * n_genes]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: marker_gene_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Score cells using marker gene expression (scType-style).
 *
 * PARAMETERS:
 *     expression      [in]  Expression matrix (cells x genes, CSR)
 *     marker_genes    [in]  Array of marker gene arrays per type [n_types]
 *     marker_counts   [in]  Number of markers per type [n_types]
 *     n_cells         [in]  Number of cells
 *     n_genes         [in]  Number of genes
 *     n_types         [in]  Number of cell types
 *     scores          [out] Marker scores [n_cells * n_types]
 *     normalize       [in]  If true, normalize scores per cell
 *
 * PRECONDITIONS:
 *     - scores has capacity >= n_cells * n_types
 *
 * POSTCONDITIONS:
 *     - scores[i * n_types + t] contains marker score for cell i and type t
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * sum(marker_counts))
 *     Space: O(n_genes) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void marker_gene_score(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    const Index* const* marker_genes,        // Marker gene arrays [n_types]
    const Index* marker_counts,              // Marker counts [n_types]
    Index n_cells,                           // Number of cells
    Index n_genes,                           // Number of genes
    Index n_types,                           // Number of cell types
    Real* scores,                            // Output scores [n_cells * n_types]
    bool normalize = true                    // Normalize scores
);

/* -----------------------------------------------------------------------------
 * FUNCTION: assign_from_scores
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Assign cell types from score matrix by selecting maximum.
 *
 * PARAMETERS:
 *     scores    [in]  Score matrix [n_cells * n_types]
 *     n_cells   [in]  Number of cells
 *     n_types   [in]  Number of cell types
 *     labels    [out] Assigned labels [n_cells]
 *     confidence [out] Confidence scores [n_cells]
 *
 * PRECONDITIONS:
 *     - labels has capacity >= n_cells
 *     - confidence has capacity >= n_cells
 *
 * POSTCONDITIONS:
 *     - labels[i] contains type with highest score
 *     - confidence[i] contains normalized max score
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * n_types)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
void assign_from_scores(
    const Real* scores,                      // Score matrix [n_cells * n_types]
    Index n_cells,                           // Number of cells
    Index n_types,                           // Number of cell types
    Array<Index> labels,                     // Output labels [n_cells]
    Array<Real> confidence                    // Output confidence [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: consensus_annotation
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Combine predictions from multiple annotation methods.
 *
 * PARAMETERS:
 *     predictions       [in]  Array of prediction arrays [n_methods]
 *     confidences       [in]  Optional confidence arrays [n_methods]
 *     n_methods         [in]  Number of methods
 *     n_cells           [in]  Number of cells
 *     n_types           [in]  Number of cell types
 *     consensus_labels  [out] Consensus labels [n_cells]
 *     consensus_confidence [out] Consensus confidence [n_cells]
 *
 * PRECONDITIONS:
 *     - All prediction arrays have length n_cells
 *
 * POSTCONDITIONS:
 *     - consensus_labels contains majority vote or weighted vote
 *     - consensus_confidence contains agreement measure
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * n_methods * n_types)
 *     Space: O(n_types) auxiliary per cell
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
void consensus_annotation(
    const Index* const* predictions,        // Prediction arrays [n_methods][n_cells]
    const Real* const* confidences,          // Optional confidence arrays [n_methods][n_cells]
    Index n_methods,                        // Number of methods
    Index n_cells,                          // Number of cells
    Index n_types,                          // Number of cell types
    Array<Index> consensus_labels,           // Output consensus labels [n_cells]
    Array<Real> consensus_confidence         // Output consensus confidence [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: detect_novel_cell_types
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Detect cells that do not match any reference type well.
 *
 * PARAMETERS:
 *     query_expression      [in]  Query expression matrix (cells x genes, CSR)
 *     reference_profiles    [in]  Reference type profiles [n_types * n_genes]
 *     assigned_labels       [in]  Previously assigned labels [n_query]
 *     n_query               [in]  Number of query cells
 *     n_types               [in]  Number of cell types
 *     n_genes               [in]  Number of genes
 *     is_novel              [out] Novelty flags [n_query]
 *     distance_threshold    [in]  Distance threshold for novelty
 *     distance_to_assigned  [out] Optional: distances to assigned type [n_query]
 *
 * PRECONDITIONS:
 *     - is_novel has capacity >= n_query
 *
 * POSTCONDITIONS:
 *     - is_novel[i] == 1 if cell i is novel
 *     - Novel cells have low similarity to assigned type profile
 *
 * COMPLEXITY:
 *     Time:  O(n_query * n_genes)
 *     Space: O(n_genes) auxiliary per query cell
 *
 * THREAD SAFETY:
 *     Safe - parallelized over query cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void detect_novel_cell_types(
    const Sparse<T, IsCSR>& query_expression,       // Query expression [n_query x n_genes]
    const Real* reference_profiles,                  // Reference profiles [n_types * n_genes]
    Array<const Index> assigned_labels,              // Assigned labels [n_query]
    Index n_query,                                   // Number of query cells
    Index n_types,                                   // Number of cell types
    Index n_genes,                                   // Number of genes
    Array<Byte> is_novel,                            // Output novelty flags [n_query]
    Real distance_threshold = config::DEFAULT_NOVELTY_THRESHOLD, // Distance threshold
    Array<Real> distance_to_assigned                 // Optional: distances [n_query]
);

} // namespace scl::kernel::annotation

