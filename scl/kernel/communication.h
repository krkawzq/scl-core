// =============================================================================
// FILE: scl/kernel/communication.h
// BRIEF: API reference for cell-cell communication analysis (CellChat/CellPhoneDB-style)
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::communication {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Index DEFAULT_N_PERM = 1000;
    constexpr Real DEFAULT_PVAL_THRESHOLD = Real(0.05);
    constexpr Real EPSILON = Real(1e-15);
    constexpr Real MIN_EXPRESSION = Real(0.1);
    constexpr Real MIN_PERCENT_EXPRESSED = Real(0.1);
    constexpr Size PARALLEL_THRESHOLD = 256;
    constexpr Size EARLY_STOP_CHECK = 100;
}

// =============================================================================
// Communication Score Types
// =============================================================================

enum class ScoreMethod {
    MeanProduct,    // Mean ligand * mean receptor
    GeometricMean,  // sqrt(mean_l * mean_r)
    MinMean,        // min(mean_l, mean_r)
    Product,        // Direct product
    Natmi           // NATMI-style scoring
};

// =============================================================================
// L-R Pair Scoring
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: lr_score_matrix
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute ligand-receptor interaction scores between all cell type pairs.
 *
 * PARAMETERS:
 *     expression       [in]  Expression matrix (cells x genes, CSR)
 *     cell_type_labels [in]  Cell type labels [n_cells]
 *     ligand_gene      [in]  Ligand gene index
 *     receptor_gene    [in]  Receptor gene index
 *     n_cells          [in]  Number of cells
 *     n_types          [in]  Number of cell types
 *     score_matrix     [out] Score matrix [n_types * n_types]
 *     method           [in]  Scoring method
 *
 * PRECONDITIONS:
 *     - score_matrix has capacity >= n_types * n_types
 *     - All cell type labels are valid
 *
 * POSTCONDITIONS:
 *     - score_matrix[s * n_types + r] contains score for sender s and receiver r
 *
 * COMPLEXITY:
 *     Time:  O(n_cells + n_types^2)
 *     Space: O(n_cells + n_types) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over pairs
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void lr_score_matrix(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    Array<const Index> cell_type_labels,     // Cell type labels [n_cells]
    Index ligand_gene,                        // Ligand gene index
    Index receptor_gene,                      // Receptor gene index
    Index n_cells,                            // Number of cells
    Index n_types,                           // Number of cell types
    Real* score_matrix,                       // Output scores [n_types * n_types]
    ScoreMethod method = ScoreMethod::MeanProduct // Scoring method
);

/* -----------------------------------------------------------------------------
 * FUNCTION: lr_score_with_permutation
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute L-R score with permutation-based p-value.
 *
 * PARAMETERS:
 *     expression       [in]  Expression matrix (cells x genes, CSR)
 *     cell_type_labels [in]  Cell type labels [n_cells]
 *     ligand_gene      [in]  Ligand gene index
 *     receptor_gene    [in]  Receptor gene index
 *     sender_type      [in]  Sender cell type index
 *     receiver_type    [in]  Receiver cell type index
 *     n_cells          [in]  Number of cells
 *     observed_score   [out] Observed L-R score
 *     p_value          [out] Permutation p-value
 *     n_permutations   [in]  Number of permutations
 *     method           [in]  Scoring method
 *     seed             [in]  Random seed
 *
 * PRECONDITIONS:
 *     - All type indices are valid
 *
 * POSTCONDITIONS:
 *     - observed_score contains observed interaction strength
 *     - p_value contains permutation-based significance
 *
 * COMPLEXITY:
 *     Time:  O(n_permutations * n_cells)
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized with thread-local RNG
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void lr_score_with_permutation(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    Array<const Index> cell_type_labels,     // Cell type labels [n_cells]
    Index ligand_gene,                        // Ligand gene index
    Index receptor_gene,                      // Receptor gene index
    Index sender_type,                        // Sender cell type
    Index receiver_type,                      // Receiver cell type
    Index n_cells,                            // Number of cells
    Real& observed_score,                     // Output observed score
    Real& p_value,                           // Output p-value
    Index n_permutations = config::DEFAULT_N_PERM, // Number of permutations
    ScoreMethod method = ScoreMethod::MeanProduct, // Scoring method
    uint64_t seed = 42                        // Random seed
);

/* -----------------------------------------------------------------------------
 * FUNCTION: batch_lr_scoring
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute L-R scores for multiple ligand-receptor pairs in parallel.
 *
 * PARAMETERS:
 *     expression       [in]  Expression matrix (cells x genes, CSR)
 *     cell_type_labels [in]  Cell type labels [n_cells]
 *     ligand_genes     [in]  Ligand gene indices [n_pairs]
 *     receptor_genes   [in]  Receptor gene indices [n_pairs]
 *     n_pairs          [in]  Number of L-R pairs
 *     n_cells          [in]  Number of cells
 *     n_types          [in]  Number of cell types
 *     scores           [out] Score matrices [n_pairs * n_types * n_types]
 *     method           [in]  Scoring method
 *
 * PRECONDITIONS:
 *     - scores has capacity >= n_pairs * n_types * n_types
 *
 * POSTCONDITIONS:
 *     - scores[p * n_types^2 + s * n_types + r] contains score for pair p
 *
 * COMPLEXITY:
 *     Time:  O(n_pairs * (n_cells + n_types^2))
 *     Space: O(n_cells * max_gene) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over pairs
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void batch_lr_scoring(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    Array<const Index> cell_type_labels,     // Cell type labels [n_cells]
    const Index* ligand_genes,               // Ligand gene indices [n_pairs]
    const Index* receptor_genes,              // Receptor gene indices [n_pairs]
    Index n_pairs,                           // Number of L-R pairs
    Index n_cells,                           // Number of cells
    Index n_types,                           // Number of cell types
    Real* scores,                            // Output scores [n_pairs * n_types^2]
    ScoreMethod method = ScoreMethod::MeanProduct // Scoring method
);

/* -----------------------------------------------------------------------------
 * FUNCTION: batch_lr_permutation_test
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute permutation p-values for multiple L-R pairs.
 *
 * PARAMETERS:
 *     expression       [in]  Expression matrix (cells x genes, CSR)
 *     cell_type_labels [in]  Cell type labels [n_cells]
 *     ligand_genes     [in]  Ligand gene indices [n_pairs]
 *     receptor_genes   [in]  Receptor gene indices [n_pairs]
 *     n_pairs          [in]  Number of L-R pairs
 *     n_cells          [in]  Number of cells
 *     n_types          [in]  Number of cell types
 *     scores           [out] Observed scores [n_pairs * n_types^2]
 *     p_values         [out] P-values [n_pairs * n_types^2]
 *     n_permutations   [in]  Number of permutations per pair
 *     method           [in]  Scoring method
 *     seed             [in]  Random seed
 *
 * PRECONDITIONS:
 *     - All output arrays have sufficient capacity
 *
 * POSTCONDITIONS:
 *     - scores and p_values computed for all pairs and type combinations
 *
 * COMPLEXITY:
 *     Time:  O(n_pairs * n_permutations * n_cells)
 *     Space: O(n_cells) auxiliary per thread
 *
 * THREAD SAFETY:
 *     Safe - parallelized over pairs
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void batch_lr_permutation_test(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    Array<const Index> cell_type_labels,     // Cell type labels [n_cells]
    const Index* ligand_genes,               // Ligand gene indices [n_pairs]
    const Index* receptor_genes,             // Receptor gene indices [n_pairs]
    Index n_pairs,                           // Number of L-R pairs
    Index n_cells,                           // Number of cells
    Index n_types,                           // Number of cell types
    Real* scores,                            // Output observed scores [n_pairs * n_types^2]
    Real* p_values,                          // Output p-values [n_pairs * n_types^2]
    Index n_permutations = config::DEFAULT_N_PERM, // Number of permutations
    ScoreMethod method = ScoreMethod::MeanProduct, // Scoring method
    uint64_t seed = 42                        // Random seed
);

/* -----------------------------------------------------------------------------
 * FUNCTION: spatial_communication_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute spatial communication scores using spatial graph.
 *
 * PARAMETERS:
 *     expression    [in]  Expression matrix (cells x genes, CSR)
 *     spatial_graph [in]  Spatial neighbor graph (CSR)
 *     ligand_gene   [in]  Ligand gene index
 *     receptor_gene [in]  Receptor gene index
 *     n_cells       [in]  Number of cells
 *     cell_scores   [out] Per-cell communication scores [n_cells]
 *
 * PRECONDITIONS:
 *     - cell_scores has capacity >= n_cells
 *     - spatial_graph represents spatial neighborhood
 *
 * POSTCONDITIONS:
 *     - cell_scores[i] contains weighted L-R interaction score for cell i
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * avg_neighbors)
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR, typename TG, bool IsCSR_G>
void spatial_communication_score(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    const Sparse<TG, IsCSR_G>& spatial_graph, // Spatial graph [n_cells x n_cells]
    Index ligand_gene,                       // Ligand gene index
    Index receptor_gene,                     // Receptor gene index
    Index n_cells,                           // Number of cells
    Real* cell_scores                        // Output scores [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: expression_specificity
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute expression specificity of a gene across cell types.
 *
 * PARAMETERS:
 *     expression       [in]  Expression matrix (cells x genes, CSR)
 *     cell_type_labels [in]  Cell type labels [n_cells]
 *     gene             [in]  Gene index
 *     n_cells          [in]  Number of cells
 *     n_types          [in]  Number of cell types
 *     specificity      [out] Specificity scores [n_types]
 *
 * PRECONDITIONS:
 *     - specificity has capacity >= n_types
 *
 * POSTCONDITIONS:
 *     - specificity[t] contains expression specificity for type t
 *
 * COMPLEXITY:
 *     Time:  O(n_cells)
 *     Space: O(n_cells + n_types) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void expression_specificity(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    Array<const Index> cell_type_labels,     // Cell type labels [n_cells]
    Index gene,                              // Gene index
    Index n_cells,                           // Number of cells
    Index n_types,                           // Number of cell types
    Real* specificity                        // Output specificity [n_types]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: filter_significant_interactions
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Filter significant L-R interactions by p-value threshold.
 *
 * PARAMETERS:
 *     p_values        [in]  P-value matrices [n_pairs * n_types^2]
 *     n_pairs         [in]  Number of L-R pairs
 *     n_types         [in]  Number of cell types
 *     p_threshold     [in]  P-value threshold
 *     pair_indices    [out] Pair indices of significant interactions [max_results]
 *     sender_types    [out] Sender type indices [max_results]
 *     receiver_types  [out] Receiver type indices [max_results]
 *     filtered_pvalues [out] Filtered p-values [max_results]
 *     max_results     [in]  Maximum number of results
 *
 * PRECONDITIONS:
 *     - All output arrays have capacity >= max_results
 *
 * POSTCONDITIONS:
 *     - Returns number of significant interactions found
 *     - Output arrays contain filtered results
 *
 * COMPLEXITY:
 *     Time:  O(n_pairs * n_types^2)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
Index filter_significant_interactions(
    const Real* p_values,                    // P-value matrices [n_pairs * n_types^2]
    Index n_pairs,                           // Number of L-R pairs
    Index n_types,                           // Number of cell types
    Real p_threshold,                        // P-value threshold
    Index* pair_indices,                     // Output pair indices [max_results]
    Index* sender_types,                     // Output sender types [max_results]
    Index* receiver_types,                   // Output receiver types [max_results]
    Real* filtered_pvalues,                 // Output p-values [max_results]
    Index max_results                        // Maximum results
);

/* -----------------------------------------------------------------------------
 * FUNCTION: aggregate_to_network
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Aggregate L-R scores into cell type communication network.
 *
 * PARAMETERS:
 *     scores         [in]  Score matrices [n_pairs * n_types^2]
 *     p_values       [in]  P-value matrices [n_pairs * n_types^2]
 *     n_pairs        [in]  Number of L-R pairs
 *     n_types        [in]  Number of cell types
 *     p_threshold    [in]  P-value threshold for filtering
 *     network_weights [out] Aggregated network weights [n_types^2]
 *     network_counts  [out] Number of significant interactions [n_types^2]
 *
 * PRECONDITIONS:
 *     - network_weights has capacity >= n_types^2
 *     - network_counts has capacity >= n_types^2
 *
 * POSTCONDITIONS:
 *     - network_weights[s * n_types + r] contains aggregated weight
 *     - network_counts[s * n_types + r] contains interaction count
 *
 * COMPLEXITY:
 *     Time:  O(n_pairs * n_types^2)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
void aggregate_to_network(
    const Real* scores,                      // Score matrices [n_pairs * n_types^2]
    const Real* p_values,                    // P-value matrices [n_pairs * n_types^2]
    Index n_pairs,                           // Number of L-R pairs
    Index n_types,                           // Number of cell types
    Real p_threshold,                        // P-value threshold
    Real* network_weights,                   // Output network weights [n_types^2]
    Index* network_counts                    // Output interaction counts [n_types^2]
);

} // namespace scl::kernel::communication

