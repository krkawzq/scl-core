// =============================================================================
// FILE: scl/kernel/scoring.h
// BRIEF: API reference for gene set scoring operations
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::scoring {

/* -----------------------------------------------------------------------------
 * ENUM: ScoringMethod
 * -----------------------------------------------------------------------------
 * VALUES:
 *     Mean         - Simple average of gene expression
 *     RankBased    - AUC-based score using expression ranks
 *     Weighted     - Weighted sum with user-provided weights
 *     SeuratModule - Seurat-style module score with control genes
 *     ZScore       - Z-score normalized average
 * -------------------------------------------------------------------------- */
enum class ScoringMethod {
    Mean,
    RankBased,
    Weighted,
    SeuratModule,
    ZScore
};

/* -----------------------------------------------------------------------------
 * ENUM: CellCyclePhase
 * -----------------------------------------------------------------------------
 * VALUES:
 *     G1  - Gap 1 phase (0)
 *     S   - Synthesis phase (1)
 *     G2M - G2/Mitosis phase (2)
 * -------------------------------------------------------------------------- */
enum class CellCyclePhase : Index {
    G1 = 0,
    S = 1,
    G2M = 2
};

/* -----------------------------------------------------------------------------
 * FUNCTION: compute_gene_means
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute mean expression for each gene across all cells.
 *
 * PARAMETERS:
 *     X          [in]  Sparse matrix (cells x genes or genes x cells)
 *     gene_means [out] Pre-allocated buffer for gene means
 *     n_cells    [in]  Number of cells
 *     n_genes    [in]  Number of genes
 *
 * PRECONDITIONS:
 *     - gene_means.len >= n_genes
 *     - Matrix dimensions match n_cells and n_genes
 *
 * POSTCONDITIONS:
 *     - gene_means[g] = mean expression of gene g across all cells
 *
 * ALGORITHM:
 *     CSR: Atomic accumulation across cells (parallel over cells)
 *     CSC: Direct column sum (parallel over genes)
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(n_genes) for atomic counters (CSR only)
 *
 * THREAD SAFETY:
 *     Safe - uses atomic operations for CSR format
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void compute_gene_means(
    const Sparse<T, IsCSR>& X,     // Input sparse matrix
    Array<Real> gene_means,        // Output gene means [n_genes]
    Index n_cells,                 // Number of cells
    Index n_genes                  // Number of genes
);

/* -----------------------------------------------------------------------------
 * FUNCTION: mean_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute mean expression score for each cell over a gene set.
 *
 * PARAMETERS:
 *     X        [in]  Sparse matrix (cells x genes)
 *     gene_set [in]  Array of gene indices in the set
 *     scores   [out] Output scores for each cell
 *     n_cells  [in]  Number of cells
 *     n_genes  [in]  Number of genes
 *
 * PRECONDITIONS:
 *     - scores.len >= n_cells
 *     - All gene indices in gene_set are in [0, n_genes)
 *
 * POSTCONDITIONS:
 *     - scores[c] = mean of X[c, g] for g in gene_set
 *
 * ALGORITHM:
 *     Uses bitset lookup for O(1) gene membership check.
 *     CSR: Parallel over cells, scan row for gene set members
 *     CSC: Parallel over genes, atomic accumulation to cells
 *
 * COMPLEXITY:
 *     Time:  O(nnz) or O(|gene_set| * avg_col_nnz)
 *     Space: O(n_genes / 64) for bitset
 *
 * THREAD SAFETY:
 *     Safe - uses atomic operations for CSC format
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void mean_score(
    const Sparse<T, IsCSR>& X,     // Input sparse matrix
    Array<const Index> gene_set,   // Gene indices in the set
    Array<Real> scores,            // Output scores [n_cells]
    Index n_cells,                 // Number of cells
    Index n_genes                  // Number of genes
);

/* -----------------------------------------------------------------------------
 * FUNCTION: weighted_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute weighted sum score for each cell over a gene set.
 *
 * PARAMETERS:
 *     X            [in]  Sparse matrix (cells x genes)
 *     gene_set     [in]  Array of gene indices
 *     gene_weights [in]  Weights for each gene in the set
 *     scores       [out] Output scores for each cell
 *     n_cells      [in]  Number of cells
 *     n_genes      [in]  Number of genes
 *
 * PRECONDITIONS:
 *     - scores.len >= n_cells
 *     - gene_weights.len >= gene_set.len
 *     - All gene indices in [0, n_genes)
 *
 * POSTCONDITIONS:
 *     - scores[c] = sum(weight[i] * X[c, gene_set[i]]) / sum(weight)
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(n_genes) for weight map
 *
 * THREAD SAFETY:
 *     Safe - uses atomic operations for CSC format
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void weighted_score(
    const Sparse<T, IsCSR>& X,         // Input sparse matrix
    Array<const Index> gene_set,       // Gene indices
    Array<const Real> gene_weights,    // Weights for each gene
    Array<Real> scores,                // Output scores [n_cells]
    Index n_cells,                     // Number of cells
    Index n_genes                      // Number of genes
);

/* -----------------------------------------------------------------------------
 * FUNCTION: auc_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute AUC-based score using expression ranks per cell.
 *
 * PARAMETERS:
 *     X        [in]  Sparse matrix (cells x genes)
 *     gene_set [in]  Array of gene indices
 *     scores   [out] Output AUC scores for each cell
 *     n_cells  [in]  Number of cells
 *     n_genes  [in]  Number of genes
 *     quantile [in]  Top quantile to consider (default 0.05)
 *
 * PRECONDITIONS:
 *     - scores.len >= n_cells
 *     - 0 < quantile <= 1
 *
 * POSTCONDITIONS:
 *     - scores[c] = fraction of gene_set genes in top quantile by expression
 *
 * ALGORITHM:
 *     Per cell (parallel):
 *     1. Extract expression values
 *     2. Compute ranks (shell sort + insertion sort)
 *     3. Count gene set genes in top quantile
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * n_genes * log(n_genes))
 *     Space: O(n_genes) per thread for workspace
 *
 * THREAD SAFETY:
 *     Safe - uses WorkspacePool for thread-local buffers
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void auc_score(
    const Sparse<T, IsCSR>& X,     // Input sparse matrix
    Array<const Index> gene_set,   // Gene indices
    Array<Real> scores,            // Output AUC scores [n_cells]
    Index n_cells,                 // Number of cells
    Index n_genes,                 // Number of genes
    Real quantile = Real(0.05)     // Top quantile threshold
);

/* -----------------------------------------------------------------------------
 * FUNCTION: module_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Seurat-style module score with expression-matched control genes.
 *
 * PARAMETERS:
 *     X                  [in]  Sparse matrix (cells x genes)
 *     gene_set           [in]  Array of gene indices
 *     scores             [out] Output module scores
 *     n_cells            [in]  Number of cells
 *     n_genes            [in]  Number of genes
 *     n_control_per_gene [in]  Control genes per target gene (default 1)
 *     n_bins             [in]  Expression bins for matching (default 25)
 *     seed               [in]  Random seed for control selection
 *
 * PRECONDITIONS:
 *     - scores.len >= n_cells
 *
 * POSTCONDITIONS:
 *     - scores[c] = mean(gene_set expression) - mean(control expression)
 *
 * ALGORITHM:
 *     1. Compute gene means and bin by expression
 *     2. For each gene in set, sample control genes from same bin
 *     3. Compute score as (gene_set mean) - (control mean)
 *
 * COMPLEXITY:
 *     Time:  O(nnz + n_genes * n_bins)
 *     Space: O(n_genes) for bins and control genes
 *
 * THREAD SAFETY:
 *     Safe - parallel cell scoring
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void module_score(
    const Sparse<T, IsCSR>& X,         // Input sparse matrix
    Array<const Index> gene_set,       // Gene indices
    Array<Real> scores,                // Output module scores [n_cells]
    Index n_cells,                     // Number of cells
    Index n_genes,                     // Number of genes
    Index n_control_per_gene = 1,      // Controls per gene
    Index n_bins = 25,                 // Expression bins
    uint64_t seed = 42                 // Random seed
);

/* -----------------------------------------------------------------------------
 * FUNCTION: zscore_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute z-score normalized gene set score.
 *
 * PARAMETERS:
 *     X        [in]  Sparse matrix (cells x genes)
 *     gene_set [in]  Array of gene indices
 *     scores   [out] Output z-scores
 *     n_cells  [in]  Number of cells
 *     n_genes  [in]  Number of genes
 *
 * PRECONDITIONS:
 *     - scores.len >= n_cells
 *
 * POSTCONDITIONS:
 *     - scores[c] = mean of z-scored expression for genes in set
 *     - z-score computed per gene across all cells
 *
 * ALGORITHM:
 *     1. Compute gene-wise mean and std
 *     2. Precompute z-score for zero expression
 *     3. Per cell: average z-scores for gene set
 *
 * COMPLEXITY:
 *     Time:  O(nnz + n_cells * |gene_set|)
 *     Space: O(n_genes + |gene_set|) per thread
 *
 * THREAD SAFETY:
 *     Safe - uses WorkspacePool for thread-local buffers
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void zscore_score(
    const Sparse<T, IsCSR>& X,     // Input sparse matrix
    Array<const Index> gene_set,   // Gene indices
    Array<Real> scores,            // Output z-scores [n_cells]
    Index n_cells,                 // Number of cells
    Index n_genes                  // Number of genes
);

/* -----------------------------------------------------------------------------
 * FUNCTION: gene_set_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Generic gene set scoring dispatcher.
 *
 * PARAMETERS:
 *     X        [in]  Sparse matrix
 *     gene_set [in]  Gene indices
 *     method   [in]  Scoring method to use
 *     scores   [out] Output scores
 *     n_cells  [in]  Number of cells
 *     n_genes  [in]  Number of genes
 *     quantile [in]  Quantile for AUC method (default 0.05)
 *
 * PRECONDITIONS:
 *     - scores.len >= n_cells
 *
 * POSTCONDITIONS:
 *     - scores computed according to specified method
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void gene_set_score(
    const Sparse<T, IsCSR>& X,     // Input sparse matrix
    Array<const Index> gene_set,   // Gene indices
    ScoringMethod method,          // Scoring method
    Array<Real> scores,            // Output scores [n_cells]
    Index n_cells,                 // Number of cells
    Index n_genes,                 // Number of genes
    Real quantile = Real(0.05)     // Quantile for AUC
);

/* -----------------------------------------------------------------------------
 * FUNCTION: differential_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute differential score between positive and negative gene sets.
 *
 * PARAMETERS:
 *     X              [in]  Sparse matrix
 *     positive_genes [in]  Positive gene indices
 *     negative_genes [in]  Negative gene indices
 *     scores         [out] Output differential scores
 *     n_cells        [in]  Number of cells
 *     n_genes        [in]  Number of genes
 *
 * POSTCONDITIONS:
 *     - scores[c] = mean(positive) - mean(negative)
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void differential_score(
    const Sparse<T, IsCSR>& X,         // Input sparse matrix
    Array<const Index> positive_genes, // Positive gene set
    Array<const Index> negative_genes, // Negative gene set
    Array<Real> scores,                // Output scores [n_cells]
    Index n_cells,                     // Number of cells
    Index n_genes                      // Number of genes
);

/* -----------------------------------------------------------------------------
 * FUNCTION: cell_cycle_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute cell cycle phase scores and assignments.
 *
 * PARAMETERS:
 *     X            [in]  Sparse matrix
 *     s_genes      [in]  S-phase gene indices
 *     g2m_genes    [in]  G2/M-phase gene indices
 *     s_scores     [out] S-phase scores
 *     g2m_scores   [out] G2/M-phase scores
 *     phase_labels [out] Phase assignment (0=G1, 1=S, 2=G2M)
 *     n_cells      [in]  Number of cells
 *     n_genes      [in]  Number of genes
 *
 * POSTCONDITIONS:
 *     - phase_labels[c] = phase with highest positive score, or G1 if both <= 0
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void cell_cycle_score(
    const Sparse<T, IsCSR>& X,     // Input sparse matrix
    Array<const Index> s_genes,    // S-phase genes
    Array<const Index> g2m_genes,  // G2/M-phase genes
    Array<Real> s_scores,          // Output S scores [n_cells]
    Array<Real> g2m_scores,        // Output G2M scores [n_cells]
    Array<Index> phase_labels,     // Output phase labels [n_cells]
    Index n_cells,                 // Number of cells
    Index n_genes                  // Number of genes
);

/* -----------------------------------------------------------------------------
 * FUNCTION: quantile_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute quantile of gene set expression per cell.
 *
 * PARAMETERS:
 *     X        [in]  Sparse matrix
 *     gene_set [in]  Gene indices
 *     quantile [in]  Quantile to compute (0 to 1)
 *     scores   [out] Output quantile values
 *     n_cells  [in]  Number of cells
 *     n_genes  [in]  Number of genes
 *
 * POSTCONDITIONS:
 *     - scores[c] = quantile-th percentile of gene set expression in cell c
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void quantile_score(
    const Sparse<T, IsCSR>& X,     // Input sparse matrix
    Array<const Index> gene_set,   // Gene indices
    Real quantile,                 // Quantile (0 to 1)
    Array<Real> scores,            // Output quantile scores [n_cells]
    Index n_cells,                 // Number of cells
    Index n_genes                  // Number of genes
);

/* -----------------------------------------------------------------------------
 * FUNCTION: multi_signature_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Score multiple gene signatures in parallel.
 *
 * PARAMETERS:
 *     X          [in]  Sparse matrix
 *     gene_sets  [in]  Array of pointers to gene set arrays
 *     set_sizes  [in]  Size of each gene set
 *     n_sets     [in]  Number of gene sets
 *     method     [in]  Scoring method
 *     all_scores [out] Output scores (n_cells x n_sets, row-major)
 *     n_cells    [in]  Number of cells
 *     n_genes    [in]  Number of genes
 *
 * POSTCONDITIONS:
 *     - all_scores[c * n_sets + s] = score for cell c, signature s
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void multi_signature_score(
    const Sparse<T, IsCSR>& X,     // Input sparse matrix
    const Index* const* gene_sets, // Array of gene set pointers
    const Index* set_sizes,        // Size of each gene set
    Index n_sets,                  // Number of gene sets
    ScoringMethod method,          // Scoring method
    Array<Real> all_scores,        // Output scores [n_cells * n_sets]
    Index n_cells,                 // Number of cells
    Index n_genes                  // Number of genes
);

} // namespace scl::kernel::scoring
