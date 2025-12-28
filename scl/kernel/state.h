// =============================================================================
// FILE: scl/kernel/state.h
// BRIEF: API reference for cell state scoring (stemness, differentiation, proliferation)
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::state {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real EPSILON = Real(1e-10);
    constexpr Size MIN_GENES_FOR_SCORE = 3;
    constexpr Real PSEUDOCOUNT = Real(1.0);
    constexpr Size PARALLEL_THRESHOLD = 64;
}

// =============================================================================
// State Scoring Functions
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: stemness_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute stemness score for each cell based on stemness gene expression.
 *
 * PARAMETERS:
 *     expression      [in]  Expression matrix (cells x genes, CSR)
 *     stemness_genes  [in]  Indices of stemness genes [n_stemness_genes]
 *     scores          [out] Z-score normalized stemness scores [n_cells]
 *
 * PRECONDITIONS:
 *     - scores.len == expression.rows()
 *     - All gene indices in stemness_genes are valid
 *
 * POSTCONDITIONS:
 *     - scores[i] contains z-score normalized stemness score for cell i
 *     - Scores have mean 0 and standard deviation 1
 *     - Matrix is unchanged
 *
 * ALGORITHM:
 *     1. For each cell, compute mean expression of stemness genes
 *     2. Z-score normalize scores across all cells
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * n_stemness_genes * log(nnz_per_cell))
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void stemness_score(
    const Sparse<T, IsCSR>& expression,    // Expression matrix [n_cells x n_genes]
    Array<const Index> stemness_genes,      // Stemness gene indices [n_stemness_genes]
    Array<Real> scores                      // Output scores [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: differentiation_potential
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute differentiation potential score (CytoTRACE-style) for each cell.
 *     Higher scores indicate greater differentiation potential.
 *
 * PARAMETERS:
 *     expression      [in]  Expression matrix (cells x genes, CSR)
 *     potency_scores  [out] Normalized potency scores [0, 1] [n_cells]
 *
 * PRECONDITIONS:
 *     - potency_scores.len == expression.rows()
 *
 * POSTCONDITIONS:
 *     - potency_scores[i] contains normalized potency score in [0, 1]
 *     - Higher scores indicate greater differentiation potential
 *     - Matrix is unchanged
 *
 * ALGORITHM:
 *     1. Count expressed genes per cell
 *     2. Compute correlation of each gene with gene count
 *     3. Select top correlated genes
 *     4. Compute weighted sum of top gene expressions
 *     5. Normalize to [0, 1] range
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * n_genes * log(nnz_per_cell) + n_genes * log(n_genes))
 *     Space: O(n_cells + n_genes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells and genes
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void differentiation_potential(
    const Sparse<T, IsCSR>& expression,    // Expression matrix [n_cells x n_genes]
    Array<Real> potency_scores               // Output scores [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: proliferation_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute proliferation score for each cell based on proliferation gene expression.
 *
 * PARAMETERS:
 *     expression          [in]  Expression matrix (cells x genes, CSR)
 *     proliferation_genes [in]  Indices of proliferation genes [n_proliferation_genes]
 *     scores              [out] Z-score normalized proliferation scores [n_cells]
 *
 * PRECONDITIONS:
 *     - scores.len == expression.rows()
 *     - All gene indices in proliferation_genes are valid
 *
 * POSTCONDITIONS:
 *     - scores[i] contains z-score normalized proliferation score for cell i
 *     - Scores have mean 0 and standard deviation 1
 *     - Matrix is unchanged
 *
 * ALGORITHM:
 *     1. For each cell, compute mean expression of proliferation genes
 *     2. Z-score normalize scores across all cells
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * n_proliferation_genes * log(nnz_per_cell))
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void proliferation_score(
    const Sparse<T, IsCSR>& expression,    // Expression matrix [n_cells x n_genes]
    Array<const Index> proliferation_genes, // Proliferation gene indices [n_proliferation_genes]
    Array<Real> scores                       // Output scores [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: stress_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute stress score for each cell based on stress gene expression.
 *
 * PARAMETERS:
 *     expression  [in]  Expression matrix (cells x genes, CSR)
 *     stress_genes [in]  Indices of stress genes [n_stress_genes]
 *     scores      [out] Z-score normalized stress scores [n_cells]
 *
 * PRECONDITIONS:
 *     - scores.len == expression.rows()
 *     - All gene indices in stress_genes are valid
 *
 * POSTCONDITIONS:
 *     - scores[i] contains z-score normalized stress score for cell i
 *     - Scores have mean 0 and standard deviation 1
 *     - Matrix is unchanged
 *
 * ALGORITHM:
 *     1. For each cell, compute mean expression of stress genes
 *     2. Z-score normalize scores across all cells
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * n_stress_genes * log(nnz_per_cell))
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void stress_score(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    Array<const Index> stress_genes,         // Stress gene indices [n_stress_genes]
    Array<Real> scores                        // Output scores [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: state_entropy
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute expression entropy (plasticity) for each cell, normalized by
 *     maximum possible entropy.
 *
 * PARAMETERS:
 *     expression      [in]  Expression matrix (cells x genes, CSR)
 *     entropy_scores  [out] Normalized entropy scores [0, 1] [n_cells]
 *
 * PRECONDITIONS:
 *     - entropy_scores.len == expression.rows()
 *
 * POSTCONDITIONS:
 *     - entropy_scores[i] contains normalized Shannon entropy for cell i
 *     - Scores are in [0, 1], where 1 indicates maximum diversity
 *     - Matrix is unchanged
 *
 * ALGORITHM:
 *     For each cell:
 *         1. Compute total expression
 *         2. Compute Shannon entropy: -sum(p_i * log(p_i))
 *         3. Normalize by maximum possible entropy (log(n_genes))
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary per cell
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void state_entropy(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    Array<Real> entropy_scores                // Output scores [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: cell_cycle_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute cell cycle phase scores (G1/S/G2M) and assign phase labels.
 *
 * PARAMETERS:
 *     expression  [in]  Expression matrix (cells x genes, CSR)
 *     s_genes     [in]  Indices of S-phase genes [n_s_genes]
 *     g2m_genes   [in]  Indices of G2/M-phase genes [n_g2m_genes]
 *     s_scores    [out] Z-score normalized S-phase scores [n_cells]
 *     g2m_scores  [out] Z-score normalized G2/M-phase scores [n_cells]
 *     phase_labels [out] Phase labels: 0=G1, 1=S, 2=G2M [n_cells]
 *
 * PRECONDITIONS:
 *     - All score arrays have length == expression.rows()
 *     - All gene indices are valid
 *
 * POSTCONDITIONS:
 *     - s_scores[i] and g2m_scores[i] are z-score normalized
 *     - phase_labels[i] indicates assigned phase (0, 1, or 2)
 *     - Matrix is unchanged
 *
 * ALGORITHM:
 *     1. Compute S-phase and G2/M-phase scores for each cell
 *     2. Z-score normalize both scores
 *     3. Assign phase: S if s_score > 0 and s_score > g2m_score,
 *        G2M if g2m_score > 0 and g2m_score > s_score, else G1
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * (n_s_genes + n_g2m_genes) * log(nnz_per_cell))
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void cell_cycle_score(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    Array<const Index> s_genes,              // S-phase gene indices [n_s_genes]
    Array<const Index> g2m_genes,            // G2/M-phase gene indices [n_g2m_genes]
    Array<Real> s_scores,                    // Output S-phase scores [n_cells]
    Array<Real> g2m_scores,                  // Output G2/M-phase scores [n_cells]
    Array<Index> phase_labels                 // Output phase labels [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: quiescence_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute quiescence score as difference between quiescence and proliferation
 *     gene expression scores.
 *
 * PARAMETERS:
 *     expression          [in]  Expression matrix (cells x genes, CSR)
 *     quiescence_genes    [in]  Indices of quiescence genes [n_quiescence_genes]
 *     proliferation_genes [in]  Indices of proliferation genes [n_proliferation_genes]
 *     scores              [out] Quiescence scores [n_cells]
 *
 * PRECONDITIONS:
 *     - scores.len == expression.rows()
 *     - All gene indices are valid
 *
 * POSTCONDITIONS:
 *     - scores[i] = quiescence_score[i] - proliferation_score[i]
 *     - Scores are z-score normalized differences
 *     - Matrix is unchanged
 *
 * ALGORITHM:
 *     1. Compute quiescence and proliferation scores separately
 *     2. Z-score normalize both
 *     3. Compute difference: quiescence - proliferation
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * (n_quiescence_genes + n_proliferation_genes) * log(nnz_per_cell))
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void quiescence_score(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    Array<const Index> quiescence_genes,      // Quiescence gene indices [n_quiescence_genes]
    Array<const Index> proliferation_genes,   // Proliferation gene indices [n_proliferation_genes]
    Array<Real> scores                         // Output scores [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: metabolic_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute glycolysis and OXPHOS (oxidative phosphorylation) scores.
 *
 * PARAMETERS:
 *     expression          [in]  Expression matrix (cells x genes, CSR)
 *     glycolysis_genes    [in]  Indices of glycolysis genes [n_glycolysis_genes]
 *     oxphos_genes        [in]  Indices of OXPHOS genes [n_oxphos_genes]
 *     glycolysis_scores   [out] Z-score normalized glycolysis scores [n_cells]
 *     oxphos_scores       [out] Z-score normalized OXPHOS scores [n_cells]
 *
 * PRECONDITIONS:
 *     - Both score arrays have length == expression.rows()
 *     - All gene indices are valid
 *
 * POSTCONDITIONS:
 *     - Both scores are z-score normalized
 *     - Matrix is unchanged
 *
 * ALGORITHM:
 *     1. Compute glycolysis and OXPHOS scores separately
 *     2. Z-score normalize both scores
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * (n_glycolysis_genes + n_oxphos_genes) * log(nnz_per_cell))
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void metabolic_score(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    Array<const Index> glycolysis_genes,     // Glycolysis gene indices [n_glycolysis_genes]
    Array<const Index> oxphos_genes,          // OXPHOS gene indices [n_oxphos_genes]
    Array<Real> glycolysis_scores,            // Output glycolysis scores [n_cells]
    Array<Real> oxphos_scores                  // Output OXPHOS scores [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: apoptosis_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute apoptosis score for each cell based on apoptosis gene expression.
 *
 * PARAMETERS:
 *     expression    [in]  Expression matrix (cells x genes, CSR)
 *     apoptosis_genes [in]  Indices of apoptosis genes [n_apoptosis_genes]
 *     scores        [out] Z-score normalized apoptosis scores [n_cells]
 *
 * PRECONDITIONS:
 *     - scores.len == expression.rows()
 *     - All gene indices in apoptosis_genes are valid
 *
 * POSTCONDITIONS:
 *     - scores[i] contains z-score normalized apoptosis score for cell i
 *     - Scores have mean 0 and standard deviation 1
 *     - Matrix is unchanged
 *
 * ALGORITHM:
 *     1. For each cell, compute mean expression of apoptosis genes
 *     2. Z-score normalize scores across all cells
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * n_apoptosis_genes * log(nnz_per_cell))
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void apoptosis_score(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    Array<const Index> apoptosis_genes,      // Apoptosis gene indices [n_apoptosis_genes]
    Array<Real> scores                         // Output scores [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: signature_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute weighted gene signature score for each cell.
 *
 * PARAMETERS:
 *     expression    [in]  Expression matrix (cells x genes, CSR)
 *     gene_indices  [in]  Indices of signature genes [n_signature_genes]
 *     gene_weights  [in]  Weights for each signature gene [n_signature_genes]
 *     scores        [out] Z-score normalized signature scores [n_cells]
 *
 * PRECONDITIONS:
 *     - scores.len == expression.rows()
 *     - gene_indices.len == gene_weights.len
 *     - All gene indices are valid
 *
 * POSTCONDITIONS:
 *     - scores[i] contains weighted signature score for cell i
 *     - Scores are z-score normalized
 *     - Matrix is unchanged
 *
 * ALGORITHM:
 *     1. For each cell, compute weighted sum of signature gene expressions
 *     2. Normalize by sum of absolute weights
 *     3. Z-score normalize across all cells
 *
 * COMPLEXITY:
 *     Time:  O(n_cells * n_signature_genes * log(nnz_per_cell))
 *     Space: O(n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void signature_score(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    Array<const Index> gene_indices,         // Signature gene indices [n_signature_genes]
    Array<const Real> gene_weights,          // Gene weights [n_signature_genes]
    Array<Real> scores                       // Output scores [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: multi_signature_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute scores for multiple gene signatures simultaneously.
 *
 * PARAMETERS:
 *     expression            [in]  Expression matrix (cells x genes, CSR)
 *     signature_gene_indices [in]  Flat array of all gene indices [total_genes]
 *     signature_offsets    [in]  Start offset for each signature [n_signatures + 1]
 *     n_signatures          [in]  Number of signatures
 *     score_matrix         [out] Score matrix [n_cells * n_signatures]
 *
 * PRECONDITIONS:
 *     - signature_offsets has length n_signatures + 1
 *     - score_matrix has capacity >= n_cells * n_signatures
 *     - All gene indices are valid
 *
 * POSTCONDITIONS:
 *     - score_matrix[i * n_signatures + s] contains z-score normalized score
 *       for cell i and signature s
 *     - Each signature column is independently z-score normalized
 *     - Matrix is unchanged
 *
 * ALGORITHM:
 *     1. For each signature, compute scores for all cells
 *     2. Z-score normalize each signature column independently
 *
 * COMPLEXITY:
 *     Time:  O(n_signatures * n_cells * avg_signature_size * log(nnz_per_cell))
 *     Space: O(n_cells * n_signatures) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over signatures
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void multi_signature_score(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    const Index* signature_gene_indices,     // Flat gene index array [total_genes]
    const Size* signature_offsets,           // Signature offsets [n_signatures + 1]
    Size n_signatures,                        // Number of signatures
    Real* score_matrix                        // Output matrix [n_cells * n_signatures]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: transcriptional_diversity
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Simpson's diversity index for expression distribution in each cell.
 *
 * PARAMETERS:
 *     expression        [in]  Expression matrix (cells x genes, CSR)
 *     diversity_scores  [out] Diversity scores [0, 1] [n_cells]
 *
 * PRECONDITIONS:
 *     - diversity_scores.len == expression.rows()
 *
 * POSTCONDITIONS:
 *     - diversity_scores[i] contains Simpson's diversity index for cell i
 *     - Scores are in [0, 1], where 1 indicates maximum diversity
 *     - Matrix is unchanged
 *
 * ALGORITHM:
 *     For each cell:
 *         1. Compute total expression and sum of squared expressions
 *         2. Simpson's index = 1 - sum(p_i^2) where p_i = value_i / total
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary per cell
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void transcriptional_diversity(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    Array<Real> diversity_scores             // Output scores [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: expression_complexity
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute expression complexity as fraction of genes expressed above threshold.
 *
 * PARAMETERS:
 *     expression          [in]  Expression matrix (cells x genes, CSR)
 *     expression_threshold [in]  Minimum expression value to count as expressed
 *     complexity_scores   [out] Complexity scores [0, 1] [n_cells]
 *
 * PRECONDITIONS:
 *     - complexity_scores.len == expression.rows()
 *
 * POSTCONDITIONS:
 *     - complexity_scores[i] = n_expressed_genes / n_total_genes for cell i
 *     - Scores are in [0, 1]
 *     - Matrix is unchanged
 *
 * ALGORITHM:
 *     For each cell:
 *         1. Count genes with expression > threshold
 *         2. Normalize by total number of genes
 *
 * COMPLEXITY:
 *     Time:  O(nnz)
 *     Space: O(1) auxiliary per cell
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void expression_complexity(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    Real expression_threshold,              // Expression threshold
    Array<Real> complexity_scores            // Output scores [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: combined_state_score
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute combined state score from multiple gene sets with weights.
 *
 * PARAMETERS:
 *     expression      [in]  Expression matrix (cells x genes, CSR)
 *     gene_sets       [in]  Array of gene set pointers [n_gene_sets]
 *     gene_set_sizes  [in]  Size of each gene set [n_gene_sets]
 *     weights         [in]  Weight for each gene set [n_gene_sets]
 *     n_gene_sets     [in]  Number of gene sets
 *     combined_scores [out] Combined scores [n_cells]
 *
 * PRECONDITIONS:
 *     - combined_scores.len == expression.rows()
 *     - All gene indices are valid
 *
 * POSTCONDITIONS:
 *     - combined_scores[i] contains weighted combination of all gene set scores
 *     - Matrix is unchanged
 *
 * ALGORITHM:
 *     1. Compute individual scores for each gene set
 *     2. Z-score normalize each gene set score
 *     3. Compute weighted combination
 *
 * COMPLEXITY:
 *     Time:  O(n_gene_sets * n_cells * avg_gene_set_size * log(nnz_per_cell))
 *     Space: O(n_gene_sets * n_cells) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over gene sets and cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void combined_state_score(
    const Sparse<T, IsCSR>& expression,     // Expression matrix [n_cells x n_genes]
    const Index* const* gene_sets,          // Array of gene set pointers [n_gene_sets]
    const Size* gene_set_sizes,              // Gene set sizes [n_gene_sets]
    const Real* weights,                     // Gene set weights [n_gene_sets]
    Size n_gene_sets,                        // Number of gene sets
    Array<Real> combined_scores               // Output scores [n_cells]
);

} // namespace scl::kernel::state

