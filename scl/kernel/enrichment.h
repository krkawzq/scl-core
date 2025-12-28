// =============================================================================
// FILE: scl/kernel/enrichment.h
// BRIEF: API reference for gene set enrichment and pathway analysis
// NOTE: Documentation only - do not include in builds
// =============================================================================
#pragma once

#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"

namespace scl::kernel::enrichment {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Index DEFAULT_N_PERMUTATIONS = 1000;
    constexpr Real DEFAULT_ALPHA = Real(0.05);
    constexpr Real EPSILON = Real(1e-15);
    constexpr Real MIN_PVALUE = Real(1e-300);
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size SIMD_THRESHOLD = 32;
    constexpr Size PREFETCH_DISTANCE = 8;
}

// =============================================================================
// Enrichment Methods
// =============================================================================

enum class EnrichmentMethod {
    Hypergeometric,
    Fisher,
    GSEA,
    GSVA,
    ORA
};

// =============================================================================
// Statistical Tests
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: hypergeometric_test
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute hypergeometric test p-value for over-representation.
 *
 * PARAMETERS:
 *     k [in] Number of successes in sample
 *     n [in] Sample size
 *     K [in] Number of successes in population
 *     N [in] Population size
 *
 * PRECONDITIONS:
 *     - k <= n <= N
 *     - K <= N
 *     - k <= K
 *
 * POSTCONDITIONS:
 *     - Returns p-value for observing k or more successes
 *
 * COMPLEXITY:
 *     Time:  O(min(k, n-k))
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
Real hypergeometric_test(
    Index k,                                 // Successes in sample
    Index n,                                 // Sample size
    Index K,                                 // Successes in population
    Index N                                  // Population size
);

/* -----------------------------------------------------------------------------
 * FUNCTION: fisher_exact_test
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Fisher's exact test p-value for 2x2 contingency table.
 *
 * PARAMETERS:
 *     a [in] Count in cell (0,0)
 *     b [in] Count in cell (0,1)
 *     c [in] Count in cell (1,0)
 *     d [in] Count in cell (1,1)
 *
 * PRECONDITIONS:
 *     - All counts >= 0
 *
 * POSTCONDITIONS:
 *     - Returns two-tailed p-value
 *
 * COMPLEXITY:
 *     Time:  O(min(a, b, c, d))
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
Real fisher_exact_test(
    Index a,                                 // Count (0,0)
    Index b,                                 // Count (0,1)
    Index c,                                 // Count (1,0)
    Index d                                  // Count (1,1)
);

/* -----------------------------------------------------------------------------
 * FUNCTION: odds_ratio
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute odds ratio from 2x2 contingency table.
 *
 * PARAMETERS:
 *     a [in] Count in cell (0,0)
 *     b [in] Count in cell (0,1)
 *     c [in] Count in cell (1,0)
 *     d [in] Count in cell (1,1)
 *
 * PRECONDITIONS:
 *     - All counts >= 0
 *     - At least one count in each row and column
 *
 * POSTCONDITIONS:
 *     - Returns (a*d) / (b*c) or infinity if denominator is zero
 *
 * COMPLEXITY:
 *     Time:  O(1)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
Real odds_ratio(
    Index a,                                 // Count (0,0)
    Index b,                                 // Count (0,1)
    Index c,                                 // Count (1,0)
    Index d                                  // Count (1,1)
);

// =============================================================================
// GSEA Methods
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: gsea
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute Gene Set Enrichment Analysis enrichment score and p-value.
 *
 * PARAMETERS:
 *     ranked_genes      [in]  Genes ranked by statistic [n_genes]
 *     in_gene_set       [in]  Boolean array indicating gene set membership [n_genes]
 *     n_genes           [in]  Total number of genes
 *     enrichment_score  [out] Enrichment score
 *     nes               [out] Normalized enrichment score
 *     p_value           [out] P-value from permutation test
 *     n_permutations    [in]  Number of permutations
 *     seed              [in]  Random seed
 *
 * PRECONDITIONS:
 *     - ranked_genes contains valid gene indices
 *     - in_gene_set.len >= n_genes
 *
 * POSTCONDITIONS:
 *     - enrichment_score contains ES
 *     - nes contains normalized ES
 *     - p_value contains permutation-based p-value
 *
 * COMPLEXITY:
 *     Time:  O(n_genes * n_permutations)
 *     Space: O(n_genes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - uses workspace pools
 * -------------------------------------------------------------------------- */
void gsea(
    Array<const Index> ranked_genes,        // Ranked gene indices [n_genes]
    Array<const bool> in_gene_set,          // Gene set membership [n_genes]
    Index n_genes,                           // Total number of genes
    Real& enrichment_score,                  // Output ES
    Real& nes,                               // Output NES
    Real& p_value,                           // Output p-value
    Index n_permutations = config::DEFAULT_N_PERMUTATIONS,
    uint64_t seed = 42
);

/* -----------------------------------------------------------------------------
 * FUNCTION: gsea_running_sum
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute GSEA running sum for visualization.
 *
 * PARAMETERS:
 *     ranked_genes  [in]  Ranked gene indices [n_genes]
 *     in_gene_set   [in]  Gene set membership [n_genes]
 *     n_genes       [in]  Total number of genes
 *     running_sum   [out] Running sum values [n_genes]
 *
 * PRECONDITIONS:
 *     - running_sum.len >= n_genes
 *
 * POSTCONDITIONS:
 *     - running_sum[i] contains cumulative enrichment at position i
 *
 * COMPLEXITY:
 *     Time:  O(n_genes)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
void gsea_running_sum(
    Array<const Index> ranked_genes,        // Ranked gene indices [n_genes]
    Array<const bool> in_gene_set,          // Gene set membership [n_genes]
    Index n_genes,                           // Total number of genes
    Array<Real> running_sum                  // Output running sum [n_genes]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: leading_edge_genes
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Identify leading edge genes (genes contributing to enrichment peak).
 *
 * PARAMETERS:
 *     ranked_genes   [in]  Ranked gene indices [n_genes]
 *     in_gene_set    [in]  Gene set membership [n_genes]
 *     n_genes        [in]  Total number of genes
 *     enrichment_score [in]  Enrichment score
 *     leading_genes  [out] Leading edge gene indices [n_genes]
 *
 * PRECONDITIONS:
 *     - leading_genes has capacity >= n_genes
 *
 * POSTCONDITIONS:
 *     - Returns number of leading edge genes
 *     - leading_genes[0..return_value) contains gene indices
 *
 * COMPLEXITY:
 *     Time:  O(n_genes)
 *     Space: O(1) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - no shared state
 * -------------------------------------------------------------------------- */
Index leading_edge_genes(
    Array<const Index> ranked_genes,        // Ranked gene indices [n_genes]
    Array<const bool> in_gene_set,          // Gene set membership [n_genes]
    Index n_genes,                           // Total number of genes
    Real enrichment_score,                    // Enrichment score
    Array<Index> leading_genes               // Output leading edge genes [n_genes]
);

// =============================================================================
// Over-Representation Analysis
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: ora
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Perform Over-Representation Analysis for multiple pathways.
 *
 * PARAMETERS:
 *     de_genes          [in]  Differentially expressed gene indices [n_de]
 *     pathway_genes     [in]  Array of pathway gene arrays [n_pathways]
 *     pathway_sizes     [in]  Size of each pathway [n_pathways]
 *     n_pathways        [in]  Number of pathways
 *     n_total_genes     [in]  Total number of genes
 *     p_values          [out] P-values [n_pathways]
 *     odds_ratios      [out] Odds ratios [n_pathways]
 *     fold_enrichments [out] Fold enrichments [n_pathways]
 *
 * PRECONDITIONS:
 *     - All output arrays have length >= n_pathways
 *     - All gene indices are valid
 *
 * POSTCONDITIONS:
 *     - p_values[p] contains hypergeometric p-value for pathway p
 *     - odds_ratios[p] contains odds ratio
 *     - fold_enrichments[p] contains fold enrichment
 *
 * COMPLEXITY:
 *     Time:  O(n_pathways * avg_pathway_size)
 *     Space: O(n_total_genes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over pathways
 * -------------------------------------------------------------------------- */
void ora(
    Array<const Index> de_genes,            // DE gene indices [n_de]
    const Index* const* pathway_genes,      // Pathway gene arrays [n_pathways]
    const Index* pathway_sizes,              // Pathway sizes [n_pathways]
    Index n_pathways,                        // Number of pathways
    Index n_total_genes,                     // Total number of genes
    Array<Real> p_values,                     // Output p-values [n_pathways]
    Array<Real> odds_ratios,                 // Output odds ratios [n_pathways]
    Array<Real> fold_enrichments             // Output fold enrichments [n_pathways]
);

// =============================================================================
// Pathway Activity
// =============================================================================

/* -----------------------------------------------------------------------------
 * FUNCTION: pathway_activity
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Compute pathway activity score for each cell.
 *
 * PARAMETERS:
 *     X              [in]  Expression matrix (cells x genes, CSR)
 *     pathway_genes  [in]  Pathway gene indices [n_pathway_genes]
 *     n_cells        [in]  Number of cells
 *     n_genes        [in]  Number of genes
 *     activity_scores [out] Activity scores [n_cells]
 *
 * PRECONDITIONS:
 *     - activity_scores.len >= n_cells
 *     - All gene indices are valid
 *
 * POSTCONDITIONS:
 *     - activity_scores[i] contains mean expression of pathway genes in cell i
 *
 * COMPLEXITY:
 *     Time:  O(nnz * n_pathway_genes / n_genes)
 *     Space: O(n_genes) auxiliary
 *
 * THREAD SAFETY:
 *     Safe - parallelized over cells
 * -------------------------------------------------------------------------- */
template <typename T, bool IsCSR>
void pathway_activity(
    const Sparse<T, IsCSR>& X,               // Expression matrix [n_cells x n_genes]
    Array<const Index> pathway_genes,        // Pathway gene indices [n_pathway_genes]
    Index n_cells,                           // Number of cells
    Index n_genes,                           // Number of genes
    Array<Real> activity_scores               // Output scores [n_cells]
);

/* -----------------------------------------------------------------------------
 * FUNCTION: benjamini_hochberg
 * -----------------------------------------------------------------------------
 * SUMMARY:
 *     Apply Benjamini-Hochberg FDR correction to enrichment p-values.
 *
 * PARAMETERS:
 *     p_values  [in]  Input p-values [n]
 *     q_values  [out] FDR-adjusted q-values [n]
 *
 * PRECONDITIONS:
 *     - q_values.len >= p_values.len
 *
 * POSTCONDITIONS:
 *     - q_values contains BH-adjusted p-values
 *
 * COMPLEXITY:
 *     Time:  O(n log n) for sorting
 *     Space: O(n) auxiliary
 *
 * THREAD SAFETY:
 *     Unsafe - sequential implementation
 * -------------------------------------------------------------------------- */
void benjamini_hochberg(
    Array<const Real> p_values,             // Input p-values [n]
    Array<Real> q_values                     // Output q-values [n]
);

} // namespace scl::kernel::enrichment

