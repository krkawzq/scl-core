#pragma once

// =============================================================================
// FILE: scl/binding/c_api/enrichment/enrichment.h
// BRIEF: C API for gene set enrichment analysis
// =============================================================================

#include "scl/binding/c_api/core/core.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Statistical Tests
// =============================================================================

// Hypergeometric test
scl_error_t scl_enrichment_hypergeometric_test(
    scl_index_t k,                    // Observed successes
    scl_index_t n,                    // Sample size
    scl_index_t K,                    // Population successes
    scl_index_t N,                    // Population size
    scl_real_t* p_value_out
);

// Fisher's exact test (2x2 contingency table)
scl_error_t scl_enrichment_fisher_exact_test(
    scl_index_t a, scl_index_t b,     // Row 1
    scl_index_t c, scl_index_t d,     // Row 2
    scl_real_t* p_value_out
);

// Odds ratio
scl_error_t scl_enrichment_odds_ratio(
    scl_index_t a, scl_index_t b,
    scl_index_t c, scl_index_t d,
    scl_real_t* odds_ratio_out
);

// =============================================================================
// Gene Set Enrichment Analysis (GSEA)
// =============================================================================

// GSEA with permutation testing
scl_error_t scl_enrichment_gsea(
    const scl_index_t* ranked_genes,  // Ranked gene indices
    const scl_real_t* ranking_scores, // Ranking scores
    const int* in_gene_set,           // Boolean array [n_genes]
    scl_index_t n_genes,
    scl_real_t* enrichment_score_out,
    scl_real_t* p_value_out,
    scl_index_t n_permutations,       // 0 = use default 1000
    uint64_t seed
);

// GSEA running sum (for plotting)
scl_error_t scl_enrichment_gsea_running_sum(
    const scl_index_t* ranked_genes,
    const int* in_gene_set,
    scl_index_t n_genes,
    scl_real_t* running_sum           // Output [n_genes]
);

// Leading edge genes
scl_error_t scl_enrichment_leading_edge_genes(
    const scl_index_t* ranked_genes,
    const int* in_gene_set,
    scl_index_t n_genes,
    scl_real_t enrichment_score,
    scl_index_t* leading_genes,       // Output indices
    scl_size_t max_leading,
    scl_index_t* n_leading_out
);

// =============================================================================
// Over-Representation Analysis (ORA)
// =============================================================================

// Single gene set ORA
scl_error_t scl_enrichment_ora_single_set(
    const scl_index_t* de_genes,      // Differentially expressed genes
    scl_size_t n_de_genes,
    const scl_index_t* pathway_genes, // Pathway genes
    scl_size_t n_pathway_genes,
    scl_index_t n_total_genes,
    scl_real_t* p_value_out,
    scl_real_t* odds_ratio_out,
    scl_real_t* fold_enrichment_out
);

// Batch ORA for multiple pathways
scl_error_t scl_enrichment_ora_batch(
    const scl_index_t* de_genes,
    scl_size_t n_de_genes,
    const scl_index_t* const* pathway_genes,  // Array of pathway gene arrays
    const scl_index_t* pathway_sizes,         // Size of each pathway
    scl_index_t n_pathways,
    scl_index_t n_total_genes,
    scl_real_t* p_values,             // Output [n_pathways]
    scl_real_t* odds_ratios,          // Output [n_pathways]
    scl_real_t* fold_enrichments      // Output [n_pathways]
);

// =============================================================================
// Multiple Testing Correction
// =============================================================================

// Benjamini-Hochberg FDR correction
scl_error_t scl_enrichment_benjamini_hochberg(
    const scl_real_t* p_values,
    scl_size_t n,
    scl_real_t* q_values              // Output adjusted p-values
);

// Bonferroni correction
scl_error_t scl_enrichment_bonferroni(
    const scl_real_t* p_values,
    scl_size_t n,
    scl_real_t* adjusted_p            // Output adjusted p-values
);

// =============================================================================
// Pathway Activity Scores
// =============================================================================

// Pathway activity score per cell
scl_error_t scl_enrichment_pathway_activity(
    scl_sparse_t X,                   // Cell x gene matrix
    const scl_index_t* pathway_genes,
    scl_size_t n_pathway_genes,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* activity_scores       // Output [n_cells]
);

// GSVA-like pathway score
scl_error_t scl_enrichment_gsva_score(
    scl_sparse_t X,
    const scl_index_t* pathway_genes,
    scl_size_t n_pathway_genes,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* gsva_scores           // Output [n_cells]
);

// Single-sample GSEA (ssGSEA)
scl_error_t scl_enrichment_ssgsea(
    scl_sparse_t X,
    const scl_index_t* pathway_genes,
    scl_size_t n_pathway_genes,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* enrichment_scores,    // Output [n_cells]
    scl_real_t weight_exponent        // 0 = use default 0.25
);

#ifdef __cplusplus
}
#endif
