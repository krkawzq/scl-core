#pragma once

#include "core_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// FILE: scl/binding/c_api/enrichment.h
// BRIEF: C API for gene set enrichment analysis
// =============================================================================

// Hypergeometric test
scl_real_t scl_enrichment_hypergeometric_test(
    scl_index_t k,
    scl_index_t n,
    scl_index_t K,
    scl_index_t N
);

// Fisher's exact test
scl_real_t scl_enrichment_fisher_exact_test(
    scl_index_t a,
    scl_index_t b,
    scl_index_t c,
    scl_index_t d
);

// Odds ratio
scl_real_t scl_enrichment_odds_ratio(
    scl_index_t a,
    scl_index_t b,
    scl_index_t c,
    scl_index_t d
);

// Gene Set Enrichment Analysis (GSEA)
scl_error_t scl_enrichment_gsea(
    const scl_index_t* ranked_genes,
    const scl_real_t* ranking_scores,
    const int* in_gene_set,
    scl_index_t n_genes,
    scl_real_t* enrichment_score,
    scl_real_t* p_value,
    scl_real_t* nes,
    scl_index_t n_permutations,
    uint64_t seed
);

// GSEA running sum
scl_error_t scl_enrichment_gsea_running_sum(
    const scl_index_t* ranked_genes,
    const int* in_gene_set,
    scl_index_t n_genes,
    scl_real_t* running_sum
);

// Leading edge genes
scl_index_t scl_enrichment_leading_edge_genes(
    const scl_index_t* ranked_genes,
    const int* in_gene_set,
    scl_index_t n_genes,
    scl_real_t enrichment_score,
    scl_index_t* leading_genes,
    scl_size_t max_leading_genes
);

// Over-Representation Analysis (ORA) - Single Set
scl_error_t scl_enrichment_ora_single_set(
    const scl_index_t* de_genes,
    scl_size_t n_de_genes,
    const scl_index_t* pathway_genes,
    scl_size_t n_pathway_genes,
    scl_index_t n_total_genes,
    scl_real_t* p_value,
    scl_real_t* odds_ratio,
    scl_real_t* fold_enrichment
);

// Batch ORA for Multiple Gene Sets
scl_error_t scl_enrichment_ora_batch(
    const scl_index_t* de_genes,
    scl_size_t n_de_genes,
    const scl_index_t* const* pathway_genes,
    const scl_index_t* pathway_sizes,
    scl_index_t n_pathways,
    scl_index_t n_total_genes,
    scl_real_t* p_values,
    scl_real_t* odds_ratios,
    scl_real_t* fold_enrichments
);

// Benjamini-Hochberg FDR Correction
scl_error_t scl_enrichment_benjamini_hochberg(
    const scl_real_t* p_values,
    scl_size_t n_pvalues,
    scl_real_t* q_values
);

// Bonferroni Correction
scl_error_t scl_enrichment_bonferroni(
    const scl_real_t* p_values,
    scl_size_t n_pvalues,
    scl_real_t* adjusted_p
);

// Pathway Activity Score (Per Cell)
scl_error_t scl_enrichment_pathway_activity(
    scl_sparse_matrix_t X,
    const scl_index_t* pathway_genes,
    scl_size_t n_pathway_genes,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* activity_scores
);

// GSVA-like Pathway Score
scl_error_t scl_enrichment_gsva_score(
    scl_sparse_matrix_t X,
    const scl_index_t* pathway_genes,
    scl_size_t n_pathway_genes,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* gsva_scores
);

// Rank Genes by Score
scl_error_t scl_enrichment_rank_genes_by_score(
    const scl_real_t* scores,
    scl_index_t n_genes,
    scl_index_t* ranked_genes
);

// Gene Set Overlap
scl_index_t scl_enrichment_gene_set_overlap(
    const scl_index_t* set1,
    scl_size_t n_set1,
    const scl_index_t* set2,
    scl_size_t n_set2,
    scl_index_t n_genes,
    scl_index_t* overlap_genes,
    scl_size_t max_overlap
);

// Jaccard Similarity Between Gene Sets
scl_real_t scl_enrichment_jaccard_similarity(
    const scl_index_t* set1,
    scl_size_t n_set1,
    const scl_index_t* set2,
    scl_size_t n_set2,
    scl_index_t n_genes
);

// Enrichment Map (Pairwise Pathway Similarity)
scl_error_t scl_enrichment_enrichment_map(
    const scl_index_t* const* pathway_genes,
    const scl_index_t* pathway_sizes,
    scl_index_t n_pathways,
    scl_index_t n_genes,
    scl_real_t* similarity_matrix
);

// Single-Sample GSEA (ssGSEA)
scl_error_t scl_enrichment_ssgsea(
    scl_sparse_matrix_t X,
    const scl_index_t* pathway_genes,
    scl_size_t n_pathway_genes,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* enrichment_scores,
    scl_real_t weight_exponent
);

// Filter Significant Pathways
scl_index_t scl_enrichment_filter_significant(
    const scl_real_t* p_values,
    scl_size_t n_pvalues,
    scl_real_t alpha,
    scl_index_t* significant_indices,
    scl_size_t max_significant
);

// Sort Pathways by P-Value
scl_error_t scl_enrichment_sort_by_pvalue(
    const scl_real_t* p_values,
    scl_size_t n_pvalues,
    scl_index_t* sorted_indices,
    scl_real_t* sorted_pvalues
);

#ifdef __cplusplus
}
#endif
