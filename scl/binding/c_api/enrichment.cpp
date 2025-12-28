#include "enrichment.h"
#include "scl/kernel/enrichment.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

static scl_error_t convert_error(scl::ErrorCode code) {
    return static_cast<scl_error_t>(code);
}

static scl::Sparse<scl::Real, true>* unwrap_matrix(scl_sparse_matrix_t mat) {
    return static_cast<scl::Sparse<scl::Real, true>*>(mat);
}

extern "C" {

scl_real_t scl_enrichment_hypergeometric_test(scl_index_t k, scl_index_t n, scl_index_t K, scl_index_t N) {
    return scl::kernel::enrichment::hypergeometric_test(k, n, K, N);
}

scl_real_t scl_enrichment_fisher_exact_test(scl_index_t a, scl_index_t b, scl_index_t c, scl_index_t d) {
    return scl::kernel::enrichment::fisher_exact_test(a, b, c, d);
}

scl_real_t scl_enrichment_odds_ratio(scl_index_t a, scl_index_t b, scl_index_t c, scl_index_t d) {
    return scl::kernel::enrichment::odds_ratio(a, b, c, d);
}

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
) {
    try {
        if (!ranked_genes || !ranking_scores || !in_gene_set || !enrichment_score || !p_value || !nes) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<const scl::Index> genes_arr(ranked_genes, static_cast<scl::Size>(n_genes));
        scl::Array<const scl::Real> scores_arr(ranking_scores, static_cast<scl::Size>(n_genes));
        scl::Array<const bool> in_set_arr(reinterpret_cast<const bool*>(in_gene_set), static_cast<scl::Size>(n_genes));
        scl::kernel::enrichment::gsea(genes_arr, scores_arr, in_set_arr, n_genes, *enrichment_score, *p_value, *nes, n_permutations, seed);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_enrichment_gsea_running_sum(
    const scl_index_t* ranked_genes,
    const int* in_gene_set,
    scl_index_t n_genes,
    scl_real_t* running_sum
) {
    try {
        if (!ranked_genes || !in_gene_set || !running_sum) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<const scl::Index> genes_arr(ranked_genes, static_cast<scl::Size>(n_genes));
        scl::Array<const bool> in_set_arr(reinterpret_cast<const bool*>(in_gene_set), static_cast<scl::Size>(n_genes));
        scl::Array<scl::Real> sum_arr(running_sum, static_cast<scl::Size>(n_genes));
        scl::kernel::enrichment::gsea_running_sum(genes_arr, in_set_arr, n_genes, sum_arr);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_index_t scl_enrichment_leading_edge_genes(
    const scl_index_t* ranked_genes,
    const int* in_gene_set,
    scl_index_t n_genes,
    scl_real_t enrichment_score,
    scl_index_t* leading_genes,
    scl_size_t max_leading_genes
) {
    try {
        if (!ranked_genes || !in_gene_set || !leading_genes) {
            return 0;
        }
        scl::Array<const scl::Index> genes_arr(ranked_genes, static_cast<scl::Size>(n_genes));
        scl::Array<const bool> in_set_arr(reinterpret_cast<const bool*>(in_gene_set), static_cast<scl::Size>(n_genes));
        scl::Array<scl::Index> leading_arr(leading_genes, max_leading_genes);
        return scl::kernel::enrichment::leading_edge_genes(genes_arr, in_set_arr, n_genes, enrichment_score, leading_arr);
    } catch (...) {
        return 0;
    }
}

scl_error_t scl_enrichment_ora_single_set(
    const scl_index_t* de_genes,
    scl_size_t n_de_genes,
    const scl_index_t* pathway_genes,
    scl_size_t n_pathway_genes,
    scl_index_t n_total_genes,
    scl_real_t* p_value,
    scl_real_t* odds_ratio,
    scl_real_t* fold_enrichment
) {
    try {
        if (!de_genes || !pathway_genes || !p_value || !odds_ratio || !fold_enrichment) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<const scl::Index> de_arr(de_genes, n_de_genes);
        scl::Array<const scl::Index> pathway_arr(pathway_genes, n_pathway_genes);
        scl::kernel::enrichment::ora_single_set(de_arr, pathway_arr, n_total_genes, *p_value, *odds_ratio, *fold_enrichment);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

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
) {
    try {
        if (!de_genes || !pathway_genes || !pathway_sizes || !p_values || !odds_ratios || !fold_enrichments) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<const scl::Index> de_arr(de_genes, n_de_genes);
        scl::Array<scl::Real> p_arr(p_values, static_cast<scl::Size>(n_pathways));
        scl::Array<scl::Real> or_arr(odds_ratios, static_cast<scl::Size>(n_pathways));
        scl::Array<scl::Real> fe_arr(fold_enrichments, static_cast<scl::Size>(n_pathways));
        scl::kernel::enrichment::ora_batch(de_arr, pathway_genes, pathway_sizes, n_pathways, n_total_genes, p_arr, or_arr, fe_arr);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_enrichment_benjamini_hochberg(
    const scl_real_t* p_values,
    scl_size_t n_pvalues,
    scl_real_t* q_values
) {
    try {
        if (!p_values || !q_values) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<const scl::Real> p_arr(p_values, n_pvalues);
        scl::Array<scl::Real> q_arr(q_values, n_pvalues);
        scl::kernel::enrichment::benjamini_hochberg(p_arr, q_arr);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_enrichment_bonferroni(
    const scl_real_t* p_values,
    scl_size_t n_pvalues,
    scl_real_t* adjusted_p
) {
    try {
        if (!p_values || !adjusted_p) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<const scl::Real> p_arr(p_values, n_pvalues);
        scl::Array<scl::Real> adj_arr(adjusted_p, n_pvalues);
        scl::kernel::enrichment::bonferroni(p_arr, adj_arr);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_enrichment_pathway_activity(
    scl_sparse_matrix_t X,
    const scl_index_t* pathway_genes,
    scl_size_t n_pathway_genes,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* activity_scores
) {
    try {
        auto* mat = unwrap_matrix(X);
        if (!mat || !pathway_genes || !activity_scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<const scl::Index> pathway_arr(pathway_genes, n_pathway_genes);
        scl::Array<scl::Real> scores_arr(activity_scores, static_cast<scl::Size>(n_cells));
        scl::kernel::enrichment::pathway_activity(*mat, pathway_arr, n_cells, n_genes, scores_arr);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_enrichment_gsva_score(
    scl_sparse_matrix_t X,
    const scl_index_t* pathway_genes,
    scl_size_t n_pathway_genes,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* gsva_scores
) {
    try {
        auto* mat = unwrap_matrix(X);
        if (!mat || !pathway_genes || !gsva_scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<const scl::Index> pathway_arr(pathway_genes, n_pathway_genes);
        scl::Array<scl::Real> scores_arr(gsva_scores, static_cast<scl::Size>(n_cells));
        scl::kernel::enrichment::gsva_score(*mat, pathway_arr, n_cells, n_genes, scores_arr);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_enrichment_rank_genes_by_score(
    const scl_real_t* scores,
    scl_index_t n_genes,
    scl_index_t* ranked_genes
) {
    try {
        if (!scores || !ranked_genes) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<const scl::Real> scores_arr(scores, static_cast<scl::Size>(n_genes));
        scl::Array<scl::Index> genes_arr(ranked_genes, static_cast<scl::Size>(n_genes));
        scl::kernel::enrichment::rank_genes_by_score(scores_arr, n_genes, genes_arr);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_index_t scl_enrichment_gene_set_overlap(
    const scl_index_t* set1,
    scl_size_t n_set1,
    const scl_index_t* set2,
    scl_size_t n_set2,
    scl_index_t n_genes,
    scl_index_t* overlap_genes,
    scl_size_t max_overlap
) {
    try {
        if (!set1 || !set2 || !overlap_genes) {
            return 0;
        }
        scl::Array<const scl::Index> set1_arr(set1, n_set1);
        scl::Array<const scl::Index> set2_arr(set2, n_set2);
        scl::Array<scl::Index> overlap_arr(overlap_genes, max_overlap);
        return scl::kernel::enrichment::gene_set_overlap(set1_arr, set2_arr, n_genes, overlap_arr);
    } catch (...) {
        return 0;
    }
}

scl_real_t scl_enrichment_jaccard_similarity(
    const scl_index_t* set1,
    scl_size_t n_set1,
    const scl_index_t* set2,
    scl_size_t n_set2,
    scl_index_t n_genes
) {
    try {
        if (!set1 || !set2) {
            return 0.0;
        }
        scl::Array<const scl::Index> set1_arr(set1, n_set1);
        scl::Array<const scl::Index> set2_arr(set2, n_set2);
        return scl::kernel::enrichment::jaccard_similarity(set1_arr, set2_arr, n_genes);
    } catch (...) {
        return 0.0;
    }
}

scl_error_t scl_enrichment_enrichment_map(
    const scl_index_t* const* pathway_genes,
    const scl_index_t* pathway_sizes,
    scl_index_t n_pathways,
    scl_index_t n_genes,
    scl_real_t* similarity_matrix
) {
    try {
        if (!pathway_genes || !pathway_sizes || !similarity_matrix) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Size total = static_cast<scl::Size>(n_pathways) * static_cast<scl::Size>(n_pathways);
        scl::Array<scl::Real> sim_arr(similarity_matrix, total);
        scl::kernel::enrichment::enrichment_map(pathway_genes, pathway_sizes, n_pathways, n_genes, sim_arr);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_error_t scl_enrichment_ssgsea(
    scl_sparse_matrix_t X,
    const scl_index_t* pathway_genes,
    scl_size_t n_pathway_genes,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* enrichment_scores,
    scl_real_t weight_exponent
) {
    try {
        auto* mat = unwrap_matrix(X);
        if (!mat || !pathway_genes || !enrichment_scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<const scl::Index> pathway_arr(pathway_genes, n_pathway_genes);
        scl::Array<scl::Real> scores_arr(enrichment_scores, static_cast<scl::Size>(n_cells));
        scl::kernel::enrichment::ssgsea(*mat, pathway_arr, n_cells, n_genes, scores_arr, weight_exponent);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

scl_index_t scl_enrichment_filter_significant(
    const scl_real_t* p_values,
    scl_size_t n_pvalues,
    scl_real_t alpha,
    scl_index_t* significant_indices,
    scl_size_t max_significant
) {
    try {
        if (!p_values || !significant_indices) {
            return 0;
        }
        scl::Array<const scl::Real> p_arr(p_values, n_pvalues);
        scl::Array<scl::Index> indices_arr(significant_indices, max_significant);
        return scl::kernel::enrichment::filter_significant(p_arr, alpha, indices_arr);
    } catch (...) {
        return 0;
    }
}

scl_error_t scl_enrichment_sort_by_pvalue(
    const scl_real_t* p_values,
    scl_size_t n_pvalues,
    scl_index_t* sorted_indices,
    scl_real_t* sorted_pvalues
) {
    try {
        if (!p_values || !sorted_indices || !sorted_pvalues) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        scl::Array<const scl::Real> p_arr(p_values, n_pvalues);
        scl::Array<scl::Index> indices_arr(sorted_indices, n_pvalues);
        scl::Array<scl::Real> sorted_arr(sorted_pvalues, n_pvalues);
        scl::kernel::enrichment::sort_by_pvalue(p_arr, indices_arr, sorted_arr);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return convert_error(e.code());
    } catch (...) {
        return SCL_ERROR_INTERNAL_ERROR;
    }
}

} // extern "C"
