// =============================================================================
// FILE: scl/binding/c_api/enrichment/enrichment.cpp
// BRIEF: C API implementation for gene set enrichment analysis
// =============================================================================

#include "scl/binding/c_api/enrichment.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/enrichment.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

namespace {
    using namespace scl::kernel::enrichment;
    constexpr scl::Index DEFAULT_N_PERMUTATIONS = 1000;
    constexpr scl::Real DEFAULT_WEIGHT_EXPONENT = scl::Real(0.25);
}

// =============================================================================
// Statistical Tests
// =============================================================================

scl_error_t scl_enrichment_hypergeometric_test(
    scl_index_t k, scl_index_t n, scl_index_t K, scl_index_t N,
    scl_real_t* p_value_out
) {
    if (!p_value_out) return SCL_ERROR_NULL_POINTER;
    try {
        *p_value_out = static_cast<scl_real_t>(
            hypergeometric_test(k, n, K, N)
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_enrichment_fisher_exact_test(
    scl_index_t a, scl_index_t b, scl_index_t c, scl_index_t d,
    scl_real_t* p_value_out
) {
    if (!p_value_out) return SCL_ERROR_NULL_POINTER;
    try {
        *p_value_out = static_cast<scl_real_t>(
            fisher_exact_test(a, b, c, d)
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_enrichment_odds_ratio(
    scl_index_t a, scl_index_t b, scl_index_t c, scl_index_t d,
    scl_real_t* odds_ratio_out
) {
    if (!odds_ratio_out) return SCL_ERROR_NULL_POINTER;
    try {
        *odds_ratio_out = static_cast<scl_real_t>(
            odds_ratio(a, b, c, d)
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// GSEA
// =============================================================================

scl_error_t scl_enrichment_gsea(
    const scl_index_t* ranked_genes,
    const scl_real_t* ranking_scores,
    const int* in_gene_set,
    scl_index_t n_genes,
    scl_real_t* enrichment_score_out,
    scl_real_t* p_value_out,
    scl_real_t* nes_out,
    scl_index_t n_permutations,
    uint64_t seed
) {
    if (!ranked_genes || !ranking_scores || !in_gene_set ||
        !enrichment_score_out || !p_value_out || !nes_out) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Index> ranked_arr(
            reinterpret_cast<const scl::Index*>(ranked_genes),
            static_cast<scl::Size>(n_genes)
        );
        scl::Array<const scl::Real> scores_arr(
            reinterpret_cast<const scl::Real*>(ranking_scores),
            static_cast<scl::Size>(n_genes)
        );
        scl::Array<const bool> in_set_arr(
            reinterpret_cast<const bool*>(in_gene_set),
            static_cast<scl::Size>(n_genes)
        );

        scl::Real es, pval, nes;
        scl::Index n_perm = (n_permutations == 0) ?
            DEFAULT_N_PERMUTATIONS : n_permutations;

        gsea(ranked_arr, scores_arr, in_set_arr, n_genes,
             es, pval, nes, n_perm, seed);

        *enrichment_score_out = static_cast<scl_real_t>(es);
        *p_value_out = static_cast<scl_real_t>(pval);
        *nes_out = static_cast<scl_real_t>(nes);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_enrichment_gsea_running_sum(
    const scl_index_t* ranked_genes,
    const int* in_gene_set,
    scl_index_t n_genes,
    scl_real_t* running_sum
) {
    if (!ranked_genes || !in_gene_set || !running_sum) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Index> ranked_arr(
            reinterpret_cast<const scl::Index*>(ranked_genes),
            static_cast<scl::Size>(n_genes)
        );
        scl::Array<const bool> in_set_arr(
            reinterpret_cast<const bool*>(in_gene_set),
            static_cast<scl::Size>(n_genes)
        );
        scl::Array<scl::Real> running_arr(
            reinterpret_cast<scl::Real*>(running_sum),
            static_cast<scl::Size>(n_genes)
        );

        gsea_running_sum(ranked_arr, in_set_arr, n_genes, running_arr);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_enrichment_leading_edge_genes(
    const scl_index_t* ranked_genes,
    const int* in_gene_set,
    scl_index_t n_genes,
    scl_real_t enrichment_score,
    scl_index_t* leading_genes,
    scl_size_t max_leading,
    scl_index_t* n_leading_out
) {
    if (!ranked_genes || !in_gene_set || !leading_genes || !n_leading_out) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Index> ranked_arr(
            reinterpret_cast<const scl::Index*>(ranked_genes),
            static_cast<scl::Size>(n_genes)
        );
        scl::Array<const bool> in_set_arr(
            reinterpret_cast<const bool*>(in_gene_set),
            static_cast<scl::Size>(n_genes)
        );
        scl::Array<scl::Index> leading_arr(
            reinterpret_cast<scl::Index*>(leading_genes),
            max_leading
        );

        *n_leading_out = leading_edge_genes(
            ranked_arr, in_set_arr, n_genes,
            static_cast<scl::Real>(enrichment_score),
            leading_arr
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// ORA
// =============================================================================

scl_error_t scl_enrichment_ora_single_set(
    const scl_index_t* de_genes,
    scl_size_t n_de_genes,
    const scl_index_t* pathway_genes,
    scl_size_t n_pathway_genes,
    scl_index_t n_total_genes,
    scl_real_t* p_value_out,
    scl_real_t* odds_ratio_out,
    scl_real_t* fold_enrichment_out
) {
    if (!de_genes || !pathway_genes || !p_value_out ||
        !odds_ratio_out || !fold_enrichment_out) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Index> de_arr(
            reinterpret_cast<const scl::Index*>(de_genes),
            n_de_genes
        );
        scl::Array<const scl::Index> pathway_arr(
            reinterpret_cast<const scl::Index*>(pathway_genes),
            n_pathway_genes
        );

        scl::Real pval, or_val, fe;
        ora_single_set(de_arr, pathway_arr, n_total_genes, pval, or_val, fe);

        *p_value_out = static_cast<scl_real_t>(pval);
        *odds_ratio_out = static_cast<scl_real_t>(or_val);
        *fold_enrichment_out = static_cast<scl_real_t>(fe);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
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
    if (!de_genes || !pathway_genes || !pathway_sizes ||
        !p_values || !odds_ratios || !fold_enrichments) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Index> de_arr(
            reinterpret_cast<const scl::Index*>(de_genes),
            n_de_genes
        );

        const scl::Index** pathways = const_cast<const scl::Index**>(
            reinterpret_cast<const scl::Index* const*>(pathway_genes)
        );

        scl::Array<scl::Real> pvals_arr(
            reinterpret_cast<scl::Real*>(p_values),
            static_cast<scl::Size>(n_pathways)
        );
        scl::Array<scl::Real> or_arr(
            reinterpret_cast<scl::Real*>(odds_ratios),
            static_cast<scl::Size>(n_pathways)
        );
        scl::Array<scl::Real> fe_arr(
            reinterpret_cast<scl::Real*>(fold_enrichments),
            static_cast<scl::Size>(n_pathways)
        );

        ora_batch(de_arr, pathways,
                 reinterpret_cast<const scl::Index*>(pathway_sizes),
                 n_pathways, n_total_genes, pvals_arr, or_arr, fe_arr);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// Multiple Testing Correction
// =============================================================================

scl_error_t scl_enrichment_benjamini_hochberg(
    const scl_real_t* p_values,
    scl_size_t n,
    scl_real_t* q_values
) {
    if (!p_values || !q_values) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Array<const scl::Real> pvals_arr(
            reinterpret_cast<const scl::Real*>(p_values), n
        );
        scl::Array<scl::Real> qvals_arr(
            reinterpret_cast<scl::Real*>(q_values), n
        );
        benjamini_hochberg(pvals_arr, qvals_arr);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_enrichment_bonferroni(
    const scl_real_t* p_values,
    scl_size_t n,
    scl_real_t* adjusted_p
) {
    if (!p_values || !adjusted_p) return SCL_ERROR_NULL_POINTER;
    try {
        scl::Array<const scl::Real> pvals_arr(
            reinterpret_cast<const scl::Real*>(p_values), n
        );
        scl::Array<scl::Real> adj_arr(
            reinterpret_cast<scl::Real*>(adjusted_p), n
        );
        bonferroni(pvals_arr, adj_arr);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

// =============================================================================
// Pathway Activity
// =============================================================================

scl_error_t scl_enrichment_pathway_activity(
    scl_sparse_t X,
    const scl_index_t* pathway_genes,
    scl_size_t n_pathway_genes,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* activity_scores
) {
    if (!X || !pathway_genes || !activity_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Index> pathway_arr(
            reinterpret_cast<const scl::Index*>(pathway_genes),
            n_pathway_genes
        );
        scl::Array<scl::Real> scores_arr(
            reinterpret_cast<scl::Real*>(activity_scores),
            static_cast<scl::Size>(n_cells)
        );

        X->visit([&](auto& mat) {
            pathway_activity(mat, pathway_arr, n_cells, n_genes, scores_arr);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_enrichment_gsva_score(
    scl_sparse_t X,
    const scl_index_t* pathway_genes,
    scl_size_t n_pathway_genes,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* gsva_scores
) {
    if (!X || !pathway_genes || !gsva_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Index> pathway_arr(
            reinterpret_cast<const scl::Index*>(pathway_genes),
            n_pathway_genes
        );
        scl::Array<scl::Real> scores_arr(
            reinterpret_cast<scl::Real*>(gsva_scores),
            static_cast<scl::Size>(n_cells)
        );

        X->visit([&](auto& mat) {
            gsva_score(mat, pathway_arr, n_cells, n_genes, scores_arr);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_enrichment_ssgsea(
    scl_sparse_t X,
    const scl_index_t* pathway_genes,
    scl_size_t n_pathway_genes,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* enrichment_scores,
    scl_real_t weight_exponent
) {
    if (!X || !pathway_genes || !enrichment_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Array<const scl::Index> pathway_arr(
            reinterpret_cast<const scl::Index*>(pathway_genes),
            n_pathway_genes
        );
        scl::Array<scl::Real> scores_arr(
            reinterpret_cast<scl::Real*>(enrichment_scores),
            static_cast<scl::Size>(n_cells)
        );
        scl::Real w = (weight_exponent == scl::Real(0)) ?
            DEFAULT_WEIGHT_EXPONENT : weight_exponent;

        X->visit([&](auto& mat) {
            ssgsea(mat, pathway_arr, n_cells, n_genes, scores_arr, w);
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
