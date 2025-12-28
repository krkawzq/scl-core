// =============================================================================
// FILE: scl/binding/c_api/kernels/markers.cpp
// BRIEF: C API implementation for marker gene selection
// =============================================================================

#include "markers.h"
#include "scl/kernel/markers.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/type.hpp"

namespace {

inline scl::ErrorCode to_error_code(scl_error_t code) {
    return static_cast<scl::ErrorCode>(code);
}

inline scl_error_t from_error_code(scl::ErrorCode code) {
    return static_cast<scl_error_t>(code);
}

inline scl::CSR* to_sparse_csr(scl_sparse_matrix_t handle) {
    return static_cast<scl::CSR*>(handle);
}

inline scl::CSC* to_sparse_csc(scl_sparse_matrix_t handle) {
    return static_cast<scl::CSC*>(handle);
}

inline scl::kernel::markers::RankingMethod to_ranking_method(scl_markers_ranking_method_t method) {
    switch (method) {
        case SCL_MARKERS_RANKING_FOLD_CHANGE: return scl::kernel::markers::RankingMethod::FoldChange;
        case SCL_MARKERS_RANKING_EFFECT_SIZE: return scl::kernel::markers::RankingMethod::EffectSize;
        case SCL_MARKERS_RANKING_P_VALUE: return scl::kernel::markers::RankingMethod::PValue;
        case SCL_MARKERS_RANKING_COMBINED: return scl::kernel::markers::RankingMethod::Combined;
        default: return scl::kernel::markers::RankingMethod::FoldChange;
    }
}

} // anonymous namespace

extern "C" {

scl_error_t scl_markers_group_mean_expression(
    scl_sparse_matrix_t X,
    const scl_index_t* group_labels,
    scl_index_t n_cells,
    scl_index_t n_groups,
    scl_index_t n_genes,
    scl_real_t* mean_expr
) {
    try {
        scl::CSR* matrix = to_sparse_csr(X);
        if (!matrix || !group_labels || !mean_expr) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> labels_array(group_labels, static_cast<scl::Size>(n_cells));
        scl::Array<scl::Real> mean_array(mean_expr, static_cast<scl::Size>(n_genes) * static_cast<scl::Size>(n_groups));
        
        scl::kernel::markers::group_mean_expression(*matrix, labels_array, n_groups, mean_array, n_genes);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_markers_percent_expressed(
    scl_sparse_matrix_t X,
    const scl_index_t* group_labels,
    scl_index_t n_cells,
    scl_index_t n_groups,
    scl_index_t n_genes,
    scl_real_t* pct_expr,
    scl_real_t threshold
) {
    try {
        scl::CSR* matrix = to_sparse_csr(X);
        if (!matrix || !group_labels || !pct_expr) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> labels_array(group_labels, static_cast<scl::Size>(n_cells));
        scl::Array<scl::Real> pct_array(pct_expr, static_cast<scl::Size>(n_genes) * static_cast<scl::Size>(n_groups));
        
        scl::kernel::markers::percent_expressed(*matrix, labels_array, n_groups, pct_array, n_genes, threshold);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_markers_log_fold_change(
    scl_sparse_matrix_t X,
    const scl_index_t* group_labels,
    scl_index_t n_cells,
    scl_index_t n_groups,
    scl_index_t target_group,
    scl_index_t n_genes,
    scl_real_t* log_fc,
    scl_real_t pseudo_count
) {
    try {
        scl::CSR* matrix = to_sparse_csr(X);
        if (!matrix || !group_labels || !log_fc) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> labels_array(group_labels, static_cast<scl::Size>(n_cells));
        scl::Array<scl::Real> log_fc_array(log_fc, static_cast<scl::Size>(n_genes));
        
        scl::kernel::markers::log_fold_change(*matrix, labels_array, n_groups, target_group, log_fc_array, n_genes, pseudo_count);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_markers_one_vs_rest_stats(
    scl_sparse_matrix_t X,
    const scl_index_t* group_labels,
    scl_index_t n_cells,
    scl_index_t n_groups,
    scl_index_t target_group,
    scl_index_t n_genes,
    scl_real_t* log_fc,
    scl_real_t* effect_size,
    scl_real_t* pct_in,
    scl_real_t* pct_out
) {
    try {
        scl::CSR* matrix = to_sparse_csr(X);
        if (!matrix || !group_labels || !log_fc || !effect_size || !pct_in || !pct_out) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> labels_array(group_labels, static_cast<scl::Size>(n_cells));
        scl::Array<scl::Real> log_fc_array(log_fc, static_cast<scl::Size>(n_genes));
        scl::Array<scl::Real> effect_array(effect_size, static_cast<scl::Size>(n_genes));
        scl::Array<scl::Real> pct_in_array(pct_in, static_cast<scl::Size>(n_genes));
        scl::Array<scl::Real> pct_out_array(pct_out, static_cast<scl::Size>(n_genes));
        
        scl::kernel::markers::one_vs_rest_stats(*matrix, labels_array, n_groups, target_group,
                                                log_fc_array, effect_array, pct_in_array, pct_out_array, n_genes);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_markers_rank_genes_groups(
    scl_sparse_matrix_t X,
    const scl_index_t* group_labels,
    scl_index_t n_cells,
    scl_index_t n_groups,
    scl_index_t n_genes,
    scl_markers_ranking_method_t method,
    scl_index_t* ranked_indices,
    scl_real_t* ranked_scores
) {
    try {
        scl::CSR* matrix = to_sparse_csr(X);
        if (!matrix || !group_labels || !ranked_indices || !ranked_scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> labels_array(group_labels, static_cast<scl::Size>(n_cells));
        scl::Array<scl::Index> indices_array(ranked_indices, static_cast<scl::Size>(n_groups) * static_cast<scl::Size>(n_genes));
        scl::Array<scl::Real> scores_array(ranked_scores, static_cast<scl::Size>(n_groups) * static_cast<scl::Size>(n_genes));
        
        scl::kernel::markers::rank_genes_groups(*matrix, labels_array, n_groups, n_genes,
                                               to_ranking_method(method), indices_array, scores_array);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_markers_tau_specificity(
    const scl_real_t* group_means,
    scl_index_t n_genes,
    scl_index_t n_groups,
    scl_real_t* tau_scores
) {
    try {
        if (!group_means || !tau_scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Real> means_array(group_means, static_cast<scl::Size>(n_genes) * static_cast<scl::Size>(n_groups));
        scl::Array<scl::Real> scores_array(tau_scores, static_cast<scl::Size>(n_genes));
        
        scl::kernel::markers::tau_specificity(means_array, n_genes, n_groups, scores_array);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_markers_gini_specificity(
    const scl_real_t* group_means,
    scl_index_t n_genes,
    scl_index_t n_groups,
    scl_real_t* gini_scores
) {
    try {
        if (!group_means || !gini_scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Real> means_array(group_means, static_cast<scl::Size>(n_genes) * static_cast<scl::Size>(n_groups));
        scl::Array<scl::Real> scores_array(gini_scores, static_cast<scl::Size>(n_genes));
        
        scl::kernel::markers::gini_specificity(means_array, n_genes, n_groups, scores_array);
        return SCL_ERROR_OK;
    } catch (const scl::Exception& e) {
        return from_error_code(e.code());
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

} // extern "C"
