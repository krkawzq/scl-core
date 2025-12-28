// =============================================================================
// FILE: scl/binding/c_api/scoring/scoring.cpp
// BRIEF: C API implementation for gene set scoring
// =============================================================================

#include "scl/binding/c_api/scoring.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/scoring.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::scoring;

extern "C" {

// =============================================================================
// Helper: Convert scoring method
// =============================================================================

static ScoringMethod convert_scoring_method(scl_scoring_method_t method) {
    switch (method) {
        case SCL_SCORING_MEAN: return ScoringMethod::Mean;
        case SCL_SCORING_RANK_BASED: return ScoringMethod::RankBased;
        case SCL_SCORING_WEIGHTED: return ScoringMethod::Weighted;
        case SCL_SCORING_SEURAT_MODULE: return ScoringMethod::SeuratModule;
        case SCL_SCORING_ZSCORE: return ScoringMethod::ZScore;
        default: return ScoringMethod::Mean;
    }
}

// =============================================================================
// Gene Set Score
// =============================================================================

scl_error_t scl_scoring_gene_set_score(
    scl_sparse_t expression,
    const scl_index_t* gene_set,
    scl_size_t n_genes_in_set,
    scl_real_t* scores,
    scl_size_t n_cells,
    scl_index_t n_genes,
    scl_scoring_method_t method,
    scl_real_t quantile)
{
    if (!expression || !gene_set || !scores) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(expression);
        
        Array<const Index> gene_set_arr(
            reinterpret_cast<const Index*>(gene_set),
            n_genes_in_set
        );
        Array<Real> scores_arr(
            reinterpret_cast<Real*>(scores),
            n_cells
        );

        wrapper->visit([&](auto& expr) {
            gene_set_score(
                expr, gene_set_arr,
                convert_scoring_method(method),
                scores_arr,
                n_cells, n_genes,
                static_cast<Real>(quantile)
            );
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Mean Score
// =============================================================================

scl_error_t scl_scoring_mean_score(
    scl_sparse_t expression,
    const scl_index_t* gene_set,
    scl_size_t n_genes_in_set,
    scl_real_t* scores,
    scl_size_t n_cells,
    scl_index_t n_genes)
{
    if (!expression || !gene_set || !scores) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(expression);
        
        Array<const Index> gene_set_arr(
            reinterpret_cast<const Index*>(gene_set),
            n_genes_in_set
        );
        Array<Real> scores_arr(
            reinterpret_cast<Real*>(scores),
            n_cells
        );

        wrapper->visit([&](auto& expr) {
            mean_score(
                expr, gene_set_arr, scores_arr,
                n_cells, n_genes
            );
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// AUC Score
// =============================================================================

scl_error_t scl_scoring_auc_score(
    scl_sparse_t expression,
    const scl_index_t* gene_set,
    scl_size_t n_genes_in_set,
    scl_real_t* scores,
    scl_size_t n_cells,
    scl_index_t n_genes,
    scl_real_t quantile)
{
    if (!expression || !gene_set || !scores) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(expression);
        
        Array<const Index> gene_set_arr(
            reinterpret_cast<const Index*>(gene_set),
            n_genes_in_set
        );
        Array<Real> scores_arr(
            reinterpret_cast<Real*>(scores),
            n_cells
        );

        wrapper->visit([&](auto& expr) {
            auc_score(
                expr, gene_set_arr, scores_arr,
                n_cells, n_genes,
                static_cast<Real>(quantile)
            );
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Module Score
// =============================================================================

scl_error_t scl_scoring_module_score(
    scl_sparse_t expression,
    const scl_index_t* gene_set,
    scl_size_t n_genes_in_set,
    scl_real_t* scores,
    scl_size_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_control_per_gene,
    scl_index_t n_bins)
{
    if (!expression || !gene_set || !scores) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(expression);
        
        Array<const Index> gene_set_arr(
            reinterpret_cast<const Index*>(gene_set),
            n_genes_in_set
        );
        Array<Real> scores_arr(
            reinterpret_cast<Real*>(scores),
            n_cells
        );

        wrapper->visit([&](auto& expr) {
            module_score(
                expr, gene_set_arr, scores_arr,
                n_cells, n_genes,
                n_control_per_gene, n_bins,
                42  // seed
            );
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

// =============================================================================
// Differential Score
// =============================================================================

scl_error_t scl_scoring_differential_score(
    scl_sparse_t expression,
    const scl_index_t* positive_genes,
    scl_size_t n_positive,
    const scl_index_t* negative_genes,
    scl_size_t n_negative,
    scl_real_t* scores,
    scl_size_t n_cells,
    scl_index_t n_genes)
{
    if (!expression || !positive_genes || !negative_genes || !scores) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(expression);
        
        Array<const Index> pos_arr(
            reinterpret_cast<const Index*>(positive_genes),
            n_positive
        );
        Array<const Index> neg_arr(
            reinterpret_cast<const Index*>(negative_genes),
            n_negative
        );
        Array<Real> scores_arr(
            reinterpret_cast<Real*>(scores),
            n_cells
        );

        wrapper->visit([&](auto& expr) {
            differential_score(
                expr, pos_arr, neg_arr, scores_arr,
                n_cells, n_genes
            );
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

