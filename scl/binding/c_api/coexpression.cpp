// =============================================================================
// FILE: scl/binding/c_api/coexpression/coexpression.cpp
// BRIEF: C API implementation for co-expression analysis
// =============================================================================

#include "scl/binding/c_api/coexpression.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/coexpression.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::coexpression;

namespace {
    CorrelationType convert_corr_type(scl_correlation_type_t t) {
        switch (t) {
            case SCL_CORR_PEARSON: return CorrelationType::Pearson;
            case SCL_CORR_SPEARMAN: return CorrelationType::Spearman;
            case SCL_CORR_BICOR: return CorrelationType::Bicor;
            default: return CorrelationType::Pearson;
        }
    }
    
    AdjacencyType convert_adj_type(scl_adjacency_type_t t) {
        switch (t) {
            case SCL_ADJ_UNSIGNED: return AdjacencyType::Unsigned;
            case SCL_ADJ_SIGNED: return AdjacencyType::Signed;
            case SCL_ADJ_SIGNED_HYBRID: return AdjacencyType::SignedHybrid;
            default: return AdjacencyType::Unsigned;
        }
    }
}

extern "C" {

scl_error_t scl_correlation_matrix(
    scl_sparse_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* corr_matrix,
    scl_correlation_type_t corr_type)
{
    if (!expression || !corr_matrix) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(expression);
        wrapper->visit([&](auto& m) {
            correlation_matrix(m, n_cells, n_genes, reinterpret_cast<Real*>(corr_matrix), convert_corr_type(corr_type));
        });
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_wgcna_adjacency(
    scl_sparse_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t power,
    scl_real_t* adjacency,
    scl_correlation_type_t corr_type,
    scl_adjacency_type_t adj_type)
{
    if (!expression || !adjacency) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(expression);
        wrapper->visit([&](auto& m) {
            wgcna_adjacency(m, n_cells, n_genes, power, reinterpret_cast<Real*>(adjacency),
                           convert_corr_type(corr_type), convert_adj_type(adj_type));
        });
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_topological_overlap_matrix(
    const scl_real_t* adjacency,
    scl_index_t n_genes,
    scl_real_t* tom)
{
    if (!adjacency || !tom) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        topological_overlap_matrix(reinterpret_cast<const Real*>(adjacency), n_genes, reinterpret_cast<Real*>(tom));
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_tom_dissimilarity(
    const scl_real_t* tom,
    scl_index_t n_genes,
    scl_real_t* dissim)
{
    if (!tom || !dissim) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        tom_dissimilarity(reinterpret_cast<const Real*>(tom), n_genes, reinterpret_cast<Real*>(dissim));
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_detect_modules(
    const scl_real_t* dissim,
    scl_index_t n_genes,
    scl_index_t* module_labels,
    scl_index_t* n_modules,
    scl_index_t min_module_size,
    scl_real_t merge_cut_height)
{
    if (!dissim || !module_labels || !n_modules) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Index n_mods = detect_modules(reinterpret_cast<const Real*>(dissim), n_genes,
                                      reinterpret_cast<Index*>(module_labels),
                                      min_module_size, merge_cut_height);
        *n_modules = n_mods;
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_module_eigengene(
    scl_sparse_t expression,
    const scl_index_t* module_labels,
    scl_index_t module_id,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* eigengene)
{
    if (!expression || !module_labels || !eigengene) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(expression);
        Array<const Index> labels_arr(reinterpret_cast<const Index*>(module_labels), static_cast<Size>(n_genes));
        Array<Real> eig_arr(reinterpret_cast<Real*>(eigengene), static_cast<Size>(n_cells));
        
        wrapper->visit([&](auto& m) {
            module_eigengene(m, labels_arr, module_id, n_cells, n_genes, eig_arr);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_all_module_eigengenes(
    scl_sparse_t expression,
    const scl_index_t* module_labels,
    scl_index_t n_modules,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_real_t* eigengenes)
{
    if (!expression || !module_labels || !eigengenes) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(expression);
        Array<const Index> labels_arr(reinterpret_cast<const Index*>(module_labels), static_cast<Size>(n_genes));
        
        wrapper->visit([&](auto& m) {
            all_module_eigengenes(m, labels_arr, n_modules, n_cells, n_genes, reinterpret_cast<Real*>(eigengenes));
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_module_trait_correlation(
    const scl_real_t* eigengenes,
    const scl_real_t* traits,
    scl_index_t n_samples,
    scl_index_t n_modules,
    scl_index_t n_traits,
    scl_real_t* correlations,
    scl_real_t* p_values)
{
    if (!eigengenes || !traits || !correlations) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        module_trait_correlation(reinterpret_cast<const Real*>(eigengenes),
                                 reinterpret_cast<const Real*>(traits),
                                 n_samples, n_modules, n_traits,
                                 reinterpret_cast<Real*>(correlations),
                                 p_values ? reinterpret_cast<Real*>(p_values) : nullptr);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_identify_hub_genes(
    const scl_real_t* adjacency,
    const scl_index_t* module_labels,
    scl_index_t module_id,
    scl_index_t n_genes,
    scl_index_t* hub_genes,
    scl_real_t* hub_scores,
    scl_index_t* n_hubs,
    scl_index_t max_hubs)
{
    if (!adjacency || !module_labels || !hub_genes || !hub_scores || !n_hubs) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Index> labels_arr(reinterpret_cast<const Index*>(module_labels), static_cast<Size>(n_genes));
        Array<Index> genes_arr(reinterpret_cast<Index*>(hub_genes), static_cast<Size>(max_hubs));
        Array<Real> scores_arr(reinterpret_cast<Real*>(hub_scores), static_cast<Size>(max_hubs));
        Index n;
        identify_hub_genes(reinterpret_cast<const Real*>(adjacency), labels_arr, module_id,
                          n_genes, genes_arr.ptr, scores_arr.ptr, max_hubs, n);
        *n_hubs = n;
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_gene_module_membership(
    scl_sparse_t expression,
    const scl_real_t* eigengenes,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_modules,
    scl_real_t* kme_matrix)
{
    if (!expression || !eigengenes || !kme_matrix) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(expression);
        wrapper->visit([&](auto& m) {
            gene_module_membership(m, reinterpret_cast<const Real*>(eigengenes),
                                  n_cells, n_genes, n_modules, reinterpret_cast<Real*>(kme_matrix));
        });
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_pick_soft_threshold(
    scl_sparse_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    const scl_real_t* powers_to_test,
    scl_index_t n_powers,
    scl_real_t* scale_free_fits,
    scl_real_t* mean_connectivity,
    scl_real_t* best_power,
    scl_correlation_type_t corr_type)
{
    if (!expression || !powers_to_test || !scale_free_fits || !mean_connectivity || !best_power) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(expression);
        Real best = wrapper->visit([&](auto& m) -> Real {
            return pick_soft_threshold(m, n_cells, n_genes,
                                      reinterpret_cast<const Real*>(powers_to_test), n_powers,
                                      reinterpret_cast<Real*>(scale_free_fits),
                                      reinterpret_cast<Real*>(mean_connectivity),
                                      convert_corr_type(corr_type));
        });
        *best_power = best;
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_blockwise_modules(
    scl_sparse_t expression,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t block_size,
    scl_real_t power,
    scl_index_t min_module_size,
    scl_index_t* module_labels,
    scl_index_t* n_modules,
    scl_correlation_type_t corr_type)
{
    if (!expression || !module_labels || !n_modules) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(expression);
        Index n_mods;
        wrapper->visit([&](auto& m) {
            blockwise_modules(m, n_cells, n_genes, block_size, power, min_module_size,
                             reinterpret_cast<Index*>(module_labels), n_mods,
                             convert_corr_type(corr_type));
        });
        *n_modules = n_mods;
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

