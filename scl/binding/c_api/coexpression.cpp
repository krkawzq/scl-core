// =============================================================================
// FILE: scl/binding/c_api/coexpression.cpp
// BRIEF: C API implementation for co-expression network analysis
// =============================================================================

#include "scl/binding/c_api/coexpression.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/coexpression.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

using namespace scl;
using namespace scl::binding;

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

/// @brief Convert C correlation type to C++ enum
[[nodiscard]] constexpr auto convert_correlation_type(
    scl_coexpr_correlation_t t) noexcept -> scl::kernel::coexpression::CorrelationType {
    switch (t) {
        case SCL_COEXPR_PEARSON:
            return scl::kernel::coexpression::CorrelationType::Pearson;
        case SCL_COEXPR_SPEARMAN:
            return scl::kernel::coexpression::CorrelationType::Spearman;
        case SCL_COEXPR_BICOR:
            return scl::kernel::coexpression::CorrelationType::Bicor;
        default:
            return scl::kernel::coexpression::CorrelationType::Pearson;
    }
}

/// @brief Convert C adjacency type to C++ enum
[[nodiscard]] constexpr auto convert_adjacency_type(
    scl_coexpr_adjacency_t t) noexcept -> scl::kernel::coexpression::AdjacencyType {
    switch (t) {
        case SCL_COEXPR_UNSIGNED:
            return scl::kernel::coexpression::AdjacencyType::Unsigned;
        case SCL_COEXPR_SIGNED:
            return scl::kernel::coexpression::AdjacencyType::Signed;
        case SCL_COEXPR_SIGNED_HYBRID:
            return scl::kernel::coexpression::AdjacencyType::SignedHybrid;
        default:
            return scl::kernel::coexpression::AdjacencyType::Unsigned;
    }
}

} // anonymous namespace

extern "C" {

// =============================================================================
// Correlation Matrix
// =============================================================================

SCL_EXPORT scl_error_t scl_coexpr_correlation_matrix(
    scl_sparse_t expression,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    scl_real_t* corr_matrix,
    const scl_coexpr_correlation_t corr_type) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(corr_matrix, "Output correlation matrix is null");
    SCL_C_API_CHECK(n_cells > 0 && n_genes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        expression->visit([&](auto& m) {
            scl::kernel::coexpression::correlation_matrix(
                m, n_cells, n_genes,
                reinterpret_cast<Real*>(corr_matrix),
                convert_correlation_type(corr_type)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// WGCNA Adjacency
// =============================================================================

SCL_EXPORT scl_error_t scl_coexpr_wgcna_adjacency(
    scl_sparse_t expression,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    const scl_real_t power,
    scl_real_t* adjacency,
    const scl_coexpr_correlation_t corr_type,
    const scl_coexpr_adjacency_t adj_type) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(adjacency, "Output adjacency matrix is null");
    SCL_C_API_CHECK(n_cells > 0 && n_genes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    SCL_C_API_CHECK(power >= 1 && power <= 20, SCL_ERROR_INVALID_ARGUMENT,
                   "Power should be in range [1, 20]");
    
    SCL_C_API_TRY
        expression->visit([&](auto& m) {
            scl::kernel::coexpression::wgcna_adjacency(
                m, n_cells, n_genes,
                static_cast<Real>(power),
                reinterpret_cast<Real*>(adjacency),
                convert_correlation_type(corr_type),
                convert_adjacency_type(adj_type)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Topological Overlap Matrix
// =============================================================================

SCL_EXPORT scl_error_t scl_coexpr_topological_overlap(
    const scl_real_t* adjacency,
    const scl_index_t n_genes,
    scl_real_t* tom) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(tom, "Output TOM matrix is null");
    SCL_C_API_CHECK(n_genes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of genes must be positive");
    
    SCL_C_API_TRY
        scl::kernel::coexpression::topological_overlap_matrix(
            reinterpret_cast<const Real*>(adjacency),
            n_genes,
            reinterpret_cast<Real*>(tom)
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_coexpr_tom_dissimilarity(
    const scl_real_t* tom,
    const scl_index_t n_genes,
    scl_real_t* dissim) {
    
    SCL_C_API_CHECK_NULL(tom, "TOM matrix is null");
    SCL_C_API_CHECK_NULL(dissim, "Output dissimilarity matrix is null");
    SCL_C_API_CHECK(n_genes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of genes must be positive");
    
    SCL_C_API_TRY
        scl::kernel::coexpression::tom_dissimilarity(
            reinterpret_cast<const Real*>(tom),
            n_genes,
            reinterpret_cast<Real*>(dissim)
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Module Detection
// =============================================================================

SCL_EXPORT scl_error_t scl_coexpr_detect_modules(
    const scl_real_t* dissim,
    const scl_index_t n_genes,
    scl_index_t* module_labels,
    scl_index_t* n_modules,
    const scl_index_t min_module_size,
    const scl_real_t merge_cut_height) {
    
    SCL_C_API_CHECK_NULL(dissim, "Dissimilarity matrix is null");
    SCL_C_API_CHECK_NULL(module_labels, "Output module labels array is null");
    SCL_C_API_CHECK_NULL(n_modules, "Output n_modules pointer is null");
    SCL_C_API_CHECK(n_genes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of genes must be positive");
    SCL_C_API_CHECK(min_module_size > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Minimum module size must be positive");
    
    SCL_C_API_TRY
        const Index n_mods = scl::kernel::coexpression::detect_modules(
            reinterpret_cast<const Real*>(dissim),
            n_genes,
            reinterpret_cast<Index*>(module_labels),
            min_module_size,
            static_cast<Real>(merge_cut_height)
        );
        
        *n_modules = n_mods;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Module Eigengenes
// =============================================================================

SCL_EXPORT scl_error_t scl_coexpr_module_eigengene(
    scl_sparse_t expression,
    const scl_index_t* module_labels,
    const scl_index_t module_id,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    scl_real_t* eigengene) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(module_labels, "Module labels array is null");
    SCL_C_API_CHECK_NULL(eigengene, "Output eigengene array is null");
    SCL_C_API_CHECK(n_cells > 0 && n_genes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        const Array<const Index> labels_arr(
            reinterpret_cast<const Index*>(module_labels),
            static_cast<Size>(n_genes)
        );
        Array<Real> eig_arr(
            reinterpret_cast<Real*>(eigengene),
            static_cast<Size>(n_cells)
        );
        
        expression->visit([&](auto& m) {
            scl::kernel::coexpression::module_eigengene(
                m, labels_arr, module_id, n_cells, n_genes, eig_arr
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_coexpr_all_eigengenes(
    scl_sparse_t expression,
    const scl_index_t* module_labels,
    const scl_index_t n_modules,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    scl_real_t* eigengenes) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(module_labels, "Module labels array is null");
    SCL_C_API_CHECK_NULL(eigengenes, "Output eigengenes array is null");
    SCL_C_API_CHECK(n_cells > 0 && n_genes > 0 && n_modules > 0,
                   SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        const Array<const Index> labels_arr(
            reinterpret_cast<const Index*>(module_labels),
            static_cast<Size>(n_genes)
        );
        
        expression->visit([&](auto& m) {
            scl::kernel::coexpression::all_module_eigengenes(
                m, labels_arr, n_modules, n_cells, n_genes,
                reinterpret_cast<Real*>(eigengenes)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Module-Trait Correlation
// =============================================================================

SCL_EXPORT scl_error_t scl_coexpr_module_trait_correlation(
    const scl_real_t* eigengenes,
    const scl_real_t* traits,
    const scl_index_t n_samples,
    const scl_index_t n_modules,
    const scl_index_t n_traits,
    scl_real_t* correlations,
    scl_real_t* p_values) {
    
    SCL_C_API_CHECK_NULL(eigengenes, "Eigengenes array is null");
    SCL_C_API_CHECK_NULL(traits, "Traits array is null");
    SCL_C_API_CHECK_NULL(correlations, "Output correlations array is null");
    SCL_C_API_CHECK(n_samples > 0 && n_modules > 0 && n_traits > 0,
                   SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        scl::kernel::coexpression::module_trait_correlation(
            reinterpret_cast<const Real*>(eigengenes),
            reinterpret_cast<const Real*>(traits),
            n_samples, n_modules, n_traits,
            reinterpret_cast<Real*>(correlations),
            p_values ? reinterpret_cast<Real*>(p_values) : nullptr
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Hub Gene Identification
// =============================================================================

SCL_EXPORT scl_error_t scl_coexpr_identify_hub_genes(
    const scl_real_t* adjacency,
    const scl_index_t* module_labels,
    const scl_index_t module_id,
    const scl_index_t n_genes,
    scl_index_t* hub_genes,
    scl_real_t* hub_scores,
    scl_index_t* n_hubs,
    const scl_index_t max_hubs) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(module_labels, "Module labels array is null");
    SCL_C_API_CHECK_NULL(hub_genes, "Output hub genes array is null");
    SCL_C_API_CHECK_NULL(hub_scores, "Output hub scores array is null");
    SCL_C_API_CHECK_NULL(n_hubs, "Output n_hubs pointer is null");
    SCL_C_API_CHECK(n_genes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of genes must be positive");
    SCL_C_API_CHECK(max_hubs > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Max hubs must be positive");
    
    SCL_C_API_TRY
        const Array<const Index> labels_arr(
            reinterpret_cast<const Index*>(module_labels),
            static_cast<Size>(n_genes)
        );
        
        Index n_hubs_result = 0;
        
        scl::kernel::coexpression::identify_hub_genes(
            reinterpret_cast<const Real*>(adjacency),
            labels_arr, module_id, n_genes,
            reinterpret_cast<Index*>(hub_genes),
            reinterpret_cast<Real*>(hub_scores),
            max_hubs, n_hubs_result
        );
        
        *n_hubs = n_hubs_result;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Gene-Module Membership
// =============================================================================

SCL_EXPORT scl_error_t scl_coexpr_gene_module_membership(
    scl_sparse_t expression,
    const scl_real_t* eigengenes,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    const scl_index_t n_modules,
    scl_real_t* kme_matrix) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(eigengenes, "Eigengenes array is null");
    SCL_C_API_CHECK_NULL(kme_matrix, "Output kME matrix is null");
    SCL_C_API_CHECK(n_cells > 0 && n_genes > 0 && n_modules > 0,
                   SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        expression->visit([&](auto& m) {
            scl::kernel::coexpression::gene_module_membership(
                m,
                reinterpret_cast<const Real*>(eigengenes),
                n_cells, n_genes, n_modules,
                reinterpret_cast<Real*>(kme_matrix)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Soft Threshold Selection
// =============================================================================

SCL_EXPORT scl_error_t scl_coexpr_pick_soft_threshold(
    scl_sparse_t expression,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    const scl_real_t* powers_to_test,
    const scl_index_t n_powers,
    scl_real_t* scale_free_fits,
    scl_real_t* mean_connectivity,
    scl_real_t* best_power,
    const scl_coexpr_correlation_t corr_type) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(powers_to_test, "Powers to test array is null");
    SCL_C_API_CHECK_NULL(scale_free_fits, "Output scale free fits array is null");
    SCL_C_API_CHECK_NULL(mean_connectivity, "Output mean connectivity array is null");
    SCL_C_API_CHECK_NULL(best_power, "Output best power pointer is null");
    SCL_C_API_CHECK(n_cells > 0 && n_genes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    SCL_C_API_CHECK(n_powers > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of powers must be positive");
    
    SCL_C_API_TRY
        Real best = Real(0);
        
        expression->visit([&](auto& m) {
            // PERFORMANCE: const_cast needed for internal sorting
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
            auto* powers_mutable = const_cast<Real*>(
                reinterpret_cast<const Real*>(powers_to_test)
            );
            
            best = scl::kernel::coexpression::pick_soft_threshold(
                m, n_cells, n_genes,
                powers_mutable, n_powers,
                reinterpret_cast<Real*>(scale_free_fits),
                reinterpret_cast<Real*>(mean_connectivity),
                convert_correlation_type(corr_type)
            );
        });
        
        *best_power = static_cast<scl_real_t>(best);
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Blockwise Module Detection
// =============================================================================

SCL_EXPORT scl_error_t scl_coexpr_blockwise_modules(
    scl_sparse_t expression,
    const scl_index_t n_cells,
    const scl_index_t n_genes,
    const scl_index_t block_size,
    const scl_real_t power,
    const scl_index_t min_module_size,
    scl_index_t* module_labels,
    scl_index_t* n_modules,
    const scl_coexpr_correlation_t corr_type) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(module_labels, "Output module labels array is null");
    SCL_C_API_CHECK_NULL(n_modules, "Output n_modules pointer is null");
    SCL_C_API_CHECK(n_cells > 0 && n_genes > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    SCL_C_API_CHECK(block_size > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Block size must be positive");
    SCL_C_API_CHECK(min_module_size > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Minimum module size must be positive");
    
    SCL_C_API_TRY
        Index n_mods = 0;
        
        expression->visit([&](auto& m) {
            scl::kernel::coexpression::blockwise_modules(
                m, n_cells, n_genes, block_size,
                static_cast<Real>(power),
                min_module_size,
                reinterpret_cast<Index*>(module_labels),
                n_mods,
                convert_correlation_type(corr_type)
            );
        });
        
        *n_modules = n_mods;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Module Preservation
// =============================================================================

SCL_EXPORT scl_error_t scl_coexpr_module_preservation(
    const scl_real_t* adjacency_ref,
    const scl_real_t* adjacency_test,
    const scl_index_t* module_labels,
    const scl_index_t n_genes,
    const scl_index_t n_modules,
    scl_real_t* zsummary) {
    
    SCL_C_API_CHECK_NULL(adjacency_ref, "Reference adjacency matrix is null");
    SCL_C_API_CHECK_NULL(adjacency_test, "Test adjacency matrix is null");
    SCL_C_API_CHECK_NULL(module_labels, "Module labels array is null");
    SCL_C_API_CHECK_NULL(zsummary, "Output z-summary array is null");
    SCL_C_API_CHECK(n_genes > 0 && n_modules > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        const Array<const Index> labels_arr(
            reinterpret_cast<const Index*>(module_labels),
            static_cast<Size>(n_genes)
        );
        
        scl::kernel::coexpression::module_preservation(
            reinterpret_cast<const Real*>(adjacency_ref),
            reinterpret_cast<const Real*>(adjacency_test),
            labels_arr, n_genes, n_modules,
            reinterpret_cast<Real*>(zsummary)
        );
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
