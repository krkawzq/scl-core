// =============================================================================
// FILE: scl/binding/c_api/association.cpp
// BRIEF: C API implementation for feature association analysis
// =============================================================================

#include "scl/binding/c_api/association.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/association.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Gene-Peak Correlation
// =============================================================================

SCL_EXPORT scl_error_t scl_association_gene_peak_correlation(
    scl_sparse_t rna_expression,
    scl_sparse_t atac_accessibility,
    scl_index_t* gene_indices,
    scl_index_t* peak_indices,
    scl_real_t* correlations,
    scl_size_t* n_correlations,
    const scl_real_t min_correlation) {
    
    SCL_C_API_CHECK_NULL(rna_expression, "RNA expression matrix is null");
    SCL_C_API_CHECK_NULL(atac_accessibility, "ATAC accessibility matrix is null");
    SCL_C_API_CHECK_NULL(gene_indices, "Output gene indices array is null");
    SCL_C_API_CHECK_NULL(peak_indices, "Output peak indices array is null");
    SCL_C_API_CHECK_NULL(correlations, "Output correlations array is null");
    SCL_C_API_CHECK_NULL(n_correlations, "Output n_correlations pointer is null");
    
    SCL_C_API_TRY
        Size n_corr = 0;
        
        rna_expression->visit([&](auto& rna) {
            atac_accessibility->visit([&](auto& atac) {
                scl::kernel::association::gene_peak_correlation(
                    rna, atac,
                    reinterpret_cast<Index*>(gene_indices),
                    reinterpret_cast<Index*>(peak_indices),
                    reinterpret_cast<Real*>(correlations),
                    n_corr,
                    static_cast<Real>(min_correlation)
                );
            });
        });
        
        *n_correlations = n_corr;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Cis-Regulatory Associations
// =============================================================================

SCL_EXPORT scl_error_t scl_association_cis_regulatory(
    scl_sparse_t rna_expression,
    scl_sparse_t atac_accessibility,
    const scl_index_t* gene_indices,
    const scl_index_t* peak_indices,
    const scl_size_t n_pairs,
    scl_real_t* correlations,
    scl_real_t* p_values) {
    
    SCL_C_API_CHECK_NULL(rna_expression, "RNA expression matrix is null");
    SCL_C_API_CHECK_NULL(atac_accessibility, "ATAC accessibility matrix is null");
    SCL_C_API_CHECK_NULL(gene_indices, "Gene indices array is null");
    SCL_C_API_CHECK_NULL(peak_indices, "Peak indices array is null");
    SCL_C_API_CHECK_NULL(correlations, "Output correlations array is null");
    SCL_C_API_CHECK_NULL(p_values, "Output p_values array is null");
    
    SCL_C_API_TRY
        rna_expression->visit([&](auto& rna) {
            atac_accessibility->visit([&](auto& atac) {
                scl::kernel::association::cis_regulatory(
                    rna, atac,
                    reinterpret_cast<const Index*>(gene_indices),
                    reinterpret_cast<const Index*>(peak_indices),
                    n_pairs,
                    reinterpret_cast<Real*>(correlations),
                    reinterpret_cast<Real*>(p_values)
                );
            });
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Enhancer-Gene Links
// =============================================================================

SCL_EXPORT scl_error_t scl_association_enhancer_gene_link(
    scl_sparse_t rna,
    scl_sparse_t atac,
    const scl_real_t correlation_threshold,
    scl_index_t* link_genes,
    scl_index_t* link_peaks,
    scl_real_t* link_correlations,
    scl_size_t* n_links) {
    
    SCL_C_API_CHECK_NULL(rna, "RNA matrix is null");
    SCL_C_API_CHECK_NULL(atac, "ATAC matrix is null");
    SCL_C_API_CHECK_NULL(link_genes, "Output link_genes array is null");
    SCL_C_API_CHECK_NULL(link_peaks, "Output link_peaks array is null");
    SCL_C_API_CHECK_NULL(link_correlations, "Output link_correlations array is null");
    SCL_C_API_CHECK_NULL(n_links, "Output n_links pointer is null");
    
    SCL_C_API_TRY
        Size n_links_result = 0;
        
        rna->visit([&](auto& r) {
            atac->visit([&](auto& a) {
                scl::kernel::association::enhancer_gene_link(
                    r, a,
                    static_cast<Real>(correlation_threshold),
                    reinterpret_cast<Index*>(link_genes),
                    reinterpret_cast<Index*>(link_peaks),
                    reinterpret_cast<Real*>(link_correlations),
                    n_links_result
                );
            });
        });
        
        *n_links = n_links_result;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Multi-modal Neighbors
// =============================================================================

SCL_EXPORT scl_error_t scl_association_multimodal_neighbors(
    scl_sparse_t modality1,
    scl_sparse_t modality2,
    const scl_real_t weight1,
    const scl_real_t weight2,
    const scl_index_t k,
    scl_index_t* neighbor_indices,
    scl_real_t* neighbor_distances) {
    
    SCL_C_API_CHECK_NULL(modality1, "Modality1 matrix is null");
    SCL_C_API_CHECK_NULL(modality2, "Modality2 matrix is null");
    SCL_C_API_CHECK_NULL(neighbor_indices, "Output neighbor indices array is null");
    SCL_C_API_CHECK_NULL(neighbor_distances, "Output neighbor distances array is null");
    
    SCL_C_API_TRY
        modality1->visit([&](auto& m1) {
            modality2->visit([&](auto& m2) {
                scl::kernel::association::multimodal_neighbors(
                    m1, m2,
                    static_cast<Real>(weight1),
                    static_cast<Real>(weight2),
                    k,
                    reinterpret_cast<Index*>(neighbor_indices),
                    reinterpret_cast<Real*>(neighbor_distances)
                );
            });
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Feature Coupling
// =============================================================================

SCL_EXPORT scl_error_t scl_association_feature_coupling(
    scl_sparse_t modality1,
    scl_sparse_t modality2,
    scl_index_t* feature1_indices,
    scl_index_t* feature2_indices,
    scl_real_t* coupling_scores,
    scl_size_t* n_couplings,
    const scl_real_t min_score) {
    
    SCL_C_API_CHECK_NULL(modality1, "Modality1 matrix is null");
    SCL_C_API_CHECK_NULL(modality2, "Modality2 matrix is null");
    SCL_C_API_CHECK_NULL(feature1_indices, "Output feature1 indices array is null");
    SCL_C_API_CHECK_NULL(feature2_indices, "Output feature2 indices array is null");
    SCL_C_API_CHECK_NULL(coupling_scores, "Output coupling scores array is null");
    SCL_C_API_CHECK_NULL(n_couplings, "Output n_couplings pointer is null");
    
    SCL_C_API_TRY
        Size n_coup = 0;
        
        modality1->visit([&](auto& m1) {
            modality2->visit([&](auto& m2) {
                scl::kernel::association::feature_coupling(
                    m1, m2,
                    reinterpret_cast<Index*>(feature1_indices),
                    reinterpret_cast<Index*>(feature2_indices),
                    reinterpret_cast<Real*>(coupling_scores),
                    n_coup,
                    static_cast<Real>(min_score)
                );
            });
        });
        
        *n_couplings = n_coup;
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Peak-to-Gene Activity
// =============================================================================

SCL_EXPORT scl_error_t scl_association_peak_to_gene_activity(
    scl_sparse_t atac,
    const scl_index_t* peak_to_gene_map,
    const scl_size_t n_peaks,
    const scl_size_t n_genes,
    scl_real_t* gene_activity) {
    
    SCL_C_API_CHECK_NULL(atac, "ATAC matrix is null");
    SCL_C_API_CHECK_NULL(peak_to_gene_map, "Peak to gene map array is null");
    SCL_C_API_CHECK_NULL(gene_activity, "Output gene activity array is null");
    
    SCL_C_API_TRY
        atac->visit([&](auto& a) {
            scl::kernel::association::peak_to_gene_activity(
                a,
                reinterpret_cast<const Index*>(peak_to_gene_map),
                n_peaks,
                n_genes,
                reinterpret_cast<Real*>(gene_activity)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Correlation in Subset
// =============================================================================

SCL_EXPORT scl_error_t scl_association_correlation_in_subset(
    scl_sparse_t data1,
    const scl_index_t feature1,
    scl_sparse_t data2,
    const scl_index_t feature2,
    const scl_index_t* cell_indices,
    const scl_size_t n_subset,
    scl_real_t* correlation) {
    
    SCL_C_API_CHECK_NULL(data1, "Data1 matrix is null");
    SCL_C_API_CHECK_NULL(data2, "Data2 matrix is null");
    SCL_C_API_CHECK_NULL(cell_indices, "Cell indices array is null");
    SCL_C_API_CHECK_NULL(correlation, "Output correlation pointer is null");
    
    SCL_C_API_TRY
        Real corr = Real(0);
        
        data1->visit([&](auto& m1) {
            data2->visit([&](auto& m2) {
                scl::kernel::association::correlation_in_subset(
                    m1, feature1,
                    m2, feature2,
                    Array<const Index>(
                        reinterpret_cast<const Index*>(cell_indices),
                        n_subset
                    ),
                    corr
                );
            });
        });
        
        *correlation = static_cast<scl_real_t>(corr);
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
