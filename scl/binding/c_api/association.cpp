// =============================================================================
// FILE: scl/binding/c_api/association.cpp
// BRIEF: C API implementation for feature association analysis
// =============================================================================

#include "scl/binding/c_api/association.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/association.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

static scl_error_t get_sparse_matrix(
    scl_sparse_t handle,
    scl::binding::SparseWrapper*& wrapper
) {
    if (!handle) {
        return SCL_ERROR_NULL_POINTER;
    }
    wrapper = static_cast<scl::binding::SparseWrapper*>(handle);
    if (!wrapper->valid()) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    return SCL_OK;
}

scl_error_t scl_association_gene_peak_correlation(
    scl_sparse_t rna_expression,
    scl_sparse_t atac_accessibility,
    scl_index_t* gene_indices,
    scl_index_t* peak_indices,
    scl_real_t* correlations,
    scl_size_t* n_correlations,
    scl_real_t min_correlation
) {
    if (!rna_expression || !atac_accessibility || !gene_indices ||
        !peak_indices || !correlations || !n_correlations) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper_rna;
        scl::binding::SparseWrapper* wrapper_atac;
        scl_error_t err1 = get_sparse_matrix(rna_expression, wrapper_rna);
        scl_error_t err2 = get_sparse_matrix(atac_accessibility, wrapper_atac);
        if (err1 != SCL_OK) return err1;
        if (err2 != SCL_OK) return err2;

        scl::Size n_corr = 0;
        wrapper_rna->visit([&](auto& rna) {
            wrapper_atac->visit([&](auto& atac) {
                scl::kernel::association::gene_peak_correlation(
                    rna, atac,
                    reinterpret_cast<scl::Index*>(gene_indices),
                    reinterpret_cast<scl::Index*>(peak_indices),
                    reinterpret_cast<scl::Real*>(correlations),
                    n_corr,
                    static_cast<scl::Real>(min_correlation)
                );
            });
        });
        *n_correlations = static_cast<scl_size_t>(n_corr);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_association_cis_regulatory(
    scl_sparse_t rna_expression,
    scl_sparse_t atac_accessibility,
    const scl_index_t* gene_indices,
    const scl_index_t* peak_indices,
    scl_size_t n_pairs,
    scl_real_t* correlations,
    scl_real_t* p_values
) {
    if (!rna_expression || !atac_accessibility || !gene_indices ||
        !peak_indices || !correlations || !p_values) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper_rna;
        scl::binding::SparseWrapper* wrapper_atac;
        scl_error_t err1 = get_sparse_matrix(rna_expression, wrapper_rna);
        scl_error_t err2 = get_sparse_matrix(atac_accessibility, wrapper_atac);
        if (err1 != SCL_OK) return err1;
        if (err2 != SCL_OK) return err2;

        wrapper_rna->visit([&](auto& rna) {
            wrapper_atac->visit([&](auto& atac) {
                scl::kernel::association::cis_regulatory(
                    rna, atac,
                    reinterpret_cast<const scl::Index*>(gene_indices),
                    reinterpret_cast<const scl::Index*>(peak_indices),
                    static_cast<scl::Size>(n_pairs),
                    reinterpret_cast<scl::Real*>(correlations),
                    reinterpret_cast<scl::Real*>(p_values)
                );
            });
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_association_enhancer_gene_link(
    scl_sparse_t rna,
    scl_sparse_t atac,
    scl_real_t correlation_threshold,
    scl_index_t* link_genes,
    scl_index_t* link_peaks,
    scl_real_t* link_correlations,
    scl_size_t* n_links
) {
    if (!rna || !atac || !link_genes || !link_peaks || !link_correlations || !n_links) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper_rna;
        scl::binding::SparseWrapper* wrapper_atac;
        scl_error_t err1 = get_sparse_matrix(rna, wrapper_rna);
        scl_error_t err2 = get_sparse_matrix(atac, wrapper_atac);
        if (err1 != SCL_OK) return err1;
        if (err2 != SCL_OK) return err2;

        scl::Size n_links_result = 0;
        wrapper_rna->visit([&](auto& r) {
            wrapper_atac->visit([&](auto& a) {
                scl::kernel::association::enhancer_gene_link(
                    r, a,
                    static_cast<scl::Real>(correlation_threshold),
                    reinterpret_cast<scl::Index*>(link_genes),
                    reinterpret_cast<scl::Index*>(link_peaks),
                    reinterpret_cast<scl::Real*>(link_correlations),
                    n_links_result
                );
            });
        });
        *n_links = static_cast<scl_size_t>(n_links_result);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_association_multimodal_neighbors(
    scl_sparse_t modality1,
    scl_sparse_t modality2,
    scl_real_t weight1,
    scl_real_t weight2,
    scl_index_t k,
    scl_index_t* neighbor_indices,
    scl_real_t* neighbor_distances
) {
    if (!modality1 || !modality2 || !neighbor_indices || !neighbor_distances) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper1;
        scl::binding::SparseWrapper* wrapper2;
        scl_error_t err1 = get_sparse_matrix(modality1, wrapper1);
        scl_error_t err2 = get_sparse_matrix(modality2, wrapper2);
        if (err1 != SCL_OK) return err1;
        if (err2 != SCL_OK) return err2;

        wrapper1->visit([&](auto& m1) {
            wrapper2->visit([&](auto& m2) {
                scl::kernel::association::multimodal_neighbors(
                    m1, m2,
                    static_cast<scl::Real>(weight1),
                    static_cast<scl::Real>(weight2),
                    static_cast<scl::Index>(k),
                    reinterpret_cast<scl::Index*>(neighbor_indices),
                    reinterpret_cast<scl::Real*>(neighbor_distances)
                );
            });
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_association_feature_coupling(
    scl_sparse_t modality1,
    scl_sparse_t modality2,
    scl_index_t* feature1_indices,
    scl_index_t* feature2_indices,
    scl_real_t* coupling_scores,
    scl_size_t* n_couplings,
    scl_real_t min_score
) {
    if (!modality1 || !modality2 || !feature1_indices ||
        !feature2_indices || !coupling_scores || !n_couplings) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper1;
        scl::binding::SparseWrapper* wrapper2;
        scl_error_t err1 = get_sparse_matrix(modality1, wrapper1);
        scl_error_t err2 = get_sparse_matrix(modality2, wrapper2);
        if (err1 != SCL_OK) return err1;
        if (err2 != SCL_OK) return err2;

        scl::Size n_coup = 0;
        wrapper1->visit([&](auto& m1) {
            wrapper2->visit([&](auto& m2) {
                scl::kernel::association::feature_coupling(
                    m1, m2,
                    reinterpret_cast<scl::Index*>(feature1_indices),
                    reinterpret_cast<scl::Index*>(feature2_indices),
                    reinterpret_cast<scl::Real*>(coupling_scores),
                    n_coup,
                    static_cast<scl::Real>(min_score)
                );
            });
        });
        *n_couplings = static_cast<scl_size_t>(n_coup);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_association_peak_to_gene_activity(
    scl_sparse_t atac,
    const scl_index_t* peak_to_gene_map,
    scl_size_t n_peaks,
    scl_size_t n_genes,
    scl_real_t* gene_activity
) {
    if (!atac || !peak_to_gene_map || !gene_activity) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(atac, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& a) {
            scl::kernel::association::peak_to_gene_activity(
                a,
                reinterpret_cast<const scl::Index*>(peak_to_gene_map),
                static_cast<scl::Size>(n_peaks),
                static_cast<scl::Size>(n_genes),
                reinterpret_cast<scl::Real*>(gene_activity)
            );
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
