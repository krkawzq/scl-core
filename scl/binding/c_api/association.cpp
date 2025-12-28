// =============================================================================
// FILE: scl/binding/c_api/association.cpp
// BRIEF: C API implementation for feature association analysis
// =============================================================================

#include "association.h"
#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/kernel/association.hpp"

namespace {
    inline scl_error_t catch_exception() {
        try {
            throw;
        } catch (const scl::DimensionError&) {
            return SCL_ERROR_DIMENSION_MISMATCH;
        } catch (const scl::ValueError&) {
            return SCL_ERROR_INVALID_ARGUMENT;
        } catch (const scl::Exception& e) {
            return static_cast<scl_error_t>(e.code());
        } catch (...) {
            return SCL_ERROR_UNKNOWN;
        }
    }

    template <typename T>
    inline scl::Sparse<T, true>* get_matrix(scl_sparse_matrix_t handle) {
        return reinterpret_cast<scl::Sparse<T, true>*>(handle);
    }
}

extern "C" scl_error_t scl_gene_peak_correlation_f32_csr(
    scl_sparse_matrix_t rna_expression,
    scl_sparse_matrix_t atac_accessibility,
    scl_index_t* gene_indices,
    scl_index_t* peak_indices,
    scl_real_t* correlations,
    scl_size_t max_correlations,
    scl_size_t* n_correlations,
    scl_real_t min_correlation
) {
    try {
        auto* rna = get_matrix<float>(rna_expression);
        auto* atac = get_matrix<float>(atac_accessibility);
        if (!rna || !atac || !gene_indices || !peak_indices || !correlations || !n_correlations) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Size n_corr = 0;
        scl::kernel::association::gene_peak_correlation(
            *rna, *atac, gene_indices, peak_indices, correlations, n_corr, min_correlation
        );
        *n_correlations = (n_corr > max_correlations) ? max_correlations : n_corr;
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_gene_peak_correlation_f64_csr(
    scl_sparse_matrix_t rna_expression,
    scl_sparse_matrix_t atac_accessibility,
    scl_index_t* gene_indices,
    scl_index_t* peak_indices,
    double* correlations,
    scl_size_t max_correlations,
    scl_size_t* n_correlations,
    double min_correlation
) {
    try {
        auto* rna = get_matrix<double>(rna_expression);
        auto* atac = get_matrix<double>(atac_accessibility);
        if (!rna || !atac || !gene_indices || !peak_indices || !correlations || !n_correlations) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Size n_corr = 0;
        scl::kernel::association::gene_peak_correlation(
            *rna, *atac, gene_indices, peak_indices, correlations, n_corr, min_correlation
        );
        *n_correlations = (n_corr > max_correlations) ? max_correlations : n_corr;
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_enhancer_gene_link_f32_csr(
    scl_sparse_matrix_t rna,
    scl_sparse_matrix_t atac,
    scl_real_t correlation_threshold,
    scl_index_t* link_genes,
    scl_index_t* link_peaks,
    scl_real_t* link_correlations,
    scl_size_t max_links,
    scl_size_t* n_links
) {
    try {
        auto* rna_mat = get_matrix<float>(rna);
        auto* atac_mat = get_matrix<float>(atac);
        if (!rna_mat || !atac_mat || !link_genes || !link_peaks || !link_correlations || !n_links) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Size n_links_val = 0;
        scl::kernel::association::enhancer_gene_link(
            *rna_mat, *atac_mat, correlation_threshold,
            link_genes, link_peaks, link_correlations, n_links_val
        );
        *n_links = (n_links_val > max_links) ? max_links : n_links_val;
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_enhancer_gene_link_f64_csr(
    scl_sparse_matrix_t rna,
    scl_sparse_matrix_t atac,
    double correlation_threshold,
    scl_index_t* link_genes,
    scl_index_t* link_peaks,
    double* link_correlations,
    scl_size_t max_links,
    scl_size_t* n_links
) {
    try {
        auto* rna_mat = get_matrix<double>(rna);
        auto* atac_mat = get_matrix<double>(atac);
        if (!rna_mat || !atac_mat || !link_genes || !link_peaks || !link_correlations || !n_links) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Size n_links_val = 0;
        scl::kernel::association::enhancer_gene_link(
            *rna_mat, *atac_mat, correlation_threshold,
            link_genes, link_peaks, link_correlations, n_links_val
        );
        *n_links = (n_links_val > max_links) ? max_links : n_links_val;
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_multimodal_neighbors_f32_csr(
    scl_sparse_matrix_t modality1,
    scl_sparse_matrix_t modality2,
    scl_real_t weight1,
    scl_real_t weight2,
    scl_index_t k,
    scl_index_t* neighbor_indices,
    scl_real_t* neighbor_distances
) {
    try {
        auto* m1 = get_matrix<float>(modality1);
        auto* m2 = get_matrix<float>(modality2);
        if (!m1 || !m2 || !neighbor_indices || !neighbor_distances) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::association::multimodal_neighbors(
            *m1, *m2, weight1, weight2, k, neighbor_indices, neighbor_distances
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_multimodal_neighbors_f64_csr(
    scl_sparse_matrix_t modality1,
    scl_sparse_matrix_t modality2,
    double weight1,
    double weight2,
    scl_index_t k,
    scl_index_t* neighbor_indices,
    double* neighbor_distances
) {
    try {
        auto* m1 = get_matrix<double>(modality1);
        auto* m2 = get_matrix<double>(modality2);
        if (!m1 || !m2 || !neighbor_indices || !neighbor_distances) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::association::multimodal_neighbors(
            *m1, *m2, weight1, weight2, k, neighbor_indices, neighbor_distances
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}
