// =============================================================================
// FILE: scl/binding/c_api/annotation.cpp
// BRIEF: C API implementation for cell type annotation
// =============================================================================

#include "annotation.h"
#include "scl/core/type.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/kernel/annotation.hpp"

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

extern "C" scl_error_t scl_reference_mapping_f32_csr(
    scl_sparse_matrix_t query_expression,
    scl_sparse_matrix_t reference_expression,
    const scl_index_t* reference_labels,
    scl_size_t n_ref,
    scl_sparse_matrix_t query_to_ref_neighbors,
    scl_index_t n_query,
    scl_index_t n_types,
    scl_index_t* query_labels,
    scl_real_t* confidence_scores
) {
    try {
        auto* query = get_matrix<float>(query_expression);
        auto* ref = get_matrix<float>(reference_expression);
        auto* neighbors = get_matrix<scl::Index>(query_to_ref_neighbors);
        if (!query || !ref || !neighbors || !reference_labels || !query_labels || !confidence_scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> ref_labels_arr(reference_labels, n_ref);
        scl::Array<scl::Index> query_labels_arr(query_labels, n_query);
        scl::Array<scl::Real> conf_arr(confidence_scores, n_query);

        scl::kernel::annotation::reference_mapping(
            *query, *ref, ref_labels_arr, *neighbors,
            n_query, static_cast<scl::Index>(n_ref), n_types,
            query_labels_arr, conf_arr
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_reference_mapping_f64_csr(
    scl_sparse_matrix_t query_expression,
    scl_sparse_matrix_t reference_expression,
    const scl_index_t* reference_labels,
    scl_size_t n_ref,
    scl_sparse_matrix_t query_to_ref_neighbors,
    scl_index_t n_query,
    scl_index_t n_types,
    scl_index_t* query_labels,
    double* confidence_scores
) {
    try {
        auto* query = get_matrix<double>(query_expression);
        auto* ref = get_matrix<double>(reference_expression);
        auto* neighbors = get_matrix<scl::Index>(query_to_ref_neighbors);
        if (!query || !ref || !neighbors || !reference_labels || !query_labels || !confidence_scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> ref_labels_arr(reference_labels, n_ref);
        scl::Array<scl::Index> query_labels_arr(query_labels, n_query);
        scl::Array<double> conf_arr(confidence_scores, n_query);

        scl::kernel::annotation::reference_mapping(
            *query, *ref, ref_labels_arr, *neighbors,
            n_query, static_cast<scl::Index>(n_ref), n_types,
            query_labels_arr, conf_arr
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_correlation_assignment_f32_csr(
    scl_sparse_matrix_t query_expression,
    scl_sparse_matrix_t reference_profiles,
    scl_index_t n_query,
    scl_index_t n_types,
    scl_index_t n_genes,
    scl_index_t* assigned_labels,
    scl_real_t* correlation_scores,
    scl_real_t* all_correlations
) {
    try {
        auto* query = get_matrix<float>(query_expression);
        auto* profiles = get_matrix<float>(reference_profiles);
        if (!query || !profiles || !assigned_labels || !correlation_scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<scl::Index> labels_arr(assigned_labels, n_query);
        scl::Array<scl::Real> scores_arr(correlation_scores, n_query);
        scl::Array<scl::Real> all_arr(all_correlations, all_correlations ? static_cast<scl::Size>(n_query) * n_types : 0);

        scl::kernel::annotation::correlation_assignment(
            *query, *profiles, n_query, n_types, n_genes,
            labels_arr, scores_arr, all_arr
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_correlation_assignment_f64_csr(
    scl_sparse_matrix_t query_expression,
    scl_sparse_matrix_t reference_profiles,
    scl_index_t n_query,
    scl_index_t n_types,
    scl_index_t n_genes,
    scl_index_t* assigned_labels,
    double* correlation_scores,
    double* all_correlations
) {
    try {
        auto* query = get_matrix<double>(query_expression);
        auto* profiles = get_matrix<double>(reference_profiles);
        if (!query || !profiles || !assigned_labels || !correlation_scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<scl::Index> labels_arr(assigned_labels, n_query);
        scl::Array<double> scores_arr(correlation_scores, n_query);
        scl::Array<double> all_arr(all_correlations, all_correlations ? static_cast<scl::Size>(n_query) * n_types : 0);

        scl::kernel::annotation::correlation_assignment(
            *query, *profiles, n_query, n_types, n_genes,
            labels_arr, scores_arr, all_arr
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_build_reference_profiles_f32_csr(
    scl_sparse_matrix_t expression,
    const scl_index_t* labels,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_types,
    scl_real_t* profiles
) {
    try {
        auto* expr = get_matrix<float>(expression);
        if (!expr || !labels || !profiles) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> labels_arr(labels, n_cells);
        scl::kernel::annotation::build_reference_profiles(
            *expr, labels_arr, n_cells, n_genes, n_types, profiles
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_build_reference_profiles_f64_csr(
    scl_sparse_matrix_t expression,
    const scl_index_t* labels,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_types,
    double* profiles
) {
    try {
        auto* expr = get_matrix<double>(expression);
        if (!expr || !labels || !profiles) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::Array<const scl::Index> labels_arr(labels, n_cells);
        scl::kernel::annotation::build_reference_profiles(
            *expr, labels_arr, n_cells, n_genes, n_types, profiles
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_marker_gene_score_f32_csr(
    scl_sparse_matrix_t expression,
    const scl_index_t* const* marker_genes,
    const scl_index_t* marker_counts,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_types,
    scl_real_t* scores,
    int normalize
) {
    try {
        auto* expr = get_matrix<float>(expression);
        if (!expr || !marker_genes || !marker_counts || !scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::annotation::marker_gene_score(
            *expr, marker_genes, marker_counts, n_cells, n_genes, n_types,
            scores, normalize != 0
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}

extern "C" scl_error_t scl_marker_gene_score_f64_csr(
    scl_sparse_matrix_t expression,
    const scl_index_t* const* marker_genes,
    const scl_index_t* marker_counts,
    scl_index_t n_cells,
    scl_index_t n_genes,
    scl_index_t n_types,
    double* scores,
    int normalize
) {
    try {
        auto* expr = get_matrix<double>(expression);
        if (!expr || !marker_genes || !marker_counts || !scores) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        scl::kernel::annotation::marker_gene_score(
            *expr, marker_genes, marker_counts, n_cells, n_genes, n_types,
            scores, normalize != 0
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return catch_exception();
    }
}
