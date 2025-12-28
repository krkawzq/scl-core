// =============================================================================
// FILE: scl/binding/c_api/metrics.cpp
// BRIEF: C API implementation for quality metrics
// =============================================================================

#include "scl/binding/c_api/metrics.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/metrics.hpp"
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

scl_error_t scl_metrics_silhouette_score(
    scl_sparse_t distances,
    const scl_index_t* labels,
    scl_size_t n_cells,
    scl_real_t* score
) {
    if (!distances || !labels || !score) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(distances, wrapper);
        if (err != SCL_OK) return err;

        scl::Real score_result = scl::Real(0);
        wrapper->visit([&](auto& dist) {
            score_result = scl::kernel::metrics::silhouette_score(
                dist,
                scl::Array<const scl::Index>(
                    reinterpret_cast<const scl::Index*>(labels),
                    static_cast<scl::Size>(n_cells)
                )
            );
        });
        *score = static_cast<scl_real_t>(score_result);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_metrics_silhouette_samples(
    scl_sparse_t distances,
    const scl_index_t* labels,
    scl_size_t n_cells,
    scl_real_t* scores
) {
    if (!distances || !labels || !scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(distances, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& dist) {
            scl::kernel::metrics::silhouette_samples(
                dist,
                scl::Array<const scl::Index>(
                    reinterpret_cast<const scl::Index*>(labels),
                    static_cast<scl::Size>(n_cells)
                ),
                scl::Array<scl::Real>(
                    reinterpret_cast<scl::Real*>(scores),
                    static_cast<scl::Size>(n_cells)
                )
            );
        });
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_metrics_adjusted_rand_index(
    const scl_index_t* labels1,
    const scl_index_t* labels2,
    scl_size_t n_cells,
    scl_real_t* ari
) {
    if (!labels1 || !labels2 || !ari) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Real ari_result = scl::kernel::metrics::adjusted_rand_index(
            scl::Array<const scl::Index>(
                reinterpret_cast<const scl::Index*>(labels1),
                static_cast<scl::Size>(n_cells)
            ),
            scl::Array<const scl::Index>(
                reinterpret_cast<const scl::Index*>(labels2),
                static_cast<scl::Size>(n_cells)
            )
        );
        *ari = static_cast<scl_real_t>(ari_result);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_metrics_normalized_mutual_information(
    const scl_index_t* labels1,
    const scl_index_t* labels2,
    scl_size_t n_cells,
    scl_real_t* nmi
) {
    if (!labels1 || !labels2 || !nmi) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Real nmi_result = scl::kernel::metrics::normalized_mutual_information(
            scl::Array<const scl::Index>(
                reinterpret_cast<const scl::Index*>(labels1),
                static_cast<scl::Size>(n_cells)
            ),
            scl::Array<const scl::Index>(
                reinterpret_cast<const scl::Index*>(labels2),
                static_cast<scl::Size>(n_cells)
            )
        );
        *nmi = static_cast<scl_real_t>(nmi_result);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_metrics_homogeneity_completeness_vmeasure(
    const scl_index_t* labels_true,
    const scl_index_t* labels_pred,
    scl_size_t n_cells,
    scl_real_t* homogeneity,
    scl_real_t* completeness,
    scl_real_t* v_measure
) {
    if (!labels_true || !labels_pred || !homogeneity || !completeness || !v_measure) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::Real h = scl::Real(0);
        scl::Real c = scl::Real(0);
        scl::Real v = scl::Real(0);

        scl::kernel::metrics::clustering_metrics(
            scl::Array<const scl::Index>(
                reinterpret_cast<const scl::Index*>(labels_true),
                static_cast<scl::Size>(n_cells)
            ),
            scl::Array<const scl::Index>(
                reinterpret_cast<const scl::Index*>(labels_pred),
                static_cast<scl::Size>(n_cells)
            ),
            h, c, v
        );

        *homogeneity = static_cast<scl_real_t>(h);
        *completeness = static_cast<scl_real_t>(c);
        *v_measure = static_cast<scl_real_t>(v);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
