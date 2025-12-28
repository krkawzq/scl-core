// =============================================================================
// FILE: scl/binding/c_api/subpopulation.cpp
// BRIEF: C API implementation for subpopulation analysis
// =============================================================================

#include "scl/binding/c_api/subpopulation.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/subpopulation.hpp"
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

scl_error_t scl_subpopulation_subclustering(
    scl_sparse_t expression,
    const scl_index_t* parent_labels,
    scl_size_t n_cells,
    scl_index_t parent_cluster,
    scl_size_t n_subclusters,
    scl_index_t* subcluster_labels,
    uint64_t seed
) {
    if (!expression || !parent_labels || !subcluster_labels) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(expression, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& expr) {
            scl::kernel::subpopulation::subclustering(
                expr,
                scl::Array<const scl::Index>(
                    reinterpret_cast<const scl::Index*>(parent_labels),
                    static_cast<scl::Size>(n_cells)
                ),
                static_cast<scl::Index>(parent_cluster),
                static_cast<scl::Size>(n_subclusters),
                scl::Array<scl::Index>(
                    reinterpret_cast<scl::Index*>(subcluster_labels),
                    static_cast<scl::Size>(n_cells)
                ),
                seed
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_subpopulation_cluster_stability(
    scl_sparse_t expression,
    const scl_index_t* original_labels,
    scl_size_t n_cells,
    scl_size_t n_bootstraps,
    scl_real_t* stability_scores,
    scl_size_t stability_scores_size,
    uint64_t seed
) {
    if (!expression || !original_labels || !stability_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(expression, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& expr) {
            scl::kernel::subpopulation::cluster_stability(
                expr,
                scl::Array<const scl::Index>(
                    reinterpret_cast<const scl::Index*>(original_labels),
                    static_cast<scl::Size>(n_cells)
                ),
                static_cast<scl::Size>(n_bootstraps),
                scl::Array<scl::Real>(
                    reinterpret_cast<scl::Real*>(stability_scores),
                    static_cast<scl::Size>(stability_scores_size)
                ),
                seed
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_subpopulation_cluster_purity(
    const scl_index_t* cluster_labels,
    const scl_index_t* true_labels,
    scl_size_t n_cells,
    scl_real_t* purity_per_cluster,
    scl_size_t purity_size
) {
    if (!cluster_labels || !true_labels || !purity_per_cluster) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::kernel::subpopulation::cluster_purity(
            scl::Array<const scl::Index>(
                reinterpret_cast<const scl::Index*>(cluster_labels),
                static_cast<scl::Size>(n_cells)
            ),
            scl::Array<const scl::Index>(
                reinterpret_cast<const scl::Index*>(true_labels),
                static_cast<scl::Size>(n_cells)
            ),
            scl::Array<scl::Real>(
                reinterpret_cast<scl::Real*>(purity_per_cluster),
                static_cast<scl::Size>(purity_size)
            )
        );
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_subpopulation_rare_cell_detection(
    scl_sparse_t expression,
    scl_sparse_t neighbors,
    scl_real_t* rarity_scores
) {
    if (!expression || !neighbors || !rarity_scores) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper_expr;
        scl::binding::SparseWrapper* wrapper_neigh;
        scl_error_t err1 = get_sparse_matrix(expression, wrapper_expr);
        scl_error_t err2 = get_sparse_matrix(neighbors, wrapper_neigh);
        if (err1 != SCL_OK) return err1;
        if (err2 != SCL_OK) return err2;

        wrapper_expr->visit([&](auto& expr) {
            wrapper_neigh->visit([&](auto& neigh) {
                scl::kernel::subpopulation::rare_cell_detection(
                    expr,
                    neigh,
                    scl::Array<scl::Real>(
                        reinterpret_cast<scl::Real*>(rarity_scores),
                        static_cast<scl::Size>(expr.rows())
                    )
                );
            });
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
