// =============================================================================
// FILE: scl/binding/c_api/markers.cpp
// BRIEF: C API implementation for marker gene selection
// =============================================================================

#include "scl/binding/c_api/markers.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/markers.hpp"
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

scl_error_t scl_markers_group_mean_expression(
    scl_sparse_t expression,
    const scl_index_t* group_labels,
    scl_size_t n_cells,
    scl_index_t n_groups,
    scl_index_t n_genes,
    scl_real_t* mean_expr
) {
    if (!expression || !group_labels || !mean_expr) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(expression, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& expr) {
            scl::kernel::markers::group_mean_expression(
                expr,
                scl::Array<const scl::Index>(
                    reinterpret_cast<const scl::Index*>(group_labels),
                    static_cast<scl::Size>(n_cells)
                ),
                static_cast<scl::Index>(n_groups),
                scl::Array<scl::Real>(
                    reinterpret_cast<scl::Real*>(mean_expr),
                    static_cast<scl::Size>(n_genes) * static_cast<scl::Size>(n_groups)
                ),
                static_cast<scl::Index>(n_genes)
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_markers_percent_expressed(
    scl_sparse_t expression,
    const scl_index_t* group_labels,
    scl_size_t n_cells,
    scl_index_t n_groups,
    scl_index_t n_genes,
    scl_real_t* pct_expr,
    scl_real_t threshold
) {
    if (!expression || !group_labels || !pct_expr) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(expression, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& expr) {
            scl::kernel::markers::percent_expressed(
                expr,
                scl::Array<const scl::Index>(
                    reinterpret_cast<const scl::Index*>(group_labels),
                    static_cast<scl::Size>(n_cells)
                ),
                static_cast<scl::Index>(n_groups),
                scl::Array<scl::Real>(
                    reinterpret_cast<scl::Real*>(pct_expr),
                    static_cast<scl::Size>(n_genes) * static_cast<scl::Size>(n_groups)
                ),
                static_cast<scl::Index>(n_genes),
                static_cast<scl::Real>(threshold)
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_markers_find_markers(
    scl_sparse_t expression,
    const scl_index_t* group_labels,
    scl_size_t n_cells,
    scl_index_t n_groups,
    scl_index_t n_genes,
    scl_index_t target_group,
    scl_index_t* marker_genes,
    scl_real_t* marker_scores,
    scl_size_t* n_markers,
    scl_real_t min_fold_change,
    scl_real_t min_pct,
    scl_index_t n_top
) {
    if (!expression || !group_labels || !marker_genes || !marker_scores || !n_markers) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(expression, wrapper);
        if (err != SCL_OK) return err;

        scl::Size n_markers_result = 0;
        wrapper->visit([&](auto& expr) {
            // This is a simplified version - the actual implementation may be more complex
            // For now, we'll use a basic marker finding approach
            scl::kernel::markers::find_markers(
                expr,
                scl::Array<const scl::Index>(
                    reinterpret_cast<const scl::Index*>(group_labels),
                    static_cast<scl::Size>(n_cells)
                ),
                static_cast<scl::Index>(n_groups),
                static_cast<scl::Index>(n_genes),
                static_cast<scl::Index>(target_group),
                scl::Array<scl::Index>(
                    reinterpret_cast<scl::Index*>(marker_genes),
                    static_cast<scl::Size>(n_top)
                ),
                scl::Array<scl::Real>(
                    reinterpret_cast<scl::Real*>(marker_scores),
                    static_cast<scl::Size>(n_top)
                ),
                n_markers_result,
                static_cast<scl::Real>(min_fold_change),
                static_cast<scl::Real>(min_pct),
                static_cast<scl::Index>(n_top)
            );
        });
        *n_markers = static_cast<scl_size_t>(n_markers_result);
        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
