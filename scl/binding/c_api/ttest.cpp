// =============================================================================
// FILE: scl/binding/c_api/ttest.cpp
// BRIEF: C API implementation for T-test
// =============================================================================

#include "scl/binding/c_api/ttest.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/ttest.hpp"
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

scl_error_t scl_ttest(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    scl_size_t n_cells,
    scl_real_t* out_t_stats,
    scl_size_t t_stats_size,
    scl_real_t* out_p_values,
    scl_size_t p_values_size,
    scl_real_t* out_log2_fc,
    scl_size_t log2_fc_size,
    int use_welch
) {
    if (!matrix || !group_ids || !out_t_stats || !out_p_values || !out_log2_fc) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(matrix, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& mat) {
            scl::kernel::ttest::ttest(
                mat,
                scl::Array<const int32_t>(
                    group_ids,
                    static_cast<scl::Size>(n_cells)
                ),
                scl::Array<scl::Real>(
                    reinterpret_cast<scl::Real*>(out_t_stats),
                    static_cast<scl::Size>(t_stats_size)
                ),
                scl::Array<scl::Real>(
                    reinterpret_cast<scl::Real*>(out_p_values),
                    static_cast<scl::Size>(p_values_size)
                ),
                scl::Array<scl::Real>(
                    reinterpret_cast<scl::Real*>(out_log2_fc),
                    static_cast<scl::Size>(log2_fc_size)
                ),
                use_welch != 0
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_ttest_compute_group_stats(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    scl_size_t n_cells,
    scl_size_t n_groups,
    scl_real_t* out_means,
    scl_size_t means_size,
    scl_real_t* out_vars,
    scl_size_t vars_size,
    scl_size_t* out_counts
) {
    if (!matrix || !group_ids || !out_means || !out_vars || !out_counts) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        scl::binding::SparseWrapper* wrapper;
        scl_error_t err = get_sparse_matrix(matrix, wrapper);
        if (err != SCL_OK) return err;

        wrapper->visit([&](auto& mat) {
            scl::kernel::ttest::compute_group_stats(
                mat,
                scl::Array<const int32_t>(
                    group_ids,
                    static_cast<scl::Size>(n_cells)
                ),
                static_cast<scl::Size>(n_groups),
                scl::Array<scl::Real>(
                    reinterpret_cast<scl::Real*>(out_means),
                    static_cast<scl::Size>(means_size)
                ),
                scl::Array<scl::Real>(
                    reinterpret_cast<scl::Real*>(out_vars),
                    static_cast<scl::Size>(vars_size)
                ),
                scl::Array<scl::Size>(
                    out_counts,
                    static_cast<scl::Size>(vars_size)
                )
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
