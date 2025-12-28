// =============================================================================
// FILE: scl/binding/c_api/correlation/correlation.cpp
// BRIEF: C API implementation for correlation analysis
// =============================================================================

#include "scl/binding/c_api/correlation.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/correlation.hpp"
#include "scl/core/type.hpp"

extern "C" {

scl_error_t scl_corr_compute_stats(
    scl_sparse_t matrix,
    scl_real_t* out_means,
    scl_real_t* out_inv_stds
) {
    if (!matrix || !out_means || !out_inv_stds) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(matrix);
        scl::Index n_rows = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<scl::Real> means(reinterpret_cast<scl::Real*>(out_means),
                                       static_cast<scl::Size>(n_rows));
            scl::Array<scl::Real> inv_stds(reinterpret_cast<scl::Real*>(out_inv_stds),
                                          static_cast<scl::Size>(n_rows));

            scl::kernel::correlation::compute_stats(m, means, inv_stds);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_corr_pearson(
    scl_sparse_t matrix,
    const scl_real_t* means,
    const scl_real_t* inv_stds,
    scl_real_t* output
) {
    if (!matrix || !means || !inv_stds || !output) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(matrix);
        scl::Index n_rows = sparse->rows();
        scl::Size n_sq = static_cast<scl::Size>(n_rows) * n_rows;

        sparse->visit([&](auto& m) {
            scl::Array<const scl::Real> means_arr(reinterpret_cast<const scl::Real*>(means),
                                                  static_cast<scl::Size>(n_rows));
            scl::Array<const scl::Real> inv_stds_arr(reinterpret_cast<const scl::Real*>(inv_stds),
                                                     static_cast<scl::Size>(n_rows));
            scl::Array<scl::Real> out(reinterpret_cast<scl::Real*>(output), n_sq);

            scl::kernel::correlation::pearson(m, means_arr, inv_stds_arr, out);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_corr_pearson_auto(
    scl_sparse_t matrix,
    scl_real_t* output
) {
    if (!matrix || !output) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(matrix);
        scl::Index n_rows = sparse->rows();
        scl::Size n_sq = static_cast<scl::Size>(n_rows) * n_rows;

        sparse->visit([&](auto& m) {
            scl::Array<scl::Real> out(reinterpret_cast<scl::Real*>(output), n_sq);
            scl::kernel::correlation::pearson(m, out);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
