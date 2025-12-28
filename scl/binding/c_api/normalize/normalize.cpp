// =============================================================================
// FILE: scl/binding/c_api/normalize/normalize.cpp
// BRIEF: C API implementation for normalization
// =============================================================================

#include "scl/binding/c_api/normalize/normalize.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/normalize.hpp"
#include "scl/core/type.hpp"

extern "C" {

scl_error_t scl_norm_compute_row_sums(
    scl_sparse_t matrix,
    scl_real_t* output
) {
    if (!matrix || !output) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(matrix);
        scl::Index n_rows = sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<scl::Real> out(reinterpret_cast<scl::Real*>(output),
                                     static_cast<scl::Size>(n_rows));
            scl::kernel::normalize::compute_row_sums(m, out);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_norm_scale_primary(
    scl_sparse_t matrix,
    const scl_real_t* scales
) {
    if (!matrix || !scales) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(matrix);
        scl::Index primary_dim = sparse->is_csr ? sparse->rows() : sparse->cols();

        sparse->visit([&](auto& m) {
            scl::Array<const scl::Real> scales_arr(reinterpret_cast<const scl::Real*>(scales),
                                                   static_cast<scl::Size>(primary_dim));
            scl::kernel::normalize::scale_primary(m, scales_arr);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_norm_primary_sums_masked(
    scl_sparse_t matrix,
    const unsigned char* mask,
    scl_real_t* output
) {
    if (!matrix || !mask || !output) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(matrix);
        scl::Index primary_dim = sparse->is_csr ? sparse->rows() : sparse->cols();
        scl::Index secondary_dim = sparse->is_csr ? sparse->cols() : sparse->rows();

        sparse->visit([&](auto& m) {
            scl::Array<const scl::Byte> mask_arr(mask, static_cast<scl::Size>(secondary_dim));
            scl::Array<scl::Real> out(reinterpret_cast<scl::Real*>(output),
                                     static_cast<scl::Size>(primary_dim));
            scl::kernel::normalize::primary_sums_masked(m, mask_arr, out);
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

scl_error_t scl_norm_detect_highly_expressed(
    scl_sparse_t matrix,
    const scl_real_t* row_sums,
    scl_real_t max_fraction,
    unsigned char* out_mask
) {
    if (!matrix || !row_sums || !out_mask) {
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* sparse = static_cast<scl_sparse_matrix*>(matrix);
        scl::Index n_rows = sparse->rows();
        scl::Index n_cols = sparse->cols();

        sparse->visit([&](auto& m) {
            scl::Array<const scl::Real> sums(reinterpret_cast<const scl::Real*>(row_sums),
                                            static_cast<scl::Size>(n_rows));
            scl::Array<scl::Byte> mask(out_mask, static_cast<scl::Size>(n_cols));
            scl::kernel::normalize::detect_highly_expressed(
                m, sums, static_cast<scl::Real>(max_fraction), mask
            );
        });

        return SCL_OK;
    } catch (...) {
        return scl::binding::handle_exception();
    }
}

} // extern "C"
