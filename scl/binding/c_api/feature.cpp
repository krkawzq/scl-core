// =============================================================================
// FILE: scl/binding/c_api/feature.cpp
// BRIEF: C API implementation for Feature Statistics
// =============================================================================

#include "scl/binding/c_api/feature.h"
#include "scl/kernel/feature.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

static scl_error_t handle_exception() {
    try {
        throw;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::TypeError&) {
        return SCL_ERROR_TYPE_ERROR;
    } catch (const scl::IOError&) {
        return SCL_ERROR_IO_ERROR;
    } catch (const scl::NotImplementedError&) {
        return SCL_ERROR_NOT_IMPLEMENTED;
    } catch (const scl::InternalError&) {
        return SCL_ERROR_INTERNAL_ERROR;
    } catch (const scl::Exception& e) {
        return static_cast<scl_error_t>(static_cast<int>(e.code()));
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_standard_moments(
    scl_sparse_matrix_t matrix,
    scl_real_t* out_means,
    scl_real_t* out_vars,
    int ddof
) {
    if (!matrix || !out_means || !out_vars) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        const scl::Index primary_dim = sparse->primary_dim();
        scl::Array<scl::Real> means_arr(out_means, static_cast<scl::Size>(primary_dim));
        scl::Array<scl::Real> vars_arr(out_vars, static_cast<scl::Size>(primary_dim));

        scl::kernel::feature::standard_moments(*sparse, means_arr, vars_arr, ddof);

        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_clipped_moments(
    scl_sparse_matrix_t matrix,
    const scl_real_t* clip_vals,
    scl_real_t* out_means,
    scl_real_t* out_vars
) {
    if (!matrix || !clip_vals || !out_means || !out_vars) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        const scl::Index primary_dim = sparse->primary_dim();
        scl::Array<const scl::Real> clip_arr(clip_vals, static_cast<scl::Size>(primary_dim));
        scl::Array<scl::Real> means_arr(out_means, static_cast<scl::Size>(primary_dim));
        scl::Array<scl::Real> vars_arr(out_vars, static_cast<scl::Size>(primary_dim));

        scl::kernel::feature::clipped_moments(*sparse, clip_arr, means_arr, vars_arr);

        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_detection_rate(
    scl_sparse_matrix_t matrix,
    scl_real_t* out_rates
) {
    if (!matrix || !out_rates) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        if (!sparse->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        const scl::Index primary_dim = sparse->primary_dim();
        scl::Array<scl::Real> rates_arr(out_rates, static_cast<scl::Size>(primary_dim));

        scl::kernel::feature::detection_rate(*sparse, rates_arr);

        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_dispersion(
    const scl_real_t* means,
    const scl_real_t* vars,
    scl_real_t* out_dispersion,
    scl_size_t n
) {
    if (!means || !vars || !out_dispersion || n == 0) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }

    try {
        scl::Array<const scl::Real> means_arr(means, static_cast<scl::Size>(n));
        scl::Array<const scl::Real> vars_arr(vars, static_cast<scl::Size>(n));
        scl::Array<scl::Real> dispersion_arr(out_dispersion, static_cast<scl::Size>(n));

        scl::kernel::feature::dispersion(means_arr, vars_arr, dispersion_arr);

        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::feature::standard_moments(scl::CSR const&, scl::Array<scl::Real>, scl::Array<scl::Real>, int);
template void scl::kernel::feature::clipped_moments(scl::CSR const&, scl::Array<const scl::Real>, scl::Array<scl::Real>, scl::Array<scl::Real>);
template void scl::kernel::feature::detection_rate(scl::CSR const&, scl::Array<scl::Real>);

} // extern "C"
