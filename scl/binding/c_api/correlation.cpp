// =============================================================================
// FILE: scl/binding/c_api/correlation.cpp
// BRIEF: C API implementation for correlation computation
// =============================================================================

#include "scl/binding/c_api/correlation.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/correlation.hpp"
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
        return SCL_ERROR_TYPE;
    } catch (const scl::IOError&) {
        return SCL_ERROR_IO;
    } catch (const scl::NotImplementedError&) {
        return SCL_ERROR_NOT_IMPLEMENTED;
    } catch (const scl::InternalError&) {
        return SCL_ERROR_INTERNAL;
    } catch (const scl::Exception& e) {
        return static_cast<scl_error_t>(static_cast<int>(e.code()));
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_correlation_compute_stats(
    scl_sparse_matrix_t matrix,
    scl_real_t* out_means,
    scl_real_t* out_inv_stds,
    scl_size_t n_rows
) {
    if (!matrix || !out_means || !out_inv_stds) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<scl::Real> means_arr(
            reinterpret_cast<scl::Real*>(out_means),
            n_rows
        );
        scl::Array<scl::Real> inv_stds_arr(
            reinterpret_cast<scl::Real*>(out_inv_stds),
            n_rows
        );
        scl::kernel::correlation::compute_stats(*sparse, means_arr, inv_stds_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_correlation_pearson_with_stats(
    scl_sparse_matrix_t matrix,
    const scl_real_t* means,
    const scl_real_t* inv_stds,
    scl_real_t* output,
    scl_size_t n_rows
) {
    if (!matrix || !means || !inv_stds || !output) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<const scl::Real> means_arr(
            reinterpret_cast<const scl::Real*>(means),
            n_rows
        );
        scl::Array<const scl::Real> inv_stds_arr(
            reinterpret_cast<const scl::Real*>(inv_stds),
            n_rows
        );
        scl::Array<scl::Real> output_arr(
            reinterpret_cast<scl::Real*>(output),
            n_rows * n_rows
        );
        scl::kernel::correlation::pearson(*sparse, means_arr, inv_stds_arr, output_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_correlation_pearson(
    scl_sparse_matrix_t matrix,
    scl_real_t* output,
    scl_size_t n_rows
) {
    if (!matrix || !output) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<scl::Real> output_arr(
            reinterpret_cast<scl::Real*>(output),
            n_rows * n_rows
        );
        scl::kernel::correlation::pearson(*sparse, output_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::correlation::compute_stats<scl::CSR>(
    const scl::CSR&,
    scl::Array<scl::Real>,
    scl::Array<scl::Real>
);

template void scl::kernel::correlation::pearson<scl::CSR>(
    const scl::CSR&,
    scl::Array<const scl::Real>,
    scl::Array<const scl::Real>,
    scl::Array<scl::Real>
);

template void scl::kernel::correlation::pearson<scl::CSR>(
    const scl::CSR&,
    scl::Array<scl::Real>
);

} // extern "C"

