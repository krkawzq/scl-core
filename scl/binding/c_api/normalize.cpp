// =============================================================================
// FILE: scl/binding/c_api/normalize.cpp
// BRIEF: C API implementation for normalization operations
// =============================================================================

#include "scl/binding/c_api/normalize.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/normalize.hpp"
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

scl_error_t scl_normalize_compute_row_sums(
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
            n_rows
        );
        scl::kernel::normalize::compute_row_sums(*sparse, output_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_normalize_scale_primary(
    scl_sparse_matrix_t matrix,
    const scl_real_t* scales,
    scl_size_t n_rows
) {
    if (!matrix || !scales) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<const scl::Real> scales_arr(
            reinterpret_cast<const scl::Real*>(scales),
            n_rows
        );
        scl::kernel::normalize::scale_primary(*sparse, scales_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_normalize_primary_sums_masked(
    scl_sparse_matrix_t matrix,
    const uint8_t* mask,
    scl_real_t* output,
    scl_size_t n_rows
) {
    if (!matrix || !mask || !output) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<const scl::Byte> mask_arr(
            reinterpret_cast<const scl::Byte*>(mask),
            static_cast<scl::Size>(sparse->secondary_dim())
        );
        scl::Array<scl::Real> output_arr(
            reinterpret_cast<scl::Real*>(output),
            n_rows
        );
        scl::kernel::normalize::primary_sums_masked(*sparse, mask_arr, output_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_normalize_detect_highly_expressed(
    scl_sparse_matrix_t matrix,
    const scl_real_t* row_sums,
    scl_real_t max_fraction,
    uint8_t* out_mask,
    scl_size_t n_cols
) {
    if (!matrix || !row_sums || !out_mask) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<const scl::Real> row_sums_arr(
            reinterpret_cast<const scl::Real*>(row_sums),
            static_cast<scl::Size>(sparse->primary_dim())
        );
        scl::Array<scl::Byte> mask_arr(
            reinterpret_cast<scl::Byte*>(out_mask),
            n_cols
        );
        scl::kernel::normalize::detect_highly_expressed(
            *sparse,
            row_sums_arr,
            static_cast<scl::Real>(max_fraction),
            mask_arr
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::normalize::compute_row_sums<scl::Real, true>(
    const scl::CSR&,
    scl::Array<scl::Real>
);

template void scl::kernel::normalize::scale_primary<scl::Real, true>(
    scl::CSR&,
    scl::Array<const scl::Real>
);

template void scl::kernel::normalize::primary_sums_masked<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Byte>,
    scl::Array<scl::Real>
);

template void scl::kernel::normalize::detect_highly_expressed<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Real>,
    scl::Real,
    scl::Array<scl::Byte>
);

} // extern "C"

