// =============================================================================
// FILE: scl/binding/c_api/scale.cpp
// BRIEF: C API implementation for scaling operations
// =============================================================================

#include "scl/binding/c_api/scale.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/scale.hpp"
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

scl_error_t scl_scale_standardize(
    scl_sparse_matrix_t matrix,
    const scl_real_t* means,
    const scl_real_t* stds,
    scl_real_t max_value,
    int zero_center,
    scl_size_t n_rows
) {
    if (!matrix || !means || !stds) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<const scl::Real> means_arr(
            reinterpret_cast<const scl::Real*>(means),
            n_rows
        );
        scl::Array<const scl::Real> stds_arr(
            reinterpret_cast<const scl::Real*>(stds),
            n_rows
        );
        scl::kernel::scale::standardize(
            *sparse,
            means_arr,
            stds_arr,
            static_cast<scl::Real>(max_value),
            zero_center != 0
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_scale_scale_rows(
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
        scl::kernel::scale::scale_rows(*sparse, scales_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_scale_shift_rows(
    scl_sparse_matrix_t matrix,
    const scl_real_t* offsets,
    scl_size_t n_rows
) {
    if (!matrix || !offsets) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<const scl::Real> offsets_arr(
            reinterpret_cast<const scl::Real*>(offsets),
            n_rows
        );
        scl::kernel::scale::shift_rows(*sparse, offsets_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::scale::standardize<scl::Real, true>(
    scl::CSR&,
    scl::Array<const scl::Real>,
    scl::Array<const scl::Real>,
    scl::Real,
    bool
);

template void scl::kernel::scale::scale_rows<scl::Real, true>(
    scl::CSR&,
    scl::Array<const scl::Real>
);

template void scl::kernel::scale::shift_rows<scl::Real, true>(
    scl::CSR&,
    scl::Array<const scl::Real>
);

} // extern "C"

