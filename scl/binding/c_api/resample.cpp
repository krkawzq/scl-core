// =============================================================================
// FILE: scl/binding/c_api/resample.cpp
// BRIEF: C API implementation for resampling operations
// =============================================================================

#include "scl/binding/c_api/resample.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/resample.hpp"
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

scl_error_t scl_resample_downsample(
    scl_sparse_matrix_t matrix,
    scl_real_t target_sum,
    uint64_t seed
) {
    if (!matrix) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::kernel::resample::downsample(
            *sparse,
            static_cast<scl::Real>(target_sum),
            seed
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_resample_downsample_variable(
    scl_sparse_matrix_t matrix,
    const scl_real_t* target_counts,
    uint64_t seed,
    scl_size_t n_rows
) {
    if (!matrix || !target_counts) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<const scl::Real> targets_arr(
            reinterpret_cast<const scl::Real*>(target_counts),
            n_rows
        );
        scl::kernel::resample::downsample_variable(*sparse, targets_arr, seed);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_resample_binomial_resample(
    scl_sparse_matrix_t matrix,
    uint64_t seed
) {
    if (!matrix) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::kernel::resample::binomial_resample(*sparse, seed);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_resample_poisson_resample(
    scl_sparse_matrix_t matrix,
    uint64_t seed
) {
    if (!matrix) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::kernel::resample::poisson_resample(*sparse, seed);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::resample::downsample<scl::Real, true>(
    scl::CSR&,
    scl::Real,
    uint64_t
);

template void scl::kernel::resample::downsample_variable<scl::Real, true>(
    scl::CSR&,
    scl::Array<const scl::Real>,
    uint64_t
);

template void scl::kernel::resample::binomial_resample<scl::Real, true>(
    scl::CSR&,
    uint64_t
);

template void scl::kernel::resample::poisson_resample<scl::Real, true>(
    scl::CSR&,
    uint64_t
);

} // extern "C"

