// =============================================================================
// FILE: scl/binding/c_api/softmax.cpp
// BRIEF: C API implementation for softmax operations
// =============================================================================

#include "scl/binding/c_api/softmax.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/softmax.hpp"
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

scl_error_t scl_softmax_inplace(
    scl_sparse_matrix_t matrix
) {
    if (!matrix) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::kernel::softmax::softmax_inplace(*sparse);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_softmax_inplace_with_temperature(
    scl_sparse_matrix_t matrix,
    scl_real_t temperature
) {
    if (!matrix) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::kernel::softmax::softmax_inplace(*sparse, static_cast<scl::Real>(temperature));
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_softmax_log_softmax_inplace(
    scl_sparse_matrix_t matrix
) {
    if (!matrix) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::kernel::softmax::log_softmax_inplace(*sparse);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_softmax_log_softmax_inplace_with_temperature(
    scl_sparse_matrix_t matrix,
    scl_real_t temperature
) {
    if (!matrix) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::kernel::softmax::log_softmax_inplace(*sparse, static_cast<scl::Real>(temperature));
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::softmax::softmax_inplace<scl::Real, true>(
    scl::CSR&
);

template void scl::kernel::softmax::softmax_inplace<scl::Real, true>(
    scl::CSR&,
    scl::Real
);

template void scl::kernel::softmax::log_softmax_inplace<scl::Real, true>(
    scl::CSR&
);

template void scl::kernel::softmax::log_softmax_inplace<scl::Real, true>(
    scl::CSR&,
    scl::Real
);

} // extern "C"

