// =============================================================================
// FILE: scl/binding/c_api/log1p.cpp
// BRIEF: C API implementation for logarithmic transforms
// =============================================================================

#include "scl/binding/c_api/log1p.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/log1p.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

// Internal helper to convert C++ exception to error code
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

scl_error_t scl_log1p_inplace(scl_sparse_matrix_t* matrix) {
    if (!matrix) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = reinterpret_cast<scl::CSR*>(matrix);
        scl::kernel::log1p::log1p_inplace(*sparse);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_log2p1_inplace(scl_sparse_matrix_t* matrix) {
    if (!matrix) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = reinterpret_cast<scl::CSR*>(matrix);
        scl::kernel::log1p::log2p1_inplace(*sparse);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_expm1_inplace(scl_sparse_matrix_t* matrix) {
    if (!matrix) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = reinterpret_cast<scl::CSR*>(matrix);
        scl::kernel::log1p::expm1_inplace(*sparse);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::log1p::log1p_inplace(scl::CSR&);
template void scl::kernel::log1p::log2p1_inplace(scl::CSR&);
template void scl::kernel::log1p::expm1_inplace(scl::CSR&);

} // extern "C"

