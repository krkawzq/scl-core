// =============================================================================
// FILE: scl/binding/c_api/gram.cpp
// BRIEF: C API implementation for Gram matrix computation
// =============================================================================

#include "scl/binding/c_api/gram.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/gram.hpp"
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

scl_error_t scl_gram_compute(
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
        scl::kernel::gram::gram(*sparse, output_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::gram::gram<scl::Real, true>(
    const scl::CSR&,
    scl::Array<scl::Real>
);

} // extern "C"

