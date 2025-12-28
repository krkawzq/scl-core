// =============================================================================
// FILE: scl/binding/c_api/mmd.cpp
// BRIEF: C API implementation for Maximum Mean Discrepancy
// =============================================================================

#include "scl/binding/c_api/mmd.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/mmd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
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

scl_error_t scl_mmd_rbf(
    scl_sparse_matrix_t matrix_x,
    scl_sparse_matrix_t matrix_y,
    scl_real_t* output,
    scl_real_t gamma
) {
    if (!matrix_x || !matrix_y || !output) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }

    try {
        auto* sparse_x = reinterpret_cast<scl::CSR*>(matrix_x);
        auto* sparse_y = reinterpret_cast<scl::CSR*>(matrix_y);

        if (!sparse_x->valid() || !sparse_y->valid()) {
            return SCL_ERROR_INVALID_ARGUMENT;
        }

        const scl::Index primary_dim = sparse_x->primary_dim();
        scl::Array<scl::Real> output_arr(output, static_cast<scl::Size>(primary_dim));

        scl::kernel::mmd::mmd_rbf(*sparse_x, *sparse_y, output_arr, static_cast<scl::Real>(gamma));

        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

// Explicit instantiation
template void scl::kernel::mmd::mmd_rbf<scl::Real, true>(
    const scl::Sparse<scl::Real, true>&,
    const scl::Sparse<scl::Real, true>&,
    scl::Array<scl::Real>,
    scl::Real
);

