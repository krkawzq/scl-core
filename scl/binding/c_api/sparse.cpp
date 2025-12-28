// =============================================================================
// FILE: scl/binding/c_api/sparse.cpp
// BRIEF: C API implementation for sparse matrix statistics
// =============================================================================

#include "scl/binding/c_api/sparse.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/sparse.hpp"
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

scl_error_t scl_sparse_primary_sums(
    scl_sparse_matrix_t matrix,
    scl_real_t* output,
    scl_size_t n_rows
) {
    if (!matrix || !output) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(matrix);
        scl::Array<scl::Real> output_arr(
            reinterpret_cast<scl::Real*>(output),
            n_rows
        );
        scl::kernel::sparse::primary_sums(*sparse, output_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_primary_means(
    scl_sparse_matrix_t matrix,
    scl_real_t* output,
    scl_size_t n_rows
) {
    if (!matrix || !output) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(matrix);
        scl::Array<scl::Real> output_arr(
            reinterpret_cast<scl::Real*>(output),
            n_rows
        );
        scl::kernel::sparse::primary_means(*sparse, output_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_primary_variances(
    scl_sparse_matrix_t matrix,
    scl_real_t* output,
    scl_size_t n_rows,
    int ddof
) {
    if (!matrix || !output) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(matrix);
        scl::Array<scl::Real> output_arr(
            reinterpret_cast<scl::Real*>(output),
            n_rows
        );
        scl::kernel::sparse::primary_variances(*sparse, output_arr, ddof);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_primary_nnz(
    scl_sparse_matrix_t matrix,
    scl_index_t* output,
    scl_size_t n_rows
) {
    if (!matrix || !output) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        const auto* sparse = static_cast<const scl::CSR*>(matrix);
        scl::Array<scl::Index> output_arr(
            reinterpret_cast<scl::Index*>(output),
            n_rows
        );
        scl::kernel::sparse::primary_nnz(*sparse, output_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::sparse::primary_sums<scl::Real, true>(
    const scl::CSR&, scl::Array<scl::Real>);
template void scl::kernel::sparse::primary_means<scl::Real, true>(
    const scl::CSR&, scl::Array<scl::Real>);
template void scl::kernel::sparse::primary_variances<scl::Real, true>(
    const scl::CSR&, scl::Array<scl::Real>, int);
template void scl::kernel::sparse::primary_nnz<scl::Real, true>(
    const scl::CSR&, scl::Array<scl::Index>);

} // extern "C"
