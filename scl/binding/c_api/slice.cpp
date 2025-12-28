// =============================================================================
// FILE: scl/binding/c_api/slice.cpp
// BRIEF: C API implementation for sparse matrix slicing
// =============================================================================

#include "scl/binding/c_api/slice.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/slice.hpp"
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

scl_index_t scl_slice_inspect_slice_primary(
    scl_sparse_matrix_t matrix,
    const scl_index_t* keep_indices,
    scl_size_t n_keep
) {
    if (!matrix || !keep_indices) {
        return 0;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<const scl::Index> keep_arr(
            reinterpret_cast<const scl::Index*>(keep_indices),
            n_keep
        );
        return scl::kernel::slice::inspect_slice_primary(*sparse, keep_arr);
    } catch (...) {
        return 0;
    }
}

scl_error_t scl_slice_materialize_slice_primary(
    scl_sparse_matrix_t matrix,
    const scl_index_t* keep_indices,
    scl_real_t* out_data,
    scl_index_t* out_indices,
    scl_index_t* out_indptr,
    scl_size_t n_keep,
    scl_size_t out_nnz
) {
    if (!matrix || !keep_indices || !out_data || !out_indices || !out_indptr) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<const scl::Index> keep_arr(
            reinterpret_cast<const scl::Index*>(keep_indices),
            n_keep
        );
        scl::Array<scl::Real> data_arr(
            reinterpret_cast<scl::Real*>(out_data),
            out_nnz
        );
        scl::Array<scl::Index> indices_arr(
            reinterpret_cast<scl::Index*>(out_indices),
            out_nnz
        );
        scl::Array<scl::Index> indptr_arr(
            reinterpret_cast<scl::Index*>(out_indptr),
            n_keep + 1
        );
        scl::kernel::slice::materialize_slice_primary(
            *sparse,
            keep_arr,
            data_arr,
            indices_arr,
            indptr_arr
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_index_t scl_slice_inspect_filter_secondary(
    scl_sparse_matrix_t matrix,
    const uint8_t* mask,
    scl_size_t n_cols
) {
    if (!matrix || !mask) {
        return 0;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<const scl::Byte> mask_arr(
            reinterpret_cast<const scl::Byte*>(mask),
            n_cols
        );
        return scl::kernel::slice::inspect_filter_secondary(*sparse, mask_arr);
    } catch (...) {
        return 0;
    }
}

scl_error_t scl_slice_materialize_filter_secondary(
    scl_sparse_matrix_t matrix,
    const uint8_t* mask,
    scl_real_t* out_data,
    scl_index_t* out_indices,
    scl_index_t* out_indptr,
    scl_size_t n_rows,
    scl_size_t n_cols,
    scl_size_t out_nnz
) {
    if (!matrix || !mask || !out_data || !out_indices || !out_indptr) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<const scl::Byte> mask_arr(
            reinterpret_cast<const scl::Byte*>(mask),
            n_cols
        );
        scl::Array<scl::Real> data_arr(
            reinterpret_cast<scl::Real*>(out_data),
            out_nnz
        );
        scl::Array<scl::Index> indices_arr(
            reinterpret_cast<scl::Index*>(out_indices),
            out_nnz
        );
        scl::Array<scl::Index> indptr_arr(
            reinterpret_cast<scl::Index*>(out_indptr),
            n_rows + 1
        );
        scl::kernel::slice::materialize_filter_secondary(
            *sparse,
            mask_arr,
            data_arr,
            indices_arr,
            indptr_arr
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template scl::Index scl::kernel::slice::inspect_slice_primary<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Index>
);

template void scl::kernel::slice::materialize_slice_primary<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Index>,
    scl::Array<scl::Real>,
    scl::Array<scl::Index>,
    scl::Array<scl::Index>
);

template scl::Index scl::kernel::slice::inspect_filter_secondary<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Byte>
);

template void scl::kernel::slice::materialize_filter_secondary<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Byte>,
    scl::Array<scl::Real>,
    scl::Array<scl::Index>,
    scl::Array<scl::Index>
);

} // extern "C"

