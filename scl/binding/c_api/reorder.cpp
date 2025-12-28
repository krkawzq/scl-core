// =============================================================================
// FILE: scl/binding/c_api/reorder.cpp
// BRIEF: C API implementation for sparse matrix reordering
// =============================================================================

#include "scl/binding/c_api/reorder.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/reorder.hpp"
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

scl_error_t scl_reorder_align_secondary(
    scl_sparse_matrix_t matrix,
    const scl_index_t* index_map,
    scl_index_t* out_lengths,
    scl_index_t new_secondary_dim,
    scl_size_t n_rows,
    scl_size_t old_secondary_dim
) {
    if (!matrix || !index_map || !out_lengths) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<const scl::Index> map_arr(
            reinterpret_cast<const scl::Index*>(index_map),
            old_secondary_dim
        );
        scl::Array<scl::Index> lengths_arr(
            reinterpret_cast<scl::Index*>(out_lengths),
            n_rows
        );
        scl::kernel::reorder::align_secondary(
            *sparse,
            map_arr,
            lengths_arr,
            static_cast<scl::Index>(new_secondary_dim)
        );
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_size_t scl_reorder_compute_filtered_nnz(
    scl_sparse_matrix_t matrix,
    const scl_index_t* index_map,
    scl_index_t new_secondary_dim,
    scl_size_t old_secondary_dim
) {
    if (!matrix || !index_map) {
        return 0;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<const scl::Index> map_arr(
            reinterpret_cast<const scl::Index*>(index_map),
            old_secondary_dim
        );
        return scl::kernel::reorder::compute_filtered_nnz(
            *sparse,
            map_arr,
            static_cast<scl::Index>(new_secondary_dim)
        );
    } catch (...) {
        return 0;
    }
}

scl_error_t scl_reorder_build_inverse_permutation(
    const scl_index_t* permutation,
    scl_index_t* inverse,
    scl_size_t n
) {
    if (!permutation || !inverse) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        scl::Array<const scl::Index> perm_arr(
            reinterpret_cast<const scl::Index*>(permutation),
            n
        );
        scl::Array<scl::Index> inv_arr(
            reinterpret_cast<scl::Index*>(inverse),
            n
        );
        scl::kernel::reorder::build_inverse_permutation(perm_arr, inv_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::reorder::align_secondary<scl::Real, true>(
    scl::CSR&,
    scl::Array<const scl::Index>,
    scl::Array<scl::Index>,
    scl::Index
);

template scl::Size scl::kernel::reorder::compute_filtered_nnz<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Index>,
    scl::Index
);

} // extern "C"

