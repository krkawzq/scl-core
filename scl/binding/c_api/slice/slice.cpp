// =============================================================================
// FILE: scl/binding/c_api/slice/slice.cpp
// BRIEF: C API implementation for sparse matrix slicing
// =============================================================================

#include "scl/binding/c_api/slice/slice.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/binding/c_api/core/sparse.h"
#include "scl/kernel/slice.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/registry.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

scl_error_t scl_slice_inspect_primary(
    scl_sparse_t matrix,
    const scl_index_t* keep_indices,
    scl_size_t n_keep,
    scl_index_t* out_nnz)
{
    if (!matrix || !keep_indices || !out_nnz) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        Array<const Index> keep_arr(
            reinterpret_cast<const Index*>(keep_indices),
            n_keep
        );

        Index nnz = 0;
        wrapper->visit([&](const auto& m) {
            nnz = scl::kernel::slice::inspect_slice_primary(m, keep_arr);
        });

        *out_nnz = nnz;
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_slice_primary(
    scl_sparse_t matrix,
    const scl_index_t* keep_indices,
    scl_size_t n_keep,
    scl_sparse_t* out_matrix)
{
    if (!matrix || !keep_indices || !out_matrix) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        Array<const Index> keep_arr(
            reinterpret_cast<const Index*>(keep_indices),
            n_keep
        );

        auto& reg = get_registry();
        auto* result_wrapper = reg.new_object<SparseWrapper>();
        if (!result_wrapper) {
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
            return SCL_ERROR_OUT_OF_MEMORY;
        }

        wrapper->visit([&](const auto& m) {
            auto sliced = scl::kernel::slice::slice_primary(m, keep_arr);
            if (wrapper->is_csr) {
                result_wrapper->matrix = CSR(std::move(sliced));
                result_wrapper->is_csr = true;
            } else {
                result_wrapper->matrix = CSC(std::move(sliced));
                result_wrapper->is_csr = false;
            }
        });

        if (!result_wrapper->valid()) {
            reg.unregister_ptr(result_wrapper);
            set_last_error(SCL_ERROR_INTERNAL, "Failed to create sliced matrix");
            return SCL_ERROR_INTERNAL;
        }

        *out_matrix = result_wrapper;
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_slice_inspect_filter_secondary(
    scl_sparse_t matrix,
    const uint8_t* mask,
    scl_size_t secondary_dim,
    scl_index_t* out_nnz)
{
    if (!matrix || !mask || !out_nnz) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        Index secondary = wrapper->is_csr ? wrapper->cols() : wrapper->rows();
        if (static_cast<scl_size_t>(secondary) != secondary_dim) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Secondary dimension mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        Array<const uint8_t> mask_arr(mask, secondary_dim);

        Index nnz = 0;
        wrapper->visit([&](const auto& m) {
            nnz = scl::kernel::slice::inspect_filter_secondary(m, mask_arr);
        });

        *out_nnz = nnz;
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_slice_filter_secondary(
    scl_sparse_t matrix,
    const uint8_t* mask,
    scl_size_t secondary_dim,
    scl_sparse_t* out_matrix)
{
    if (!matrix || !mask || !out_matrix) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        Index secondary = wrapper->is_csr ? wrapper->cols() : wrapper->rows();
        if (static_cast<scl_size_t>(secondary) != secondary_dim) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Secondary dimension mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        Array<const uint8_t> mask_arr(mask, secondary_dim);

        auto& reg = get_registry();
        auto* result_wrapper = reg.new_object<SparseWrapper>();
        if (!result_wrapper) {
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
            return SCL_ERROR_OUT_OF_MEMORY;
        }

        wrapper->visit([&](const auto& m) {
            auto filtered = scl::kernel::slice::filter_secondary(m, mask_arr);
            if (wrapper->is_csr) {
                result_wrapper->matrix = CSR(std::move(filtered));
                result_wrapper->is_csr = true;
            } else {
                result_wrapper->matrix = CSC(std::move(filtered));
                result_wrapper->is_csr = false;
            }
        });

        if (!result_wrapper->valid()) {
            reg.unregister_ptr(result_wrapper);
            set_last_error(SCL_ERROR_INTERNAL, "Failed to create filtered matrix");
            return SCL_ERROR_INTERNAL;
        }

        *out_matrix = result_wrapper;
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

