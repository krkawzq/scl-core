// =============================================================================
// FILE: scl/binding/c_api/sparse_kernel/sparse_kernel.cpp
// BRIEF: C API implementation for sparse matrix statistics
// =============================================================================

#include "scl/binding/c_api/sparse_kernel.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/binding/c_api/core/sparse.h"
#include "scl/kernel/sparse.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/registry.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

scl_error_t scl_sparse_kernel_primary_sums(
    scl_sparse_t matrix,
    scl_real_t* output,
    scl_size_t primary_dim)
{
    if (!matrix || !output) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        Index pdim = wrapper->rows();
        if (static_cast<scl_size_t>(pdim) != primary_dim) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Primary dimension mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        Array<Real> output_arr(
            reinterpret_cast<Real*>(output),
            primary_dim
        );

        wrapper->visit([&](const auto& m) {
            scl::kernel::sparse::primary_sums(m, output_arr);
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_kernel_primary_means(
    scl_sparse_t matrix,
    scl_real_t* output,
    scl_size_t primary_dim)
{
    if (!matrix || !output) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        Index pdim = wrapper->rows();
        if (static_cast<scl_size_t>(pdim) != primary_dim) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Primary dimension mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        Array<Real> output_arr(
            reinterpret_cast<Real*>(output),
            primary_dim
        );

        wrapper->visit([&](const auto& m) {
            scl::kernel::sparse::primary_means(m, output_arr);
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_kernel_primary_variances(
    scl_sparse_t matrix,
    scl_real_t* output,
    scl_size_t primary_dim,
    int ddof)
{
    if (!matrix || !output) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        Index pdim = wrapper->rows();
        if (static_cast<scl_size_t>(pdim) != primary_dim) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Primary dimension mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        Array<Real> output_arr(
            reinterpret_cast<Real*>(output),
            primary_dim
        );

        wrapper->visit([&](const auto& m) {
            scl::kernel::sparse::primary_variances(m, output_arr, ddof);
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_kernel_primary_nnz(
    scl_sparse_t matrix,
    scl_index_t* output,
    scl_size_t primary_dim)
{
    if (!matrix || !output) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        Index pdim = wrapper->rows();
        if (static_cast<scl_size_t>(pdim) != primary_dim) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Primary dimension mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }

        Array<Index> output_arr(
            reinterpret_cast<Index*>(output),
            primary_dim
        );

        wrapper->visit([&](const auto& m) {
            scl::kernel::sparse::primary_nnz(m, output_arr);
        });

        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_kernel_eliminate_zeros(
    scl_sparse_t matrix,
    scl_sparse_t* out_matrix,
    scl_real_t tolerance)
{
    if (!matrix || !out_matrix) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        auto& reg = get_registry();
        auto* result_wrapper = reg.new_object<SparseWrapper>();
        if (!result_wrapper) {
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
            return SCL_ERROR_OUT_OF_MEMORY;
        }

        wrapper->visit([&](const auto& m) {
            auto result = scl::kernel::sparse::eliminate_zeros(m, static_cast<Real>(tolerance));
            if (wrapper->is_csr) {
                result_wrapper->matrix = CSR(std::move(result));
                result_wrapper->is_csr = true;
            } else {
                result_wrapper->matrix = CSC(std::move(result));
                result_wrapper->is_csr = false;
            }
        });

        if (!result_wrapper->valid()) {
            reg.unregister_ptr(result_wrapper);
            set_last_error(SCL_ERROR_INTERNAL, "Failed to create result matrix");
            return SCL_ERROR_INTERNAL;
        }

        *out_matrix = result_wrapper;
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sparse_kernel_prune(
    scl_sparse_t matrix,
    scl_sparse_t* out_matrix,
    scl_real_t threshold,
    int keep_structure)
{
    if (!matrix || !out_matrix) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }

    try {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        auto& reg = get_registry();
        auto* result_wrapper = reg.new_object<SparseWrapper>();
        if (!result_wrapper) {
            set_last_error(SCL_ERROR_OUT_OF_MEMORY, "Failed to allocate wrapper");
            return SCL_ERROR_OUT_OF_MEMORY;
        }

        wrapper->visit([&](const auto& m) {
            auto result = scl::kernel::sparse::prune(
                m,
                static_cast<Real>(threshold),
                keep_structure != 0
            );
            if (wrapper->is_csr) {
                result_wrapper->matrix = CSR(std::move(result));
                result_wrapper->is_csr = true;
            } else {
                result_wrapper->matrix = CSC(std::move(result));
                result_wrapper->is_csr = false;
            }
        });

        if (!result_wrapper->valid()) {
            reg.unregister_ptr(result_wrapper);
            set_last_error(SCL_ERROR_INTERNAL, "Failed to create result matrix");
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

