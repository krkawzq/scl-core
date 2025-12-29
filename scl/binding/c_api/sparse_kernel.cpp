// =============================================================================
// FILE: scl/binding/c_api/sparse_kernel/sparse_kernel.cpp
// BRIEF: C API implementation for sparse matrix statistics
// =============================================================================

#include "scl/binding/c_api/sparse_kernel.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/sparse.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/registry.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

SCL_EXPORT scl_error_t scl_sparse_kernel_primary_sums(
    scl_sparse_t matrix,
    scl_real_t* output,
    const scl_size_t primary_dim) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(output, "Output array is null");
    SCL_C_API_CHECK(primary_dim > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Primary dimension must be positive");

    SCL_C_API_TRY
        Array<Real> output_arr(reinterpret_cast<Real*>(output), primary_dim);

        matrix->visit([&](const auto& m) {
            scl::kernel::sparse::primary_sums(m, output_arr);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_sparse_kernel_primary_means(
    scl_sparse_t matrix,
    scl_real_t* output,
    scl_size_t primary_dim)
{
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(output, "Output array is null");
    SCL_C_API_CHECK(primary_dim > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Primary dimension must be positive");

    SCL_C_API_TRY
        Array<Real> output_arr(reinterpret_cast<Real*>(output), primary_dim);

        matrix->visit([&](const auto& m) {
            scl::kernel::sparse::primary_means(m, output_arr);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
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

SCL_EXPORT scl_error_t scl_sparse_kernel_eliminate_zeros(
    scl_sparse_t matrix,
    scl_sparse_t* out_matrix,
    const scl_real_t tolerance) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(out_matrix, "Output matrix pointer is null");

    SCL_C_API_TRY
        auto& reg = get_registry();
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate result matrix handle");
        
        handle->is_csr = matrix->is_csr;
        
        if (matrix->is_csr) {
            auto& m = matrix->as_csr();
            handle->matrix = scl::kernel::sparse::eliminate_zeros(m, static_cast<Real>(tolerance));
        } else {
            auto& m = matrix->as_csc();
            handle->matrix = scl::kernel::sparse::eliminate_zeros(m, static_cast<Real>(tolerance));
        }
        
        SCL_CHECK_ARG(handle->valid(), "Failed to eliminate zeros");
        
        *out_matrix = handle;
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_sparse_kernel_prune(
    scl_sparse_t matrix,
    scl_sparse_t* out_matrix,
    const scl_real_t threshold,
    const int keep_structure) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(out_matrix, "Output matrix pointer is null");

    SCL_C_API_TRY
        auto& reg = get_registry();
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate result matrix handle");
        
        handle->is_csr = matrix->is_csr;
        
        if (matrix->is_csr) {
            auto& m = matrix->as_csr();
            handle->matrix = scl::kernel::sparse::prune(
                m, static_cast<Real>(threshold), keep_structure != 0
            );
        } else {
            auto& m = matrix->as_csc();
            handle->matrix = scl::kernel::sparse::prune(
                m, static_cast<Real>(threshold), keep_structure != 0
            );
        }
        
        SCL_CHECK_ARG(handle->valid(), "Failed to prune matrix");
        
        *out_matrix = handle;
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"

