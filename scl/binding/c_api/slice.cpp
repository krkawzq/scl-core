// =============================================================================
// FILE: scl/binding/c_api/slice.cpp
// BRIEF: C API implementation for sparse matrix slicing
// =============================================================================

#include "scl/binding/c_api/slice.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/slice.hpp"
#include "scl/core/type.hpp"
#include "scl/core/registry.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Inspect Primary Slice
// =============================================================================

SCL_EXPORT scl_error_t scl_slice_inspect_primary(
    scl_sparse_t matrix,
    const scl_index_t* keep_indices,
    const scl_size_t n_keep,
    scl_index_t* out_nnz) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(keep_indices, "Keep indices pointer is null");
    SCL_C_API_CHECK_NULL(out_nnz, "Output nnz pointer is null");
    SCL_C_API_CHECK(n_keep > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of indices must be positive");

    SCL_C_API_TRY
        const Size n_keep_sz = static_cast<Size>(n_keep);
        Array<const Index> keep_arr(reinterpret_cast<const Index*>(keep_indices), n_keep_sz);

        Index nnz = 0;
        matrix->visit([&](const auto& m) {
            nnz = scl::kernel::slice::inspect_slice_primary(m, keep_arr);
        });

        *out_nnz = static_cast<scl_index_t>(nnz);
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Slice Primary Copy
// =============================================================================

SCL_EXPORT scl_error_t scl_slice_primary_copy(
    scl_sparse_t src,
    const scl_index_t* keep_indices,
    const scl_size_t n_keep,
    scl_sparse_t* out) {
    
    SCL_C_API_CHECK_NULL(src, "Source matrix is null");
    SCL_C_API_CHECK_NULL(keep_indices, "Keep indices pointer is null");
    SCL_C_API_CHECK_NULL(out, "Output pointer is null");
    SCL_C_API_CHECK(n_keep > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of indices must be positive");

    SCL_C_API_TRY
        const Size n_keep_sz = static_cast<Size>(n_keep);
        Array<const Index> keep_arr(reinterpret_cast<const Index*>(keep_indices), n_keep_sz);
        
        auto& reg = get_registry();
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate slice matrix handle");
        
        handle->is_csr = src->is_csr;
        
        if (src->is_csr) {
            auto& m = src->as_csr();
            handle->matrix = scl::kernel::slice::slice_primary(m, keep_arr);
        } else {
            auto& m = src->as_csc();
            handle->matrix = scl::kernel::slice::slice_primary(m, keep_arr);
        }
        
        SCL_CHECK_ARG(handle->valid(), "Failed to slice matrix");
        
        *out = handle;
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
