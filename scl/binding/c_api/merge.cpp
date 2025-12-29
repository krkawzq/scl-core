// =============================================================================
// FILE: scl/binding/c_api/merge.cpp
// BRIEF: C API implementation for matrix merging
// =============================================================================

#include "scl/binding/c_api/merge.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/merge.hpp"
#include "scl/core/registry.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Vertical Stack
// =============================================================================

SCL_EXPORT scl_error_t scl_merge_vstack(
    scl_sparse_t matrix1,
    scl_sparse_t matrix2,
    scl_sparse_t* result) {
    
    SCL_C_API_CHECK_NULL(matrix1, "Matrix1 handle is null");
    SCL_C_API_CHECK_NULL(matrix2, "Matrix2 handle is null");
    SCL_C_API_CHECK_NULL(result, "Result handle pointer is null");

    SCL_C_API_TRY
        SCL_C_API_CHECK(matrix1->is_csr_format() == matrix2->is_csr_format(),
                       SCL_ERROR_TYPE_MISMATCH,
                       "Matrices must have the same format");

        const bool is_csr = matrix1->is_csr_format();
        
        auto& reg = get_registry();
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate merged matrix handle");
        
        handle->is_csr = is_csr;
        
        if (is_csr) {
            auto& m1 = matrix1->as_csr();
            auto& m2 = matrix2->as_csr();
            handle->matrix = scl::kernel::merge::vstack(m1, m2);
        } else {
            auto& m1 = matrix1->as_csc();
            auto& m2 = matrix2->as_csc();
            handle->matrix = scl::kernel::merge::vstack(m1, m2);
        }
        
        SCL_CHECK_ARG(handle->valid(), "Failed to merge matrices");
        
        *result = handle;
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Horizontal Stack
// =============================================================================

SCL_EXPORT scl_error_t scl_merge_hstack(
    scl_sparse_t matrix1,
    scl_sparse_t matrix2,
    scl_sparse_t* result) {
    SCL_C_API_CHECK_NULL(matrix1, "Matrix1 handle is null");
    SCL_C_API_CHECK_NULL(matrix2, "Matrix2 handle is null");
    SCL_C_API_CHECK_NULL(result, "Result handle pointer is null");

    SCL_C_API_TRY
        SCL_C_API_CHECK(matrix1->is_csr_format() == matrix2->is_csr_format(),
                       SCL_ERROR_TYPE_MISMATCH,
                       "Matrices must have the same format");

        const bool is_csr = matrix1->is_csr_format();
        
        auto& reg = get_registry();
        auto* handle = reg.new_object<scl_sparse_matrix>();
        SCL_CHECK_NULL(handle, "Failed to allocate merged matrix handle");
        
        handle->is_csr = is_csr;
        
        if (is_csr) {
            auto& m1 = matrix1->as_csr();
            auto& m2 = matrix2->as_csr();
            handle->matrix = scl::kernel::merge::hstack(m1, m2);
        } else {
            auto& m1 = matrix1->as_csc();
            auto& m2 = matrix2->as_csc();
            handle->matrix = scl::kernel::merge::hstack(m1, m2);
        }
        
        SCL_CHECK_ARG(handle->valid(), "Failed to merge matrices");
        
        *result = handle;
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
