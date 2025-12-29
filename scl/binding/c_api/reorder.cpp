// =============================================================================
// FILE: scl/binding/c_api/reorder.cpp
// BRIEF: C API implementation for matrix reordering
// =============================================================================

#include "scl/binding/c_api/reorder.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/reorder.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Align Secondary Dimension
// =============================================================================

SCL_EXPORT scl_error_t scl_reorder_align_secondary(
    scl_sparse_t matrix,
    const scl_index_t* index_map,
    scl_size_t old_dim,
    scl_index_t new_secondary_dim,
    scl_index_t* out_lengths)
{
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(index_map, "Index map pointer is null");
    SCL_C_API_CHECK_NULL(out_lengths, "Output lengths buffer pointer is null");
    SCL_C_API_CHECK(old_dim > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Old dimension must be positive");
    SCL_C_API_CHECK(new_secondary_dim > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "New secondary dimension must be positive");

    SCL_C_API_TRY {
        const Index primary_dim = matrix->rows();
        const Size primary_dim_sz = static_cast<Size>(primary_dim);
        const Size old_dim_sz = static_cast<Size>(old_dim);
        
        Array<const Index> map_arr(reinterpret_cast<const Index*>(index_map), old_dim_sz);
        Array<Index> lengths_arr(reinterpret_cast<Index*>(out_lengths), primary_dim_sz);
        
        matrix->visit([&](auto& m) {
            scl::kernel::reorder::align_secondary(m, map_arr, lengths_arr, new_secondary_dim);
        });
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

// =============================================================================
// Compute Filtered NNZ
// =============================================================================

SCL_EXPORT scl_error_t scl_reorder_compute_filtered_nnz(
    scl_sparse_t matrix,
    const scl_index_t* index_map,
    scl_size_t old_dim,
    scl_index_t new_secondary_dim,
    scl_size_t* out_nnz)
{
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(index_map, "Index map pointer is null");
    SCL_C_API_CHECK_NULL(out_nnz, "Output NNZ pointer is null");
    SCL_C_API_CHECK(old_dim > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Old dimension must be positive");
    SCL_C_API_CHECK(new_secondary_dim > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "New secondary dimension must be positive");

    SCL_C_API_TRY {
        const Size old_dim_sz = static_cast<Size>(old_dim);
        Array<const Index> map_arr(reinterpret_cast<const Index*>(index_map), old_dim_sz);
        
        const Size nnz = matrix->visit([&](auto& m) -> Size {
            return scl::kernel::reorder::compute_filtered_nnz(m, map_arr, new_secondary_dim);
        });
        
        *out_nnz = nnz;
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

// =============================================================================
// Build Inverse Permutation
// =============================================================================

SCL_EXPORT scl_error_t scl_reorder_build_inverse_permutation(
    const scl_index_t* permutation,
    scl_size_t n,
    scl_index_t* inverse)
{
    SCL_C_API_CHECK_NULL(permutation, "Permutation array pointer is null");
    SCL_C_API_CHECK_NULL(inverse, "Inverse array pointer is null");
    SCL_C_API_CHECK(n > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Array size must be positive");

    SCL_C_API_TRY {
        const Size n_sz = static_cast<Size>(n);
        Array<const Index> perm_arr(reinterpret_cast<const Index*>(permutation), n_sz);
        Array<Index> inv_arr(reinterpret_cast<Index*>(inverse), n_sz);
        
        scl::kernel::reorder::build_inverse_permutation(perm_arr, inv_arr);
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

} // extern "C"
