// =============================================================================
// FILE: scl/binding/c_api/reorder/reorder.cpp
// BRIEF: C API implementation for matrix reordering
// =============================================================================

#include "scl/binding/c_api/reorder.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/reorder.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::reorder;

extern "C" {

scl_error_t scl_reorder_align_secondary(
    scl_sparse_t matrix,
    const scl_index_t* index_map,
    scl_size_t old_dim,
    scl_index_t new_secondary_dim,
    scl_index_t* out_lengths)
{
    if (!matrix || !index_map || !out_lengths) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(matrix);
        
        if (!wrapper->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        const Index primary_dim = wrapper->rows();
        Array<const Index> map_arr(reinterpret_cast<const Index*>(index_map), old_dim);
        Array<Index> lengths_arr(reinterpret_cast<Index*>(out_lengths), static_cast<Size>(primary_dim));
        
        wrapper->visit([&](auto& m) {
            align_secondary(m, map_arr, lengths_arr, new_secondary_dim);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_reorder_compute_filtered_nnz(
    scl_sparse_t matrix,
    const scl_index_t* index_map,
    scl_size_t old_dim,
    scl_index_t new_secondary_dim,
    scl_size_t* out_nnz)
{
    if (!matrix || !index_map || !out_nnz) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(matrix);
        
        if (!wrapper->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<const Index> map_arr(reinterpret_cast<const Index*>(index_map), old_dim);
        
        Size nnz = wrapper->visit([&](auto& m) -> Size {
            return compute_filtered_nnz(m, map_arr, new_secondary_dim);
        });
        
        *out_nnz = nnz;
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_reorder_build_inverse_permutation(
    const scl_index_t* permutation,
    scl_size_t n,
    scl_index_t* inverse)
{
    if (!permutation || !inverse) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Index> perm_arr(reinterpret_cast<const Index*>(permutation), n);
        Array<Index> inv_arr(reinterpret_cast<Index*>(inverse), n);
        build_inverse_permutation(perm_arr, inv_arr);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

