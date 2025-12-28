// =============================================================================
// FILE: scl/binding/c_api/bbknn/bbknn.cpp
// BRIEF: C API implementation for BBKNN
// =============================================================================

#include "scl/binding/c_api/bbknn/bbknn.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/bbknn.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::bbknn;

extern "C" {

scl_error_t scl_bbknn(
    scl_sparse_t matrix,
    const int32_t* batch_labels,
    scl_size_t n_cells,
    scl_size_t n_batches,
    scl_size_t k,
    scl_index_t* out_indices,
    scl_real_t* out_distances)
{
    if (!matrix || !batch_labels || !out_indices || !out_distances) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(matrix);
        
        if (!wrapper->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        const Size neighbors_per_cell = n_batches * k;
        const Size total_neighbors = n_cells * neighbors_per_cell;
        
        Array<const int32_t> batch_arr(batch_labels, n_cells);
        Array<Index> indices_arr(reinterpret_cast<Index*>(out_indices), total_neighbors);
        Array<Real> distances_arr(reinterpret_cast<Real*>(out_distances), total_neighbors);
        
        wrapper->visit([&](auto& m) {
            bbknn(m, batch_arr, n_batches, k, indices_arr, distances_arr);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_bbknn_with_norms(
    scl_sparse_t matrix,
    const int32_t* batch_labels,
    scl_size_t n_cells,
    scl_size_t n_batches,
    scl_size_t k,
    const scl_real_t* norms_sq,
    scl_index_t* out_indices,
    scl_real_t* out_distances)
{
    if (!matrix || !batch_labels || !norms_sq || !out_indices || !out_distances) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(matrix);
        
        if (!wrapper->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        const Size neighbors_per_cell = n_batches * k;
        const Size total_neighbors = n_cells * neighbors_per_cell;
        
        Array<const int32_t> batch_arr(batch_labels, n_cells);
        Array<const Real> norms_arr(reinterpret_cast<const Real*>(norms_sq), n_cells);
        Array<Index> indices_arr(reinterpret_cast<Index*>(out_indices), total_neighbors);
        Array<Real> distances_arr(reinterpret_cast<Real*>(out_distances), total_neighbors);
        
        wrapper->visit([&](auto& m) {
            bbknn(m, batch_arr, n_batches, k, indices_arr, distances_arr, norms_arr);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

