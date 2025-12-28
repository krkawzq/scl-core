// =============================================================================
// FILE: scl/binding/c_api/neighbors/neighbors.cpp
// BRIEF: C API implementation for K-nearest neighbors
// =============================================================================

#include "scl/binding/c_api/neighbors/neighbors.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/neighbors.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"

#include <exception>

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::neighbors;

extern "C" {

scl_error_t scl_neighbors_compute_norms(
    scl_sparse_t matrix,
    scl_real_t* norms_sq)
{
    if (!matrix || !norms_sq) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(matrix);
        
        if (!wrapper->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        const Index n = wrapper->rows();
        Array<Real> norms_arr(reinterpret_cast<Real*>(norms_sq), static_cast<Size>(n));
        
        wrapper->visit([&](auto& m) {
            compute_norms(m, norms_arr);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_knn(
    scl_sparse_t matrix,
    const scl_real_t* norms_sq,
    scl_size_t n_samples,
    scl_size_t k,
    scl_index_t* out_indices,
    scl_real_t* out_distances)
{
    if (!matrix || !norms_sq || !out_indices || !out_distances) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(matrix);
        
        if (!wrapper->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        const Size total_neighbors = n_samples * k;
        Array<const Real> norms_arr(reinterpret_cast<const Real*>(norms_sq), n_samples);
        Array<Index> indices_arr(reinterpret_cast<Index*>(out_indices), total_neighbors);
        Array<Real> distances_arr(reinterpret_cast<Real*>(out_distances), total_neighbors);
        
        wrapper->visit([&](auto& m) {
            knn(m, norms_arr, k, indices_arr, distances_arr);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_knn_simple(
    scl_sparse_t matrix,
    scl_size_t n_samples,
    scl_size_t k,
    scl_index_t* out_indices,
    scl_real_t* out_distances)
{
    if (!matrix || !out_indices || !out_distances) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(matrix);
        
        if (!wrapper->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        // Compute norms first
        Real* norms_sq = scl::memory::aligned_alloc<Real>(n_samples, SCL_ALIGNMENT);
        Array<Real> norms_arr(norms_sq, n_samples);
        
        wrapper->visit([&](auto& m) {
            compute_norms(m, norms_arr);
        });
        
        // Compute KNN
        const Size total_neighbors = n_samples * k;
        Array<const Real> norms_const_arr(norms_sq, n_samples);
        Array<Index> indices_arr(reinterpret_cast<Index*>(out_indices), total_neighbors);
        Array<Real> distances_arr(reinterpret_cast<Real*>(out_distances), total_neighbors);
        
        wrapper->visit([&](auto& m) {
            knn(m, norms_const_arr, k, indices_arr, distances_arr);
        });
        
        scl::memory::aligned_free(norms_sq, SCL_ALIGNMENT);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

