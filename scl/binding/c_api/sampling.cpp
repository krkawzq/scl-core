// =============================================================================
// FILE: scl/binding/c_api/sampling/sampling.cpp
// BRIEF: C API implementation for advanced sampling
// =============================================================================

#include "scl/binding/c_api/sampling.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/sampling.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

using namespace scl;
using namespace scl::binding;
using namespace scl::kernel::sampling;

extern "C" {

scl_error_t scl_sampling_geometric_sketching(
    scl_sparse_t data,
    scl_size_t target_size,
    scl_index_t* selected_indices,
    scl_size_t* n_selected,
    uint64_t seed)
{
    if (!data || !selected_indices || !n_selected) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(data);
        
        if (!wrapper->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<Index> indices_arr(reinterpret_cast<Index*>(selected_indices), target_size);
        
        wrapper->visit([&](auto& m) {
            geometric_sketching(m, target_size, indices_arr.ptr, *n_selected, seed);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sampling_density_preserving(
    scl_sparse_t data,
    scl_sparse_t neighbors,
    scl_size_t target_size,
    scl_index_t* selected_indices,
    scl_size_t* n_selected)
{
    if (!data || !neighbors || !selected_indices || !n_selected) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper_data = static_cast<scl_sparse_matrix*>(data);
        auto* wrapper_neighbors = static_cast<scl_sparse_matrix*>(neighbors);
        
        if (!wrapper_data->valid() || !wrapper_neighbors->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<Index> indices_arr(reinterpret_cast<Index*>(selected_indices), target_size);
        
        wrapper_data->visit([&](auto& md) {
            wrapper_neighbors->visit([&](auto& mn) {
                density_preserving(md, mn, target_size, indices_arr.ptr, *n_selected);
            });
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sampling_landmark_selection(
    scl_sparse_t data,
    scl_size_t n_landmarks,
    scl_index_t* landmark_indices,
    scl_size_t* n_selected,
    uint64_t seed)
{
    if (!data || !landmark_indices || !n_selected) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(data);
        
        if (!wrapper->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<Index> indices_arr(reinterpret_cast<Index*>(landmark_indices), n_landmarks);
        
        wrapper->visit([&](auto& m) {
            landmark_selection(m, n_landmarks, indices_arr.ptr, *n_selected, seed);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sampling_representative_cells(
    scl_sparse_t data,
    const scl_index_t* cluster_labels,
    scl_size_t n_cells,
    scl_size_t per_cluster,
    scl_index_t* representatives,
    scl_size_t* n_selected,
    scl_size_t max_count,
    uint64_t seed)
{
    if (!data || !cluster_labels || !representatives || !n_selected) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        auto* wrapper = static_cast<scl_sparse_matrix*>(data);
        
        if (!wrapper->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<const Index> labels_arr(reinterpret_cast<const Index*>(cluster_labels), n_cells);
        Array<Index> reps_arr(reinterpret_cast<Index*>(representatives), max_count);
        
        wrapper->visit([&](auto& m) {
            representative_cells(m, labels_arr, per_cluster, reps_arr.ptr, *n_selected, seed);
        });
        
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sampling_balanced(
    const scl_index_t* labels,
    scl_size_t n,
    scl_size_t target_size,
    scl_index_t* selected_indices,
    scl_size_t* n_selected,
    uint64_t seed)
{
    if (!labels || !selected_indices || !n_selected) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Index> labels_arr(reinterpret_cast<const Index*>(labels), n);
        Array<Index> indices_arr(reinterpret_cast<Index*>(selected_indices), target_size);
        balanced_sampling(labels_arr, target_size, indices_arr.ptr, *n_selected, seed);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sampling_stratified(
    const scl_real_t* values,
    scl_size_t n,
    scl_size_t n_strata,
    scl_size_t target_size,
    scl_index_t* selected_indices,
    scl_size_t* n_selected,
    uint64_t seed)
{
    if (!values || !selected_indices || !n_selected) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Real> values_arr(reinterpret_cast<const Real*>(values), n);
        Array<Index> indices_arr(reinterpret_cast<Index*>(selected_indices), target_size);
        stratified_sampling(values_arr, n_strata, target_size, indices_arr.ptr, *n_selected, seed);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sampling_uniform(
    scl_size_t n,
    scl_size_t target_size,
    scl_index_t* selected_indices,
    scl_size_t* n_selected,
    uint64_t seed)
{
    if (!selected_indices || !n_selected) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<Index> indices_arr(reinterpret_cast<Index*>(selected_indices), target_size);
        uniform_sampling(n, target_size, indices_arr.ptr, *n_selected, seed);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sampling_importance(
    const scl_real_t* weights,
    scl_size_t n,
    scl_size_t target_size,
    scl_index_t* selected_indices,
    scl_size_t* n_selected,
    uint64_t seed)
{
    if (!weights || !selected_indices || !n_selected) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<const Real> weights_arr(reinterpret_cast<const Real*>(weights), n);
        Array<Index> indices_arr(reinterpret_cast<Index*>(selected_indices), target_size);
        importance_sampling(weights_arr, target_size, indices_arr.ptr, *n_selected, seed);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_sampling_reservoir(
    scl_size_t stream_size,
    scl_size_t reservoir_size,
    scl_index_t* reservoir,
    scl_size_t* n_selected,
    uint64_t seed)
{
    if (!reservoir || !n_selected) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        Array<Index> res_arr(reinterpret_cast<Index*>(reservoir), reservoir_size);
        reservoir_sampling(stream_size, reservoir_size, res_arr.ptr, *n_selected, seed);
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

