// =============================================================================
// FILE: scl/binding/c_api/propagation/propagation.cpp
// BRIEF: C API implementation for label propagation
// =============================================================================

#include "scl/binding/c_api/propagation.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/propagation.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

scl_error_t scl_propagation_label_propagation(
    scl_sparse_t adjacency,
    scl_index_t* labels,
    scl_index_t n,
    scl_index_t max_iter,
    uint64_t seed)
{
    if (!adjacency || !labels) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!adjacency->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<Index> labels_arr(
            reinterpret_cast<Index*>(labels),
            static_cast<Size>(n)
        );
        
        adjacency->visit([&](auto& adj) {
            scl::kernel::propagation::label_propagation(
                adj,
                labels_arr,
                max_iter,
                seed
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_propagation_label_spreading(
    scl_sparse_t adjacency,
    scl_real_t* label_probs,
    scl_index_t n,
    scl_index_t n_classes,
    scl_real_t alpha,
    scl_index_t max_iter,
    scl_real_t tolerance)
{
    if (!adjacency || !label_probs) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!adjacency->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Size total_size = static_cast<Size>(n) * static_cast<Size>(n_classes);
        Array<Real> label_probs_arr(
            reinterpret_cast<Real*>(label_probs),
            total_size
        );
        
        adjacency->visit([&](auto& adj) {
            scl::kernel::propagation::label_spreading(
                adj,
                label_probs_arr,
                n,
                n_classes,
                static_cast<Real>(alpha),
                max_iter,
                static_cast<Real>(tolerance)
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_propagation_harmonic_function(
    scl_sparse_t adjacency,
    scl_real_t* label_probs,
    scl_index_t n,
    scl_index_t n_classes,
    scl_real_t alpha)
{
    if (!adjacency || !label_probs) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!adjacency->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Size total_size = static_cast<Size>(n) * static_cast<Size>(n_classes);
        Array<Real> label_probs_arr(
            reinterpret_cast<Real*>(label_probs),
            total_size
        );
        
        adjacency->visit([&](auto& adj) {
            scl::kernel::propagation::harmonic_function(
                adj,
                label_probs_arr,
                n,
                n_classes,
                static_cast<Real>(alpha)
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_propagation_confidence(
    scl_sparse_t adjacency,
    scl_real_t* confidences,
    scl_index_t n,
    scl_real_t alpha,
    scl_index_t max_iter,
    scl_real_t tolerance)
{
    if (!adjacency || !confidences) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!adjacency->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Array<Real> confidences_arr(
            reinterpret_cast<Real*>(confidences),
            static_cast<Size>(n)
        );
        
        adjacency->visit([&](auto& adj) {
            scl::kernel::propagation::confidence_propagation(
                adj,
                confidences_arr,
                n,
                static_cast<Real>(alpha),
                max_iter,
                static_cast<Real>(tolerance)
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

