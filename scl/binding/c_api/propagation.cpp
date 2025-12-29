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

SCL_EXPORT scl_error_t scl_propagation_label_spreading(
    scl_sparse_t adjacency,
    scl_real_t* label_probs,
    const unsigned char* is_labeled,
    const scl_index_t n,
    const scl_index_t n_classes,
    const scl_real_t alpha,
    const scl_index_t max_iter,
    const scl_real_t tolerance) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(label_probs, "Label probabilities array is null");
    SCL_C_API_CHECK_NULL(is_labeled, "Is labeled mask array is null");
    SCL_C_API_CHECK(n > 0 && n_classes > 0 && max_iter > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        const Size total_size = static_cast<Size>(n) * static_cast<Size>(n_classes);
        Array<Real> label_probs_arr(reinterpret_cast<Real*>(label_probs), total_size);
        const bool* is_labeled_bool = reinterpret_cast<const bool*>(is_labeled);
        
        adjacency->visit([&](auto& adj) {
            scl::kernel::propagation::label_spreading(
                adj,
                label_probs_arr,
                n_classes,
                is_labeled_bool,
                static_cast<Real>(alpha),
                max_iter,
                static_cast<Real>(tolerance)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_propagation_harmonic_function(
    scl_sparse_t adjacency,
    scl_real_t* values,
    const unsigned char* is_known,
    const scl_index_t n,
    const scl_index_t max_iter,
    const scl_real_t tolerance) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(values, "Values array is null");
    SCL_C_API_CHECK_NULL(is_known, "Is known mask array is null");
    SCL_C_API_CHECK(n > 0 && max_iter > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        Array<Real> values_arr(reinterpret_cast<Real*>(values), static_cast<Size>(n));
        const bool* is_known_bool = reinterpret_cast<const bool*>(is_known);
        
        adjacency->visit([&](auto& adj) {
            scl::kernel::propagation::harmonic_function(
                adj,
                values_arr,
                is_known_bool,
                max_iter,
                static_cast<Real>(tolerance)
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

SCL_EXPORT scl_error_t scl_propagation_confidence(
    scl_sparse_t adjacency,
    scl_index_t* labels,
    scl_real_t* confidences,
    const scl_index_t n,
    const scl_index_t n_classes,
    const scl_real_t alpha,
    const scl_index_t max_iter) {
    
    SCL_C_API_CHECK_NULL(adjacency, "Adjacency matrix is null");
    SCL_C_API_CHECK_NULL(labels, "Labels array is null");
    SCL_C_API_CHECK_NULL(confidences, "Confidences array is null");
    SCL_C_API_CHECK(n > 0 && n_classes > 0 && max_iter > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");
    
    SCL_C_API_TRY
        Array<Index> labels_arr(reinterpret_cast<Index*>(labels), static_cast<Size>(n));
        Array<Real> confidences_arr(reinterpret_cast<Real*>(confidences), static_cast<Size>(n));
        
        adjacency->visit([&](auto& adj) {
            scl::kernel::propagation::confidence_propagation(
                adj,
                labels_arr,
                confidences_arr,
                n_classes,
                static_cast<Real>(alpha),
                max_iter
            );
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"

