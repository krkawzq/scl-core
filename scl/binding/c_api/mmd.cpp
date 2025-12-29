// =============================================================================
// FILE: scl/binding/c_api/mmd.cpp
// BRIEF: C API implementation for MMD
// =============================================================================

#include "scl/binding/c_api/mmd.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/mmd.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// MMD RBF
// =============================================================================

SCL_EXPORT scl_error_t scl_mmd_rbf(
    scl_sparse_t mat_x,
    scl_sparse_t mat_y,
    scl_real_t* output,
    const scl_real_t gamma) {
    
    SCL_C_API_CHECK_NULL(mat_x, "Matrix X handle is null");
    SCL_C_API_CHECK_NULL(mat_y, "Matrix Y handle is null");
    SCL_C_API_CHECK_NULL(output, "Output pointer is null");

    SCL_C_API_TRY
        SCL_C_API_CHECK(mat_x->is_csr == mat_y->is_csr,
                       SCL_ERROR_TYPE_MISMATCH,
                       "Matrices must have the same format");
        
        // mmd_rbf expects output array with size = primary_dim
        const Index primary_dim = mat_x->rows();
        
        if (mat_x->is_csr) {
            auto& mx = mat_x->as_csr();
            auto& my = mat_y->as_csr();
            
            using T = typename std::remove_reference_t<decltype(mx)>::ValueType;
            auto temp_output_ptr = scl::memory::aligned_alloc<T>(static_cast<Size>(primary_dim), SCL_ALIGNMENT);
            T* temp_output = temp_output_ptr.get();
            Array<T> temp_arr(temp_output, static_cast<Size>(primary_dim));
            
            scl::kernel::mmd::mmd_rbf(mx, my, temp_arr, static_cast<T>(gamma));
            
            // Copy to output
            for (Index i = 0; i < primary_dim; ++i) {
                output[i] = static_cast<scl_real_t>(temp_output[i]);
            }
        } else {
            auto& mx = mat_x->as_csc();
            auto& my = mat_y->as_csc();
            
            using T = typename std::remove_reference_t<decltype(mx)>::ValueType;
            auto temp_output_ptr = scl::memory::aligned_alloc<T>(static_cast<Size>(primary_dim), SCL_ALIGNMENT);
            T* temp_output = temp_output_ptr.get();
            Array<T> temp_arr(temp_output, static_cast<Size>(primary_dim));
            
            scl::kernel::mmd::mmd_rbf(mx, my, temp_arr, static_cast<T>(gamma));
            
            // Copy to output
            for (Index i = 0; i < primary_dim; ++i) {
                output[i] = static_cast<scl_real_t>(temp_output[i]);
            }
        }
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
