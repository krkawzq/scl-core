// =============================================================================
// FILE: scl/binding/c_api/stat/auroc.cpp
// BRIEF: C API implementation for AUROC computation
// =============================================================================

#include "scl/binding/c_api/stat/auroc.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/stat/auroc.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

scl_error_t scl_stat_auroc(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    scl_real_t* out_auroc,
    scl_real_t* out_p_values,
    scl_index_t primary_dim)
{
    if (!matrix || !group_ids || !out_auroc || !out_p_values) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Index actual_primary_dim = matrix->is_csr ? matrix->rows() : matrix->cols();
        Index secondary_dim = matrix->is_csr ? matrix->cols() : matrix->rows();
        
        if (static_cast<scl_index_t>(actual_primary_dim) != primary_dim) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Primary dimension mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }
        
        Array<const int32_t> group_ids_arr(
            group_ids,
            static_cast<Size>(secondary_dim)
        );
        Array<Real> out_auroc_arr(
            reinterpret_cast<Real*>(out_auroc),
            static_cast<Size>(primary_dim)
        );
        Array<Real> out_p_values_arr(
            reinterpret_cast<Real*>(out_p_values),
            static_cast<Size>(primary_dim)
        );
        
        matrix->visit([&](auto& m) {
            scl::kernel::stat::auroc::auroc(
                m,
                group_ids_arr,
                out_auroc_arr,
                out_p_values_arr
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_stat_auroc_with_fc(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    scl_real_t* out_auroc,
    scl_real_t* out_p_values,
    scl_real_t* out_log2_fc,
    scl_index_t primary_dim)
{
    if (!matrix || !group_ids || !out_auroc || !out_p_values || !out_log2_fc) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Index actual_primary_dim = matrix->is_csr ? matrix->rows() : matrix->cols();
        Index secondary_dim = matrix->is_csr ? matrix->cols() : matrix->rows();
        
        if (static_cast<scl_index_t>(actual_primary_dim) != primary_dim) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Primary dimension mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }
        
        Array<const int32_t> group_ids_arr(
            group_ids,
            static_cast<Size>(secondary_dim)
        );
        Array<Real> out_auroc_arr(
            reinterpret_cast<Real*>(out_auroc),
            static_cast<Size>(primary_dim)
        );
        Array<Real> out_p_values_arr(
            reinterpret_cast<Real*>(out_p_values),
            static_cast<Size>(primary_dim)
        );
        Array<Real> out_log2_fc_arr(
            reinterpret_cast<Real*>(out_log2_fc),
            static_cast<Size>(primary_dim)
        );
        
        matrix->visit([&](auto& m) {
            scl::kernel::stat::auroc::auroc_with_fc(
                m,
                group_ids_arr,
                out_auroc_arr,
                out_p_values_arr,
                out_log2_fc_arr
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

