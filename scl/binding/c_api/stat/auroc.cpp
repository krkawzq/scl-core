// =============================================================================
// FILE: scl/binding/c_api/stat/auroc.cpp
// BRIEF: C API implementation for AUROC computation
// =============================================================================

#include "scl/binding/c_api/stat/auroc.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/stat/auroc.hpp"
#include "scl/core/type.hpp"

namespace scl::binding {
    using namespace scl::kernel::stat::auroc;
}

extern "C" {

// =============================================================================
// AUROC
// =============================================================================

scl_error_t scl_stat_auroc(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    scl_real_t* out_auroc,
    scl_real_t* out_p_values,
    scl_index_t primary_dim)
{
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(group_ids, "Group IDs pointer is null");
    SCL_C_API_CHECK_NULL(out_auroc, "Output AUROC pointer is null");
    SCL_C_API_CHECK_NULL(out_p_values, "Output p-values pointer is null");

    SCL_C_API_TRY {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        SCL_C_API_CHECK(wrapper->valid(), SCL_ERROR_INVALID_ARGUMENT,
                       "Invalid sparse matrix");

        const Index actual_primary_dim = wrapper->is_csr_format() 
                                       ? wrapper->rows() 
                                       : wrapper->cols();
        const Index secondary_dim = wrapper->is_csr_format() 
                                  ? wrapper->cols() 
                                  : wrapper->rows();
        
        SCL_C_API_CHECK(static_cast<scl_index_t>(actual_primary_dim) == primary_dim,
                       SCL_ERROR_DIMENSION_MISMATCH,
                       "Primary dimension mismatch");

        const Size primary_dim_sz = static_cast<Size>(primary_dim);
        const Size secondary_dim_sz = static_cast<Size>(secondary_dim);
        
        Array<const int32_t> group_ids_arr(group_ids, secondary_dim_sz);
        Array<Real> auroc_arr(reinterpret_cast<Real*>(out_auroc), primary_dim_sz);
        Array<Real> p_values_arr(reinterpret_cast<Real*>(out_p_values), primary_dim_sz);
        
        wrapper->visit([&](auto& m) {
            auroc(m, group_ids_arr, auroc_arr, p_values_arr);
        });
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

// =============================================================================
// AUROC with Fold Change
// =============================================================================

scl_error_t scl_stat_auroc_with_fc(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    scl_real_t* out_auroc,
    scl_real_t* out_p_values,
    scl_real_t* out_log2_fc,
    scl_index_t primary_dim)
{
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(group_ids, "Group IDs pointer is null");
    SCL_C_API_CHECK_NULL(out_auroc, "Output AUROC pointer is null");
    SCL_C_API_CHECK_NULL(out_p_values, "Output p-values pointer is null");
    SCL_C_API_CHECK_NULL(out_log2_fc, "Output log2 fold change pointer is null");

    SCL_C_API_TRY {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        SCL_C_API_CHECK(wrapper->valid(), SCL_ERROR_INVALID_ARGUMENT,
                       "Invalid sparse matrix");

        const Index actual_primary_dim = wrapper->is_csr_format() 
                                       ? wrapper->rows() 
                                       : wrapper->cols();
        const Index secondary_dim = wrapper->is_csr_format() 
                                  ? wrapper->cols() 
                                  : wrapper->rows();
        
        SCL_C_API_CHECK(static_cast<scl_index_t>(actual_primary_dim) == primary_dim,
                       SCL_ERROR_DIMENSION_MISMATCH,
                       "Primary dimension mismatch");

        const Size primary_dim_sz = static_cast<Size>(primary_dim);
        const Size secondary_dim_sz = static_cast<Size>(secondary_dim);
        
        Array<const int32_t> group_ids_arr(group_ids, secondary_dim_sz);
        Array<Real> auroc_arr(reinterpret_cast<Real*>(out_auroc), primary_dim_sz);
        Array<Real> p_values_arr(reinterpret_cast<Real*>(out_p_values), primary_dim_sz);
        Array<Real> log2_fc_arr(reinterpret_cast<Real*>(out_log2_fc), primary_dim_sz);
        
        wrapper->visit([&](auto& m) {
            auroc_with_fc(m, group_ids_arr, auroc_arr, p_values_arr, log2_fc_arr);
        });
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

} // extern "C"
