// =============================================================================
// FILE: scl/binding/c_api/group.cpp
// BRIEF: C API implementation for group aggregation statistics
// =============================================================================

#include "scl/binding/c_api/group.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/group.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Group Statistics
// =============================================================================

SCL_EXPORT scl_error_t scl_group_stats(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    const scl_size_t n_groups,
    const scl_size_t* group_sizes,
    scl_real_t* out_means,
    scl_real_t* out_vars,
    const int ddof,
    const int include_zeros) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(group_ids, "Group IDs pointer is null");
    SCL_C_API_CHECK_NULL(group_sizes, "Group sizes pointer is null");
    SCL_C_API_CHECK_NULL(out_means, "Output means pointer is null");
    SCL_C_API_CHECK_NULL(out_vars, "Output variances pointer is null");
    SCL_C_API_CHECK(n_groups > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of groups must be positive");

    SCL_C_API_TRY
        const Index primary_dim = matrix->is_csr_format() 
                                 ? matrix->rows() 
                                 : matrix->cols();
        const Index secondary_dim = matrix->is_csr_format() 
                                  ? matrix->cols() 
                                  : matrix->rows();
        
        const Size primary_dim_sz = static_cast<Size>(primary_dim);
        const Size secondary_dim_sz = static_cast<Size>(secondary_dim);
        const Size n_groups_sz = static_cast<Size>(n_groups);
        const Size total_size = primary_dim_sz * n_groups_sz;
        
        Array<const int32_t> group_ids_arr(group_ids, secondary_dim_sz);
        Array<const Size> group_sizes_arr(reinterpret_cast<const Size*>(group_sizes), n_groups_sz);
        Array<Real> means_arr(reinterpret_cast<Real*>(out_means), total_size);
        Array<Real> vars_arr(reinterpret_cast<Real*>(out_vars), total_size);
        
        matrix->visit([&](auto& m) {
            scl::kernel::group::group_stats(m, group_ids_arr, n_groups_sz, group_sizes_arr,
                      means_arr, vars_arr, ddof, include_zeros != 0);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
