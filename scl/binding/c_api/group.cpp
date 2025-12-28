// =============================================================================
// FILE: scl/binding/c_api/group/group.cpp
// BRIEF: C API implementation for group aggregation statistics
// =============================================================================

#include "scl/binding/c_api/group.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/group.hpp"

#include <span>

using namespace scl;
using namespace scl::binding;

extern "C" {

scl_error_t scl_group_stats(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    scl_size_t n_groups,
    const scl_size_t* group_sizes,
    scl_real_t* out_means,
    scl_real_t* out_vars,
    int ddof,
    int include_zeros)
{
    if (!matrix || !group_ids || !group_sizes || !out_means || !out_vars) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    if (n_groups == 0) {
        set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Number of groups must be > 0");
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        if (!matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Index primary_dim = matrix->is_csr ? matrix->rows() : matrix->cols();
        Index secondary_dim = matrix->is_csr ? matrix->cols() : matrix->rows();
        
        Size total_size = static_cast<Size>(primary_dim) * static_cast<Size>(n_groups);
        
        Array<const int32_t> group_ids_arr(
            reinterpret_cast<const int32_t*>(group_ids),
            static_cast<Size>(secondary_dim)
        );
        Array<const Size> group_sizes_arr(
            reinterpret_cast<const Size*>(group_sizes),
            static_cast<Size>(n_groups)
        );
        Array<Real> out_means_arr(
            reinterpret_cast<Real*>(out_means),
            total_size
        );
        Array<Real> out_vars_arr(
            reinterpret_cast<Real*>(out_vars),
            total_size
        );
        
        matrix->visit([&](auto& m) {
            scl::kernel::group::group_stats(
                m,
                group_ids_arr,
                static_cast<Size>(n_groups),
                group_sizes_arr,
                out_means_arr,
                out_vars_arr,
                ddof,
                include_zeros != 0
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

