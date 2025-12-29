// =============================================================================
// FILE: scl/binding/c_api/ttest.cpp
// BRIEF: C API implementation for T-test
// =============================================================================

#include "scl/binding/c_api/ttest.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/ttest.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// T-Test
// =============================================================================

SCL_EXPORT scl_error_t scl_ttest(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    const scl_size_t n_cells,
    scl_real_t* out_t_stats,
    const scl_size_t t_stats_size,
    scl_real_t* out_p_values,
    const scl_size_t p_values_size,
    scl_real_t* out_log2_fc,
    const scl_size_t log2_fc_size,
    const int use_welch) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(group_ids, "Group IDs pointer is null");
    SCL_C_API_CHECK_NULL(out_t_stats, "Output t-statistics pointer is null");
    SCL_C_API_CHECK_NULL(out_p_values, "Output p-values pointer is null");
    SCL_C_API_CHECK_NULL(out_log2_fc, "Output log2 fold change pointer is null");
    SCL_C_API_CHECK(n_cells > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Number of cells must be positive");

    SCL_C_API_TRY
        const Index primary_dim = matrix->rows();
        const Size primary_dim_sz = static_cast<Size>(primary_dim);
        const Size n_cells_sz = static_cast<Size>(n_cells);

        // Validate output array sizes
        SCL_C_API_CHECK(t_stats_size == primary_dim_sz, SCL_ERROR_DIMENSION_MISMATCH,
                       "t_stats_size must match primary dimension");
        SCL_C_API_CHECK(p_values_size == primary_dim_sz, SCL_ERROR_DIMENSION_MISMATCH,
                       "p_values_size must match primary dimension");
        SCL_C_API_CHECK(log2_fc_size == primary_dim_sz, SCL_ERROR_DIMENSION_MISMATCH,
                       "log2_fc_size must match primary dimension");

        // Wrap C arrays with Array views
        Array<const int32_t> groups_arr(group_ids, n_cells_sz);
        Array<Real> t_arr(reinterpret_cast<Real*>(out_t_stats), primary_dim_sz);
        Array<Real> p_arr(reinterpret_cast<Real*>(out_p_values), primary_dim_sz);
        Array<Real> fc_arr(reinterpret_cast<Real*>(out_log2_fc), primary_dim_sz);
        
        const bool use_welch_flag = (use_welch != 0);
        
        // Dispatch to kernel implementation
        matrix->visit([&](auto& m) {
            scl::kernel::ttest::ttest(m, groups_arr, t_arr, p_arr, fc_arr, use_welch_flag);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Compute Group Statistics
// =============================================================================

SCL_EXPORT scl_error_t scl_ttest_compute_group_stats(
    scl_sparse_t matrix,
    const int32_t* group_ids,
    const scl_size_t n_cells,
    const scl_size_t n_groups,
    scl_real_t* out_means,
    const scl_size_t means_size,
    scl_real_t* out_vars,
    const scl_size_t vars_size,
    scl_size_t* out_counts) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(group_ids, "Group IDs pointer is null");
    SCL_C_API_CHECK_NULL(out_means, "Output means pointer is null");
    SCL_C_API_CHECK_NULL(out_vars, "Output variances pointer is null");
    SCL_C_API_CHECK_NULL(out_counts, "Output counts pointer is null");
    SCL_C_API_CHECK(n_cells > 0 && n_groups > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Dimensions must be positive");

    SCL_C_API_TRY
        const Index primary_dim = matrix->rows();
        const Size primary_dim_sz = static_cast<Size>(primary_dim);
        const Size n_cells_sz = static_cast<Size>(n_cells);
        const Size n_groups_sz = static_cast<Size>(n_groups);
        const Size expected_size = primary_dim_sz * n_groups_sz;

        // Validate output array sizes
        SCL_C_API_CHECK(means_size == expected_size, SCL_ERROR_DIMENSION_MISMATCH,
                       "means_size must be primary_dim * n_groups");
        SCL_C_API_CHECK(vars_size == expected_size, SCL_ERROR_DIMENSION_MISMATCH,
                       "vars_size must be primary_dim * n_groups");

        // Wrap C arrays with Array views
        Array<const int32_t> groups_arr(group_ids, n_cells_sz);
        Array<Real> means_arr(reinterpret_cast<Real*>(out_means), expected_size);
        Array<Real> vars_arr(reinterpret_cast<Real*>(out_vars), expected_size);
        Array<Size> counts_arr(out_counts, expected_size);
        
        // Dispatch to kernel implementation
        matrix->visit([&](auto& m) {
            scl::kernel::ttest::compute_group_stats(m, groups_arr, n_groups_sz, means_arr, vars_arr, counts_arr);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
