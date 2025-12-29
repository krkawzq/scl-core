// =============================================================================
// FILE: scl/binding/c_api/markers.cpp
// BRIEF: C API implementation for marker gene selection
// =============================================================================

#include "scl/binding/c_api/markers.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/markers.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Group Mean Expression
// =============================================================================

SCL_EXPORT scl_error_t scl_markers_group_mean_expression(
    scl_sparse_t expression,
    const scl_index_t* group_labels,
    const scl_size_t n_cells,
    const scl_index_t n_groups,
    const scl_index_t n_genes,
    scl_real_t* mean_expr) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(group_labels, "Group labels array is null");
    SCL_C_API_CHECK_NULL(mean_expr, "Output mean expression array is null");
    SCL_C_API_CHECK(n_cells > 0 && n_groups > 0 && n_genes > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");

    SCL_C_API_TRY
        expression->visit([&](auto& expr) {
            scl::kernel::markers::group_mean_expression(
                expr,
                Array<const Index>(
                    reinterpret_cast<const Index*>(group_labels),
                    static_cast<Size>(n_cells)
                ),
                static_cast<Index>(n_groups),
                Array<Real>(
                    reinterpret_cast<Real*>(mean_expr),
                    static_cast<Size>(n_genes) * static_cast<Size>(n_groups)
                ),
                static_cast<Index>(n_genes)
            );
        });

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Percent Expressed
// =============================================================================

SCL_EXPORT scl_error_t scl_markers_percent_expressed(
    scl_sparse_t expression,
    const scl_index_t* group_labels,
    const scl_size_t n_cells,
    const scl_index_t n_groups,
    const scl_index_t n_genes,
    scl_real_t* pct_expr,
    const scl_real_t threshold) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(group_labels, "Group labels array is null");
    SCL_C_API_CHECK_NULL(pct_expr, "Output percent expressed array is null");
    SCL_C_API_CHECK(n_cells > 0 && n_groups > 0 && n_genes > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");

    SCL_C_API_TRY
        expression->visit([&](auto& expr) {
            scl::kernel::markers::percent_expressed(
                expr,
                Array<const Index>(
                    reinterpret_cast<const Index*>(group_labels),
                    static_cast<Size>(n_cells)
                ),
                static_cast<Index>(n_groups),
                Array<Real>(
                    reinterpret_cast<Real*>(pct_expr),
                    static_cast<Size>(n_genes) * static_cast<Size>(n_groups)
                ),
                static_cast<Index>(n_genes),
                static_cast<Real>(threshold)
            );
        });

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Fold Change
// =============================================================================

SCL_EXPORT scl_error_t scl_markers_fold_change(
    scl_sparse_t expression,
    const scl_index_t* group_labels,
    const scl_size_t n_cells,
    const scl_index_t target_group,
    const scl_index_t n_groups,
    const scl_index_t n_genes,
    scl_real_t* fold_changes) {
    
    SCL_C_API_CHECK_NULL(expression, "Expression matrix is null");
    SCL_C_API_CHECK_NULL(group_labels, "Group labels array is null");
    SCL_C_API_CHECK_NULL(fold_changes, "Output fold changes array is null");
    SCL_C_API_CHECK(n_cells > 0 && n_groups > 0 && n_genes > 0,
                   SCL_ERROR_INVALID_ARGUMENT, "Dimensions must be positive");
    SCL_C_API_CHECK(target_group >= 0 && target_group < n_groups,
                   SCL_ERROR_INVALID_ARGUMENT, "Target group index out of range");

    SCL_C_API_TRY
        expression->visit([&](auto& expr) {
            scl::kernel::markers::log_fold_change(
                expr,
                Array<const Index>(
                    reinterpret_cast<const Index*>(group_labels),
                    static_cast<Size>(n_cells)
                ),
                static_cast<Index>(n_groups),
                target_group,
                Array<Real>(
                    reinterpret_cast<Real*>(fold_changes),
                    static_cast<Size>(n_genes)
                ),
                static_cast<Index>(n_genes)
            );
        });

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

} // extern "C"
