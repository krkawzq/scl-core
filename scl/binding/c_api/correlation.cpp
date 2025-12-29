// =============================================================================
// FILE: scl/binding/c_api/correlation.cpp
// BRIEF: C API implementation for correlation analysis
// =============================================================================

#include "scl/binding/c_api/correlation.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/correlation.hpp"
#include "scl/core/type.hpp"

namespace scl::binding {
    using namespace scl::kernel::correlation;
}

extern "C" {

// =============================================================================
// Compute Statistics
// =============================================================================

scl_error_t scl_corr_compute_stats(
    scl_sparse_t matrix,
    scl_real_t* out_means,
    scl_real_t* out_inv_stds)
{
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(out_means, "Output means pointer is null");
    SCL_C_API_CHECK_NULL(out_inv_stds, "Output inverse stds pointer is null");

    SCL_C_API_TRY {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        SCL_C_API_CHECK(wrapper->valid(), SCL_ERROR_INVALID_ARGUMENT,
                       "Invalid sparse matrix");

        const Index n_rows = wrapper->rows();
        const Size n_rows_sz = static_cast<Size>(n_rows);
        
        Array<Real> means_arr(reinterpret_cast<Real*>(out_means), n_rows_sz);
        Array<Real> inv_stds_arr(reinterpret_cast<Real*>(out_inv_stds), n_rows_sz);

        wrapper->visit([&](auto& m) {
            compute_stats(m, means_arr, inv_stds_arr);
        });

        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

// =============================================================================
// Pearson Correlation
// =============================================================================

scl_error_t scl_corr_pearson(
    scl_sparse_t matrix,
    const scl_real_t* means,
    const scl_real_t* inv_stds,
    scl_real_t* output)
{
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(means, "Means pointer is null");
    SCL_C_API_CHECK_NULL(inv_stds, "Inverse stds pointer is null");
    SCL_C_API_CHECK_NULL(output, "Output pointer is null");

    SCL_C_API_TRY {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        SCL_C_API_CHECK(wrapper->valid(), SCL_ERROR_INVALID_ARGUMENT,
                       "Invalid sparse matrix");

        const Index n_rows = wrapper->rows();
        const Size n_rows_sz = static_cast<Size>(n_rows);
        const Size n_sq = n_rows_sz * n_rows_sz;

        Array<const Real> means_arr(reinterpret_cast<const Real*>(means), n_rows_sz);
        Array<const Real> inv_stds_arr(reinterpret_cast<const Real*>(inv_stds), n_rows_sz);
        Array<Real> out_arr(reinterpret_cast<Real*>(output), n_sq);

        wrapper->visit([&](auto& m) {
            pearson(m, means_arr, inv_stds_arr, out_arr);
        });

        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

// =============================================================================
// Pearson Correlation (Auto)
// =============================================================================

scl_error_t scl_corr_pearson_auto(
    scl_sparse_t matrix,
    scl_real_t* output)
{
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(output, "Output pointer is null");

    SCL_C_API_TRY {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        SCL_C_API_CHECK(wrapper->valid(), SCL_ERROR_INVALID_ARGUMENT,
                       "Invalid sparse matrix");

        const Index n_rows = wrapper->rows();
        const Size n_rows_sz = static_cast<Size>(n_rows);
        const Size n_sq = n_rows_sz * n_rows_sz;

        Array<Real> out_arr(reinterpret_cast<Real*>(output), n_sq);

        wrapper->visit([&](auto& m) {
            pearson(m, out_arr);
        });

        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

} // extern "C"
