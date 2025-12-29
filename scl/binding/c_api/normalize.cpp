// =============================================================================
// FILE: scl/binding/c_api/normalize.cpp
// BRIEF: C API implementation for normalization
// =============================================================================

#include "scl/binding/c_api/normalize.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/normalize.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Compute Row Sums
// =============================================================================

SCL_EXPORT scl_error_t scl_norm_compute_row_sums(
    scl_sparse_t matrix,
    scl_real_t* output) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(output, "Output pointer is null");

    SCL_C_API_TRY
        const Index n_rows = matrix->rows();
        const Size n_rows_sz = static_cast<Size>(n_rows);
        Array<Real> out_arr(reinterpret_cast<Real*>(output), n_rows_sz);

        matrix->visit([&](auto& m) {
            scl::kernel::normalize::compute_row_sums(m, out_arr);
        });

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Scale Primary Dimension
// =============================================================================

SCL_EXPORT scl_error_t scl_norm_scale_primary(
    scl_sparse_t matrix,
    const scl_real_t* scales) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(scales, "Scales pointer is null");

    SCL_C_API_TRY
        const Index primary_dim = matrix->is_csr_format() 
                                ? matrix->rows() 
                                : matrix->cols();
        const Size primary_dim_sz = static_cast<Size>(primary_dim);
        
        Array<const Real> scales_arr(reinterpret_cast<const Real*>(scales), primary_dim_sz);

        matrix->visit([&](auto& m) {
            scl::kernel::normalize::scale_primary(m, scales_arr);
        });

        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Primary Sums Masked
// =============================================================================

scl_error_t scl_norm_primary_sums_masked(
    scl_sparse_t matrix,
    const unsigned char* mask,
    scl_real_t* output)
{
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(mask, "Mask pointer is null");
    SCL_C_API_CHECK_NULL(output, "Output pointer is null");

    SCL_C_API_TRY {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        SCL_C_API_CHECK(wrapper->valid(), SCL_ERROR_INVALID_ARGUMENT,
                       "Invalid sparse matrix");

        const Index primary_dim = wrapper->is_csr_format() 
                                ? wrapper->rows() 
                                : wrapper->cols();
        const Index secondary_dim = wrapper->is_csr_format() 
                                   ? wrapper->cols() 
                                   : wrapper->rows();
        
        const Size primary_dim_sz = static_cast<Size>(primary_dim);
        const Size secondary_dim_sz = static_cast<Size>(secondary_dim);

        Array<const Byte> mask_arr(mask, secondary_dim_sz);
        Array<Real> out_arr(reinterpret_cast<Real*>(output), primary_dim_sz);

        matrix->visit([&](auto& m) {
            scl::kernel::normalize::primary_sums_masked(m, mask_arr, out_arr);
        });

        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

// =============================================================================
// Detect Highly Expressed
// =============================================================================

scl_error_t scl_norm_detect_highly_expressed(
    scl_sparse_t matrix,
    const scl_real_t* row_sums,
    scl_real_t max_fraction,
    unsigned char* out_mask)
{
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(row_sums, "Row sums pointer is null");
    SCL_C_API_CHECK_NULL(out_mask, "Output mask pointer is null");

    SCL_C_API_TRY {
        auto* wrapper = static_cast<SparseWrapper*>(matrix);
        
        SCL_C_API_CHECK(wrapper->valid(), SCL_ERROR_INVALID_ARGUMENT,
                       "Invalid sparse matrix");

        const Index n_rows = wrapper->rows();
        const Index n_cols = wrapper->cols();
        const Size n_rows_sz = static_cast<Size>(n_rows);
        const Size n_cols_sz = static_cast<Size>(n_cols);

        Array<const Real> sums_arr(reinterpret_cast<const Real*>(row_sums), n_rows_sz);
        Array<Byte> mask_arr(out_mask, n_cols_sz);

        matrix->visit([&](auto& m) {
            scl::kernel::normalize::detect_highly_expressed(m, sums_arr, static_cast<Real>(max_fraction), mask_arr);
        });

        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

} // extern "C"
