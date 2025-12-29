// =============================================================================
// FILE: scl/binding/c_api/qc.cpp
// BRIEF: C API implementation for quality control metrics
// =============================================================================

#include "scl/binding/c_api/qc.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/qc.hpp"
#include "scl/core/type.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

// =============================================================================
// Compute Basic QC Metrics
// =============================================================================

SCL_EXPORT scl_error_t scl_qc_compute_basic(
    scl_sparse_t matrix,
    scl_index_t* out_n_genes,
    scl_real_t* out_total_counts,
    const scl_index_t primary_dim) {
    
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(out_n_genes, "Output n_genes pointer is null");
    SCL_C_API_CHECK_NULL(out_total_counts, "Output total_counts pointer is null");
    SCL_C_API_CHECK(primary_dim > 0, SCL_ERROR_INVALID_ARGUMENT,
                   "Primary dimension must be positive");

    SCL_C_API_TRY
        const Index actual_primary_dim = matrix->is_csr_format() 
                                       ? matrix->rows() 
                                       : matrix->cols();
        
        SCL_C_API_CHECK(static_cast<scl_index_t>(actual_primary_dim) == primary_dim,
                       SCL_ERROR_DIMENSION_MISMATCH,
                       "Primary dimension mismatch");

        const Size primary_dim_sz = static_cast<Size>(primary_dim);
        Array<Index> n_genes_arr(reinterpret_cast<Index*>(out_n_genes), primary_dim_sz);
        Array<Real> counts_arr(reinterpret_cast<Real*>(out_total_counts), primary_dim_sz);
        
        matrix->visit([&](auto& m) {
            scl::kernel::qc::compute_basic_qc(m, n_genes_arr, counts_arr);
        });
        
        SCL_C_API_RETURN_OK;
    SCL_C_API_CATCH
}

// =============================================================================
// Compute Subset Percentage
// =============================================================================

SCL_EXPORT scl_error_t scl_qc_compute_subset_pct(
    scl_sparse_t matrix,
    const uint8_t* subset_mask,
    scl_real_t* out_pcts,
    const scl_index_t primary_dim) {
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(subset_mask, "Subset mask pointer is null");
    SCL_C_API_CHECK_NULL(out_pcts, "Output percentages pointer is null");

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
        
        Array<const uint8_t> mask_arr(subset_mask, secondary_dim_sz);
        Array<Real> pcts_arr(reinterpret_cast<Real*>(out_pcts), primary_dim_sz);
        
        matrix->visit([&](auto& m) {
            scl::kernel::qc::compute_subset_pct(m, mask_arr, pcts_arr);
        });
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

// =============================================================================
// Compute Fused QC Metrics
// =============================================================================

scl_error_t scl_qc_compute_fused(
    scl_sparse_t matrix,
    const uint8_t* subset_mask,
    scl_index_t* out_n_genes,
    scl_real_t* out_total_counts,
    scl_real_t* out_pcts,
    scl_index_t primary_dim)
{
    SCL_C_API_CHECK_NULL(matrix, "Matrix handle is null");
    SCL_C_API_CHECK_NULL(subset_mask, "Subset mask pointer is null");
    SCL_C_API_CHECK_NULL(out_n_genes, "Output n_genes pointer is null");
    SCL_C_API_CHECK_NULL(out_total_counts, "Output total_counts pointer is null");
    SCL_C_API_CHECK_NULL(out_pcts, "Output percentages pointer is null");

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
        
        Array<const uint8_t> mask_arr(subset_mask, secondary_dim_sz);
        Array<Index> n_genes_arr(reinterpret_cast<Index*>(out_n_genes), primary_dim_sz);
        Array<Real> counts_arr(reinterpret_cast<Real*>(out_total_counts), primary_dim_sz);
        Array<Real> pcts_arr(reinterpret_cast<Real*>(out_pcts), primary_dim_sz);
        
        wrapper->visit([&](auto& m) {
            scl::kernel::qc::compute_fused_qc(m, mask_arr, n_genes_arr, counts_arr, pcts_arr);
        });
        
        SCL_C_API_RETURN_OK;
    }
    SCL_C_API_CATCH
}

} // extern "C"
