// =============================================================================
// FILE: scl/binding/c_api/qc/qc.cpp
// BRIEF: C API implementation for quality control metrics
// =============================================================================

#include "scl/binding/c_api/qc/qc.h"
#include "scl/binding/c_api/core/internal.hpp"
#include "scl/kernel/qc.hpp"

using namespace scl;
using namespace scl::binding;

extern "C" {

scl_error_t scl_qc_compute_basic(
    scl_sparse_t matrix,
    scl_index_t* out_n_genes,
    scl_real_t* out_total_counts,
    scl_index_t primary_dim)
{
    if (!matrix || !out_n_genes || !out_total_counts) {
        set_last_error(SCL_ERROR_NULL_POINTER, "Null pointer argument");
        return SCL_ERROR_NULL_POINTER;
    }
    
    try {
        if (!matrix->valid()) {
            set_last_error(SCL_ERROR_INVALID_ARGUMENT, "Invalid sparse matrix");
            return SCL_ERROR_INVALID_ARGUMENT;
        }
        
        Index actual_primary_dim = matrix->is_csr ? matrix->rows() : matrix->cols();
        if (static_cast<scl_index_t>(actual_primary_dim) != primary_dim) {
            set_last_error(SCL_ERROR_DIMENSION_MISMATCH, "Primary dimension mismatch");
            return SCL_ERROR_DIMENSION_MISMATCH;
        }
        
        Array<Index> out_n_genes_arr(
            reinterpret_cast<Index*>(out_n_genes),
            static_cast<Size>(primary_dim)
        );
        Array<Real> out_total_counts_arr(
            reinterpret_cast<Real*>(out_total_counts),
            static_cast<Size>(primary_dim)
        );
        
        matrix->visit([&](auto& m) {
            scl::kernel::qc::compute_basic_qc(
                m,
                out_n_genes_arr,
                out_total_counts_arr
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_qc_compute_subset_pct(
    scl_sparse_t matrix,
    const uint8_t* subset_mask,
    scl_real_t* out_pcts,
    scl_index_t primary_dim)
{
    if (!matrix || !subset_mask || !out_pcts) {
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
        
        Array<const uint8_t> subset_mask_arr(
            subset_mask,
            static_cast<Size>(secondary_dim)
        );
        Array<Real> out_pcts_arr(
            reinterpret_cast<Real*>(out_pcts),
            static_cast<Size>(primary_dim)
        );
        
        matrix->visit([&](auto& m) {
            scl::kernel::qc::compute_subset_pct(
                m,
                subset_mask_arr,
                out_pcts_arr
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_qc_compute_fused(
    scl_sparse_t matrix,
    const uint8_t* subset_mask,
    scl_index_t* out_n_genes,
    scl_real_t* out_total_counts,
    scl_real_t* out_pcts,
    scl_index_t primary_dim)
{
    if (!matrix || !subset_mask || !out_n_genes || !out_total_counts || !out_pcts) {
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
        
        Array<const uint8_t> subset_mask_arr(
            subset_mask,
            static_cast<Size>(secondary_dim)
        );
        Array<Index> out_n_genes_arr(
            reinterpret_cast<Index*>(out_n_genes),
            static_cast<Size>(primary_dim)
        );
        Array<Real> out_total_counts_arr(
            reinterpret_cast<Real*>(out_total_counts),
            static_cast<Size>(primary_dim)
        );
        Array<Real> out_pcts_arr(
            reinterpret_cast<Real*>(out_pcts),
            static_cast<Size>(primary_dim)
        );
        
        matrix->visit([&](auto& m) {
            scl::kernel::qc::compute_fused_qc(
                m,
                subset_mask_arr,
                out_n_genes_arr,
                out_total_counts_arr,
                out_pcts_arr
            );
        });
        
        clear_last_error();
        return SCL_OK;
    } catch (...) {
        return handle_exception();
    }
}

} // extern "C"

