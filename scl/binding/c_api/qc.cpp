// =============================================================================
// FILE: scl/binding/c_api/qc.cpp
// BRIEF: C API implementation for quality control metrics
// =============================================================================

#include "scl/binding/c_api/qc.h"
#include "scl/binding/c_api/core_types.h"
#include "scl/kernel/qc.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/type.hpp"
#include "scl/core/error.hpp"

#include <exception>

extern "C" {

static scl_error_t handle_exception() {
    try {
        throw;
    } catch (const scl::DimensionError&) {
        return SCL_ERROR_DIMENSION_MISMATCH;
    } catch (const scl::ValueError&) {
        return SCL_ERROR_INVALID_ARGUMENT;
    } catch (const scl::TypeError&) {
        return SCL_ERROR_TYPE;
    } catch (const scl::IOError&) {
        return SCL_ERROR_IO;
    } catch (const scl::NotImplementedError&) {
        return SCL_ERROR_NOT_IMPLEMENTED;
    } catch (const scl::InternalError&) {
        return SCL_ERROR_INTERNAL;
    } catch (const scl::Exception& e) {
        return static_cast<scl_error_t>(static_cast<int>(e.code()));
    } catch (...) {
        return SCL_ERROR_UNKNOWN;
    }
}

scl_error_t scl_qc_compute_basic(
    scl_sparse_matrix_t matrix,
    scl_index_t* out_n_genes,
    scl_real_t* out_total_counts,
    scl_size_t n_rows
) {
    if (!matrix || !out_n_genes || !out_total_counts) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<scl::Index> n_genes_arr(
            reinterpret_cast<scl::Index*>(out_n_genes),
            n_rows
        );
        scl::Array<scl::Real> counts_arr(
            reinterpret_cast<scl::Real*>(out_total_counts),
            n_rows
        );
        scl::kernel::qc::compute_basic_qc(*sparse, n_genes_arr, counts_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_qc_compute_subset_pct(
    scl_sparse_matrix_t matrix,
    const uint8_t* subset_mask,
    scl_real_t* out_pcts,
    scl_size_t n_rows,
    scl_size_t n_cols
) {
    if (!matrix || !subset_mask || !out_pcts) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<const scl::Byte> mask_arr(
            reinterpret_cast<const scl::Byte*>(subset_mask),
            n_cols
        );
        scl::Array<scl::Real> pcts_arr(
            reinterpret_cast<scl::Real*>(out_pcts),
            n_rows
        );
        scl::kernel::qc::compute_subset_pct(*sparse, mask_arr, pcts_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

scl_error_t scl_qc_compute_fused(
    scl_sparse_matrix_t matrix,
    const uint8_t* subset_mask,
    scl_index_t* out_n_genes,
    scl_real_t* out_total_counts,
    scl_real_t* out_pcts,
    scl_size_t n_rows,
    scl_size_t n_cols
) {
    if (!matrix || !subset_mask || !out_n_genes || !out_total_counts || !out_pcts) {
        return SCL_ERROR_INVALID_ARGUMENT;
    }
    
    try {
        auto* sparse = static_cast<scl::CSR*>(matrix);
        scl::Array<const scl::Byte> mask_arr(
            reinterpret_cast<const scl::Byte*>(subset_mask),
            n_cols
        );
        scl::Array<scl::Index> n_genes_arr(
            reinterpret_cast<scl::Index*>(out_n_genes),
            n_rows
        );
        scl::Array<scl::Real> counts_arr(
            reinterpret_cast<scl::Real*>(out_total_counts),
            n_rows
        );
        scl::Array<scl::Real> pcts_arr(
            reinterpret_cast<scl::Real*>(out_pcts),
            n_rows
        );
        scl::kernel::qc::compute_fused_qc(*sparse, mask_arr, n_genes_arr, counts_arr, pcts_arr);
        return SCL_ERROR_OK;
    } catch (...) {
        return handle_exception();
    }
}

// Explicit instantiation
template void scl::kernel::qc::compute_basic_qc<scl::Real, true>(
    const scl::CSR&,
    scl::Array<scl::Index>,
    scl::Array<scl::Real>
);

template void scl::kernel::qc::compute_subset_pct<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Byte>,
    scl::Array<scl::Real>
);

template void scl::kernel::qc::compute_fused_qc<scl::Real, true>(
    const scl::CSR&,
    scl::Array<const scl::Byte>,
    scl::Array<scl::Index>,
    scl::Array<scl::Real>,
    scl::Array<scl::Real>
);

} // extern "C"

