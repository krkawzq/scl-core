// =============================================================================
/// @file qc_normalize.cpp
/// @brief Quality Control and Normalization Operations
///
/// Provides QC metrics computation and normalization functions.
// =============================================================================

#include "error.hpp"
#include "scl/kernel/core.hpp"

extern "C" {

// =============================================================================
// Quality Control Metrics (qc.hpp)
// =============================================================================

int scl_compute_basic_qc_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index /*cols*/,
    scl::Index* out_n_genes,
    scl::Real* out_total_counts
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, 0
        );
        scl::kernel::qc::compute_basic_qc(
            matrix,
            scl::Array<scl::Index>(out_n_genes, static_cast<scl::Size>(rows)),
            scl::Array<scl::Real>(out_total_counts, static_cast<scl::Size>(rows))
        );
    )
}

int scl_compute_basic_qc_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    scl::Index* out_n_cells,
    scl::Real* out_total_counts
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::qc::compute_basic_qc(
            matrix,
            scl::Array<scl::Index>(out_n_cells, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_total_counts, static_cast<scl::Size>(cols))
        );
    )
}

// =============================================================================
// Normalization Operations (normalize.hpp)
// =============================================================================

int scl_scale_primary_csr(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index /*cols*/,
    const scl::Real* scales
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, 0
        );
        scl::kernel::normalize::scale_primary(
            matrix,
            scl::Array<const scl::Real>(scales, static_cast<scl::Size>(rows))
        );
    )
}

int scl_scale_primary_csc(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    const scl::Real* scales
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::normalize::scale_primary(
            matrix,
            scl::Array<const scl::Real>(scales, static_cast<scl::Size>(cols))
        );
    )
}

} // extern "C"
