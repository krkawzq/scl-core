// =============================================================================
/// @file gram_correlation.cpp
/// @brief Gram Matrix and Pearson Correlation (gram.hpp, correlation.hpp)
///
/// Provides gram matrix computation and correlation analysis functions.
// =============================================================================

#include "error.hpp"
#include "scl/kernel/core.hpp"

extern "C" {

// =============================================================================
// Gram Matrix
// =============================================================================

int scl_gram_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::Size output_size = static_cast<scl::Size>(cols) * static_cast<scl::Size>(cols);
        scl::kernel::gram::gram(matrix, scl::Array<scl::Real>(output, output_size));
    )
}

int scl_gram_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index /*cols*/,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, 0
        );
        scl::Size output_size = static_cast<scl::Size>(rows) * static_cast<scl::Size>(rows);
        scl::kernel::gram::gram(matrix, scl::Array<scl::Real>(output, output_size));
    )
}

// =============================================================================
// Pearson Correlation
// =============================================================================

int scl_pearson_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    scl::Real* output,
    scl::Real* workspace_means,
    scl::Real* workspace_inv_stds
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::correlation::detail::compute_stats(
            matrix,
            scl::Array<scl::Real>(workspace_means, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(workspace_inv_stds, static_cast<scl::Size>(cols))
        );
        scl::Size output_size = static_cast<scl::Size>(cols) * static_cast<scl::Size>(cols);
        scl::kernel::gram::gram(matrix, scl::Array<scl::Real>(output, output_size));
        // Note: Full correlation implementation requires additional transformation
    )
}

int scl_pearson_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index /*cols*/,
    scl::Real* output,
    scl::Real* workspace_means,
    scl::Real* workspace_inv_stds
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, 0
        );
        scl::kernel::correlation::detail::compute_stats(
            matrix,
            scl::Array<scl::Real>(workspace_means, static_cast<scl::Size>(rows)),
            scl::Array<scl::Real>(workspace_inv_stds, static_cast<scl::Size>(rows))
        );
        scl::Size output_size = static_cast<scl::Size>(rows) * static_cast<scl::Size>(rows);
        scl::kernel::gram::gram(matrix, scl::Array<scl::Real>(output, output_size));
        // Note: Full correlation implementation requires additional transformation
    )
}

} // extern "C"
