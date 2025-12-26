// =============================================================================
/// @file sparse.cpp
/// @brief Sparse Matrix Statistics (sparse.hpp)
///
/// Provides sparse matrix statistics computation functions.
// =============================================================================

#include "error.hpp"
#include "scl/kernel/core.hpp"

extern "C" {

// =============================================================================
// Sparse Matrix Statistics
// =============================================================================

int scl_primary_sums_csr(
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
        scl::kernel::sparse::primary_sums(
            matrix,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(rows))
        );
    )
}

int scl_primary_sums_csc(
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
        scl::kernel::sparse::primary_sums(
            matrix,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(cols))
        );
    )
}

int scl_primary_means_csr(
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
        scl::kernel::sparse::primary_means(
            matrix,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(rows))
        );
    )
}

int scl_primary_means_csc(
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
        scl::kernel::sparse::primary_means(
            matrix,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(cols))
        );
    )
}

int scl_primary_variances_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index /*cols*/,
    int ddof,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, 0
        );
        scl::kernel::sparse::primary_variances(
            matrix,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(rows)),
            ddof
        );
    )
}

int scl_primary_variances_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    int ddof,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::sparse::primary_variances(
            matrix,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(cols)),
            ddof
        );
    )
}

int scl_primary_nnz_counts_csr(
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            nullptr,
            nullptr,
            const_cast<scl::Index*>(indptr),
            rows, 0
        );
        scl::kernel::sparse::primary_nnz_counts(
            matrix,
            scl::Array<scl::Index>(output, static_cast<scl::Size>(rows))
        );
    )
}

int scl_primary_nnz_counts_csc(
    const scl::Index* indptr,
    scl::Index cols,
    scl::Index* output
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            nullptr,
            nullptr,
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::sparse::primary_nnz_counts(
            matrix,
            scl::Array<scl::Index>(output, static_cast<scl::Size>(cols))
        );
    )
}

} // extern "C"
