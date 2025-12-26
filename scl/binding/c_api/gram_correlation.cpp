// =============================================================================
/// @file gram_correlation.cpp
/// @brief Gram Matrix and Pearson Correlation (gram.hpp, correlation.hpp)
///
/// Provides gram matrix computation and correlation analysis functions.
/// Includes both standard (CustomSparse) and mapped (MappedCustomSparse) versions.
// =============================================================================

#include "scl/binding/c_api/error.hpp"
#include "scl/core/error.hpp"
#include "scl/kernel/core.hpp"
#include "scl/io/mmatrix.hpp"

#include <vector>

extern "C" {

// =============================================================================
// Gram Matrix - Standard
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
// Pearson Correlation - Standard
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
        scl::kernel::correlation::compute_stats(
            matrix,
            scl::Array<scl::Real>(workspace_means, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(workspace_inv_stds, static_cast<scl::Size>(cols))
        );
        scl::Size output_size = static_cast<scl::Size>(cols) * static_cast<scl::Size>(cols);
        scl::kernel::gram::gram(matrix, scl::Array<scl::Real>(output, output_size));
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
        scl::kernel::correlation::compute_stats(
            matrix,
            scl::Array<scl::Real>(workspace_means, static_cast<scl::Size>(rows)),
            scl::Array<scl::Real>(workspace_inv_stds, static_cast<scl::Size>(rows))
        );
        scl::Size output_size = static_cast<scl::Size>(rows) * static_cast<scl::Size>(rows);
        scl::kernel::gram::gram(matrix, scl::Array<scl::Real>(output, output_size));
    )
}

// =============================================================================
// Gram Matrix - Mapped
// =============================================================================

int scl_gram_mapped_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        auto mapped_data = scl::io::MappedArray<scl::Real>::from_ptr(
            const_cast<scl::Real*>(data), static_cast<scl::Size>(nnz));
        auto mapped_indices = scl::io::MappedArray<scl::Index>::from_ptr(
            const_cast<scl::Index*>(indices), static_cast<scl::Size>(nnz));
        auto mapped_indptr = scl::io::MappedArray<scl::Index>::from_ptr(
            const_cast<scl::Index*>(indptr), static_cast<scl::Size>(cols + 1));
        
        scl::io::MappedCustomSparse<scl::Real, false> matrix(
            std::move(mapped_data),
            std::move(mapped_indices),
            std::move(mapped_indptr),
            rows, cols
        );
        
        scl::Size output_size = static_cast<scl::Size>(cols) * static_cast<scl::Size>(cols);
        scl::kernel::gram::mapped::gram_mapped(
            matrix, 
            scl::Array<scl::Real>(output, output_size)
        );
    )
}

int scl_gram_mapped_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        auto mapped_data = scl::io::MappedArray<scl::Real>::from_ptr(
            const_cast<scl::Real*>(data), static_cast<scl::Size>(nnz));
        auto mapped_indices = scl::io::MappedArray<scl::Index>::from_ptr(
            const_cast<scl::Index*>(indices), static_cast<scl::Size>(nnz));
        auto mapped_indptr = scl::io::MappedArray<scl::Index>::from_ptr(
            const_cast<scl::Index*>(indptr), static_cast<scl::Size>(rows + 1));
        
        scl::io::MappedCustomSparse<scl::Real, true> matrix(
            std::move(mapped_data),
            std::move(mapped_indices),
            std::move(mapped_indptr),
            rows, cols
        );
        
        scl::Size output_size = static_cast<scl::Size>(rows) * static_cast<scl::Size>(rows);
        scl::kernel::gram::mapped::gram_mapped(
            matrix, 
            scl::Array<scl::Real>(output, output_size)
        );
    )
}

// =============================================================================
// Pearson Correlation - Mapped
// =============================================================================

int scl_pearson_mapped_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        auto mapped_data = scl::io::MappedArray<scl::Real>::from_ptr(
            const_cast<scl::Real*>(data), static_cast<scl::Size>(nnz));
        auto mapped_indices = scl::io::MappedArray<scl::Index>::from_ptr(
            const_cast<scl::Index*>(indices), static_cast<scl::Size>(nnz));
        auto mapped_indptr = scl::io::MappedArray<scl::Index>::from_ptr(
            const_cast<scl::Index*>(indptr), static_cast<scl::Size>(cols + 1));
        
        scl::io::MappedCustomSparse<scl::Real, false> matrix(
            std::move(mapped_data),
            std::move(mapped_indices),
            std::move(mapped_indptr),
            rows, cols
        );
        
        scl::Size output_size = static_cast<scl::Size>(cols) * static_cast<scl::Size>(cols);
        scl::Size n = static_cast<scl::Size>(cols);
        
        std::vector<scl::Real> means(n);
        std::vector<scl::Real> inv_stds(n);
        
        scl::kernel::correlation::mapped::compute_stats_mapped_dispatch<scl::io::MappedCustomSparse<scl::Real, false>, false>(
            matrix,
            scl::Array<scl::Real>(means.data(), n),
            scl::Array<scl::Real>(inv_stds.data(), n)
        );
        
        scl::kernel::correlation::mapped::pearson_mapped_dispatch<scl::io::MappedCustomSparse<scl::Real, false>, false>(
            matrix,
            scl::Array<const scl::Real>(means.data(), n),
            scl::Array<const scl::Real>(inv_stds.data(), n),
            scl::Array<scl::Real>(output, output_size)
        );
    )
}

int scl_pearson_mapped_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* output
) {
    SCL_C_API_WRAPPER(
        auto mapped_data = scl::io::MappedArray<scl::Real>::from_ptr(
            const_cast<scl::Real*>(data), static_cast<scl::Size>(nnz));
        auto mapped_indices = scl::io::MappedArray<scl::Index>::from_ptr(
            const_cast<scl::Index*>(indices), static_cast<scl::Size>(nnz));
        auto mapped_indptr = scl::io::MappedArray<scl::Index>::from_ptr(
            const_cast<scl::Index*>(indptr), static_cast<scl::Size>(rows + 1));
        
        scl::io::MappedCustomSparse<scl::Real, true> matrix(
            std::move(mapped_data),
            std::move(mapped_indices),
            std::move(mapped_indptr),
            rows, cols
        );
        
        scl::Size output_size = static_cast<scl::Size>(rows) * static_cast<scl::Size>(rows);
        scl::Size n = static_cast<scl::Size>(rows);
        
        std::vector<scl::Real> means(n);
        std::vector<scl::Real> inv_stds(n);
        
        scl::kernel::correlation::mapped::compute_stats_mapped_dispatch<scl::io::MappedCustomSparse<scl::Real, true>, true>(
            matrix,
            scl::Array<scl::Real>(means.data(), n),
            scl::Array<scl::Real>(inv_stds.data(), n)
        );
        
        scl::kernel::correlation::mapped::pearson_mapped_dispatch<scl::io::MappedCustomSparse<scl::Real, true>, true>(
            matrix,
            scl::Array<const scl::Real>(means.data(), n),
            scl::Array<const scl::Real>(inv_stds.data(), n),
            scl::Array<scl::Real>(output, output_size)
        );
    )
}

} // extern "C"
