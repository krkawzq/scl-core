// =============================================================================
/// @file sparse.cpp
/// @brief Sparse Matrix Statistics (sparse.hpp)
///
/// Provides sparse matrix statistics computation functions.
/// Includes both standard (CustomSparse) and mapped (MappedCustomSparse) versions.
// =============================================================================

#include "scl/binding/c_api/error.hpp"
#include "scl/core/error.hpp"
#include "scl/kernel/core.hpp"
#include "scl/kernel/sparse_mapped_impl.hpp"
#include "scl/io/mmatrix.hpp"

extern "C" {

// =============================================================================
// Standard Sparse Matrix Statistics (CustomSparse)
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

// =============================================================================
// Mapped Sparse Matrix Statistics (MappedCustomSparse)
// =============================================================================

int scl_primary_sums_mapped_csr(
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
        
        scl::kernel::sparse::mapped::primary_sums_mapped(
            matrix,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(rows))
        );
    )
}

int scl_primary_sums_mapped_csc(
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
        
        scl::kernel::sparse::mapped::primary_sums_mapped(
            matrix,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(cols))
        );
    )
}

int scl_primary_means_mapped_csr(
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
        
        scl::kernel::sparse::mapped::primary_means_mapped(
            matrix,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(rows))
        );
    )
}

int scl_primary_means_mapped_csc(
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
        
        scl::kernel::sparse::mapped::primary_means_mapped(
            matrix,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(cols))
        );
    )
}

int scl_primary_variances_mapped_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    int ddof,
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
        
        scl::kernel::sparse::mapped::primary_variances_mapped(
            matrix,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(rows)),
            ddof
        );
    )
}

int scl_primary_variances_mapped_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    int ddof,
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
        
        scl::kernel::sparse::mapped::primary_variances_mapped(
            matrix,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(cols)),
            ddof
        );
    )
}

int scl_primary_nnz_counts_mapped_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Index* output
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
        
        scl::kernel::sparse::mapped::primary_nnz_mapped(
            matrix,
            scl::Array<scl::Index>(output, static_cast<scl::Size>(rows))
        );
    )
}

int scl_primary_nnz_counts_mapped_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Index* output
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
        
        scl::kernel::sparse::mapped::primary_nnz_mapped(
            matrix,
            scl::Array<scl::Index>(output, static_cast<scl::Size>(cols))
        );
    )
}

} // extern "C"
