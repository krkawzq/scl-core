// =============================================================================
/// @file qc_normalize.cpp
/// @brief Quality Control and Normalization Operations
///
/// Provides QC metrics computation and normalization functions.
/// Includes both standard (CustomSparse) and mapped (MappedCustomSparse) versions.
// =============================================================================

#include "error.hpp"
#include "scl/kernel/core.hpp"
#include "scl/kernel/qc_mapped_impl.hpp"
#include "scl/kernel/normalize_mapped_impl.hpp"
#include "scl/io/mmatrix.hpp"

#include <cstring>

extern "C" {

// =============================================================================
// Quality Control Metrics - Standard (qc.hpp)
// =============================================================================

int scl_compute_basic_qc_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index* out_n_genes,
    scl::Real* out_total_counts
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols
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
    scl::Index rows,
    scl::Index cols,
    scl::Index* out_n_cells,
    scl::Real* out_total_counts
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols
        );
        scl::kernel::qc::compute_basic_qc(
            matrix,
            scl::Array<scl::Index>(out_n_cells, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_total_counts, static_cast<scl::Size>(cols))
        );
    )
}

// =============================================================================
// Normalization Operations - Standard (normalize.hpp)
// =============================================================================

int scl_scale_primary_csr(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    const scl::Real* scales
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols
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
    scl::Index rows,
    scl::Index cols,
    const scl::Real* scales
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols
        );
        scl::kernel::normalize::scale_primary(
            matrix,
            scl::Array<const scl::Real>(scales, static_cast<scl::Size>(cols))
        );
    )
}

// =============================================================================
// Quality Control Metrics - Mapped
// =============================================================================

int scl_compute_basic_qc_mapped_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Index* out_n_genes,
    scl::Real* out_total_counts
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
        
        scl::kernel::qc::mapped::compute_basic_qc_mapped(
            matrix,
            scl::Array<scl::Index>(out_n_genes, static_cast<scl::Size>(rows)),
            scl::Array<scl::Real>(out_total_counts, static_cast<scl::Size>(rows))
        );
    )
}

int scl_compute_basic_qc_mapped_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Index* out_n_cells,
    scl::Real* out_total_counts
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
        
        scl::kernel::qc::mapped::compute_basic_qc_mapped(
            matrix,
            scl::Array<scl::Index>(out_n_cells, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_total_counts, static_cast<scl::Size>(cols))
        );
    )
}

// =============================================================================
// Normalization Operations - Mapped
// =============================================================================

int scl_compute_row_sums_mapped_csr(
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
        
        scl::kernel::normalize::mapped::compute_row_sums_mapped(
            matrix,
            scl::Array<scl::Real>(output, static_cast<scl::Size>(rows))
        );
    )
}

int scl_scale_rows_mapped_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    const scl::Real* scales,
    scl::Real* out_data,
    scl::Index* out_indices,
    scl::Index* out_indptr
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
        
        auto result = scl::kernel::normalize::mapped::scale_rows_mapped(
            matrix,
            scl::Array<const scl::Real>(scales, static_cast<scl::Size>(rows))
        );
        
        std::memcpy(out_data, result.data.data(), result.data.size() * sizeof(scl::Real));
        std::memcpy(out_indices, result.indices.data(), result.indices.size() * sizeof(scl::Index));
        std::memcpy(out_indptr, result.indptr.data(), result.indptr.size() * sizeof(scl::Index));
    )
}

int scl_normalize_rows_mapped_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real target_sum,
    scl::Real* out_data,
    scl::Index* out_indices,
    scl::Index* out_indptr
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
        
        auto result = scl::kernel::normalize::mapped::normalize_rows_mapped(
            matrix,
            target_sum
        );
        
        std::memcpy(out_data, result.data.data(), result.data.size() * sizeof(scl::Real));
        std::memcpy(out_indices, result.indices.data(), result.indices.size() * sizeof(scl::Index));
        std::memcpy(out_indptr, result.indptr.data(), result.indptr.size() * sizeof(scl::Index));
    )
}

} // extern "C"
