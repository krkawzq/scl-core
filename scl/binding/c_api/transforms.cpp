// =============================================================================
/// @file transforms.cpp
/// @brief Log Transforms and Mathematical Transformations (log1p.hpp)
///
/// Provides logarithmic and exponential transformation functions.
/// Includes both standard (CustomSparse) and mapped (MappedCustomSparse) versions.
// =============================================================================

#include "error.hpp"
#include "scl/kernel/core.hpp"
#include "scl/kernel/log1p_mapped_impl.hpp"
#include "scl/io/mmatrix.hpp"

#include <cstring>

extern "C" {

// =============================================================================
// Log Transforms - Standard (In-Place)
// =============================================================================

int scl_log1p_inplace_array(scl::Real* data, scl::Size size) {
    SCL_C_API_WRAPPER(
        scl::kernel::log1p_inplace(scl::Array<scl::Real>(data, size));
    )
}

int scl_log1p_inplace_csr(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSR matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            rows, cols
        );
        scl::kernel::log1p_inplace(matrix);
    )
}

int scl_log2p1_inplace_array(scl::Real* data, scl::Size size) {
    SCL_C_API_WRAPPER(
        scl::kernel::log2p1_inplace(scl::Array<scl::Real>(data, size));
    )
}

int scl_expm1_inplace_array(scl::Real* data, scl::Size size) {
    SCL_C_API_WRAPPER(
        scl::kernel::expm1_inplace(scl::Array<scl::Real>(data, size));
    )
}

// =============================================================================
// Log Transforms - Mapped (Out-of-Place)
// =============================================================================

int scl_log1p_mapped_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
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
        
        auto result = scl::kernel::log1p::mapped::log1p_mapped(matrix);
        
        std::memcpy(out_data, result.data.data(), result.data.size() * sizeof(scl::Real));
        std::memcpy(out_indices, result.indices.data(), result.indices.size() * sizeof(scl::Index));
        std::memcpy(out_indptr, result.indptr.data(), result.indptr.size() * sizeof(scl::Index));
    )
}

int scl_log1p_mapped_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
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
            const_cast<scl::Index*>(indptr), static_cast<scl::Size>(cols + 1));
        
        scl::io::MappedCustomSparse<scl::Real, false> matrix(
            std::move(mapped_data),
            std::move(mapped_indices),
            std::move(mapped_indptr),
            rows, cols
        );
        
        auto result = scl::kernel::log1p::mapped::log1p_mapped(matrix);
        
        std::memcpy(out_data, result.data.data(), result.data.size() * sizeof(scl::Real));
        std::memcpy(out_indices, result.indices.data(), result.indices.size() * sizeof(scl::Index));
        std::memcpy(out_indptr, result.indptr.data(), result.indptr.size() * sizeof(scl::Index));
    )
}

int scl_log2p1_mapped_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
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
        
        auto result = scl::kernel::log1p::mapped::log2p1_mapped(matrix);
        
        std::memcpy(out_data, result.data.data(), result.data.size() * sizeof(scl::Real));
        std::memcpy(out_indices, result.indices.data(), result.indices.size() * sizeof(scl::Index));
        std::memcpy(out_indptr, result.indptr.data(), result.indptr.size() * sizeof(scl::Index));
    )
}

int scl_expm1_mapped_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
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
        
        auto result = scl::kernel::log1p::mapped::expm1_mapped(matrix);
        
        std::memcpy(out_data, result.data.data(), result.data.size() * sizeof(scl::Real));
        std::memcpy(out_indices, result.indices.data(), result.indices.size() * sizeof(scl::Index));
        std::memcpy(out_indptr, result.indptr.data(), result.indptr.size() * sizeof(scl::Index));
    )
}

} // extern "C"
