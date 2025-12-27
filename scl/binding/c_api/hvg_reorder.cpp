// =============================================================================
/// @file hvg_reorder.cpp
/// @brief HVG Selection and Reordering (hvg.hpp, reorder.hpp, resample.hpp)
///
/// Provides highly variable gene selection and matrix reordering functions.
/// Includes both standard (CustomSparse) and mapped (MappedCustomSparse) versions.
// =============================================================================

#include "error.hpp"
#include "scl/kernel/core.hpp"
#include "scl/kernel/hvg_mapped_impl.hpp"
#include "scl/kernel/reorder_mapped_impl.hpp"
#include "scl/kernel/resample_mapped_impl.hpp"
#include "scl/io/mmatrix.hpp"

#include <cstring>

extern "C" {

// =============================================================================
// HVG Selection - Standard
// =============================================================================

int scl_hvg_by_dispersion_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index cols,
    scl::Size n_top,
    scl::Index* out_indices,
    uint8_t* out_mask,
    scl::Real* out_dispersions
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::hvg::select_by_dispersion(
            matrix,
            n_top,
            scl::Array<scl::Index>(out_indices, n_top),
            scl::Array<uint8_t>(out_mask, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_dispersions, static_cast<scl::Size>(cols))
        );
    )
}

int scl_hvg_by_variance_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Size n_top,
    scl::Index* out_indices,
    uint8_t* out_mask
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::hvg::select_by_variance(
            matrix,
            n_top,
            scl::Array<scl::Index>(out_indices, n_top),
            scl::Array<uint8_t>(out_mask, static_cast<scl::Size>(cols))
        );
    )
}

// =============================================================================
// Reordering - Standard
// =============================================================================

int scl_align_secondary_csc(
    scl::Real* data,
    scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    const scl::Index* index_map,
    scl::Index* out_lengths,
    scl::Index new_cols
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            data,
            indices,
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::reorder::align_secondary(
            matrix,
            scl::Array<const scl::Index>(index_map, static_cast<scl::Size>(cols)),
            scl::Array<scl::Index>(out_lengths, static_cast<scl::Size>(cols)),
            new_cols
        );
    )
}

// =============================================================================
// Resampling - Standard
// =============================================================================

int scl_downsample_counts_csc(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Real target_sum,
    uint64_t seed
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::resample::downsample_counts(matrix, target_sum, seed);
    )
}

// =============================================================================
// HVG Selection - Mapped
// =============================================================================

int scl_hvg_by_dispersion_mapped_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Size n_top,
    scl::Index* out_indices,
    uint8_t* out_mask,
    scl::Real* out_dispersions
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
        
        scl::kernel::hvg::mapped::select_by_dispersion_mapped(
            matrix,
            n_top,
            scl::Array<scl::Index>(out_indices, n_top),
            scl::Array<uint8_t>(out_mask, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_dispersions, static_cast<scl::Size>(cols))
        );
    )
}

int scl_hvg_by_variance_mapped_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Size n_top,
    scl::Index* out_indices,
    uint8_t* out_mask
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
        
        scl::kernel::hvg::mapped::select_by_variance_mapped(
            matrix,
            n_top,
            scl::Array<scl::Index>(out_indices, n_top),
            scl::Array<uint8_t>(out_mask, static_cast<scl::Size>(cols))
        );
    )
}

// =============================================================================
// Resampling - Mapped
// =============================================================================

int scl_downsample_counts_mapped_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real target_sum,
    uint64_t seed,
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
        
        auto result = scl::kernel::resample::mapped::downsample_mapped(
            matrix,
            target_sum,
            seed
        );
        
        std::memcpy(out_data, result.data.data(), result.data.size() * sizeof(scl::Real));
        std::memcpy(out_indices, result.indices.data(), result.indices.size() * sizeof(scl::Index));
        std::memcpy(out_indptr, result.indptr.data(), result.indptr.size() * sizeof(scl::Index));
    )
}

int scl_binomial_resample_mapped_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real p,
    uint64_t seed,
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
        
        auto result = scl::kernel::resample::mapped::binomial_resample_mapped(
            matrix,
            p,
            seed
        );
        
        std::memcpy(out_data, result.data.data(), result.data.size() * sizeof(scl::Real));
        std::memcpy(out_indices, result.indices.data(), result.indices.size() * sizeof(scl::Index));
        std::memcpy(out_indptr, result.indptr.data(), result.indptr.size() * sizeof(scl::Index));
    )
}

} // extern "C"
