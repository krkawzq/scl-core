// =============================================================================
/// @file feature.cpp
/// @brief Feature Statistics (feature.hpp)
///
/// Provides feature-level statistical computations.
/// Includes both standard (CustomSparse) and mapped (MappedCustomSparse) versions.
// =============================================================================

#include "error.hpp"
#include "scl/kernel/core.hpp"
#include "scl/kernel/feature_mapped_impl.hpp"
#include "scl/io/mmatrix.hpp"

extern "C" {

// =============================================================================
// Feature Statistics - Standard
// =============================================================================

int scl_standard_moments_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Real* out_means,
    scl::Real* out_vars,
    int ddof
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::feature::standard_moments(
            matrix,
            scl::Array<scl::Real>(out_means, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_vars, static_cast<scl::Size>(cols)),
            ddof
        );
    )
}

int scl_clipped_moments_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    const scl::Real* clip_vals,
    scl::Real* out_means,
    scl::Real* out_vars
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::feature::clipped_moments(
            matrix,
            scl::Array<const scl::Real>(clip_vals, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_means, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_vars, static_cast<scl::Size>(cols))
        );
    )
}

int scl_detection_rate_csc(
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Real* out_rates
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            nullptr,
            nullptr,
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::feature::detection_rate(
            matrix,
            scl::Array<scl::Real>(out_rates, static_cast<scl::Size>(cols))
        );
    )
}

int scl_dispersion(
    const scl::Real* means,
    const scl::Real* vars,
    scl::Size size,
    scl::Real* out_dispersion
) {
    SCL_C_API_WRAPPER(
        scl::kernel::feature::dispersion(
            scl::Array<const scl::Real>(means, size),
            scl::Array<const scl::Real>(vars, size),
            scl::Array<scl::Real>(out_dispersion, size)
        );
    )
}

// =============================================================================
// Feature Statistics - Mapped
// =============================================================================

int scl_standard_moments_mapped_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* out_means,
    scl::Real* out_vars,
    int ddof
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
        
        scl::kernel::feature::mapped::standard_moments_mapped(
            matrix,
            scl::Array<scl::Real>(out_means, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_vars, static_cast<scl::Size>(cols)),
            ddof
        );
    )
}

int scl_clipped_moments_mapped_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    const scl::Real* clip_vals,
    scl::Real* out_means,
    scl::Real* out_vars
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
        
        scl::kernel::feature::mapped::clipped_moments_mapped(
            matrix,
            scl::Array<const scl::Real>(clip_vals, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_means, static_cast<scl::Size>(cols)),
            scl::Array<scl::Real>(out_vars, static_cast<scl::Size>(cols))
        );
    )
}

int scl_detection_rate_mapped_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    scl::Real* out_rates
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
        
        scl::kernel::feature::mapped::detection_rate_mapped(
            matrix,
            scl::Array<scl::Real>(out_rates, static_cast<scl::Size>(cols))
        );
    )
}

} // extern "C"
