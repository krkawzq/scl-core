// =============================================================================
/// @file hvg_reorder.cpp
/// @brief HVG Selection and Reordering (hvg.hpp, reorder.hpp, resample.hpp)
///
/// Provides highly variable gene selection and matrix reordering functions.
// =============================================================================

#include "error.hpp"
#include "scl/kernel/core.hpp"

extern "C" {

// =============================================================================
// HVG Selection
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
    scl::Index /*rows*/,
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
// Reordering
// =============================================================================

int scl_align_secondary_csc(
    scl::Real* data,
    scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
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
// Resampling
// =============================================================================

int scl_downsample_counts_csc(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
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

} // extern "C"
