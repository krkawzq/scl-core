// =============================================================================
/// @file group_scale.cpp
/// @brief Group Aggregations and Standardization (group.hpp, scale.hpp)
///
/// Provides group statistics computation and data standardization functions.
// =============================================================================

#include "error.hpp"
#include "scl/kernel/core.hpp"

extern "C" {

// =============================================================================
// Group Aggregations
// =============================================================================

int scl_group_stats_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    const int32_t* group_ids,
    scl::Size n_groups,
    const scl::Size* group_sizes,
    scl::Real* out_means,
    scl::Real* out_vars,
    int ddof,
    bool include_zeros
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            const_cast<scl::Real*>(data),
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::Size output_size = static_cast<scl::Size>(cols) * n_groups;
        scl::kernel::group::group_stats(
            matrix,
            scl::Array<const int32_t>(group_ids, static_cast<scl::Size>(0)), // rows=0
            n_groups,
            scl::Array<const scl::Size>(group_sizes, n_groups),
            scl::Array<scl::Real>(out_means, output_size),
            scl::Array<scl::Real>(out_vars, output_size),
            ddof,
            include_zeros
        );
    )
}

int scl_count_group_sizes(
    const int32_t* group_ids,
    scl::Size n_elements,
    scl::Size n_groups,
    scl::Size* out_sizes
) {
    SCL_C_API_WRAPPER(
        scl::kernel::group::count_group_sizes(
            scl::Array<const int32_t>(group_ids, n_elements),
            n_groups,
            scl::Array<scl::Size>(out_sizes, n_groups)
        );
    )
}

// =============================================================================
// Standardization
// =============================================================================

int scl_standardize_csc(
    scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index /*rows*/,
    scl::Index cols,
    const scl::Real* means,
    const scl::Real* stds,
    scl::Real max_value,
    bool zero_center
) {
    SCL_C_API_WRAPPER(
        scl::CustomCSC matrix(
            data,
            const_cast<scl::Index*>(indices),
            const_cast<scl::Index*>(indptr),
            0, cols
        );
        scl::kernel::scale::standardize(
            matrix,
            scl::Array<const scl::Real>(means, static_cast<scl::Size>(cols)),
            scl::Array<const scl::Real>(stds, static_cast<scl::Size>(cols)),
            max_value,
            zero_center
        );
    )
}

} // extern "C"
