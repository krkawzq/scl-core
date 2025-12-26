// =============================================================================
/// @file group_scale.cpp
/// @brief Group Aggregations and Standardization (group.hpp, scale.hpp)
///
/// Provides group statistics computation and data standardization functions.
/// Includes both standard (CustomSparse) and mapped (MappedCustomSparse) versions.
// =============================================================================

#include "error.hpp"
#include "scl/kernel/core.hpp"
#include "scl/kernel/group_mapped_impl.hpp"
#include "scl/kernel/scale_mapped_impl.hpp"
#include "scl/io/mmatrix.hpp"

#include <cstring>

extern "C" {

// =============================================================================
// Group Aggregations - Standard
// =============================================================================

int scl_group_stats_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
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
            rows, cols
        );
        scl::Size output_size = static_cast<scl::Size>(cols) * n_groups;
        scl::kernel::group::group_stats(
            matrix,
            scl::Array<const int32_t>(group_ids, static_cast<scl::Size>(rows)),
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
// Standardization - Standard
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

// =============================================================================
// Group Aggregations - Mapped
// =============================================================================

int scl_group_stats_mapped_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    const int32_t* group_ids,
    scl::Size n_groups,
    const scl::Size* group_sizes,
    scl::Real* out_means,
    scl::Real* out_vars,
    int ddof,
    bool include_zeros
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
        
        scl::Size output_size = static_cast<scl::Size>(cols) * n_groups;
        scl::kernel::group::mapped::group_stats_mapped(
            matrix,
            scl::Array<const int32_t>(group_ids, static_cast<scl::Size>(rows)),
            n_groups,
            scl::Array<const scl::Size>(group_sizes, n_groups),
            scl::Array<scl::Real>(out_means, output_size),
            scl::Array<scl::Real>(out_vars, output_size),
            ddof,
            include_zeros
        );
    )
}

int scl_group_stats_mapped_csr(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    const int32_t* group_ids,
    scl::Size n_groups,
    const scl::Size* group_sizes,
    scl::Real* out_means,
    scl::Real* out_vars,
    int ddof,
    bool include_zeros
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
        
        scl::Size output_size = static_cast<scl::Size>(rows) * n_groups;
        scl::kernel::group::mapped::group_stats_mapped(
            matrix,
            scl::Array<const int32_t>(group_ids, static_cast<scl::Size>(cols)),
            n_groups,
            scl::Array<const scl::Size>(group_sizes, n_groups),
            scl::Array<scl::Real>(out_means, output_size),
            scl::Array<scl::Real>(out_vars, output_size),
            ddof,
            include_zeros
        );
    )
}

// =============================================================================
// Standardization - Mapped
// =============================================================================

int scl_standardize_mapped_csc(
    const scl::Real* data,
    const scl::Index* indices,
    const scl::Index* indptr,
    scl::Index rows,
    scl::Index cols,
    scl::Index nnz,
    const scl::Real* means,
    const scl::Real* stds,
    scl::Real max_value,
    bool zero_center,
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
        
        auto result = scl::kernel::scale::mapped::standardize_mapped_custom(
            matrix,
            scl::Array<const scl::Real>(means, static_cast<scl::Size>(cols)),
            scl::Array<const scl::Real>(stds, static_cast<scl::Size>(cols)),
            max_value,
            zero_center
        );
        
        std::memcpy(out_data, result.data.data(), result.data.size() * sizeof(scl::Real));
        std::memcpy(out_indices, result.indices.data(), result.indices.size() * sizeof(scl::Index));
        std::memcpy(out_indptr, result.indptr.data(), result.indptr.size() * sizeof(scl::Index));
    )
}

} // extern "C"
