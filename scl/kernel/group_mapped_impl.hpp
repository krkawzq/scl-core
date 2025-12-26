#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
/// @file group_mapped_impl.hpp
/// @brief Mapped Backend Group Aggregation
///
/// Group statistics are read-only operations, can stream directly from
/// mapped data. No materialization needed.
///
/// Supported operations:
/// - Group statistics: mean and variance per group per row
// =============================================================================

namespace scl::kernel::group::mapped {

namespace detail {
constexpr size_t PREFETCH_DISTANCE = 64;
}

// =============================================================================
// MappedCustomSparse Group Statistics
// =============================================================================

/// @brief Compute group statistics for MappedCustomSparse
///
/// Streaming algorithm - processes data in chunks for cache efficiency.
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void group_stats_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<const Size> group_sizes,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof,
    bool include_zeros
) {
    const Index n_primary = scl::primary_size(matrix);
    const Size total_size = static_cast<Size>(n_primary) * n_groups;

    SCL_CHECK_DIM(out_means.len == total_size, "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len == total_size, "Vars size mismatch");

    // Zero initialize
    for (Size i = 0; i < total_size; ++i) {
        out_means[i] = 0.0;
        out_vars[i] = 0.0;
    }

    // Prefetch hint
    kernel::mapped::hint_prefetch(matrix);

    // Process in chunks for cache efficiency
    constexpr Size CHUNK_SIZE = 256;
    const Size n_chunks = (n_primary + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Index chunk_start = static_cast<Index>(chunk_id * CHUNK_SIZE);
        Index chunk_end = std::min(chunk_start + static_cast<Index>(CHUNK_SIZE), n_primary);

        scl::threading::parallel_for(chunk_start, chunk_end, [&](Index p) {
            auto values = scl::primary_values(matrix, p);
            auto indices = scl::primary_indices(matrix, p);
            Index len = static_cast<Index>(values.len);

            Real* mean_ptr = out_means.ptr + (p * n_groups);
            Real* var_ptr = out_vars.ptr + (p * n_groups);

            if (len == 0) {
                for (Size g = 0; g < n_groups; ++g) {
                    mean_ptr[g] = 0.0;
                    var_ptr[g] = 0.0;
                }
                return;
            }

            const T* SCL_RESTRICT vals = values.ptr;
            const Index* SCL_RESTRICT inds = indices.ptr;

            // Accumulation with prefetch
            for (Index k = 0; k < len; ++k) {
                if (k + detail::PREFETCH_DISTANCE < len) {
                    Index future_idx = inds[k + detail::PREFETCH_DISTANCE];
                    SCL_PREFETCH_READ(&group_ids[future_idx], 0);
                }

                Index idx = inds[k];
                int32_t g = group_ids[idx];

                if (g >= 0 && static_cast<Size>(g) < n_groups) {
                    Real v = static_cast<Real>(vals[k]);
                    mean_ptr[g] += v;
                    var_ptr[g] += v * v;
                }
            }

            // Finalize
            for (Size g = 0; g < n_groups; ++g) {
                Real sum = mean_ptr[g];
                Real sum_sq = var_ptr[g];
                Real N = include_zeros ? static_cast<Real>(group_sizes[g]) : static_cast<Real>(len);

                if (N <= static_cast<Real>(ddof)) {
                    mean_ptr[g] = 0.0;
                    var_ptr[g] = 0.0;
                    continue;
                }

                Real mu = sum / N;
                Real variance = (sum_sq - N * mu * mu) / (N - static_cast<Real>(ddof));
                if (variance < 0.0) variance = 0.0;

                mean_ptr[g] = mu;
                var_ptr[g] = variance;
            }
        });
    }
}

/// @brief Compute group sums for MappedCustomSparse (simplified version)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void group_sums_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<Real> out_sums
) {
    const Index n_primary = scl::primary_size(matrix);
    const Size total_size = static_cast<Size>(n_primary) * n_groups;

    SCL_CHECK_DIM(out_sums.len == total_size, "Sums size mismatch");

    // Zero initialize
    for (Size i = 0; i < total_size; ++i) {
        out_sums[i] = 0.0;
    }

    kernel::mapped::hint_prefetch(matrix);

    constexpr Size CHUNK_SIZE = 256;
    const Size n_chunks = (n_primary + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Index chunk_start = static_cast<Index>(chunk_id * CHUNK_SIZE);
        Index chunk_end = std::min(chunk_start + static_cast<Index>(CHUNK_SIZE), n_primary);

        scl::threading::parallel_for(chunk_start, chunk_end, [&](Index p) {
            auto values = scl::primary_values(matrix, p);
            auto indices = scl::primary_indices(matrix, p);

            if (values.len == 0) return;

            Real* sum_ptr = out_sums.ptr + (p * n_groups);
            const T* SCL_RESTRICT vals = values.ptr;
            const Index* SCL_RESTRICT inds = indices.ptr;

            for (Size k = 0; k < values.len; ++k) {
                Index idx = inds[k];
                int32_t g = group_ids[idx];

                if (g >= 0 && static_cast<Size>(g) < n_groups) {
                    sum_ptr[g] += static_cast<Real>(vals[k]);
                }
            }
        });
    }
}

// =============================================================================
// MappedVirtualSparse Group Statistics
// =============================================================================

/// @brief Compute group statistics for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void group_stats_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<const Size> group_sizes,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof,
    bool include_zeros
) {
    const Index n_primary = scl::primary_size(matrix);
    const Size total_size = static_cast<Size>(n_primary) * n_groups;

    SCL_CHECK_DIM(out_means.len == total_size, "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len == total_size, "Vars size mismatch");

    // Zero initialize
    for (Size i = 0; i < total_size; ++i) {
        out_means[i] = 0.0;
        out_vars[i] = 0.0;
    }

    constexpr Size CHUNK_SIZE = 256;
    const Size n_chunks = (n_primary + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Index chunk_start = static_cast<Index>(chunk_id * CHUNK_SIZE);
        Index chunk_end = std::min(chunk_start + static_cast<Index>(CHUNK_SIZE), n_primary);

        scl::threading::parallel_for(chunk_start, chunk_end, [&](Index p) {
            auto values = scl::primary_values(matrix, p);
            auto indices = scl::primary_indices(matrix, p);
            Index len = static_cast<Index>(values.len);

            Real* mean_ptr = out_means.ptr + (p * n_groups);
            Real* var_ptr = out_vars.ptr + (p * n_groups);

            if (len == 0) {
                for (Size g = 0; g < n_groups; ++g) {
                    mean_ptr[g] = 0.0;
                    var_ptr[g] = 0.0;
                }
                return;
            }

            const T* SCL_RESTRICT vals = values.ptr;
            const Index* SCL_RESTRICT inds = indices.ptr;

            // Accumulation
            for (Index k = 0; k < len; ++k) {
                Index idx = inds[k];
                int32_t g = group_ids[idx];

                if (g >= 0 && static_cast<Size>(g) < n_groups) {
                    Real v = static_cast<Real>(vals[k]);
                    mean_ptr[g] += v;
                    var_ptr[g] += v * v;
                }
            }

            // Finalize
            for (Size g = 0; g < n_groups; ++g) {
                Real sum = mean_ptr[g];
                Real sum_sq = var_ptr[g];
                Real N = include_zeros ? static_cast<Real>(group_sizes[g]) : static_cast<Real>(len);

                if (N <= static_cast<Real>(ddof)) {
                    mean_ptr[g] = 0.0;
                    var_ptr[g] = 0.0;
                    continue;
                }

                Real mu = sum / N;
                Real variance = (sum_sq - N * mu * mu) / (N - static_cast<Real>(ddof));
                if (variance < 0.0) variance = 0.0;

                mean_ptr[g] = mu;
                var_ptr[g] = variance;
            }
        });
    }
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate mapped fast path
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void group_stats_mapped_dispatch(
    const MatrixT& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<const Size> group_sizes,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof,
    bool include_zeros
) {
    group_stats_mapped(matrix, group_ids, n_groups, group_sizes,
                       out_means, out_vars, ddof, include_zeros);
}

} // namespace scl::kernel::group::mapped
