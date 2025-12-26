#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cstring>
#include <vector>

// =============================================================================
/// @file group_mapped_impl.hpp
/// @brief Group Aggregation for Memory-Mapped Sparse Matrices
///
/// ## Key Optimizations
///
/// 1. Streaming Access
///    - Prefetch hints for OS page cache
///    - Sequential primary dimension traversal
///
/// 2. SIMD Zero Initialization
///    - Vectorized memset for output buffers
///
/// 3. Thread-Local Counting
///    - Avoid recomputing group counts during finalization
///    - Stack-allocated for small n_groups (<=256)
///
/// 4. 4-Way Unrolled Accumulation
///    - Reduce loop overhead for dense rows
///
/// ## Output Layout
///
/// Primary-Dimension-Major-Group order:
/// Index = primary_idx * n_groups + group_idx
// =============================================================================

namespace scl::kernel::group::mapped {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 64;
    constexpr Size SIMD_ZERO_THRESHOLD = 64;
    constexpr Size STACK_GROUP_LIMIT = 256;
}

// =============================================================================
// SECTION 2: Utilities
// =============================================================================

namespace detail {

/// @brief SIMD zero initialization
template <typename T>
SCL_FORCE_INLINE void zero_fill_simd(T* ptr, Size len) {
    if (len < config::SIMD_ZERO_THRESHOLD) {
        std::memset(ptr, 0, len * sizeof(T));
        return;
    }

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto zero = s::Zero(d);
    Size k = 0;

    for (; k + lanes <= len; k += lanes) {
        s::Store(zero, d, ptr + k);
    }

    for (; k < len; ++k) {
        ptr[k] = T(0);
    }
}

/// @brief Finalize mean and variance
SCL_FORCE_INLINE void finalize_stats(
    Real* SCL_RESTRICT mean_ptr,
    Real* SCL_RESTRICT var_ptr,
    const Size* SCL_RESTRICT group_sizes,
    const Size* SCL_RESTRICT nnz_counts,
    Size n_groups,
    int ddof,
    bool include_zeros
) {
    for (Size g = 0; g < n_groups; ++g) {
        Real sum = mean_ptr[g];
        Real sum_sq = var_ptr[g];
        Real N = include_zeros ? static_cast<Real>(group_sizes[g])
                               : static_cast<Real>(nnz_counts[g]);

        if (N <= static_cast<Real>(ddof)) {
            mean_ptr[g] = Real(0);
            var_ptr[g] = Real(0);
            continue;
        }

        Real mu = sum / N;
        Real variance = (sum_sq - N * mu * mu) / (N - static_cast<Real>(ddof));
        if (variance < Real(0)) variance = Real(0);

        mean_ptr[g] = mu;
        var_ptr[g] = variance;
    }
}

} // namespace detail

// =============================================================================
// SECTION 3: MappedCustomSparse Group Statistics
// =============================================================================

/// @brief Group statistics for MappedCustomSparse
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

    SCL_CHECK_DIM(out_means.len >= total_size, "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len >= total_size, "Vars size mismatch");

    // SIMD zero initialize
    detail::zero_fill_simd(out_means.ptr, total_size);
    detail::zero_fill_simd(out_vars.ptr, total_size);

    // Prefetch hint
    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        auto values = scl::primary_values(matrix, static_cast<Index>(p));
        auto indices = scl::primary_indices(matrix, static_cast<Index>(p));
        Index len = static_cast<Index>(values.len);

        Real* mean_ptr = out_means.ptr + (p * n_groups);
        Real* var_ptr = out_vars.ptr + (p * n_groups);

        if (len == 0) {
            detail::finalize_stats(mean_ptr, var_ptr, group_sizes.ptr, nullptr,
                                   n_groups, ddof, include_zeros);
            return;
        }

        const T* SCL_RESTRICT vals = values.ptr;
        const Index* SCL_RESTRICT inds = indices.ptr;

        // Thread-local nonzero counts
        Size nnz_counts_local[config::STACK_GROUP_LIMIT];
        Size* nnz_counts = nullptr;
        std::vector<Size> nnz_counts_heap;

        if (!include_zeros) {
            if (n_groups <= config::STACK_GROUP_LIMIT) {
                nnz_counts = nnz_counts_local;
            } else {
                nnz_counts_heap.resize(n_groups, 0);
                nnz_counts = nnz_counts_heap.data();
            }
            std::memset(nnz_counts, 0, n_groups * sizeof(Size));
        }

        // 4-way unrolled accumulation
        Index k = 0;
        for (; k + 4 <= len; k += 4) {
            if (k + config::PREFETCH_DISTANCE < len) {
                SCL_PREFETCH_READ(&group_ids[inds[k + config::PREFETCH_DISTANCE]], 0);
            }

            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                Index idx = inds[k + u];
                int32_t g = group_ids[idx];

                if (SCL_LIKELY(g >= 0 && static_cast<Size>(g) < n_groups)) {
                    Real v = static_cast<Real>(vals[k + u]);
                    mean_ptr[g] += v;
                    var_ptr[g] += v * v;
                    if (!include_zeros) nnz_counts[g]++;
                }
            }
        }

        for (; k < len; ++k) {
            Index idx = inds[k];
            int32_t g = group_ids[idx];

            if (SCL_LIKELY(g >= 0 && static_cast<Size>(g) < n_groups)) {
                Real v = static_cast<Real>(vals[k]);
                mean_ptr[g] += v;
                var_ptr[g] += v * v;
                if (!include_zeros) nnz_counts[g]++;
            }
        }

        detail::finalize_stats(mean_ptr, var_ptr, group_sizes.ptr, nnz_counts,
                               n_groups, ddof, include_zeros);
    });
}

/// @brief Group sums for MappedCustomSparse
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

    SCL_CHECK_DIM(out_sums.len >= total_size, "Sums size mismatch");

    detail::zero_fill_simd(out_sums.ptr, total_size);

    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        auto values = scl::primary_values(matrix, static_cast<Index>(p));
        auto indices = scl::primary_indices(matrix, static_cast<Index>(p));

        if (values.len == 0) return;

        Real* sum_ptr = out_sums.ptr + (p * n_groups);
        const T* SCL_RESTRICT vals = values.ptr;
        const Index* SCL_RESTRICT inds = indices.ptr;
        Size len = values.len;

        // 4-way unrolled
        Size k = 0;
        for (; k + 4 <= len; k += 4) {
            if (k + config::PREFETCH_DISTANCE < len) {
                SCL_PREFETCH_READ(&group_ids[inds[k + config::PREFETCH_DISTANCE]], 0);
            }

            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                Index idx = inds[k + u];
                int32_t g = group_ids[idx];

                if (SCL_LIKELY(g >= 0 && static_cast<Size>(g) < n_groups)) {
                    sum_ptr[g] += static_cast<Real>(vals[k + u]);
                }
            }
        }

        for (; k < len; ++k) {
            Index idx = inds[k];
            int32_t g = group_ids[idx];

            if (SCL_LIKELY(g >= 0 && static_cast<Size>(g) < n_groups)) {
                sum_ptr[g] += static_cast<Real>(vals[k]);
            }
        }
    });
}

// =============================================================================
// SECTION 4: MappedVirtualSparse Group Statistics
// =============================================================================

/// @brief Group statistics for MappedVirtualSparse
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

    SCL_CHECK_DIM(out_means.len >= total_size, "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len >= total_size, "Vars size mismatch");

    detail::zero_fill_simd(out_means.ptr, total_size);
    detail::zero_fill_simd(out_vars.ptr, total_size);

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_primary), [&](size_t p) {
        auto values = scl::primary_values(matrix, static_cast<Index>(p));
        auto indices = scl::primary_indices(matrix, static_cast<Index>(p));
        Index len = static_cast<Index>(values.len);

        Real* mean_ptr = out_means.ptr + (p * n_groups);
        Real* var_ptr = out_vars.ptr + (p * n_groups);

        if (len == 0) {
            detail::finalize_stats(mean_ptr, var_ptr, group_sizes.ptr, nullptr,
                                   n_groups, ddof, include_zeros);
            return;
        }

        const T* SCL_RESTRICT vals = values.ptr;
        const Index* SCL_RESTRICT inds = indices.ptr;

        Size nnz_counts_local[config::STACK_GROUP_LIMIT];
        Size* nnz_counts = nullptr;
        std::vector<Size> nnz_counts_heap;

        if (!include_zeros) {
            if (n_groups <= config::STACK_GROUP_LIMIT) {
                nnz_counts = nnz_counts_local;
            } else {
                nnz_counts_heap.resize(n_groups, 0);
                nnz_counts = nnz_counts_heap.data();
            }
            std::memset(nnz_counts, 0, n_groups * sizeof(Size));
        }

        // 4-way unrolled
        Index k = 0;
        for (; k + 4 <= len; k += 4) {
            if (k + config::PREFETCH_DISTANCE < len) {
                SCL_PREFETCH_READ(&group_ids[inds[k + config::PREFETCH_DISTANCE]], 0);
            }

            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                Index idx = inds[k + u];
                int32_t g = group_ids[idx];

                if (SCL_LIKELY(g >= 0 && static_cast<Size>(g) < n_groups)) {
                    Real v = static_cast<Real>(vals[k + u]);
                    mean_ptr[g] += v;
                    var_ptr[g] += v * v;
                    if (!include_zeros) nnz_counts[g]++;
                }
            }
        }

        for (; k < len; ++k) {
            Index idx = inds[k];
            int32_t g = group_ids[idx];

            if (SCL_LIKELY(g >= 0 && static_cast<Size>(g) < n_groups)) {
                Real v = static_cast<Real>(vals[k]);
                mean_ptr[g] += v;
                var_ptr[g] += v * v;
                if (!include_zeros) nnz_counts[g]++;
            }
        }

        detail::finalize_stats(mean_ptr, var_ptr, group_sizes.ptr, nnz_counts,
                               n_groups, ddof, include_zeros);
    });
}

// =============================================================================
// SECTION 5: Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate mapped implementation
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
