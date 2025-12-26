#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// Mapped backend support
#include "scl/kernel/mapped_common.hpp"
#include "scl/kernel/group_mapped_impl.hpp"

#include <cstring>

// =============================================================================
/// @file group_fast_impl.hpp
/// @brief Extreme Performance Group Aggregation
///
/// ## Key Optimizations
///
/// 1. SIMD Zero Initialization
///    - Vectorized memset for output buffers
///
/// 2. Thread-Local Per-Group Counting
///    - Avoid recomputing group counts in finalization
///    - Precompute nonzero count per group in accumulation
///
/// 3. 4-Way Unrolled Accumulation
///    - Reduce loop overhead
///    - Better instruction-level parallelism
///
/// 4. Prefetch for Random Access
///    - group_ids access is random based on indices
///    - Prefetch future group_ids
///
/// ## Output Layout
///
/// Primary-Dimension-Major-Group order:
/// Index = primary_idx * n_groups + group_idx
///
/// Performance Target: 2-3x faster than generic
// =============================================================================

namespace scl::kernel::group::fast {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 64;
    constexpr Size SIMD_ZERO_THRESHOLD = 64;  // Use SIMD for arrays >= this size
}

// =============================================================================
// SECTION 2: SIMD Utilities
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

    // SIMD store
    for (; k + lanes <= len; k += lanes) {
        s::Store(zero, d, ptr + k);
    }

    // Scalar remainder
    for (; k < len; ++k) {
        ptr[k] = T(0);
    }
}

/// @brief Finalize mean and variance from sum and sum_sq
SCL_FORCE_INLINE void finalize_stats(
    Real* SCL_RESTRICT mean_ptr,
    Real* SCL_RESTRICT var_ptr,
    const Size* SCL_RESTRICT group_sizes,
    const Size* SCL_RESTRICT nnz_counts,  // Can be nullptr if include_zeros
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
// SECTION 3: CustomSparse Fast Path
// =============================================================================

/// @brief Group statistics for CustomSparse
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void group_stats_custom(
    const CustomSparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<const Size> group_sizes,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof,
    bool include_zeros
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size total_size = static_cast<Size>(primary_dim) * n_groups;

    SCL_CHECK_DIM(out_means.len >= total_size, "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len >= total_size, "Vars size mismatch");

    // SIMD zero initialize
    detail::zero_fill_simd(out_means.ptr, total_size);
    detail::zero_fill_simd(out_vars.ptr, total_size);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Index len = end - start;

        Real* mean_ptr = out_means.ptr + (p * n_groups);
        Real* var_ptr = out_vars.ptr + (p * n_groups);

        if (len == 0) {
            // Already zero-initialized, just finalize
            detail::finalize_stats(mean_ptr, var_ptr, group_sizes.ptr, nullptr,
                                   n_groups, ddof, include_zeros);
            return;
        }

        const T* SCL_RESTRICT values = matrix.data + start;
        const Index* SCL_RESTRICT indices = matrix.indices + start;

        // Thread-local nonzero counts (only needed if !include_zeros)
        Size nnz_counts_local[256];  // Stack buffer for small n_groups
        Size* nnz_counts = nullptr;
        std::vector<Size> nnz_counts_heap;

        if (!include_zeros) {
            if (n_groups <= 256) {
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
            // Prefetch
            if (k + config::PREFETCH_DISTANCE < len) {
                SCL_PREFETCH_READ(&group_ids[indices[k + config::PREFETCH_DISTANCE]], 0);
            }

            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                Index idx = indices[k + u];
                int32_t g = group_ids[idx];

                if (SCL_LIKELY(g >= 0 && static_cast<Size>(g) < n_groups)) {
                    Real v = static_cast<Real>(values[k + u]);
                    mean_ptr[g] += v;
                    var_ptr[g] += v * v;
                    if (!include_zeros) nnz_counts[g]++;
                }
            }
        }

        // Scalar remainder
        for (; k < len; ++k) {
            Index idx = indices[k];
            int32_t g = group_ids[idx];

            if (SCL_LIKELY(g >= 0 && static_cast<Size>(g) < n_groups)) {
                Real v = static_cast<Real>(values[k]);
                mean_ptr[g] += v;
                var_ptr[g] += v * v;
                if (!include_zeros) nnz_counts[g]++;
            }
        }

        // Finalize
        detail::finalize_stats(mean_ptr, var_ptr, group_sizes.ptr, nnz_counts,
                               n_groups, ddof, include_zeros);
    });
}

// =============================================================================
// SECTION 4: VirtualSparse Fast Path
// =============================================================================

/// @brief Group statistics for VirtualSparse
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void group_stats_virtual(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<const Size> group_sizes,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof,
    bool include_zeros
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size total_size = static_cast<Size>(primary_dim) * n_groups;

    SCL_CHECK_DIM(out_means.len >= total_size, "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len >= total_size, "Vars size mismatch");

    // SIMD zero initialize
    detail::zero_fill_simd(out_means.ptr, total_size);
    detail::zero_fill_simd(out_vars.ptr, total_size);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index len = matrix.lengths[p];

        Real* mean_ptr = out_means.ptr + (p * n_groups);
        Real* var_ptr = out_vars.ptr + (p * n_groups);

        if (len == 0) {
            detail::finalize_stats(mean_ptr, var_ptr, group_sizes.ptr, nullptr,
                                   n_groups, ddof, include_zeros);
            return;
        }

        // Single pointer dereference
        const T* SCL_RESTRICT values = static_cast<const T*>(matrix.data_ptrs[p]);
        const Index* SCL_RESTRICT indices = static_cast<const Index*>(matrix.indices_ptrs[p]);

        // Thread-local nonzero counts
        Size nnz_counts_local[256];
        Size* nnz_counts = nullptr;
        std::vector<Size> nnz_counts_heap;

        if (!include_zeros) {
            if (n_groups <= 256) {
                nnz_counts = nnz_counts_local;
            } else {
                nnz_counts_heap.resize(n_groups, 0);
                nnz_counts = nnz_counts_heap.data();
            }
            std::memset(nnz_counts, 0, n_groups * sizeof(Size));
        }

        // 4-way unrolled accumulation (same as Custom)
        Index k = 0;
        for (; k + 4 <= len; k += 4) {
            if (k + config::PREFETCH_DISTANCE < len) {
                SCL_PREFETCH_READ(&group_ids[indices[k + config::PREFETCH_DISTANCE]], 0);
            }

            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                Index idx = indices[k + u];
                int32_t g = group_ids[idx];

                if (SCL_LIKELY(g >= 0 && static_cast<Size>(g) < n_groups)) {
                    Real v = static_cast<Real>(values[k + u]);
                    mean_ptr[g] += v;
                    var_ptr[g] += v * v;
                    if (!include_zeros) nnz_counts[g]++;
                }
            }
        }

        for (; k < len; ++k) {
            Index idx = indices[k];
            int32_t g = group_ids[idx];

            if (SCL_LIKELY(g >= 0 && static_cast<Size>(g) < n_groups)) {
                Real v = static_cast<Real>(values[k]);
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

/// @brief Auto-dispatch to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void group_stats_fast(
    const MatrixT& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<const Size> group_sizes,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof,
    bool include_zeros
) {
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        scl::kernel::group::mapped::group_stats_mapped_dispatch<MatrixT, IsCSR>(
            matrix, group_ids, n_groups, group_sizes, out_means, out_vars, ddof, include_zeros);
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        group_stats_custom(matrix, group_ids, n_groups, group_sizes,
                           out_means, out_vars, ddof, include_zeros);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        group_stats_virtual(matrix, group_ids, n_groups, group_sizes,
                            out_means, out_vars, ddof, include_zeros);
    }
}

} // namespace scl::kernel::group::fast
