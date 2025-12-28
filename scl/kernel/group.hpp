#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
// FILE: scl/kernel/group.hpp
// BRIEF: Group aggregation statistics with SIMD optimization
// =============================================================================

namespace scl::kernel::group {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Size PREFETCH_DISTANCE = 64;
}

// =============================================================================
// SIMD Utilities
// =============================================================================

namespace detail {

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
// Group Statistics Implementation
// =============================================================================

template <typename T, bool IsCSR>
void group_stats(
    const Sparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<const Size> group_sizes,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof = 1,
    bool include_zeros = true
) {
    const Index primary_dim = matrix.primary_dim();
    const Size total_size = static_cast<Size>(primary_dim) * n_groups;

    SCL_CHECK_DIM(out_means.len >= total_size, "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len >= total_size, "Vars size mismatch");

    scl::memory::zero(out_means);
    scl::memory::zero(out_vars);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);

        Real* mean_ptr = out_means.ptr + (p * n_groups);
        Real* var_ptr = out_vars.ptr + (p * n_groups);

        if (len == 0) {
            detail::finalize_stats(mean_ptr, var_ptr, group_sizes.ptr, nullptr,
                                   n_groups, ddof, include_zeros);
            return;
        }

        const auto values = matrix.primary_values_unsafe(idx);
        const auto indices = matrix.primary_indices_unsafe(idx);

        Size nnz_counts_local[256];
        Size* nnz_counts = nullptr;
        Size* nnz_counts_heap = nullptr;

        if (!include_zeros) {
            if (n_groups <= 256) {
                nnz_counts = nnz_counts_local;
            } else {
                nnz_counts_heap = scl::memory::aligned_alloc<Size>(n_groups, SCL_ALIGNMENT);
                nnz_counts = nnz_counts_heap;
            }
            scl::algo::zero(nnz_counts, n_groups);
        }

        Index k = 0;
        for (; k + 4 <= len; k += 4) {
            if (static_cast<Size>(k) + config::PREFETCH_DISTANCE < static_cast<Size>(len)) {
                SCL_PREFETCH_READ(&group_ids[indices[k + config::PREFETCH_DISTANCE]], 0);
            }

            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                Index idx_val = indices[k + u];
                int32_t g = group_ids[idx_val];

                if (SCL_LIKELY(g >= 0 && static_cast<Size>(g) < n_groups)) {
                    Real v = static_cast<Real>(values[k + u]);
                    mean_ptr[g] += v;
                    var_ptr[g] += v * v;
                    if (!include_zeros) nnz_counts[g]++;
                }
            }
        }

        for (; k < len; ++k) {
            Index idx_val = indices[k];
            int32_t g = group_ids[idx_val];

            if (SCL_LIKELY(g >= 0 && static_cast<Size>(g) < n_groups)) {
                Real v = static_cast<Real>(values[k]);
                mean_ptr[g] += v;
                var_ptr[g] += v * v;
                if (!include_zeros) nnz_counts[g]++;
            }
        }

        detail::finalize_stats(mean_ptr, var_ptr, group_sizes.ptr, nnz_counts,
                               n_groups, ddof, include_zeros);

        if (nnz_counts_heap) {
            scl::memory::aligned_free(nnz_counts_heap, SCL_ALIGNMENT);
        }
    });
}

} // namespace scl::kernel::group
