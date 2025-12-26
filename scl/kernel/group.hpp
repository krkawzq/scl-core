#pragma once

#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// Backend implementations
#include "scl/kernel/group_fast_impl.hpp"

#include <cmath>
#include <cstring>

// =============================================================================
/// @file group.hpp
/// @brief Group-wise Aggregations on Sparse Matrices
///
/// ## Algorithm
///
/// Efficiently computes statistics (mean, variance, counts) per group.
/// Works with both CSR and CSC layouts using unified accessors.
///
/// ## Output Layout
///
/// Primary-Dimension-Major-Group order:
/// Index = primary_idx * n_groups + group_idx
///
/// This allows efficient row-major access when iterating over primary dimension.
///
/// ## Supported Operations
///
/// - group_count_nonzero: Count nnz per group
/// - group_mean: Mean per group (with/without zeros)
/// - group_stats: Mean and variance per group
///
/// ## Backend Dispatch
///
/// - MappedSparseLike -> group_mapped_impl.hpp
/// - CustomSparseLike -> group_fast_impl.hpp
/// - VirtualSparseLike -> group_fast_impl.hpp
/// - Generic -> This file (fallback)
///
/// ## Performance
///
/// - Parallel over primary dimension
/// - Thread-local counting for include_zeros=false
/// - SIMD zero initialization
/// - 4-way unrolled accumulation
// =============================================================================

namespace scl::kernel::group {

// =============================================================================
// SECTION 1: Helper Functions
// =============================================================================

/// @brief Count number of elements in each group
///
/// @param group_ids Array of group labels (length = secondary_dim)
/// @param n_groups Total number of groups
/// @param out_sizes Output array (size = n_groups)
SCL_FORCE_INLINE void count_group_sizes(
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<Size> out_sizes
) {
    SCL_CHECK_DIM(out_sizes.size() >= n_groups, "Output size mismatch");

    // Zero init
    std::memset(out_sizes.ptr, 0, n_groups * sizeof(Size));

    // Count
    for (Size i = 0; i < group_ids.size(); ++i) {
        int32_t g = group_ids[i];
        if (g >= 0 && static_cast<Size>(g) < n_groups) {
            out_sizes[g]++;
        }
    }
}

// =============================================================================
// SECTION 2: Generic Implementations (Fallback)
// =============================================================================

namespace detail {

/// @brief Generic group count nonzero
template <typename MatrixT>
    requires AnySparse<MatrixT>
void group_count_nonzero_generic(
    const MatrixT& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<Real> out_counts
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size total_size = static_cast<Size>(primary_dim) * n_groups;

    SCL_CHECK_DIM(out_counts.size() >= total_size, "Output buffer size mismatch");

    // Zero initialize
    std::memset(out_counts.ptr, 0, total_size * sizeof(Real));

    // Parallel over primary dimension
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index primary_idx = static_cast<Index>(p);
        auto indices = scl::primary_indices(matrix, primary_idx);
        Index len = scl::primary_length(matrix, primary_idx);

        Real* res_ptr = out_counts.ptr + (p * n_groups);

        for (Index k = 0; k < len; ++k) {
            Index secondary_idx = indices[k];
            int32_t g = group_ids[secondary_idx];

            if (g >= 0 && static_cast<Size>(g) < n_groups) {
                res_ptr[g] += Real(1);
            }
        }
    });
}

/// @brief Generic group mean
template <typename MatrixT>
    requires AnySparse<MatrixT>
void group_mean_generic(
    const MatrixT& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<const Size> group_sizes,
    Array<Real> out_means,
    bool include_zeros
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size total_size = static_cast<Size>(primary_dim) * n_groups;

    SCL_CHECK_DIM(out_means.size() >= total_size, "Output buffer size mismatch");
    if (include_zeros) {
        SCL_CHECK_DIM(group_sizes.size() >= n_groups, "Group sizes required");
    }

    std::memset(out_means.ptr, 0, total_size * sizeof(Real));

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index primary_idx = static_cast<Index>(p);
        auto indices = scl::primary_indices(matrix, primary_idx);
        auto values = scl::primary_values(matrix, primary_idx);
        Index len = scl::primary_length(matrix, primary_idx);

        Real* res_ptr = out_means.ptr + (p * n_groups);

        // Accumulate sums
        for (Index k = 0; k < len; ++k) {
            Index secondary_idx = indices[k];
            int32_t g = group_ids[secondary_idx];
            if (g >= 0 && static_cast<Size>(g) < n_groups) {
                res_ptr[g] += values[k];
            }
        }

        // Normalize
        if (include_zeros) {
            for (Size g = 0; g < n_groups; ++g) {
                Size N = group_sizes[g];
                if (N > 0) {
                    res_ptr[g] /= static_cast<Real>(N);
                }
            }
        }
    });

    if (!include_zeros) {
        SCL_ASSERT(false, "group_mean with include_zeros=false requires group_stats");
    }
}

/// @brief Generic group stats
template <typename MatrixT>
    requires AnySparse<MatrixT>
void group_stats_generic(
    const MatrixT& matrix,
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

    SCL_CHECK_DIM(out_means.size() >= total_size, "Means buffer size mismatch");
    SCL_CHECK_DIM(out_vars.size() >= total_size, "Vars buffer size mismatch");

    std::memset(out_means.ptr, 0, total_size * sizeof(Real));
    std::memset(out_vars.ptr, 0, total_size * sizeof(Real));

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index primary_idx = static_cast<Index>(p);
        auto indices = scl::primary_indices(matrix, primary_idx);
        auto values = scl::primary_values(matrix, primary_idx);
        Index len = scl::primary_length(matrix, primary_idx);

        Real* mean_ptr = out_means.ptr + (p * n_groups);
        Real* var_ptr = out_vars.ptr + (p * n_groups);

        // Thread-local nnz counts
        std::vector<Size> nnz_counts;
        if (!include_zeros) {
            nnz_counts.resize(n_groups, 0);
        }

        // Accumulate Sum and SumSq
        for (Index k = 0; k < len; ++k) {
            Index secondary_idx = indices[k];
            int32_t g = group_ids[secondary_idx];
            if (g >= 0 && static_cast<Size>(g) < n_groups) {
                Real v = values[k];
                mean_ptr[g] += v;
                var_ptr[g] += v * v;
                if (!include_zeros) nnz_counts[g]++;
            }
        }

        // Finalize
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
            mean_ptr[g] = mu;

            Real variance = (sum_sq - N * mu * mu) / (N - static_cast<Real>(ddof));
            if (variance < Real(0)) variance = Real(0);
            var_ptr[g] = variance;
        }
    });
}

} // namespace detail

// =============================================================================
// SECTION 3: Public API
// =============================================================================

/// @brief Count non-zero elements per group
///
/// @param matrix Input sparse matrix
/// @param group_ids Group labels for secondary dimension
/// @param n_groups Number of groups
/// @param out_counts Output buffer (size = primary_dim * n_groups), PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void group_count_nonzero(
    const MatrixT& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<Real> out_counts
) {
    detail::group_count_nonzero_generic(matrix, group_ids, n_groups, out_counts);
}

/// @brief Calculate group means
///
/// @param matrix Input sparse matrix
/// @param group_ids Group labels for secondary dimension
/// @param n_groups Number of groups
/// @param group_sizes Total size of each group
/// @param out_means Output buffer (size = primary_dim * n_groups), PRE-ALLOCATED
/// @param include_zeros If true, include implicit zeros
template <typename MatrixT>
    requires AnySparse<MatrixT>
void group_mean(
    const MatrixT& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<const Size> group_sizes,
    Array<Real> out_means,
    bool include_zeros = true
) {
    detail::group_mean_generic(matrix, group_ids, n_groups, group_sizes,
                               out_means, include_zeros);
}

/// @brief Calculate group mean and variance
///
/// Uses equation: Var = (SumSq - N * mu^2) / (N - ddof)
///
/// @param matrix Input sparse matrix
/// @param group_ids Group labels for secondary dimension
/// @param n_groups Number of groups
/// @param group_sizes Total size of each group
/// @param out_means Output means (size = primary_dim * n_groups), PRE-ALLOCATED
/// @param out_vars Output variances (size = primary_dim * n_groups), PRE-ALLOCATED
/// @param ddof Delta degrees of freedom (default 1)
/// @param include_zeros If true, include implicit zeros
template <typename MatrixT>
    requires AnySparse<MatrixT>
void group_stats(
    const MatrixT& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<const Size> group_sizes,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof = 1,
    bool include_zeros = true
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR> ||
                  CustomSparseLike<MatrixT, IsCSR> ||
                  VirtualSparseLike<MatrixT, IsCSR>) {
        fast::group_stats_fast<MatrixT, IsCSR>(
            matrix, group_ids, n_groups, group_sizes,
            out_means, out_vars, ddof, include_zeros);
    } else {
        detail::group_stats_generic(
            matrix, group_ids, n_groups, group_sizes,
            out_means, out_vars, ddof, include_zeros);
    }
}

} // namespace scl::kernel::group
