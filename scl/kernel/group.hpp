#pragma once

#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
/// @file group.hpp
/// @brief Group-wise Aggregations on Sparse Matrices
///
/// Efficiently computes statistics (mean, variance, counts) per group.
/// Works with both CSR and CSC layouts using unified accessors.
///
/// Output Layout: Primary-Dimension-Major-Group order
/// Index = primary_idx * n_groups + group_idx
///
/// Performance:
/// - Parallel over primary dimension
/// - Zero memory allocation in hot loop
/// - Cache-friendly access pattern
// =============================================================================

namespace scl::kernel::group {

// =============================================================================
// Helper Functions
// =============================================================================

/// @brief Count number of elements in each group
///
/// @param group_ids Array of group labels
/// @param n_groups Total number of groups
/// @param out_sizes Output array (size = n_groups)
SCL_FORCE_INLINE void count_group_sizes(
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<Size> out_sizes
) {
    SCL_CHECK_DIM(out_sizes.size() == n_groups, "Output size mismatch");
    
    // Zero init
    for (Size i = 0; i < n_groups; ++i) {
        out_sizes[i] = 0;
    }

    // Count
    for (Size i = 0; i < group_ids.size(); ++i) {
        int32_t g = group_ids[i];
        if (g >= 0 && static_cast<Size>(g) < n_groups) {
            out_sizes[g]++;
        }
    }
}

// =============================================================================
// Core Algorithms (Unified for CSR/CSC)
// =============================================================================

/// @brief Count non-zero elements per group (unified for CSR/CSC)
///
/// For CSC: group_ids are row labels, output is per-column
/// For CSR: group_ids are column labels, output is per-row
///
/// @param matrix Input sparse matrix
/// @param group_ids Group labels for secondary dimension elements
/// @param n_groups Number of groups
/// @param out_counts Output buffer (size = primary_dim * n_groups)
template <typename MatrixT>
    requires AnySparse<MatrixT>
SCL_FORCE_INLINE void group_count_nonzero(
    const MatrixT& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<Real> out_counts
) {
    const Index primary_dim = scl::primary_size(matrix);
    SCL_CHECK_DIM(out_counts.size() == static_cast<Size>(primary_dim) * n_groups, 
                  "Output buffer size mismatch");

    // Initialize to zero
    scl::threading::parallel_for(0, out_counts.size(), [&](size_t i) {
        out_counts[i] = 0.0;
    });

    // Parallel over primary dimension
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index primary_idx = static_cast<Index>(p);
        auto indices = scl::primary_indices(matrix, primary_idx);
        Index len = scl::primary_length(matrix, primary_idx);
        
        Real* res_ptr = out_counts.ptr + (p * n_groups);

        for (Index k = 0; k < len; ++k) {
            Index secondary_idx = indices[k];
            int32_t g = group_ids[secondary_idx];
            
            if (g >= 0 && static_cast<Size>(g) < n_groups) {
                res_ptr[g] += 1.0;
            }
        }
    });
}

/// @brief Calculate group means (unified for CSR/CSC)
///
/// @param matrix Input sparse matrix
/// @param group_ids Group labels for secondary dimension
/// @param n_groups Number of groups
/// @param group_sizes Total size of each group (required if include_zeros=true)
/// @param out_means Output buffer (size = primary_dim * n_groups)
/// @param include_zeros If true, include implicit zeros in mean calculation
template <typename MatrixT>
    requires AnySparse<MatrixT>
SCL_FORCE_INLINE void group_mean(
    const MatrixT& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<const Size> group_sizes,
    Array<Real> out_means,
    bool include_zeros = true
) {
    const Index primary_dim = scl::primary_size(matrix);
    SCL_CHECK_DIM(out_means.size() == static_cast<Size>(primary_dim) * n_groups, 
                  "Output buffer size mismatch");
    if (include_zeros) {
        SCL_CHECK_DIM(group_sizes.size() == n_groups, 
                      "Group sizes required for include_zeros=true");
    }

    // Initialize
    scl::threading::parallel_for(0, out_means.size(), [&](size_t i) {
        out_means[i] = 0.0;
    });

    // Accumulate sums
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index primary_idx = static_cast<Index>(p);
        auto indices = scl::primary_indices(matrix, primary_idx);
        auto values  = scl::primary_values(matrix, primary_idx);
        Index len = scl::primary_length(matrix, primary_idx);
        
        Real* res_ptr = out_means.ptr + (p * n_groups);
        
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
                } else {
                    res_ptr[g] = 0.0;
                }
            }
        }
    });
    
    if (!include_zeros) {
        SCL_ASSERT(false, "group_mean with include_zeros=false requires group_stats");
    }
}

/// @brief Calculate group mean and variance (unified for CSR/CSC)
///
/// Uses equation: Var = (SumSq - Sum^2 / N) / (N - ddof)
///
/// @param matrix Input sparse matrix
/// @param group_ids Group labels for secondary dimension
/// @param n_groups Number of groups
/// @param group_sizes Total size of each group
/// @param out_means Output means (size = primary_dim * n_groups)
/// @param out_vars Output variances (size = primary_dim * n_groups)
/// @param ddof Delta degrees of freedom (default 1)
/// @param include_zeros If true, include implicit zeros
template <typename MatrixT>
    requires AnySparse<MatrixT>
SCL_FORCE_INLINE void group_stats(
    const MatrixT& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<const Size> group_sizes,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof = 1,
    bool include_zeros = true
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size total_size = static_cast<Size>(primary_dim) * n_groups;
    SCL_CHECK_DIM(out_means.size() == total_size, "Means buffer size mismatch");
    SCL_CHECK_DIM(out_vars.size() == total_size, "Vars buffer size mismatch");

    // Initialize
    scl::threading::parallel_for(0, total_size, [&](size_t i) {
        out_means[i] = 0.0;
        out_vars[i] = 0.0;
    });

    // Accumulate Sum and SumSq
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index primary_idx = static_cast<Index>(p);
        auto indices = scl::primary_indices(matrix, primary_idx);
        auto values  = scl::primary_values(matrix, primary_idx);
        Index len = scl::primary_length(matrix, primary_idx);
        
        Real* mean_ptr = out_means.ptr + (p * n_groups);
        Real* var_ptr  = out_vars.ptr + (p * n_groups);
        
        for (Index k = 0; k < len; ++k) {
            Index secondary_idx = indices[k];
            int32_t g = group_ids[secondary_idx];
            if (g >= 0 && static_cast<Size>(g) < n_groups) {
                Real v = values[k];
                mean_ptr[g] += v;
                var_ptr[g]  += v * v;
            }
        }

        // Finalize calculation
        for (Size g = 0; g < n_groups; ++g) {
            Real sum = mean_ptr[g];
            Real sum_sq = var_ptr[g];
            Real N = 0.0;

            if (include_zeros) {
                N = static_cast<Real>(group_sizes[g]);
            } else {
                Index count = 0;
                for (Index k = 0; k < len; ++k) {
                    if (group_ids[indices[k]] == static_cast<int32_t>(g)) {
                        count++;
                    }
                }
                N = static_cast<Real>(count);
            }

            if (N <= static_cast<Real>(ddof)) {
                mean_ptr[g] = 0.0;
                var_ptr[g] = 0.0;
                continue;
            }

            // Mean
            Real mu = sum / N;
            mean_ptr[g] = mu;

            // Variance
            Real variance = (sum_sq - N * mu * mu) / (N - static_cast<Real>(ddof));
            if (variance < 0.0) variance = 0.0;
            
            var_ptr[g] = variance;
        }
    });
}

} // namespace scl::kernel::group
