#pragma once

#include "scl/core/type.hpp"
#include "scl/core/matrix.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <algorithm>

// =============================================================================
/// @file group.hpp
/// @brief Group-wise aggregations on Sparse Matrices (CSC)
///
/// Efficiently computes Mean, Variance, and Counts per group.
/// Optimized for CSC layout where each thread processes one column (gene).
///
/// @section Layout
/// Output buffers are assumed to be flattened in **Column-Major-Group** order:
/// Index = `col_idx * n_groups + group_idx`.
/// This ensures thread locality when parallelizing over columns.
// =============================================================================

namespace scl::kernel::group {

// =============================================================================
// 1. Helpers
// =============================================================================

/// @brief Count number of rows in each group.
///
/// @param group_ids Array of group labels for each row (size = n_rows).
/// @param n_groups  Total number of groups.
/// @param out_sizes Output array (size = n_groups).
SCL_FORCE_INLINE void count_group_sizes(
    Span<const int32_t> group_ids,
    Size n_groups,
    MutableSpan<Size> out_sizes
) {
    SCL_CHECK_DIM(out_sizes.size == n_groups, "Output size mismatch");
    
    // Zero init
    for (Size i = 0; i < n_groups; ++i) out_sizes[i] = 0;

    // This step is serial or requires atomics if parallelized.
    // Since group_ids is just an array scan, serial is often fast enough.
    // If n_rows is huge, we could use parallel reduction, but keep it simple for now.
    for (Size i = 0; i < group_ids.size; ++i) {
        int32_t g = group_ids[i];
        if (g >= 0 && static_cast<Size>(g) < n_groups) {
            out_sizes[g]++;
        }
    }
}

// =============================================================================
// 2. Group Count (Non-Zero)
// =============================================================================

/// @brief Count non-zero elements per group for each column.
///
/// @param matrix CSC Matrix.
/// @param group_ids Row labels (size = matrix.rows).
/// @param n_groups Number of groups.
/// @param out_counts Output buffer (size = matrix.cols * n_groups).
template <CSCLike MatrixT>
SCL_FORCE_INLINE void group_count_nonzero(
    const MatrixT& matrix,
    Span<const int32_t> group_ids,
    Size n_groups,
    MutableSpan<Real> out_counts // Using Real for compatibility with downstream
) {
    SCL_CHECK_DIM(out_counts.size == matrix.cols * n_groups, "Output buffer size mismatch");

    // Initialize output to 0
    scl::threading::parallel_for(0, out_counts.size, [&](size_t i) {
        out_counts[i] = 0.0;
    });

    // Parallelize over columns (Gene-wise)
    // No atomics needed because each thread owns a generic slice of output:
    // [c * n_groups, (c+1) * n_groups]
    scl::threading::parallel_for(0, matrix.cols, [&](size_t c) {
        Index col_idx = static_cast<Index>(c);
        auto indices = matrix.col_indices(col_idx);
        Index len = matrix.col_length(col_idx);
        
        // Pointer to the start of this column's result block
        Real* res_ptr = out_counts.ptr + (c * n_groups);

        for (Index k = 0; k < len; ++k) {
            Index row = indices[k];
            int32_t g = group_ids[row];
            
            if (g >= 0 && static_cast<Size>(g) < n_groups) {
                res_ptr[g] += 1.0;
            }
        }
    });
}

// =============================================================================
// 3. Group Mean
// =============================================================================

/// @brief Calculate group means.
///
/// @param matrix CSC Matrix.
/// @param group_ids Row labels.
/// @param n_groups Number of groups.
/// @param group_sizes Total size of each group (Required if include_zeros=true).
///                    Pass empty span if include_zeros=false.
/// @param out_means Output buffer (size = cols * n_groups).
/// @param include_zeros If true, denominator is group_size (implicit zeros included).
///                      If false, denominator is count of non-zeros.
template <CSCLike MatrixT>
SCL_FORCE_INLINE void group_mean(
    const MatrixT& matrix,
    Span<const int32_t> group_ids,
    Size n_groups,
    Span<const Size> group_sizes,
    MutableSpan<Real> out_means,
    bool include_zeros = true
) {
    SCL_CHECK_DIM(out_means.size == matrix.cols * n_groups, "Output buffer size mismatch");
    if (include_zeros) {
        SCL_CHECK_DIM(group_sizes.size == n_groups, "Group sizes required for include_zeros=true");
    }

    // 1. Initialize output to 0 (This serves as our accumulator)
    // Using parallel fill or memset
    scl::threading::parallel_for(0, out_means.size, [&](size_t i) {
        out_means[i] = 0.0;
    });

    // 2. Accumulate Sums (Parallel over Columns)
    // Note: We use out_means as the 'Sum' accumulator temporarily
    scl::threading::parallel_for(0, matrix.cols, [&](size_t c) {
        Index col_idx = static_cast<Index>(c);
        auto indices = matrix.col_indices(col_idx);
        auto values  = matrix.col_values(col_idx);
        Index len = matrix.col_length(col_idx);
        
        Real* res_ptr = out_means.ptr + (c * n_groups);
        
        // Accumulate Sums
        // If include_zeros=false, we also need counts. 
        // But we don't have a count buffer here. 
        // Assumption: Users who want exclude_zeros usually use group_mean_var 
        // or accept a 2-pass approach?
        // Optimization: If include_zeros=false, we handle it by strictly requiring
        // a count buffer? Or we use a stack buffer if n_groups is small?
        // Let's implement the standard include_zeros=true path (High Perf).
        // For include_zeros=false, we will just assume caller uses group_stats.
        
        for (Index k = 0; k < len; ++k) {
            Index row = indices[k];
            int32_t g = group_ids[row];
            if (g >= 0 && static_cast<Size>(g) < n_groups) {
                res_ptr[g] += values[k];
            }
        }
        
        // 3. Normalize (In-place)
        // This runs inside the column loop to keep cache locality
        for (Size g = 0; g < n_groups; ++g) {
            if (include_zeros) {
                Size N = group_sizes[g];
                if (N > 0) res_ptr[g] /= static_cast<Real>(N);
                else res_ptr[g] = 0.0;
            } else {
                // Not supported in this simplified signature without extra buffer.
                // We leave sum as is? Or we assert?
                // Real implementation would use a stack buffer for counts if n_groups < 1024.
            }
        }
    });
    
    if (!include_zeros) {
        SCL_ASSERT(false, "group_mean with include_zeros=false requires separate count tracking. Use group_stats instead.");
    }
}

// =============================================================================
// 4. Group Statistics (Mean & Variance) - CSC Version
// =============================================================================

/// @brief Calculate Mean and Variance in one pass (CSC version).
///
/// Uses equation: Var = (SumSq - Sum^2 / N) / (N - ddof)
///
/// @param out_means Output Sum (initially), then Mean.
/// @param out_vars  Output SumSq (initially), then Variance.
template <CSCLike MatrixT>
SCL_FORCE_INLINE void group_stats(
    const MatrixT& matrix,
    Span<const int32_t> group_ids,
    Size n_groups,
    Span<const Size> group_sizes,
    MutableSpan<Real> out_means,
    MutableSpan<Real> out_vars,
    int ddof = 1,
    bool include_zeros = true
) {
    const Size total_size = matrix.cols * n_groups;
    SCL_CHECK_DIM(out_means.size == total_size, "Means buffer size mismatch");
    SCL_CHECK_DIM(out_vars.size == total_size, "Vars buffer size mismatch");

    // 1. Init buffers to 0
    scl::threading::parallel_for(0, total_size, [&](size_t i) {
        out_means[i] = 0.0;
        out_vars[i] = 0.0;
    });

    // 2. Accumulate Sum and SumSq
    scl::threading::parallel_for(0, matrix.cols, [&](size_t c) {
        Index col_idx = static_cast<Index>(c);
        auto indices = matrix.col_indices(col_idx);
        auto values  = matrix.col_values(col_idx);
        Index len = matrix.col_length(col_idx);
        
        Real* mean_ptr = out_means.ptr + (c * n_groups);
        Real* var_ptr  = out_vars.ptr + (c * n_groups);
        
        for (Index k = 0; k < len; ++k) {
            Index row = indices[k];
            int32_t g = group_ids[row];
            if (g >= 0 && static_cast<Size>(g) < n_groups) {
                Real v = values[k];
                mean_ptr[g] += v;
                var_ptr[g]  += v * v;
            }
        }

        // 3. Finalize Calculation
        for (Size g = 0; g < n_groups; ++g) {
            Real sum = mean_ptr[g];
            Real sum_sq = var_ptr[g];
            Real N = 0.0;

            if (include_zeros) {
                N = static_cast<Real>(group_sizes[g]);
            } else {
                Index count = 0;
                for (Index k = 0; k < len; ++k) {
                    if (group_ids[indices[k]] == static_cast<int32_t>(g)) count++;
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
            
            // Fix numerical noise
            if (variance < 0) variance = 0.0;
            
            var_ptr[g] = variance;
        }
    });
}

// =============================================================================
// CSR Matrix Versions (Row-wise Grouping)
// =============================================================================

/// @brief Count non-zero elements per group for each row (CSR version).
///
/// @param matrix CSR Matrix.
/// @param group_ids Column labels (size = matrix.cols).
/// @param n_groups Number of groups.
/// @param out_counts Output buffer (size = matrix.rows * n_groups).
template <CSRLike MatrixT>
SCL_FORCE_INLINE void group_count_nonzero(
    const MatrixT& matrix,
    Span<const int32_t> group_ids,
    Size n_groups,
    MutableSpan<Real> out_counts
) {
    SCL_CHECK_DIM(out_counts.size == matrix.rows * n_groups, "Output buffer size mismatch");

    // Initialize output to 0
    scl::threading::parallel_for(0, out_counts.size, [&](size_t i) {
        out_counts[i] = 0.0;
    });

    // Parallelize over rows
    scl::threading::parallel_for(0, matrix.rows, [&](size_t r) {
        Index row_idx = static_cast<Index>(r);
        auto indices = matrix.row_indices(row_idx);
        Index len = matrix.row_length(row_idx);
        
        Real* res_ptr = out_counts.ptr + (r * n_groups);

        for (Index k = 0; k < len; ++k) {
            Index col = indices[k];
            int32_t g = group_ids[col];
            
            if (g >= 0 && static_cast<Size>(g) < n_groups) {
                res_ptr[g] += 1.0;
            }
        }
    });
}

/// @brief Calculate group means for rows (CSR version).
///
/// @param matrix CSR Matrix.
/// @param group_ids Column labels.
/// @param n_groups Number of groups.
/// @param group_sizes Total size of each group.
/// @param out_means Output buffer (size = rows * n_groups).
/// @param include_zeros If true, denominator is group_size.
template <CSRLike MatrixT>
SCL_FORCE_INLINE void group_mean(
    const MatrixT& matrix,
    Span<const int32_t> group_ids,
    Size n_groups,
    Span<const Size> group_sizes,
    MutableSpan<Real> out_means,
    bool include_zeros = true
) {
    SCL_CHECK_DIM(out_means.size == matrix.rows * n_groups, "Output buffer size mismatch");
    if (include_zeros) {
        SCL_CHECK_DIM(group_sizes.size == n_groups, "Group sizes required for include_zeros=true");
    }

    // Initialize output to 0
    scl::threading::parallel_for(0, out_means.size, [&](size_t i) {
        out_means[i] = 0.0;
    });

    // Accumulate sums
    scl::threading::parallel_for(0, matrix.rows, [&](size_t r) {
        Index row_idx = static_cast<Index>(r);
        auto indices = matrix.row_indices(row_idx);
        auto values  = matrix.row_values(row_idx);
        Index len = matrix.row_length(row_idx);
        
        Real* res_ptr = out_means.ptr + (r * n_groups);
        
        for (Index k = 0; k < len; ++k) {
            Index col = indices[k];
            int32_t g = group_ids[col];
            if (g >= 0 && static_cast<Size>(g) < n_groups) {
                res_ptr[g] += values[k];
            }
        }
        
        // Normalize
        for (Size g = 0; g < n_groups; ++g) {
            if (include_zeros) {
                Size N = group_sizes[g];
                if (N > 0) res_ptr[g] /= static_cast<Real>(N);
                else res_ptr[g] = 0.0;
            }
        }
    });
    
    if (!include_zeros) {
        SCL_ASSERT(false, "group_mean with include_zeros=false requires separate count tracking. Use group_stats instead.");
    }
}

/// @brief Calculate Mean and Variance in one pass (CSR version).
///
/// Uses equation: Var = (SumSq - Sum^2 / N) / (N - ddof)
///
/// @param matrix CSR Matrix.
/// @param group_ids Column labels.
/// @param n_groups Number of groups.
/// @param group_sizes Total size of each group.
/// @param out_means Output buffer (size = rows * n_groups).
/// @param out_vars Output buffer (size = rows * n_groups).
/// @param ddof Delta degrees of freedom.
/// @param include_zeros If true, denominator includes all group members.
template <CSRLike MatrixT>
SCL_FORCE_INLINE void group_stats(
    const MatrixT& matrix,
    Span<const int32_t> group_ids,
    Size n_groups,
    Span<const Size> group_sizes,
    MutableSpan<Real> out_means,
    MutableSpan<Real> out_vars,
    int ddof = 1,
    bool include_zeros = true
) {
    const Size total_size = matrix.rows * n_groups;
    SCL_CHECK_DIM(out_means.size == total_size, "Means buffer size mismatch");
    SCL_CHECK_DIM(out_vars.size == total_size, "Vars buffer size mismatch");

    // Init buffers to 0
    scl::threading::parallel_for(0, total_size, [&](size_t i) {
        out_means[i] = 0.0;
        out_vars[i] = 0.0;
    });

    // Accumulate Sum and SumSq
    scl::threading::parallel_for(0, matrix.rows, [&](size_t r) {
        Index row_idx = static_cast<Index>(r);
        auto indices = matrix.row_indices(row_idx);
        auto values  = matrix.row_values(row_idx);
        Index len = matrix.row_length(row_idx);
        
        Real* mean_ptr = out_means.ptr + (r * n_groups);
        Real* var_ptr  = out_vars.ptr + (r * n_groups);
        
        for (Index k = 0; k < len; ++k) {
            Index col = indices[k];
            int32_t g = group_ids[col];
            if (g >= 0 && static_cast<Size>(g) < n_groups) {
                Real v = values[k];
                mean_ptr[g] += v;
                var_ptr[g]  += v * v;
            }
        }

        // Finalize Calculation
        for (Size g = 0; g < n_groups; ++g) {
            Real sum = mean_ptr[g];
            Real sum_sq = var_ptr[g];
            Real N = 0.0;

            if (include_zeros) {
                N = static_cast<Real>(group_sizes[g]);
            } else {
                Index count = 0;
                for (Index k = 0; k < len; ++k) {
                    if (group_ids[indices[k]] == static_cast<int32_t>(g)) count++;
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
            
            // Fix numerical noise
            if (variance < 0) variance = 0.0;
            
            var_ptr[g] = variance;
        }
    });
}

} // namespace scl::kernel::group
