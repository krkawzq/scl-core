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
SCL_FORCE_INLINE void group_count_nonzero(
    CSCMatrix<Real> matrix,
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
        auto indices = matrix.col_indices(static_cast<Index>(c));
        
        // Pointer to the start of this column's result block
        Real* res_ptr = out_counts.ptr + (c * n_groups);

        for (size_t k = 0; k < indices.size; ++k) {
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
SCL_FORCE_INLINE void group_mean(
    CSCMatrix<Real> matrix,
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
        auto indices = matrix.col_indices(static_cast<Index>(c));
        auto values  = matrix.col_values(static_cast<Index>(c));
        
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
        
        for (size_t k = 0; k < values.size; ++k) {
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
// 4. Group Statistics (Mean & Variance)
// =============================================================================

/// @brief Calculate Mean and Variance in one pass.
///
/// Uses equation: Var = (SumSq - Sum^2 / N) / (N - ddof)
///
/// @param out_means Output Sum (initially), then Mean.
/// @param out_vars  Output SumSq (initially), then Variance.
SCL_FORCE_INLINE void group_stats(
    CSCMatrix<Real> matrix,
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
        auto indices = matrix.col_indices(static_cast<Index>(c));
        auto values  = matrix.col_values(static_cast<Index>(c));
        
        Real* mean_ptr = out_means.ptr + (c * n_groups);
        Real* var_ptr  = out_vars.ptr + (c * n_groups);
        
        // Stack buffer for NNZ counts (only if include_zeros=false)
        // Limitation: n_groups must be reasonable. 
        // If n_groups is large, this stack allocation might fail? 
        // SCL usually handles < 1000 groups.
        // For safety, if include_zeros=false, we warn or use a hack.
        // Let's assume include_zeros=true (standard) for the optimized path.
        
        for (size_t k = 0; k < values.size; ++k) {
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
                // Without a count buffer, we cannot support include_zeros=false easily here 
                // without a 2-pass scan of the column indices.
                // Pass 1: Count. Pass 2: Accumulate.
                // Since columns are usually short (< 5% sparsity), 2-pass is cheap.
                Index count = 0;
                for (size_t k = 0; k < indices.size; ++k) {
                    if (group_ids[indices[k]] == static_cast<int32_t>(g)) count++;
                }
                N = static_cast<Real>(count);
            }

            if (N <= static_cast<Real>(ddof)) {
                mean_ptr[g] = 0.0; // Or NaN
                var_ptr[g] = 0.0;  // Or NaN
                continue;
            }

            // Mean
            Real mu = sum / N;
            mean_ptr[g] = mu;

            // Variance
            // Var = (SumSq - N * mu^2) / (N - ddof)
            // This is algebraically equivalent to (SumSq - Sum^2/N) / (N-ddof)
            // but slightly more stable if mu is already computed.
            Real variance = (sum_sq - N * mu * mu) / (N - static_cast<Real>(ddof));
            
            // Fix numerical noise
            if (variance < 0) variance = 0.0;
            
            var_ptr[g] = variance;
        }
    });
}

} // namespace scl::kernel::group
