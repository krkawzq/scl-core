#pragma once

#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/threading/parallel_for.hpp"

// Backend implementations
#include "scl/kernel/ttest_fast_impl.hpp"
#include "scl/kernel/ttest_mapped_impl.hpp"

#include <cmath>
#include <vector>

// =============================================================================
/// @file ttest.hpp
/// @brief Differential Expression via T-Test
///
/// ## Supported Operations
///
/// 1. Welch's T-Test (unequal variance)
/// 2. Pooled T-Test (equal variance)
///
/// ## Key Optimizations
///
/// 1. Fused Group Statistics
///    - Single pass: sum, sum_sq, count per group
///    - 4-way unrolled accumulation
///
/// 2. Fast erfc Approximation
///    - Polynomial approximation (max error < 1.5e-7)
///    - Avoids expensive std::erfc calls
///
/// 3. Backend Dispatch
///    - CustomSparseLike -> ttest_fast_impl.hpp
///    - VirtualSparseLike -> ttest_fast_impl.hpp
///    - MappedSparseLike -> ttest_mapped_impl.hpp
///
/// ## Architecture
///
/// 1. Aggregation: Compute mean/var per group (single pass)
/// 2. Testing: Compute T-statistics for Ref vs each Target
///
/// Performance: Dominated by group aggregation O(nnz)
// =============================================================================

namespace scl::kernel::diff_expr {

// =============================================================================
// SECTION 1: Generic Implementation (Fallback)
// =============================================================================

namespace detail {

/// @brief Fast erfc approximation (Abramowitz & Stegun)
SCL_FORCE_INLINE Real fast_erfc(Real x) {
    Real sign = Real(1);
    if (x < Real(0)) {
        sign = Real(-1);
        x = -x;
    }
    
    Real t = Real(1) / (Real(1) + Real(0.3275911) * x);
    Real t2 = t * t;
    Real t3 = t2 * t;
    Real t4 = t3 * t;
    Real t5 = t4 * t;
    
    Real poly = Real(0.254829592) * t 
              - Real(0.284496736) * t2 
              + Real(1.421413741) * t3 
              - Real(1.453152027) * t4 
              + Real(1.061405429) * t5;
    
    Real result = poly * std::exp(-x * x);
    return (sign > Real(0)) ? result : (Real(2) - result);
}

/// @brief Compute p-value from t-statistic
SCL_FORCE_INLINE Real t_to_pvalue(Real t_stat) {
    return Real(2) * fast_erfc(std::abs(t_stat) / Real(M_SQRT2));
}

/// @brief Count group sizes from group_ids
SCL_FORCE_INLINE void count_group_sizes(
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<Size> out_sizes
) {
    SCL_CHECK_DIM(out_sizes.len >= n_groups, "Output size mismatch");
    
    std::fill(out_sizes.ptr, out_sizes.ptr + n_groups, Size(0));
    
    for (Size i = 0; i < group_ids.len; ++i) {
        int32_t g = group_ids[i];
        if (g >= 0 && static_cast<Size>(g) < n_groups) {
            out_sizes[g]++;
        }
    }
}

/// @brief Generic group statistics computation
template <typename MatrixT>
    requires AnySparse<MatrixT>
void compute_group_stats_generic(
    const MatrixT& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<Real> out_means,
    Array<Real> out_vars,
    Array<Size> out_counts
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size total_size = static_cast<Size>(primary_dim) * n_groups;
    
    SCL_CHECK_DIM(out_means.len >= total_size, "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len >= total_size, "Vars size mismatch");
    SCL_CHECK_DIM(out_counts.len >= total_size, "Counts size mismatch");
    
    std::fill(out_means.ptr, out_means.ptr + total_size, Real(0));
    std::fill(out_vars.ptr, out_vars.ptr + total_size, Real(0));
    std::fill(out_counts.ptr, out_counts.ptr + total_size, Size(0));
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        auto values = scl::primary_values(matrix, static_cast<Index>(p));
        auto indices = scl::primary_indices(matrix, static_cast<Index>(p));
        
        Real* mean_ptr = out_means.ptr + (p * n_groups);
        Real* var_ptr = out_vars.ptr + (p * n_groups);
        Size* count_ptr = out_counts.ptr + (p * n_groups);
        
        for (Size k = 0; k < values.len; ++k) {
            Index idx = indices[k];
            int32_t g = group_ids[idx];
            
            if (g >= 0 && static_cast<Size>(g) < n_groups) {
                Real v = static_cast<Real>(values[k]);
                mean_ptr[g] += v;
                var_ptr[g] += v * v;
                count_ptr[g]++;
            }
        }
    });
    
    // Finalize
    scl::threading::parallel_for(Size(0), total_size, [&](size_t i) {
        Size n = out_counts[i];
        if (n > 0) {
            Real mean = out_means[i] / static_cast<Real>(n);
            Real sum_sq = out_vars[i];
            Real var = (n > 1) ? (sum_sq / static_cast<Real>(n) - mean * mean) : Real(0);
            if (n > 1) {
                var = var * static_cast<Real>(n) / static_cast<Real>(n - 1);
            }
            out_means[i] = mean;
            out_vars[i] = var;
        }
    });
}

/// @brief Generic t-test computation
template <typename MatrixT>
    requires AnySparse<MatrixT>
void ttest_generic(
    const MatrixT& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<const Size> group_sizes,
    Array<Real> out_t_stats,
    Array<Real> out_p_values,
    Array<Real> out_log2_fc,
    bool use_welch
) {
    const Index n_features = scl::primary_size(matrix);
    const Size n_targets = n_groups - 1;
    const Size output_size = static_cast<Size>(n_features) * n_targets;
    
    SCL_CHECK_DIM(out_t_stats.len >= output_size, "T-stats size mismatch");
    SCL_CHECK_DIM(out_p_values.len >= output_size, "P-values size mismatch");
    SCL_CHECK_DIM(out_log2_fc.len >= output_size, "Log2FC size mismatch");
    
    std::vector<Real> means(static_cast<Size>(n_features) * n_groups);
    std::vector<Real> vars(static_cast<Size>(n_features) * n_groups);
    std::vector<Size> counts(static_cast<Size>(n_features) * n_groups);
    
    compute_group_stats_generic(
        matrix, group_ids, n_groups,
        Array<Real>(means.data(), means.size()),
        Array<Real>(vars.data(), vars.size()),
        Array<Size>(counts.data(), counts.size())
    );
    
    const Size N_ref = group_sizes[0];
    const Real inv_N_ref = (N_ref > 0) ? (Real(1) / static_cast<Real>(N_ref)) : Real(0);
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(n_features), [&](size_t i) {
        Real mean_ref = means[i * n_groups + 0];
        Real var_ref = vars[i * n_groups + 0];
        
        for (Size t = 0; t < n_targets; ++t) {
            Size target_group = t + 1;
            Size N_target = group_sizes[target_group];
            
            Real mean_target = means[i * n_groups + target_group];
            Real var_target = vars[i * n_groups + target_group];
            
            Real mean_diff = mean_target - mean_ref;
            
            constexpr Real PSEUDOCOUNT = Real(1e-9);
            Real log2_fc = std::log2((mean_target + PSEUDOCOUNT) / (mean_ref + PSEUDOCOUNT));
            
            Real t_stat = Real(0);
            Real p_value = Real(1);
            
            if (N_ref > 0 && N_target > 0) {
                Real inv_N_target = Real(1) / static_cast<Real>(N_target);
                
                if (use_welch) {
                    Real se_sq = var_ref * inv_N_ref + var_target * inv_N_target;
                    if (se_sq > Real(0)) {
                        t_stat = mean_diff / std::sqrt(se_sq);
                        p_value = t_to_pvalue(t_stat);
                    }
                } else {
                    Real df_ref = static_cast<Real>(N_ref - 1);
                    Real df_target = static_cast<Real>(N_target - 1);
                    Real pooled_var = (df_ref * var_ref + df_target * var_target) / (df_ref + df_target);
                    Real se = std::sqrt(pooled_var * (inv_N_ref + inv_N_target));
                    if (se > Real(0)) {
                        t_stat = mean_diff / se;
                        p_value = t_to_pvalue(t_stat);
                    }
                }
            }
            
            Size out_idx = i * n_targets + t;
            out_t_stats[out_idx] = t_stat;
            out_p_values[out_idx] = p_value;
            out_log2_fc[out_idx] = log2_fc;
        }
    });
}

} // namespace detail

// =============================================================================
// SECTION 2: Public API with Backend Dispatch
// =============================================================================

/// @brief Count group sizes from group labels
///
/// @param group_ids Group labels for each observation
/// @param n_groups Total number of groups
/// @param out_sizes Output sizes [size = n_groups], PRE-ALLOCATED
SCL_FORCE_INLINE void count_group_sizes(
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<Size> out_sizes
) {
    detail::count_group_sizes(group_ids, n_groups, out_sizes);
}

/// @brief T-test for differential expression
///
/// Compares Group 0 (Reference) against Groups 1..K (Targets).
///
/// @param matrix Input sparse matrix
/// @param group_ids Group labels (0=Ref, 1..K=Targets)
/// @param n_groups Total groups (K+1)
/// @param out_t_stats T-statistics [size = n_features * n_targets], PRE-ALLOCATED
/// @param out_p_values P-values [size = n_features * n_targets], PRE-ALLOCATED
/// @param out_log2_fc Log2 fold changes [size = n_features * n_targets], PRE-ALLOCATED
/// @param use_welch Use Welch's t-test (default true)
template <typename MatrixT>
    requires AnySparse<MatrixT>
void ttest(
    const MatrixT& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<Real> out_t_stats,
    Array<Real> out_p_values,
    Array<Real> out_log2_fc,
    bool use_welch = true
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;
    
    // Count group sizes
    std::vector<Size> group_sizes(n_groups);
    detail::count_group_sizes(group_ids, n_groups, Array<Size>(group_sizes.data(), n_groups));
    
    // Dispatch to backend
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        ttest::mapped::ttest_mapped_dispatch<MatrixT, IsCSR>(
            matrix, group_ids, n_groups,
            Array<const Size>(group_sizes.data(), n_groups),
            out_t_stats, out_p_values, out_log2_fc, use_welch
        );
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR> || VirtualSparseLike<MatrixT, IsCSR>) {
        fast::ttest_fast_dispatch<MatrixT, IsCSR>(
            matrix, group_ids, n_groups,
            Array<const Size>(group_sizes.data(), n_groups),
            out_t_stats, out_p_values, out_log2_fc, use_welch
        );
    } else {
        detail::ttest_generic(
            matrix, group_ids, n_groups,
            Array<const Size>(group_sizes.data(), n_groups),
            out_t_stats, out_p_values, out_log2_fc, use_welch
        );
    }
}

/// @brief T-test with pre-computed group sizes
///
/// Useful when group sizes are already known (avoids re-counting).
template <typename MatrixT>
    requires AnySparse<MatrixT>
void ttest_with_sizes(
    const MatrixT& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<const Size> group_sizes,
    Array<Real> out_t_stats,
    Array<Real> out_p_values,
    Array<Real> out_log2_fc,
    bool use_welch = true
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;
    
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        ttest::mapped::ttest_mapped_dispatch<MatrixT, IsCSR>(
            matrix, group_ids, n_groups, group_sizes,
            out_t_stats, out_p_values, out_log2_fc, use_welch
        );
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR> || VirtualSparseLike<MatrixT, IsCSR>) {
        fast::ttest_fast_dispatch<MatrixT, IsCSR>(
            matrix, group_ids, n_groups, group_sizes,
            out_t_stats, out_p_values, out_log2_fc, use_welch
        );
    } else {
        detail::ttest_generic(
            matrix, group_ids, n_groups, group_sizes,
            out_t_stats, out_p_values, out_log2_fc, use_welch
        );
    }
}

} // namespace scl::kernel::diff_expr
