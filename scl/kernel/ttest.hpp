#pragma once

#include "scl/core/type.hpp"
#include "scl/core/matrix.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/kernel/group.hpp" // Reuse efficient group aggregation
#include "scl/math/fast/ttest.hpp"
#include "scl/core/memory.hpp"
#include "scl/threading/parallel_for.hpp"

#include <vector>

// =============================================================================
/// @file ttest.hpp
/// @brief Differential Expression Kernels (T-Test)
///
/// Computes pairwise differential expression (Reference Group vs Target Groups).
///
/// Architecture:
/// 1. Aggregation Phase: Calls group_stats to compute Mean/Var for all groups.
/// 2. Testing Phase: Iterates features to compute T-stats for Ref vs Targets.
// =============================================================================

namespace scl::kernel::diff_expr {

// =============================================================================
// Layer 1: Virtual Interface (ISparse-based, Generic but Slower)
// =============================================================================

/// @brief Run One-vs-Rest or One-vs-Many T-Test (Virtual Interface, CSC).
///
/// Generic implementation using ISparse base class.
/// Works with any sparse matrix type but may have virtual call overhead.
///
/// Compares Group 0 (Reference) against Groups 1..K (Targets).
///
/// @param matrix CSC sparse matrix (via ISparse interface)
/// @param group_ids Row labels. 0=Ref, 1..K=Targets.
/// @param n_groups Total groups (K+1).
/// @param out_t_stats Output: T-statistics [size = n_features * n_targets].
/// @param out_p_values Output: P-values [size = n_features * n_targets].
/// @param out_log2_fc Output: Log2 fold changes [size = n_features * n_targets].
/// @param out_mean_diff Output: Mean differences [size = n_features * n_targets].
/// @param workspace Temporary buffer for group stats.
///                 Size: matrix.cols() * n_groups * 2 * sizeof(Real).
/// @param use_welch If true, use Welch's t-test (unequal variance). Default true.
template <typename T>
SCL_FORCE_INLINE void ttest(
    const ICSC<T>& matrix,
    Span<const int32_t> group_ids,
    Size n_groups,
    MutableSpan<Real> out_t_stats,
    MutableSpan<Real> out_p_values,
    MutableSpan<Real> out_log2_fc,
    MutableSpan<Real> out_mean_diff,
    MutableSpan<Byte> workspace,
    bool use_welch = true
) {
    const Size n_features = static_cast<Size>(matrix.cols());
    const Size n_targets = n_groups - 1;

    // Validate Dimensions
    SCL_CHECK_DIM(out_t_stats.size == n_features * n_targets, "T-stats output size mismatch");
    SCL_CHECK_DIM(out_p_values.size == n_features * n_targets, "P-values output size mismatch");
    SCL_CHECK_DIM(out_log2_fc.size == n_features * n_targets, "Log2FC output size mismatch");
    SCL_CHECK_DIM(out_mean_diff.size == n_features * n_targets, "Mean diff output size mismatch");
    SCL_CHECK_DIM(workspace.size >= n_features * n_groups * 2 * sizeof(Real), "Workspace too small");

    // 1. Calculate Group Sizes
    // We need precise N for each group to compute degrees of freedom.
    // Small vector on stack/heap is fine (n_groups is small, e.g., 20).
    std::vector<Size> group_sizes(n_groups);
    scl::kernel::group::count_group_sizes(group_ids, n_groups, 
                                          MutableSpan<Size>(group_sizes.data(), n_groups));

    // 2. Aggregation Phase (Heavy Lifting)
    // Reuse the highly optimized kernel from group.hpp
    // Workspace Layout:
    // [Means (Features x Groups)] [Vars (Features x Groups)]
    Real* means_ptr = reinterpret_cast<Real*>(workspace.ptr);
    Real* vars_ptr  = means_ptr + (n_features * n_groups);

    MutableSpan<Real> means_span(means_ptr, n_features * n_groups);
    MutableSpan<Real> vars_span(vars_ptr, n_features * n_groups);

    // Compute Mean and Variance for all groups in one pass
    scl::kernel::group::group_stats(
        matrix, group_ids, n_groups, 
        Span<const Size>(group_sizes.data(), n_groups),
        means_span, vars_span
    );

    // 3. Testing Phase (Parallel over Features)
    // Reference Group is Index 0.
    const Size N_ref = group_sizes[0];
    
    scl::threading::parallel_for(0, n_features, [&](size_t i) {
        // Pointers to this feature's stats
        // Layout is Column-Major-Group: [Feature * n_groups + Group]
        const Real* f_means = means_ptr + (i * n_groups);
        const Real* f_vars  = vars_ptr  + (i * n_groups);
        
        Real mu_ref = f_means[0];
        Real var_ref = f_vars[0];
        
        // Loop over Target Groups (1..K)
        // Usually K is small, so scalar loop inside feature loop is fine.
        for (Size t = 0; t < n_targets; ++t) {
            Size group_idx = t + 1;
            Size N_tar = group_sizes[group_idx];
            
            Real mu_tar = f_means[group_idx];
            Real var_tar = f_vars[group_idx];

            // Default results (if N is insufficient)
            Real t_stat = 0.0;
            Real p_val = 1.0;
            Real lfc = 0.0;
            Real diff = mu_tar - mu_ref;

            // Log2FC (with pseudocount 1e-9)
            // Using fast ApproxLog2 could be an optimization, but std::log2 is safer here.
            constexpr Real eps = 1e-9;
            lfc = std::log2((mu_tar + eps) / (mu_ref + eps));

            if (N_ref > 1 && N_tar > 1) {
                // Compute T-Statistic
                if (use_welch) {
                    Real se = scl::math::fast::ttest::se_welch(
                        var_ref, static_cast<Real>(N_ref), 
                        var_tar, static_cast<Real>(N_tar)
                    );
                    Real df = scl::math::fast::ttest::df_welch(
                        var_ref, static_cast<Real>(N_ref), 
                        var_tar, static_cast<Real>(N_tar)
                    );
                    if (se > 1e-12) {
                        t_stat = diff / se;
                        p_val = scl::math::fast::ttest::p_value_approx(t_stat, df);
                    }
                } else {
                    // Student's T
                    Real se = scl::math::fast::ttest::se_pooled(
                        var_ref, static_cast<Real>(N_ref), 
                        var_tar, static_cast<Real>(N_tar)
                    );
                    Real df = static_cast<Real>(N_ref + N_tar - 2);
                    if (se > 1e-12) {
                        t_stat = diff / se;
                        p_val = scl::math::fast::ttest::p_value_approx(t_stat, df);
                    }
                }
            }

            // Write Output
            // Output Layout: Feature-Major [Feature * n_targets + Target]
            // This aligns with the outer loop (i)
            size_t out_idx = i * n_targets + t;
            
            out_t_stats[out_idx] = t_stat;
            out_p_values[out_idx] = p_val;
            out_log2_fc[out_idx] = lfc;
            out_mean_diff[out_idx] = diff;
        }
    });
}

/// @brief Run One-vs-Rest or One-vs-Many T-Test (Virtual Interface, CSR).
///
/// Generic implementation using ISparse base class.
/// This version works on CSR matrices for row-wise (sample-wise) differential analysis.
///
/// Compares Group 0 (Reference) against Groups 1..K (Targets).
///
/// @param matrix CSR sparse matrix (via ISparse interface)
/// @param group_ids Column labels. 0=Ref, 1..K=Targets.
/// @param n_groups Total groups (K+1).
/// @param out_t_stats Output: T-statistics [size = n_features * n_targets].
/// @param out_p_values Output: P-values [size = n_features * n_targets].
/// @param out_log2_fc Output: Log2 fold changes [size = n_features * n_targets].
/// @param out_mean_diff Output: Mean differences [size = n_features * n_targets].
/// @param workspace Temporary buffer for group stats.
///                 Size: matrix.rows() * n_groups * 2 * sizeof(Real).
/// @param use_welch If true, use Welch's t-test (unequal variance). Default true.
template <typename T>
SCL_FORCE_INLINE void ttest(
    const ICSR<T>& matrix,
    Span<const int32_t> group_ids,
    Size n_groups,
    MutableSpan<Real> out_t_stats,
    MutableSpan<Real> out_p_values,
    MutableSpan<Real> out_log2_fc,
    MutableSpan<Real> out_mean_diff,
    MutableSpan<Byte> workspace,
    bool use_welch = true
) {
    const Size n_features = static_cast<Size>(matrix.rows());
    const Size n_targets = n_groups - 1;

    // Validate Dimensions
    SCL_CHECK_DIM(out_t_stats.size == n_features * n_targets, "T-stats output size mismatch");
    SCL_CHECK_DIM(out_p_values.size == n_features * n_targets, "P-values output size mismatch");
    SCL_CHECK_DIM(out_log2_fc.size == n_features * n_targets, "Log2FC output size mismatch");
    SCL_CHECK_DIM(out_mean_diff.size == n_features * n_targets, "Mean diff output size mismatch");
    SCL_CHECK_DIM(workspace.size >= n_features * n_groups * 2 * sizeof(Real), "Workspace too small");

    // 1. Calculate Group Sizes
    std::vector<Size> group_sizes(n_groups);
    scl::kernel::group::count_group_sizes(group_ids, n_groups, 
                                          MutableSpan<Size>(group_sizes.data(), n_groups));

    // 2. Aggregation Phase (Heavy Lifting)
    // Workspace Layout:
    // [Means (Features x Groups)] [Vars (Features x Groups)]
    Real* means_ptr = reinterpret_cast<Real*>(workspace.ptr);
    Real* vars_ptr  = means_ptr + (n_features * n_groups);

    MutableSpan<Real> means_span(means_ptr, n_features * n_groups);
    MutableSpan<Real> vars_span(vars_ptr, n_features * n_groups);

    // Compute Mean and Variance for all groups in one pass
    scl::kernel::group::group_stats(
        matrix, group_ids, n_groups, 
        Span<const Size>(group_sizes.data(), n_groups),
        means_span, vars_span
    );

    // 3. Testing Phase (Parallel over Features)
    const Size N_ref = group_sizes[0];
    
    scl::threading::parallel_for(0, n_features, [&](size_t i) {
        // Pointers to this feature's stats
        // Layout is Row-Major-Group: [Feature * n_groups + Group]
        const Real* f_means = means_ptr + (i * n_groups);
        const Real* f_vars  = vars_ptr  + (i * n_groups);
        
        Real mu_ref = f_means[0];
        Real var_ref = f_vars[0];
        
        // Loop over Target Groups (1..K)
        for (Size t = 0; t < n_targets; ++t) {
            Size group_idx = t + 1;
            Size N_tar = group_sizes[group_idx];
            
            Real mu_tar = f_means[group_idx];
            Real var_tar = f_vars[group_idx];

            // Default results (if N is insufficient)
            Real t_stat = 0.0;
            Real p_val = 1.0;
            Real lfc = 0.0;
            Real diff = mu_tar - mu_ref;

            // Log2FC (with pseudocount 1e-9)
            constexpr Real eps = 1e-9;
            lfc = std::log2((mu_tar + eps) / (mu_ref + eps));

            if (N_ref > 1 && N_tar > 1) {
                // Compute T-Statistic
                if (use_welch) {
                    Real se = scl::math::fast::ttest::se_welch(
                        var_ref, static_cast<Real>(N_ref), 
                        var_tar, static_cast<Real>(N_tar)
                    );
                    Real df = scl::math::fast::ttest::df_welch(
                        var_ref, static_cast<Real>(N_ref), 
                        var_tar, static_cast<Real>(N_tar)
                    );
                    if (se > 1e-12) {
                        t_stat = diff / se;
                        p_val = scl::math::fast::ttest::p_value_approx(t_stat, df);
                    }
                } else {
                    // Student's T
                    Real se = scl::math::fast::ttest::se_pooled(
                        var_ref, static_cast<Real>(N_ref), 
                        var_tar, static_cast<Real>(N_tar)
                    );
                    Real df = static_cast<Real>(N_ref + N_tar - 2);
                    if (se > 1e-12) {
                        t_stat = diff / se;
                        p_val = scl::math::fast::ttest::p_value_approx(t_stat, df);
                    }
                }
            }

            // Write Output
            size_t out_idx = i * n_targets + t;
            
            out_t_stats[out_idx] = t_stat;
            out_p_values[out_idx] = p_val;
            out_log2_fc[out_idx] = lfc;
            out_mean_diff[out_idx] = diff;
        }
    });
}

// =============================================================================
// Layer 2: Concept-Based (CSCLike/CSRLike, Optimized for Custom/Virtual)
// =============================================================================

/// @brief Run One-vs-Rest or One-vs-Many T-Test (Concept-based, Optimized, CSC).
///
/// High-performance implementation for CSCLike matrices.
/// Uses unified accessors for zero-overhead abstraction.
///
/// Compares Group 0 (Reference) against Groups 1..K (Targets).
///
/// @tparam MatrixT Any CSC-like matrix type (CustomSparse or VirtualSparse)
/// @param matrix Input CSC-like Matrix (Gene-wise).
/// @param group_ids Row labels. 0=Ref, 1..K=Targets.
/// @param n_groups Total groups (K+1).
/// @param out_t_stats Output: T-statistics [size = n_features * n_targets].
/// @param out_p_values Output: P-values [size = n_features * n_targets].
/// @param out_log2_fc Output: Log2 fold changes [size = n_features * n_targets].
/// @param out_mean_diff Output: Mean differences [size = n_features * n_targets].
/// @param workspace Temporary buffer for group stats.
///                 Size: matrix.cols * n_groups * 2 * sizeof(Real).
/// @param use_welch If true, use Welch's t-test (unequal variance). Default true.
template <CSCLike MatrixT>
SCL_FORCE_INLINE void ttest(
    const MatrixT& matrix,
    Span<const int32_t> group_ids,
    Size n_groups,
    MutableSpan<Real> out_t_stats,
    MutableSpan<Real> out_p_values,
    MutableSpan<Real> out_log2_fc,
    MutableSpan<Real> out_mean_diff,
    MutableSpan<Byte> workspace,
    bool use_welch = true
) {
    const Size n_features = static_cast<Size>(scl::cols(matrix));
    const Size n_targets = n_groups - 1;

    // Validate Dimensions
    SCL_CHECK_DIM(out_t_stats.size == n_features * n_targets, "T-stats output size mismatch");
    SCL_CHECK_DIM(out_p_values.size == n_features * n_targets, "P-values output size mismatch");
    SCL_CHECK_DIM(out_log2_fc.size == n_features * n_targets, "Log2FC output size mismatch");
    SCL_CHECK_DIM(out_mean_diff.size == n_features * n_targets, "Mean diff output size mismatch");
    SCL_CHECK_DIM(workspace.size >= n_features * n_groups * 2 * sizeof(Real), "Workspace too small");

    // 1. Calculate Group Sizes
    // We need precise N for each group to compute degrees of freedom.
    // Small vector on stack/heap is fine (n_groups is small, e.g., 20).
    std::vector<Size> group_sizes(n_groups);
    scl::kernel::group::count_group_sizes(group_ids, n_groups, 
                                          MutableSpan<Size>(group_sizes.data(), n_groups));

    // 2. Aggregation Phase (Heavy Lifting)
    // Reuse the highly optimized kernel from group.hpp
    // Workspace Layout:
    // [Means (Features x Groups)] [Vars (Features x Groups)]
    Real* means_ptr = reinterpret_cast<Real*>(workspace.ptr);
    Real* vars_ptr  = means_ptr + (n_features * n_groups);

    MutableSpan<Real> means_span(means_ptr, n_features * n_groups);
    MutableSpan<Real> vars_span(vars_ptr, n_features * n_groups);

    // Compute Mean and Variance for all groups in one pass
    scl::kernel::group::group_stats(
        matrix, group_ids, n_groups, 
        Span<const Size>(group_sizes.data(), n_groups),
        means_span, vars_span
    );

    // 3. Testing Phase (Parallel over Features)
    // Reference Group is Index 0.
    const Size N_ref = group_sizes[0];
    
    scl::threading::parallel_for(0, n_features, [&](size_t i) {
        // Pointers to this feature's stats
        // Layout is Column-Major-Group: [Feature * n_groups + Group]
        const Real* f_means = means_ptr + (i * n_groups);
        const Real* f_vars  = vars_ptr  + (i * n_groups);
        
        Real mu_ref = f_means[0];
        Real var_ref = f_vars[0];
        
        // Loop over Target Groups (1..K)
        // Usually K is small, so scalar loop inside feature loop is fine.
        for (Size t = 0; t < n_targets; ++t) {
            Size group_idx = t + 1;
            Size N_tar = group_sizes[group_idx];
            
            Real mu_tar = f_means[group_idx];
            Real var_tar = f_vars[group_idx];

            // Default results (if N is insufficient)
            Real t_stat = 0.0;
            Real p_val = 1.0;
            Real lfc = 0.0;
            Real diff = mu_tar - mu_ref;

            // Log2FC (with pseudocount 1e-9)
            // Using fast ApproxLog2 could be an optimization, but std::log2 is safer here.
            constexpr Real eps = 1e-9;
            lfc = std::log2((mu_tar + eps) / (mu_ref + eps));

            if (N_ref > 1 && N_tar > 1) {
                // Compute T-Statistic
                if (use_welch) {
                    Real se = scl::math::fast::ttest::se_welch(
                        var_ref, static_cast<Real>(N_ref), 
                        var_tar, static_cast<Real>(N_tar)
                    );
                    Real df = scl::math::fast::ttest::df_welch(
                        var_ref, static_cast<Real>(N_ref), 
                        var_tar, static_cast<Real>(N_tar)
                    );
                    if (se > 1e-12) {
                        t_stat = diff / se;
                        p_val = scl::math::fast::ttest::p_value_approx(t_stat, df);
                    }
                } else {
                    // Student's T
                    Real se = scl::math::fast::ttest::se_pooled(
                        var_ref, static_cast<Real>(N_ref), 
                        var_tar, static_cast<Real>(N_tar)
                    );
                    Real df = static_cast<Real>(N_ref + N_tar - 2);
                    if (se > 1e-12) {
                        t_stat = diff / se;
                        p_val = scl::math::fast::ttest::p_value_approx(t_stat, df);
                    }
                }
            }

            // Write Output
            // Output Layout: Feature-Major [Feature * n_targets + Target]
            // This aligns with the outer loop (i)
            size_t out_idx = i * n_targets + t;
            
            out_t_stats[out_idx] = t_stat;
            out_p_values[out_idx] = p_val;
            out_log2_fc[out_idx] = lfc;
            out_mean_diff[out_idx] = diff;
        }
    });
}

/// @brief Run One-vs-Rest or One-vs-Many T-Test (Concept-based, Optimized, CSR).
///
/// High-performance implementation for CSRLike matrices.
/// This version works on CSR matrices for row-wise (sample-wise) differential analysis.
///
/// Compares Group 0 (Reference) against Groups 1..K (Targets).
///
/// @tparam MatrixT Any CSR-like matrix type (CustomSparse or VirtualSparse)
/// @param matrix Input CSR-like Matrix (Sample-wise).
/// @param group_ids Column labels. 0=Ref, 1..K=Targets.
/// @param n_groups Total groups (K+1).
/// @param out_t_stats Output: T-statistics [size = n_features * n_targets].
/// @param out_p_values Output: P-values [size = n_features * n_targets].
/// @param out_log2_fc Output: Log2 fold changes [size = n_features * n_targets].
/// @param out_mean_diff Output: Mean differences [size = n_features * n_targets].
/// @param workspace Temporary buffer for group stats.
///                 Size: matrix.rows * n_groups * 2 * sizeof(Real).
/// @param use_welch If true, use Welch's t-test (unequal variance). Default true.
template <CSRLike MatrixT>
SCL_FORCE_INLINE void ttest(
    const MatrixT& matrix,
    Span<const int32_t> group_ids,
    Size n_groups,
    MutableSpan<Real> out_t_stats,
    MutableSpan<Real> out_p_values,
    MutableSpan<Real> out_log2_fc,
    MutableSpan<Real> out_mean_diff,
    MutableSpan<Byte> workspace,
    bool use_welch = true
) {
    const Size n_features = static_cast<Size>(scl::rows(matrix));
    const Size n_targets = n_groups - 1;

    // Validate Dimensions
    SCL_CHECK_DIM(out_t_stats.size == n_features * n_targets, "T-stats output size mismatch");
    SCL_CHECK_DIM(out_p_values.size == n_features * n_targets, "P-values output size mismatch");
    SCL_CHECK_DIM(out_log2_fc.size == n_features * n_targets, "Log2FC output size mismatch");
    SCL_CHECK_DIM(out_mean_diff.size == n_features * n_targets, "Mean diff output size mismatch");
    SCL_CHECK_DIM(workspace.size >= n_features * n_groups * 2 * sizeof(Real), "Workspace too small");

    // 1. Calculate Group Sizes
    std::vector<Size> group_sizes(n_groups);
    scl::kernel::group::count_group_sizes(group_ids, n_groups, 
                                          MutableSpan<Size>(group_sizes.data(), n_groups));

    // 2. Aggregation Phase (Heavy Lifting)
    // Workspace Layout:
    // [Means (Features x Groups)] [Vars (Features x Groups)]
    Real* means_ptr = reinterpret_cast<Real*>(workspace.ptr);
    Real* vars_ptr  = means_ptr + (n_features * n_groups);

    MutableSpan<Real> means_span(means_ptr, n_features * n_groups);
    MutableSpan<Real> vars_span(vars_ptr, n_features * n_groups);

    // Compute Mean and Variance for all groups in one pass
    scl::kernel::group::group_stats(
        matrix, group_ids, n_groups, 
        Span<const Size>(group_sizes.data(), n_groups),
        means_span, vars_span
    );

    // 3. Testing Phase (Parallel over Features)
    const Size N_ref = group_sizes[0];
    
    scl::threading::parallel_for(0, n_features, [&](size_t i) {
        // Pointers to this feature's stats
        // Layout is Row-Major-Group: [Feature * n_groups + Group]
        const Real* f_means = means_ptr + (i * n_groups);
        const Real* f_vars  = vars_ptr  + (i * n_groups);
        
        Real mu_ref = f_means[0];
        Real var_ref = f_vars[0];
        
        // Loop over Target Groups (1..K)
        for (Size t = 0; t < n_targets; ++t) {
            Size group_idx = t + 1;
            Size N_tar = group_sizes[group_idx];
            
            Real mu_tar = f_means[group_idx];
            Real var_tar = f_vars[group_idx];

            // Default results (if N is insufficient)
            Real t_stat = 0.0;
            Real p_val = 1.0;
            Real lfc = 0.0;
            Real diff = mu_tar - mu_ref;

            // Log2FC (with pseudocount 1e-9)
            constexpr Real eps = 1e-9;
            lfc = std::log2((mu_tar + eps) / (mu_ref + eps));

            if (N_ref > 1 && N_tar > 1) {
                // Compute T-Statistic
                if (use_welch) {
                    Real se = scl::math::fast::ttest::se_welch(
                        var_ref, static_cast<Real>(N_ref), 
                        var_tar, static_cast<Real>(N_tar)
                    );
                    Real df = scl::math::fast::ttest::df_welch(
                        var_ref, static_cast<Real>(N_ref), 
                        var_tar, static_cast<Real>(N_tar)
                    );
                    if (se > 1e-12) {
                        t_stat = diff / se;
                        p_val = scl::math::fast::ttest::p_value_approx(t_stat, df);
                    }
                } else {
                    // Student's T
                    Real se = scl::math::fast::ttest::se_pooled(
                        var_ref, static_cast<Real>(N_ref), 
                        var_tar, static_cast<Real>(N_tar)
                    );
                    Real df = static_cast<Real>(N_ref + N_tar - 2);
                    if (se > 1e-12) {
                        t_stat = diff / se;
                        p_val = scl::math::fast::ttest::p_value_approx(t_stat, df);
                    }
                }
            }

            // Write Output
            size_t out_idx = i * n_targets + t;
            
            out_t_stats[out_idx] = t_stat;
            out_p_values[out_idx] = p_val;
            out_log2_fc[out_idx] = lfc;
            out_mean_diff[out_idx] = diff;
        }
    });
}

} // namespace scl::kernel::diff_expr
