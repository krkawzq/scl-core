#pragma once

#include "scl/core/type.hpp"
#include "scl/core/error.hpp"
#include "scl/kernel/group.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <vector>

// =============================================================================
/// @file ttest.hpp
/// @brief Differential Expression via T-Test
///
/// Computes pairwise differential expression (Reference vs Target groups).
///
/// Architecture:
/// 1. Aggregation: Use group_stats to compute mean/var for all groups
/// 2. Testing: Compute T-statistics for Ref vs each Target
///
/// Performance: Dominated by group aggregation O(nnz)
// =============================================================================

namespace scl::kernel::diff_expr {

/// @brief T-test for differential expression (unified for CSR/CSC)
///
/// Compares Group 0 (Reference) against Groups 1..K (Targets).
///
/// @param matrix Input sparse matrix
/// @param group_ids Group labels (0=Ref, 1..K=Targets)
/// @param n_groups Total groups (K+1)
/// @param out_t_stats T-statistics [size = n_features * n_targets]
/// @param out_p_values P-values [size = n_features * n_targets]
/// @param out_log2_fc Log2 fold changes [size = n_features * n_targets]
/// @param use_welch Use Welch's t-test (unequal variance)
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
    const Index n_features = scl::primary_size(matrix);
    const Size n_targets = n_groups - 1;

    SCL_CHECK_DIM(out_t_stats.size() == static_cast<Size>(n_features) * n_targets, 
                  "T-stats size mismatch");
    SCL_CHECK_DIM(out_p_values.size() == static_cast<Size>(n_features) * n_targets, 
                  "P-values size mismatch");
    SCL_CHECK_DIM(out_log2_fc.size() == static_cast<Size>(n_features) * n_targets, 
                  "Log2FC size mismatch");

    // Count group sizes
    std::vector<Size> group_sizes(n_groups);
    scl::kernel::group::count_group_sizes(
        group_ids, n_groups, 
        Array<Size>(group_sizes.data(), n_groups)
    );

    // Compute group statistics
    std::vector<Real> means(n_features * n_groups);
    std::vector<Real> vars(n_features * n_groups);
    
    scl::kernel::group::group_stats(
        matrix, group_ids, n_groups,
        Array<const Size>(group_sizes.data(), n_groups),
        Array<Real>(means.data(), means.size()),
        Array<Real>(vars.data(), vars.size())
    );

    // Compute T-tests
    const Size N_ref = group_sizes[0];
    
    scl::threading::parallel_for(0, static_cast<size_t>(n_features), [&](size_t i) {
        Real mean_ref = means[i * n_groups + 0];
        Real var_ref = vars[i * n_groups + 0];
        
        for (Size t = 0; t < n_targets; ++t) {
            Size target_group = t + 1;
            Size N_target = group_sizes[target_group];
            
            Real mean_target = means[i * n_groups + target_group];
            Real var_target = vars[i * n_groups + target_group];
            
            Real mean_diff = mean_target - mean_ref;
            Real log2_fc = std::log2((mean_target + 1e-9) / (mean_ref + 1e-9));
            
            // Compute T-statistic
            Real t_stat = 0.0;
            Real p_value = 1.0;
            
            if (use_welch) {
                // Welch's t-test
                Real se_sq = (var_ref / N_ref) + (var_target / N_target);
                if (se_sq > 0) {
                    t_stat = mean_diff / std::sqrt(se_sq);
                    // Simplified p-value (would need proper t-distribution)
                    p_value = 2.0 * std::erfc(std::abs(t_stat) / std::sqrt(2.0));
                }
            } else {
                // Pooled t-test
                Real pooled_var = ((N_ref - 1) * var_ref + (N_target - 1) * var_target) / 
                                 (N_ref + N_target - 2);
                Real se = std::sqrt(pooled_var * (1.0 / N_ref + 1.0 / N_target));
                if (se > 0) {
                    t_stat = mean_diff / se;
                    p_value = 2.0 * std::erfc(std::abs(t_stat) / std::sqrt(2.0));
                }
            }
            
            Size out_idx = i * n_targets + t;
            out_t_stats[out_idx] = t_stat;
            out_p_values[out_idx] = p_value;
            out_log2_fc[out_idx] = log2_fc;
        }
    });
}

} // namespace scl::kernel::diff_expr
