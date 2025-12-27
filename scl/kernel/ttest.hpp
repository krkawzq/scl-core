#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/kernel/ttest.hpp
// BRIEF: T-test computation with SIMD optimization
// =============================================================================

namespace scl::kernel::ttest {

namespace detail {

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

SCL_FORCE_INLINE Real t_to_pvalue(Real t_stat) {
    return Real(2) * fast_erfc(std::abs(t_stat) / Real(M_SQRT2));
}

} // namespace detail

// =============================================================================
// Group Statistics
// =============================================================================

template <typename T, bool IsCSR>
void compute_group_stats(
    const Sparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<Real> out_means,
    Array<Real> out_vars,
    Array<Size> out_counts
) {
    const Index primary_dim = matrix.primary_dim();
    const Size total_size = static_cast<Size>(primary_dim) * n_groups;
    
    SCL_CHECK_DIM(out_means.len >= total_size, "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len >= total_size, "Vars size mismatch");
    SCL_CHECK_DIM(out_counts.len >= total_size, "Counts size mismatch");

    scl::memory::zero(out_means);
    scl::memory::zero(out_vars);
    scl::algo::zero(out_counts.ptr, total_size);
    
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);
        
        if (len_sz == 0) return;
        
        Real* mean_ptr = out_means.ptr + (p * n_groups);
        Real* var_ptr = out_vars.ptr + (p * n_groups);
        Size* count_ptr = out_counts.ptr + (p * n_groups);
        
        auto values = matrix.primary_values(idx);
        auto indices = matrix.primary_indices(idx);
        
        Size k = 0;
        for (; k + 4 <= len_sz; k += 4) {
            for (int u = 0; u < 4; ++u) {
                Index idx_col = indices[k + u];
                int32_t g = group_ids[idx_col];
                
                if (g >= 0 && static_cast<Size>(g) < n_groups) {
                    Real v = static_cast<Real>(values[k + u]);
                    mean_ptr[g] += v;
                    var_ptr[g] += v * v;
                    count_ptr[g]++;
                }
            }
        }
        
        for (; k < len_sz; ++k) {
            Index idx_col = indices[k];
            int32_t g = group_ids[idx_col];
            
            if (g >= 0 && static_cast<Size>(g) < n_groups) {
                Real v = static_cast<Real>(values[k]);
                mean_ptr[g] += v;
                var_ptr[g] += v * v;
                count_ptr[g]++;
            }
        }
    });
    
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

// =============================================================================
// T-Test Computation
// =============================================================================

template <typename T, bool IsCSR>
void ttest(
    const Sparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<const Size> group_sizes,
    Array<Real> out_t_stats,
    Array<Real> out_p_values,
    Array<Real> out_log2_fc,
    bool use_welch
) {
    const Index n_features = matrix.primary_dim();
    const Size n_targets = n_groups - 1;
    const Size output_size = static_cast<Size>(n_features) * n_targets;
    const Size stats_size = static_cast<Size>(n_features) * n_groups;

    SCL_CHECK_DIM(out_t_stats.len >= output_size, "T-stats size mismatch");
    SCL_CHECK_DIM(out_p_values.len >= output_size, "P-values size mismatch");
    SCL_CHECK_DIM(out_log2_fc.len >= output_size, "Log2FC size mismatch");

    Real* means = scl::memory::aligned_alloc<Real>(stats_size, SCL_ALIGNMENT);
    Real* vars = scl::memory::aligned_alloc<Real>(stats_size, SCL_ALIGNMENT);
    Size* counts = scl::memory::aligned_alloc<Size>(stats_size, SCL_ALIGNMENT);

    compute_group_stats(
        matrix, group_ids, n_groups,
        Array<Real>(means, stats_size),
        Array<Real>(vars, stats_size),
        Array<Size>(counts, stats_size)
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
                        p_value = detail::t_to_pvalue(t_stat);
                    }
                } else {
                    Real df_ref = static_cast<Real>(N_ref - 1);
                    Real df_target = static_cast<Real>(N_target - 1);
                    Real pooled_var = (df_ref * var_ref + df_target * var_target) / (df_ref + df_target);
                    Real se = std::sqrt(pooled_var * (inv_N_ref + inv_N_target));
                    if (se > Real(0)) {
                        t_stat = mean_diff / se;
                        p_value = detail::t_to_pvalue(t_stat);
                    }
                }
            }

            Size out_idx = i * n_targets + t;
            out_t_stats[out_idx] = t_stat;
            out_p_values[out_idx] = p_value;
            out_log2_fc[out_idx] = log2_fc;
        }
    });

    scl::memory::aligned_free(means, SCL_ALIGNMENT);
    scl::memory::aligned_free(vars, SCL_ALIGNMENT);
    scl::memory::aligned_free(counts, SCL_ALIGNMENT);
}

} // namespace scl::kernel::ttest

