#pragma once

#include "scl/kernel/stat/stat_base.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/memory.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/kernel/stat/oneway_anova.hpp
// BRIEF: One-way ANOVA F-test for k groups
// =============================================================================

namespace scl::kernel::stat::oneway_anova {

// =============================================================================
// Group Counting (k groups)
// =============================================================================

inline void count_k_groups(
    Array<const int32_t> group_ids,
    Size n_groups,
    Size* out_counts
) {
    for (Size g = 0; g < n_groups; ++g) {
        out_counts[g] = 0;
    }

    for (Size i = 0; i < group_ids.len; ++i) {
        int32_t g = group_ids[static_cast<Index>(i)];
        if (g >= 0 && static_cast<Size>(g) < n_groups) {
            out_counts[g]++;
        }
    }
}

// =============================================================================
// One-way ANOVA
// =============================================================================

template <typename T, bool IsCSR>
void oneway_anova(
    const Sparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Size n_groups,
    Array<Real> out_F_stats,
    Array<Real> out_p_values
) {
    const Index primary_dim = matrix.primary_dim();
    const Size N_features = static_cast<Size>(primary_dim);

    SCL_CHECK_ARG(n_groups >= 2, "One-way ANOVA requires at least 2 groups");

    // Count group sizes
    auto group_sizes = scl::memory::aligned_alloc<Size>(n_groups, SCL_ALIGNMENT);
    count_k_groups(group_ids, n_groups, group_sizes.get());

    Size N_total = 0;
    Size valid_groups = 0;
    for (Size g = 0; g < n_groups; ++g) {
        N_total += group_sizes[g];
        if (group_sizes[g] > 0) valid_groups++;
    }

    SCL_CHECK_ARG(valid_groups >= 2, "One-way ANOVA: At least 2 groups must have members");
    SCL_CHECK_ARG(N_total > n_groups, "One-way ANOVA: N must be > k");

    // Degrees of freedom
    Size df_between = valid_groups - 1;
    Size df_within = N_total - valid_groups;

    // Precompute group size info
    auto inv_group_sizes = scl::memory::aligned_alloc<double>(n_groups, SCL_ALIGNMENT);
    for (Size g = 0; g < n_groups; ++g) {
        inv_group_sizes[g] = (group_sizes[g] > 0) ? (1.0 / static_cast<double>(group_sizes[g])) : 0.0;
    }
    double inv_N = 1.0 / static_cast<double>(N_total);

    // Workspace: counts + sums + sum_sqs per group
    const size_t n_threads = scl::threading::get_num_threads_runtime();
    Size workspace_per_thread = n_groups * 3;
    scl::threading::WorkspacePool<double> work_pool;
    work_pool.init(n_threads, workspace_per_thread);

    scl::threading::parallel_for(Size(0), N_features, [&](size_t p, size_t thread_rank) {
        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);
        const Size len_sz = static_cast<Size>(len);

        auto values = matrix.primary_values_unsafe(idx);
        auto indices = matrix.primary_indices_unsafe(idx);

        double* workspace = work_pool.get(thread_rank);
        Size* counts = reinterpret_cast<Size*>(workspace);
        double* sums = workspace + n_groups;
        double* sum_sqs = workspace + 2 * n_groups;

        // Initialize
        for (Size g = 0; g < n_groups; ++g) {
            counts[g] = 0;
            sums[g] = 0.0;
            sum_sqs[g] = 0.0;
        }

        // Accumulate statistics per group
        for (Size k = 0; k < len_sz; ++k) {
            Index sec_idx = indices[k];
            int32_t g = group_ids[sec_idx];

            if (g >= 0 && static_cast<Size>(g) < n_groups) {
                auto v = static_cast<double>(values[k]);
                counts[g]++;
                sums[g] += v;
                sum_sqs[g] += v * v;
            }
        }

        // Compute grand mean (including zeros)
        double grand_sum = 0.0;
        for (Size g = 0; g < n_groups; ++g) {
            grand_sum += sums[g];
        }
        double grand_mean = grand_sum * inv_N;

        // Compute group means (including zeros)
        auto group_means = scl::memory::aligned_alloc<double>(n_groups, SCL_ALIGNMENT);
        for (Size g = 0; g < n_groups; ++g) {
            group_means[g] = sums[g] * inv_group_sizes[g];
        }

        // SS_between = sum(n_g * (mean_g - grand_mean)^2)
        double SS_between = 0.0;
        for (Size g = 0; g < n_groups; ++g) {
            if (group_sizes[g] > 0) {
                double diff = group_means[g] - grand_mean;
                SS_between += static_cast<double>(group_sizes[g]) * diff * diff;
            }
        }

        // SS_within = sum over all observations of (x - mean_g)^2
        // For sparse data: non-zeros contribute (x - mean_g)^2, zeros contribute mean_g^2
        double SS_within = 0.0;
        for (Size g = 0; g < n_groups; ++g) {
            if (group_sizes[g] > 0) {
                double mean_g = group_means[g];
                Size n_nz = counts[g];
                Size n_zeros_g = group_sizes[g] - n_nz;
                
                // Non-zero contribution: sum(x^2) - 2*mean*sum(x) + n*mean^2
                // = sum_sq - 2*mean*sum + n*mean^2
                // = sum_sq - n*mean^2 (since sum = n*mean)
                double ss_nz = sum_sqs[g] - static_cast<double>(n_nz) * mean_g * mean_g;
                
                // Zero contribution: each zero contributes (0 - mean_g)^2 = mean_g^2
                double ss_zeros = static_cast<double>(n_zeros_g) * mean_g * mean_g;
                
                SS_within += ss_nz + ss_zeros;
            }
        }

        if (SS_within < 0.0) SS_within = 0.0;

        // F statistic
        double MS_between = (df_between > 0) ? (SS_between / static_cast<double>(df_between)) : 0.0;
        double MS_within = (df_within > 0) ? (SS_within / static_cast<double>(df_within)) : 0.0;

        double F = 0.0;
        double p_val = 1.0;

        if (MS_within > config::SIGMA_MIN) {
            F = MS_between / MS_within;
            p_val = static_cast<double>(pvalue::f_pvalue(static_cast<Real>(F), df_between, df_within));
        }

        out_F_stats[static_cast<Index>(p)] = static_cast<Real>(F);
        out_p_values[static_cast<Index>(p)] = static_cast<Real>(p_val);
    });
}

} // namespace scl::kernel::stat::oneway_anova
