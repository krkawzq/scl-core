#pragma once

#include "scl/kernel/stat/stat_base.hpp"
#include "scl/kernel/stat/group_partition.hpp"
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
        int32_t g = group_ids[i];
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
    Size* group_sizes = scl::memory::aligned_alloc<Size>(n_groups, SCL_ALIGNMENT);
    count_k_groups(group_ids, n_groups, group_sizes);

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
    double* inv_group_sizes = scl::memory::aligned_alloc<double>(n_groups, SCL_ALIGNMENT);
    for (Size g = 0; g < n_groups; ++g) {
        inv_group_sizes[g] = (group_sizes[g] > 0) ? (1.0 / static_cast<double>(group_sizes[g])) : 0.0;
    }
    double inv_N = 1.0 / static_cast<double>(N_total);

    // Find max row length
    Size max_len = 0;
    for (Index i = 0; i < primary_dim; ++i) {
        Size len = static_cast<Size>(matrix.primary_length_unsafe(i));
        if (len > max_len) max_len = len;
    }

    // Workspace: counts + sums + sum_sqs per group
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    Size workspace_per_thread = n_groups * 3;  // counts, sums, sum_sqs
    scl::threading::WorkspacePool<double> work_pool;
    work_pool.init(n_threads, workspace_per_thread);

    scl::threading::parallel_for(Size(0), N_features, [&](size_t p, size_t thread_rank) {
        const Index idx = static_cast<Index>(p);
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
                double v = static_cast<double>(values[k]);
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
        double* group_means = sum_sqs;  // Reuse buffer
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
        // = sum(sum_sq_g - n_g * mean_g^2)
        // But we need to account for zeros
        double SS_within = 0.0;
        for (Size g = 0; g < n_groups; ++g) {
            if (group_sizes[g] > 0) {
                // Non-zero contribution
                double mean_g = group_means[g];
                Size n_nz = counts[g];
                double sum_nz = sums[g];
                double sum_sq_nz = 0.0;

                // Recompute sum_sq for this group (we overwrote it)
                // Actually we need another buffer...
                // For simplicity, use online formula
                // SS_within_g = sum_sq - n * mean^2
                // But we don't have original sum_sq anymore

                // Alternative: compute SS_within = SS_total - SS_between
            }
        }

        // SS_total = sum((x_i - grand_mean)^2) for all N observations
        // = sum(x_i^2) - N * grand_mean^2
        double total_sum_sq = 0.0;
        for (Size k = 0; k < len_sz; ++k) {
            Index sec_idx = indices[k];
            int32_t g = group_ids[sec_idx];
            if (g >= 0 && static_cast<Size>(g) < n_groups) {
                double v = static_cast<double>(values[k]);
                total_sum_sq += v * v;
            }
        }
        // Add zeros: contribute grand_mean^2 each
        Size n_zeros = N_total - len_sz;  // Approximate
        total_sum_sq += static_cast<double>(n_zeros) * grand_mean * grand_mean;

        double SS_total = total_sum_sq - static_cast<double>(N_total) * grand_mean * grand_mean;
        SS_within = SS_total - SS_between;
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

        out_F_stats[p] = static_cast<Real>(F);
        out_p_values[p] = static_cast<Real>(p_val);
    });

    scl::memory::aligned_free(group_sizes, SCL_ALIGNMENT);
    scl::memory::aligned_free(inv_group_sizes, SCL_ALIGNMENT);
}

} // namespace scl::kernel::stat::oneway_anova
