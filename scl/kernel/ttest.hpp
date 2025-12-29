#pragma once

#include "scl/kernel/stat/stat_base.hpp"
#include "scl/kernel/stat/group_partition.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/kernel/ttest.hpp
// BRIEF: T-test with mask-based group partitioning
// =============================================================================

namespace scl::kernel::ttest {

using namespace scl::kernel::stat;

// =============================================================================
// Group Counting
// =============================================================================

inline void count_groups(
    Array<const int32_t> group_ids,
    Size& out_n1,
    Size& out_n2
) {
    out_n1 = scl::vectorize::count(group_ids, int32_t(0));
    out_n2 = scl::vectorize::count(group_ids, int32_t(1));
}

// =============================================================================
// T-Test Constants
// =============================================================================

struct TTestConstants {
    double n1d;
    double n2d;
    double inv_n1;
    double inv_n2;
    double pooled_df;   // n1 + n2 - 2

    TTestConstants(Size n1, Size n2)
        : n1d(static_cast<double>(n1))
        , n2d(static_cast<double>(n2))
        , inv_n1((n1 > 0) ? (1.0 / n1d) : 0.0)
        , inv_n2((n2 > 0) ? (1.0 / n2d) : 0.0)
        , pooled_df(n1d + n2d - 2.0)
    {}
};

// =============================================================================
// T-Test Computation
// =============================================================================

template <typename T, bool IsCSR>
void ttest(
    const Sparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Array<Real> out_t_stats,
    Array<Real> out_p_values,
    Array<Real> out_log2_fc,
    bool use_welch = true
) {
    const Index primary_dim = matrix.primary_dim();
    const Size N = static_cast<Size>(primary_dim);

    Size n1_total = 0;
    Size n2_total = 0;
    count_groups(group_ids, n1_total, n2_total);

    SCL_CHECK_ARG(n1_total > 0 && n2_total > 0,
                  "T-test: Both groups must have at least one member");

    const TTestConstants c(n1_total, n2_total);

    // Find max row length for buffer allocation
    Size max_len = 0;
    for (Index i = 0; i < primary_dim; ++i) {
        Size len = static_cast<Size>(matrix.primary_length_unsafe(i));
        if (len > max_len) max_len = len;
    }

    // Pre-allocate dual buffer pool
    const Size n_threads = scl::threading::get_num_threads_runtime();
    scl::threading::DualWorkspacePool<T> buf_pool;
    buf_pool.init(n_threads, max_len);

    scl::threading::parallel_for(Size(0), N, [&](Size p, Size thread_rank) {
        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);
        const Size len_sz = static_cast<Size>(len);

        auto values = matrix.primary_values_unsafe(idx);
        auto indices = matrix.primary_indices_unsafe(idx);

        T* SCL_RESTRICT buf1 = buf_pool.get1(thread_rank);
        T* SCL_RESTRICT buf2 = buf_pool.get2(thread_rank);
        Size n1 = 0, n2 = 0;
        double sum1 = 0.0, sum_sq1 = 0.0;
        double sum2 = 0.0, sum_sq2 = 0.0;

        // Partition by group with moment accumulation
        partition::partition_two_groups_moments(
            values.ptr, indices.ptr, len_sz, group_ids.ptr,
            buf1, n1, buf2, n2,
            sum1, sum_sq1, sum2, sum_sq2
        );

        // Compute means (including zeros)
        double mean1 = sum1 * c.inv_n1;
        double mean2 = sum2 * c.inv_n2;

        // Log2 fold change
        out_log2_fc[static_cast<Index>(p)] = compute_log2_fc(mean1, mean2);

        // Compute variances (with zero adjustment)
        double var1 = 0.0;
        double var2 = 0.0;

        // For group 1
        if (n1_total > 1) {
            double mean1_nz = (n1 > 0) ? (sum1 / static_cast<double>(n1)) : 0.0;
            double var_numer1 = sum_sq1 - static_cast<double>(n1) * mean1_nz * mean1_nz;
            // Adjust for zeros
            Size n_zeros1 = n1_total - n1;
            var_numer1 += static_cast<double>(n_zeros1) * mean1 * mean1;
            var1 = var_numer1 / (c.n1d - 1.0);
            if (var1 < 0.0) var1 = 0.0;
        } else {
            var1 = 0.0;
        }

        // For group 2
        if (n2_total > 1) {
            double mean2_nz = (n2 > 0) ? (sum2 / static_cast<double>(n2)) : 0.0;
            double var_numer2 = sum_sq2 - static_cast<double>(n2) * mean2_nz * mean2_nz;
            Size n_zeros2 = n2_total - n2;
            var_numer2 += static_cast<double>(n_zeros2) * mean2 * mean2;
            var2 = var_numer2 / (c.n2d - 1.0);
            if (var2 < 0.0) var2 = 0.0;
        } else {
            var2 = 0.0;
        }

        // Compute t-statistic
        double mean_diff = mean2 - mean1;
        double t_stat = 0.0;
        double p_val = 1.0;

        if (use_welch) {
            // Welch's t-test (unequal variances)
            double se_sq = var1 * c.inv_n1 + var2 * c.inv_n2;
            if (se_sq > config::SIGMA_MIN) {
                t_stat = mean_diff / std::sqrt(se_sq);
                p_val = static_cast<double>(pvalue::t_two_sided(static_cast<Real>(t_stat)));
            }
        } else {
            // Student's t-test (pooled variance)
            if (c.pooled_df > 0.0) {
                double df1 = c.n1d - 1.0;
                double df2 = c.n2d - 1.0;
                double pooled_var = (df1 * var1 + df2 * var2) / c.pooled_df;
                double se = std::sqrt(pooled_var * (c.inv_n1 + c.inv_n2));
                if (se > config::SIGMA_MIN) {
                    t_stat = mean_diff / se;
                    p_val = static_cast<double>(pvalue::t_two_sided(static_cast<Real>(t_stat)));
                }
            }
        }

        out_t_stats[static_cast<Index>(p)] = static_cast<Real>(t_stat);
        out_p_values[static_cast<Index>(p)] = static_cast<Real>(p_val);
    });
}

// =============================================================================
// Legacy Interface: compute_group_stats (for backward compatibility)
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

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](Size p) {
        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);
        const Size len_sz = static_cast<Size>(len);

        if (len_sz == 0) return;

        Real* mean_ptr = out_means.ptr + (p * n_groups);
        Real* var_ptr = out_vars.ptr + (p * n_groups);
        Size* count_ptr = out_counts.ptr + (p * n_groups);

        auto values = matrix.primary_values_unsafe(idx);
        auto indices = matrix.primary_indices_unsafe(idx);

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

    // Finalize statistics
    scl::threading::parallel_for(Size(0), total_size, [&](Size i) {
        Size n = out_counts[static_cast<Index>(i)];
        if (n > 0) {
            Real mean = out_means[static_cast<Index>(i)] / static_cast<Real>(n);
            Real sum_sq = out_vars[static_cast<Index>(i)];
            Real var = (n > 1) ? (sum_sq / static_cast<Real>(n) - mean * mean) : Real(0);

            if (n > 1) {
                var = var * static_cast<Real>(n) / static_cast<Real>(n - 1);
            }

            out_means[static_cast<Index>(i)] = mean;
            out_vars[static_cast<Index>(i)] = var;
        }
    });
}

} // namespace scl::kernel::ttest
