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
// FILE: scl/kernel/stat/effect_size.hpp
// BRIEF: Unified effect size computation (Cohen's d, Hedges' g, Glass' delta, CLES)
// =============================================================================

namespace scl::kernel::stat::effect_size {

// =============================================================================
// Effect Size Types
// =============================================================================

enum class EffectSizeType {
    CohensD,     // (mean2 - mean1) / pooled_sd
    HedgesG,     // Bias-corrected Cohen's d
    GlassDelta,  // (mean2 - mean1) / sd1 (control group SD)
    CLES         // Common Language Effect Size (from AUROC)
};

// =============================================================================
// Scalar Effect Size Computation
// =============================================================================

SCL_FORCE_INLINE Real compute_cohens_d(
    double mean1, double var1, Size n1,
    double mean2, double var2, Size n2
) {
    if (n1 <= 1 || n2 <= 1) {
        return Real(0);
    }

    double df1 = static_cast<double>(n1 - 1);
    double df2 = static_cast<double>(n2 - 1);
    double pooled_var = (df1 * var1 + df2 * var2) / (df1 + df2);
    double pooled_sd = std::sqrt(pooled_var);

    if (pooled_sd < config::SIGMA_MIN) {
        return Real(0);
    }

    return static_cast<Real>((mean2 - mean1) / pooled_sd);
}

SCL_FORCE_INLINE Real compute_hedges_g(
    double mean1, double var1, Size n1,
    double mean2, double var2, Size n2
) {
    Real d = compute_cohens_d(mean1, var1, n1, mean2, var2, n2);

    // Hedges' correction factor: J = 1 - 3/(4*df - 1)
    Size df = n1 + n2 - 2;
    if (df < 2) {
        return d;
    }

    double J = 1.0 - 3.0 / (4.0 * static_cast<double>(df) - 1.0);
    return static_cast<Real>(static_cast<double>(d) * J);
}

SCL_FORCE_INLINE Real compute_glass_delta(
    double mean1, double var1, Size n1,
    double mean2, double var2, Size n2
) {
    (void)var2;  // Unused
    (void)n2;    // Unused

    if (n1 <= 1) {
        return Real(0);
    }

    double sd1 = std::sqrt(var1);
    if (sd1 < config::SIGMA_MIN) {
        return Real(0);
    }

    return static_cast<Real>((mean2 - mean1) / sd1);
}

// CLES from AUROC: probability that random X2 > random X1
SCL_FORCE_INLINE Real auroc_to_cles(Real auroc) {
    return auroc;  // AUROC and CLES are equivalent
}

// =============================================================================
// Generic Effect Size from Statistics
// =============================================================================

SCL_FORCE_INLINE Real compute_effect_size(
    double mean1, double var1, Size n1,
    double mean2, double var2, Size n2,
    EffectSizeType type
) {
    switch (type) {
        case EffectSizeType::CohensD:
            return compute_cohens_d(mean1, var1, n1, mean2, var2, n2);
        case EffectSizeType::HedgesG:
            return compute_hedges_g(mean1, var1, n1, mean2, var2, n2);
        case EffectSizeType::GlassDelta:
            return compute_glass_delta(mean1, var1, n1, mean2, var2, n2);
        case EffectSizeType::CLES:
            // CLES requires AUROC, not directly computable from moments
            return Real(0.5);
        default:
            return Real(0);
    }
}

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
// Batch Effect Size Computation
// =============================================================================

template <typename T, bool IsCSR>
void effect_size(
    const Sparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Array<Real> out_effect_size,
    EffectSizeType type = EffectSizeType::CohensD
) {
    const Index primary_dim = matrix.primary_dim();
    const Size N = static_cast<Size>(primary_dim);

    Size n1_total, n2_total;
    count_groups(group_ids, n1_total, n2_total);

    SCL_CHECK_ARG(n1_total > 0 && n2_total > 0,
                  "Effect size: Both groups must have at least one member");

    const GroupConstants c(n1_total, n2_total);

    Size max_len = 0;
    for (Index i = 0; i < primary_dim; ++i) {
        Size len = static_cast<Size>(matrix.primary_length(i));
        if (len > max_len) max_len = len;
    }

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    scl::threading::DualWorkspacePool<T> buf_pool;
    buf_pool.init(n_threads, max_len);

    scl::threading::parallel_for(Size(0), N, [&](size_t p, size_t thread_rank) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);

        auto values = matrix.primary_values(idx);
        auto indices = matrix.primary_indices(idx);

        T* SCL_RESTRICT buf1 = buf_pool.get1(thread_rank);
        T* SCL_RESTRICT buf2 = buf_pool.get2(thread_rank);
        Size n1 = 0, n2 = 0;
        double sum1 = 0.0, sum_sq1 = 0.0;
        double sum2 = 0.0, sum_sq2 = 0.0;

        partition::partition_two_groups_moments(
            values.ptr, indices.ptr, len_sz, group_ids.ptr,
            buf1, n1, buf2, n2,
            sum1, sum_sq1, sum2, sum_sq2
        );

        // Compute means (including zeros)
        double mean1 = sum1 * c.inv_n1;
        double mean2 = sum2 * c.inv_n2;

        // Compute variances (with zero adjustment)
        double var1 = 0.0, var2 = 0.0;

        if (n1_total > 1) {
            double mean1_nz = (n1 > 0) ? (sum1 / static_cast<double>(n1)) : 0.0;
            double var_numer1 = sum_sq1 - static_cast<double>(n1) * mean1_nz * mean1_nz;
            Size n_zeros1 = n1_total - n1;
            var_numer1 += static_cast<double>(n_zeros1) * mean1 * mean1;
            var1 = var_numer1 / (c.n1d - 1.0);
            if (var1 < 0.0) var1 = 0.0;
        }

        if (n2_total > 1) {
            double mean2_nz = (n2 > 0) ? (sum2 / static_cast<double>(n2)) : 0.0;
            double var_numer2 = sum_sq2 - static_cast<double>(n2) * mean2_nz * mean2_nz;
            Size n_zeros2 = n2_total - n2;
            var_numer2 += static_cast<double>(n_zeros2) * mean2 * mean2;
            var2 = var_numer2 / (c.n2d - 1.0);
            if (var2 < 0.0) var2 = 0.0;
        }

        out_effect_size[p] = compute_effect_size(
            mean1, var1, n1_total,
            mean2, var2, n2_total,
            type
        );
    });
}

// =============================================================================
// Combined T-Test and Effect Size
// =============================================================================

template <typename T, bool IsCSR>
void ttest_with_effect_size(
    const Sparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Array<Real> out_t_stats,
    Array<Real> out_p_values,
    Array<Real> out_log2_fc,
    Array<Real> out_effect_size,
    EffectSizeType es_type = EffectSizeType::CohensD,
    bool use_welch = true
) {
    const Index primary_dim = matrix.primary_dim();
    const Size N = static_cast<Size>(primary_dim);

    Size n1_total, n2_total;
    count_groups(group_ids, n1_total, n2_total);

    SCL_CHECK_ARG(n1_total > 0 && n2_total > 0,
                  "T-test: Both groups must have at least one member");

    const GroupConstants c(n1_total, n2_total);
    const double pooled_df = c.n1d + c.n2d - 2.0;

    Size max_len = 0;
    for (Index i = 0; i < primary_dim; ++i) {
        Size len = static_cast<Size>(matrix.primary_length(i));
        if (len > max_len) max_len = len;
    }

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    scl::threading::DualWorkspacePool<T> buf_pool;
    buf_pool.init(n_threads, max_len);

    scl::threading::parallel_for(Size(0), N, [&](size_t p, size_t thread_rank) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);

        auto values = matrix.primary_values(idx);
        auto indices = matrix.primary_indices(idx);

        T* SCL_RESTRICT buf1 = buf_pool.get1(thread_rank);
        T* SCL_RESTRICT buf2 = buf_pool.get2(thread_rank);
        Size n1 = 0, n2 = 0;
        double sum1 = 0.0, sum_sq1 = 0.0;
        double sum2 = 0.0, sum_sq2 = 0.0;

        partition::partition_two_groups_moments(
            values.ptr, indices.ptr, len_sz, group_ids.ptr,
            buf1, n1, buf2, n2,
            sum1, sum_sq1, sum2, sum_sq2
        );

        double mean1 = sum1 * c.inv_n1;
        double mean2 = sum2 * c.inv_n2;

        out_log2_fc[p] = compute_log2_fc(mean1, mean2);

        double var1 = 0.0, var2 = 0.0;

        if (n1_total > 1) {
            double mean1_nz = (n1 > 0) ? (sum1 / static_cast<double>(n1)) : 0.0;
            double var_numer1 = sum_sq1 - static_cast<double>(n1) * mean1_nz * mean1_nz;
            Size n_zeros1 = n1_total - n1;
            var_numer1 += static_cast<double>(n_zeros1) * mean1 * mean1;
            var1 = var_numer1 / (c.n1d - 1.0);
            if (var1 < 0.0) var1 = 0.0;
        }

        if (n2_total > 1) {
            double mean2_nz = (n2 > 0) ? (sum2 / static_cast<double>(n2)) : 0.0;
            double var_numer2 = sum_sq2 - static_cast<double>(n2) * mean2_nz * mean2_nz;
            Size n_zeros2 = n2_total - n2;
            var_numer2 += static_cast<double>(n_zeros2) * mean2 * mean2;
            var2 = var_numer2 / (c.n2d - 1.0);
            if (var2 < 0.0) var2 = 0.0;
        }

        // T-statistic
        double mean_diff = mean2 - mean1;
        double t_stat = 0.0;
        double p_val = 1.0;

        if (use_welch) {
            double se_sq = var1 * c.inv_n1 + var2 * c.inv_n2;
            if (se_sq > config::SIGMA_MIN) {
                t_stat = mean_diff / std::sqrt(se_sq);
                p_val = static_cast<double>(pvalue::t_two_sided(static_cast<Real>(t_stat)));
            }
        } else {
            if (pooled_df > 0.0) {
                double df1 = c.n1d - 1.0;
                double df2 = c.n2d - 1.0;
                double pooled_var = (df1 * var1 + df2 * var2) / pooled_df;
                double se = std::sqrt(pooled_var * (c.inv_n1 + c.inv_n2));
                if (se > config::SIGMA_MIN) {
                    t_stat = mean_diff / se;
                    p_val = static_cast<double>(pvalue::t_two_sided(static_cast<Real>(t_stat)));
                }
            }
        }

        out_t_stats[p] = static_cast<Real>(t_stat);
        out_p_values[p] = static_cast<Real>(p_val);

        // Effect size
        out_effect_size[p] = compute_effect_size(
            mean1, var1, n1_total,
            mean2, var2, n2_total,
            es_type
        );
    });
}

} // namespace scl::kernel::stat::effect_size
