#pragma once

#include "scl/kernel/stat/stat_base.hpp"
#include "scl/kernel/stat/rank_utils.hpp"
#include "scl/kernel/stat/group_partition.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/sort.hpp"
#include "scl/core/error.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"

// =============================================================================
// FILE: scl/kernel/stat/auroc.hpp
// BRIEF: AUROC (Area Under ROC Curve) computation
// =============================================================================

namespace scl::kernel::stat::auroc {

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
// AUROC Test (independent kernel)
// =============================================================================

template <typename T, bool IsCSR>
void auroc(
    const Sparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Array<Real> out_auroc,
    Array<Real> out_p_values
) {
    const Index primary_dim = matrix.primary_dim();
    const Size N = static_cast<Size>(primary_dim);

    Size n1_total, n2_total;
    count_groups(group_ids, n1_total, n2_total);

    SCL_CHECK_ARG(n1_total > 0 && n2_total > 0,
                  "AUROC: Both groups must have at least one member");

    const MWUConstants c(n1_total, n2_total);

    // Find max row length for buffer allocation
    Size max_len = 0;
    for (Index i = 0; i < primary_dim; ++i) {
        Size len = static_cast<Size>(matrix.primary_length(i));
        if (len > max_len) max_len = len;
    }

    // Pre-allocate dual buffer pool
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
        double sum1 = 0.0, sum2 = 0.0;

        // Partition by group
        partition::partition_two_groups(
            values.ptr, indices.ptr, len_sz, group_ids.ptr,
            buf1, n1, buf2, n2, sum1, sum2
        );

        if (SCL_UNLIKELY(n1 == 0 && n2 == 0)) {
            out_auroc[p] = Real(0.5);
            out_p_values[p] = Real(1);
            return;
        }

        // Sort
        if (SCL_LIKELY(n1 > 1)) {
            scl::sort::sort(Array<T>(buf1, n1));
        }
        if (SCL_LIKELY(n2 > 1)) {
            scl::sort::sort(Array<T>(buf2, n2));
        }

        // Compute rank sum
        double R1, tie_sum;
        rank::compute_rank_sum_sparse(
            buf1, n1, n1_total,
            buf2, n2, n2_total,
            R1, tie_sum
        );

        // Compute U and AUROC
        Real u_stat, p_val;
        rank::compute_u_and_pvalue(R1, tie_sum, c, u_stat, p_val);

        out_auroc[p] = rank::compute_auroc(
            static_cast<double>(u_stat), c.n1d, c.n2d
        );
        out_p_values[p] = p_val;
    });
}

// =============================================================================
// AUROC with Log2 Fold Change
// =============================================================================

template <typename T, bool IsCSR>
void auroc_with_fc(
    const Sparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Array<Real> out_auroc,
    Array<Real> out_p_values,
    Array<Real> out_log2_fc
) {
    const Index primary_dim = matrix.primary_dim();
    const Size N = static_cast<Size>(primary_dim);

    Size n1_total, n2_total;
    count_groups(group_ids, n1_total, n2_total);

    SCL_CHECK_ARG(n1_total > 0 && n2_total > 0,
                  "AUROC: Both groups must have at least one member");

    const MWUConstants c(n1_total, n2_total);

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
        double sum1 = 0.0, sum2 = 0.0;

        partition::partition_two_groups(
            values.ptr, indices.ptr, len_sz, group_ids.ptr,
            buf1, n1, buf2, n2, sum1, sum2
        );

        // Compute log2 fold change
        double mean1 = sum1 * c.inv_n1;
        double mean2 = sum2 * c.inv_n2;
        out_log2_fc[p] = compute_log2_fc(mean1, mean2);

        if (SCL_UNLIKELY(n1 == 0 && n2 == 0)) {
            out_auroc[p] = Real(0.5);
            out_p_values[p] = Real(1);
            return;
        }

        if (SCL_LIKELY(n1 > 1)) {
            scl::sort::sort(Array<T>(buf1, n1));
        }
        if (SCL_LIKELY(n2 > 1)) {
            scl::sort::sort(Array<T>(buf2, n2));
        }

        double R1, tie_sum;
        rank::compute_rank_sum_sparse(
            buf1, n1, n1_total,
            buf2, n2, n2_total,
            R1, tie_sum
        );

        Real u_stat, p_val;
        rank::compute_u_and_pvalue(R1, tie_sum, c, u_stat, p_val);

        out_auroc[p] = rank::compute_auroc(
            static_cast<double>(u_stat), c.n1d, c.n2d
        );
        out_p_values[p] = p_val;
    });
}

} // namespace scl::kernel::stat::auroc
