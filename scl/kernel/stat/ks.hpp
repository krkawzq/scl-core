#pragma once

#include "scl/kernel/stat/rank_utils.hpp"
#include "scl/kernel/stat/group_partition.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/sort.hpp"
#include "scl/core/error.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/kernel/stat/ks.hpp
// BRIEF: Kolmogorov-Smirnov two-sample test
// =============================================================================

namespace scl::kernel::stat::ks {

// =============================================================================
// KS P-Value Computation
// =============================================================================

namespace detail {

// Kolmogorov distribution approximation for p-value
SCL_FORCE_INLINE Real ks_pvalue(Real D, Size n1, Size n2) {
    if (D <= Real(0)) {
        return Real(1);
    }

    // Effective sample size
    auto n1d = static_cast<double>(n1);
    auto n2d = static_cast<double>(n2);
    auto n_eff = (n1d * n2d) / (n1d + n2d);

    // Asymptotic approximation
    double lambda = (std::sqrt(n_eff) + 0.12 + 0.11 / std::sqrt(n_eff)) * static_cast<double>(D);

    // Kolmogorov distribution: P(D > d) = 2 * sum_{k=1}^inf (-1)^{k+1} * exp(-2*k^2*lambda^2)
    double sum = 0.0;
    double lambda_sq = lambda * lambda;

    for (int k = 1; k <= 100; ++k) {
        double term = std::exp(-2.0 * static_cast<double>(k * k) * lambda_sq);
        if (k % 2 == 1) {
            sum += term;
        } else {
            sum -= term;
        }
        if (term < 1e-12) break;
    }

    double p = 2.0 * sum;
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;

    return static_cast<Real>(p);
}

// Merge for KS: track max ECDF difference (sparse version)
template <typename T>
SCL_FORCE_INLINE void compute_ks_sparse(
    const T* SCL_RESTRICT a, Size na_nz, Size n1_total,
    const T* SCL_RESTRICT b, Size nb_nz, Size n2_total,
    double& out_D
) {
    double inv_n1 = (n1_total > 0) ? (1.0 / static_cast<double>(n1_total)) : 0.0;
    double inv_n2 = (n2_total > 0) ? (1.0 / static_cast<double>(n2_total)) : 0.0;

    Size a_zeros = n1_total - na_nz;
    Size b_zeros = n2_total - nb_nz;

    // Find negative boundaries
    Size na_neg = rank::find_negative_boundary(a, na_nz);
    Size nb_neg = rank::find_negative_boundary(b, nb_nz);

    double F1 = 0.0, F2 = 0.0;
    double max_diff = 0.0;
    Size pa = 0, pb = 0;

    // Process negative values
    while (pa < na_neg || pb < nb_neg) {
        T v1 = (pa < na_neg) ? a[pa] : std::numeric_limits<T>::max();
        T v2 = (pb < nb_neg) ? b[pb] : std::numeric_limits<T>::max();
        T val = (v1 < v2) ? v1 : v2;

        while (pa < na_neg && a[pa] == val) { pa++; }
        while (pb < nb_neg && b[pb] == val) { pb++; }

        F1 = static_cast<double>(pa) * inv_n1;
        F2 = static_cast<double>(pb) * inv_n2;
        double diff = std::abs(F1 - F2);
        if (diff > max_diff) max_diff = diff;
    }

    // Process zeros
    Size a_at_zero = pa + a_zeros;
    Size b_at_zero = pb + b_zeros;
    F1 = static_cast<double>(a_at_zero) * inv_n1;
    F2 = static_cast<double>(b_at_zero) * inv_n2;
    double diff = std::abs(F1 - F2);
    if (diff > max_diff) max_diff = diff;

    // Process positive values
    pa = na_neg;
    pb = nb_neg;
    while (pa < na_nz || pb < nb_nz) {
        T v1 = (pa < na_nz) ? a[pa] : std::numeric_limits<T>::max();
        T v2 = (pb < nb_nz) ? b[pb] : std::numeric_limits<T>::max();
        T val = (v1 < v2) ? v1 : v2;

        while (pa < na_nz && a[pa] == val) { pa++; }
        while (pb < nb_nz && b[pb] == val) { pb++; }

        // Update ECDF accounting for zeros
        F1 = static_cast<double>(a_zeros + pa) * inv_n1;
        F2 = static_cast<double>(b_zeros + pb) * inv_n2;
        diff = std::abs(F1 - F2);
        if (diff > max_diff) max_diff = diff;
    }

    out_D = max_diff;
}

} // namespace detail

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
// KS Test
// =============================================================================

template <typename T, bool IsCSR>
void ks_test(
    const Sparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Array<Real> out_D_stats,
    Array<Real> out_p_values
) {
    const Index primary_dim = matrix.primary_dim();
    const Size N = static_cast<Size>(primary_dim);

    Size n1_total{}, n2_total{};
    count_groups(group_ids, n1_total, n2_total);

    SCL_CHECK_ARG(n1_total > 0 && n2_total > 0,
                  "KS test: Both groups must have at least one member");

    Size max_len = 0;
    for (Index i = 0; i < primary_dim; ++i) {
        Size len = static_cast<Size>(matrix.primary_length_unsafe(i));
        if (len > max_len) max_len = len;
    }

    const size_t n_threads = scl::threading::get_num_threads_runtime();
    scl::threading::DualWorkspacePool<T> buf_pool;
    buf_pool.init(n_threads, max_len);

    scl::threading::parallel_for(Size(0), N, [&](size_t p, size_t thread_rank) {
        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);
        const Size len_sz = static_cast<Size>(len);

        auto values = matrix.primary_values_unsafe(idx);
        auto indices = matrix.primary_indices_unsafe(idx);

        T* SCL_RESTRICT buf1 = buf_pool.get1(thread_rank);
        T* SCL_RESTRICT buf2 = buf_pool.get2(thread_rank);
        Size n1 = 0, n2 = 0;

        // Partition by group
        partition::partition_two_groups_simple(
            values.ptr, indices.ptr, len_sz, group_ids.ptr,
            buf1, n1, buf2, n2
        );

        if (SCL_UNLIKELY(n1 == 0 && n2 == 0)) {
            out_D_stats[static_cast<Index>(p)] = Real(0);
            out_p_values[static_cast<Index>(p)] = Real(1);
            return;
        }

        // Sort
        if (SCL_LIKELY(n1 > 1)) {
            scl::sort::sort(Array<T>(buf1, n1));
        }
        if (SCL_LIKELY(n2 > 1)) {
            scl::sort::sort(Array<T>(buf2, n2));
        }

        // Compute KS statistic
        double D = NAN;
        detail::compute_ks_sparse(
            buf1, n1, n1_total,
            buf2, n2, n2_total,
            D
        );

        out_D_stats[static_cast<Index>(p)] = static_cast<Real>(D);
        out_p_values[static_cast<Index>(p)] = detail::ks_pvalue(static_cast<Real>(D), n1_total, n2_total);
    });
}

} // namespace scl::kernel::stat::ks
