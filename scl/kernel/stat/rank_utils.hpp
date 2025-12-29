#pragma once

#include "scl/kernel/stat/stat_base.hpp"
#include "scl/core/algo.hpp"

#include <limits>

// =============================================================================
// FILE: scl/kernel/stat/rank_utils.hpp
// BRIEF: Ranking utilities for non-parametric statistical tests
// =============================================================================

namespace scl::kernel::stat::rank {

// =============================================================================
// Find Negative Boundary
// =============================================================================

template <typename T>
SCL_FORCE_INLINE SCL_HOT Size find_negative_boundary(const T* SCL_RESTRICT arr, Size n) {
    if (SCL_UNLIKELY(n == 0)) return 0;

    // Fast path: check endpoints
    if (SCL_LIKELY(arr[0] >= T(0))) return 0;
    if (SCL_UNLIKELY(arr[n - 1] < T(0))) return n;

    if (n >= config::BINARY_SEARCH_THRESHOLD) {
        // Binary search for first element >= 0
        const T* result = scl::algo::lower_bound(arr, arr + n, T(0));
        return static_cast<Size>(result - arr);
    } else {
        // 4-way unrolled linear scan for small arrays
        Size k = 0;
        for (; k + 4 <= n; k += 4) {
            if (arr[k + 0] >= T(0)) return k + 0;
            if (arr[k + 1] >= T(0)) return k + 1;
            if (arr[k + 2] >= T(0)) return k + 2;
            if (arr[k + 3] >= T(0)) return k + 3;
        }

        for (; k < n; ++k) {
            if (arr[k] >= T(0)) return k;
        }

        return n;
    }
}

// =============================================================================
// Merge with Tie Handling (for MWU rank sum)
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void merge_with_ties(
    const T* SCL_RESTRICT a, Size& pa, Size pa_end,
    const T* SCL_RESTRICT b, Size& pb, Size pb_end,
    Size& rank,
    double& R1,
    double& tie_sum
) {
    while (pa < pa_end || pb < pb_end) {
        // Prefetch ahead
        if (SCL_LIKELY(pa + config::PREFETCH_DISTANCE < pa_end)) {
            SCL_PREFETCH_READ(&a[pa + config::PREFETCH_DISTANCE], 0);
        }
        if (SCL_LIKELY(pb + config::PREFETCH_DISTANCE < pb_end)) {
            SCL_PREFETCH_READ(&b[pb + config::PREFETCH_DISTANCE], 0);
        }

        T v1 = (pa < pa_end) ? a[pa] : std::numeric_limits<T>::max();
        T v2 = (pb < pb_end) ? b[pb] : std::numeric_limits<T>::max();
        T val = (v1 < v2) ? v1 : v2;

        Size count1 = 0;
        while (pa < pa_end && a[pa] == val) { count1++; pa++; }

        Size count2 = 0;
        while (pb < pb_end && b[pb] == val) { count2++; pb++; }

        Size t = count1 + count2;
        double avg_rank = static_cast<double>(rank) + static_cast<double>(t - 1) * 0.5;

        R1 += static_cast<double>(count1) * avg_rank;

        if (SCL_UNLIKELY(t > 1)) {
            auto td = static_cast<double>(t);
            // t^3 - t = t * (t^2 - 1)
            tie_sum += td * (td * td - 1.0);
        }

        rank += t;
    }
}

// =============================================================================
// Merge for KS Test (track cumulative distribution difference)
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void merge_for_ks(
    const T* SCL_RESTRICT a, Size na,
    const T* SCL_RESTRICT b, Size nb,
    double inv_n1, double inv_n2,
    double& out_D
) {
    Size pa = 0, pb = 0;
    double F1 = 0.0, F2 = 0.0;
    double max_diff = 0.0;

    while (pa < na || pb < nb) {
        if (SCL_LIKELY(pa + config::PREFETCH_DISTANCE < na)) {
            SCL_PREFETCH_READ(&a[pa + config::PREFETCH_DISTANCE], 0);
        }
        if (SCL_LIKELY(pb + config::PREFETCH_DISTANCE < nb)) {
            SCL_PREFETCH_READ(&b[pb + config::PREFETCH_DISTANCE], 0);
        }

        T v1 = (pa < na) ? a[pa] : std::numeric_limits<T>::max();
        T v2 = (pb < nb) ? b[pb] : std::numeric_limits<T>::max();
        T val = (v1 < v2) ? v1 : v2;

        // Count ties in each group
        while (pa < na && a[pa] == val) { pa++; }
        while (pb < nb && b[pb] == val) { pb++; }

        // Update ECDFs
        F1 = static_cast<double>(pa) * inv_n1;
        F2 = static_cast<double>(pb) * inv_n2;

        double diff = std::abs(F1 - F2);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    out_D = max_diff;
}

// =============================================================================
// Compute Rank Sum for Sparse Data (handles zeros explicitly)
// =============================================================================

template <typename T>
SCL_FORCE_INLINE SCL_HOT void compute_rank_sum_sparse(
    const T* SCL_RESTRICT a, Size na_nz, Size n1_total,
    const T* SCL_RESTRICT b, Size nb_nz, Size n2_total,
    double& out_R1,
    double& out_tie_sum
) {
    double R1 = 0.0;
    double tie_sum = 0.0;

    Size a_zeros = n1_total - na_nz;
    Size b_zeros = n2_total - nb_nz;
    Size total_zeros = a_zeros + b_zeros;

    // Binary search for negative boundary
    Size na_neg = find_negative_boundary(a, na_nz);
    Size nb_neg = find_negative_boundary(b, nb_nz);

    Size rank = 1;
    Size p1 = 0, p2 = 0;

    // Merge negative values
    merge_with_ties(a, p1, na_neg, b, p2, nb_neg, rank, R1, tie_sum);

    // Handle zeros
    if (SCL_UNLIKELY(total_zeros > 0)) {
        double avg_rank = static_cast<double>(rank) + static_cast<double>(total_zeros - 1) * 0.5;
        R1 += static_cast<double>(a_zeros) * avg_rank;

        if (SCL_UNLIKELY(total_zeros > 1)) {
            auto tz = static_cast<double>(total_zeros);
            tie_sum += tz * (tz * tz - 1.0);
        }

        rank += total_zeros;
    }

    // Merge positive values
    p1 = na_neg;
    p2 = nb_neg;
    merge_with_ties(a, p1, na_nz, b, p2, nb_nz, rank, R1, tie_sum);

    out_R1 = R1;
    out_tie_sum = tie_sum;
}

// =============================================================================
// Compute Rank Sums for K Groups (for Kruskal-Wallis)
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void compute_rank_sums_k_groups(
    const T* SCL_RESTRICT sorted_values,
    const Size* SCL_RESTRICT group_assignments,
    Size n_total,
    Size n_groups,
    double* SCL_RESTRICT out_rank_sums,
    double& out_tie_sum
) {
    // Initialize rank sums
    for (Size g = 0; g < n_groups; ++g) {
        out_rank_sums[g] = 0.0;
    }
    double tie_sum = 0.0;

    Size i = 0;
    Size rank = 1;

    while (i < n_total) {
        T val = sorted_values[i];
        Size tie_start = i;

        // Count ties
        while (i < n_total && sorted_values[i] == val) {
            ++i;
        }

        Size tie_count = i - tie_start;
        double avg_rank = static_cast<double>(rank) + static_cast<double>(tie_count - 1) * 0.5;

        // Assign average rank to each group
        for (Size j = tie_start; j < i; ++j) {
            Size g = group_assignments[j];
            if (g < n_groups) {
                out_rank_sums[g] += avg_rank;
            }
        }

        // Update tie correction
        if (tie_count > 1) {
            auto t = static_cast<double>(tie_count);
            tie_sum += t * (t * t - 1.0);
        }

        rank += tie_count;
    }

    out_tie_sum = tie_sum;
}

// =============================================================================
// Compute U and P-value (MWU specific)
// =============================================================================

SCL_FORCE_INLINE SCL_HOT void compute_u_and_pvalue(
    double R1,
    double tie_sum,
    const MWUConstants& c,
    Real& out_u,
    Real& out_pval
) {
    double U = R1 - c.half_n1_n1p1;

    // Tie-corrected variance
    double tie_term = tie_sum * c.inv_N_Nm1;
    double var = c.var_base * (c.N_p1 - tie_term);
    double sigma = SCL_LIKELY(var > 0.0) ? std::sqrt(var) : 0.0;

    out_u = static_cast<Real>(U);

    if (SCL_UNLIKELY(sigma <= config::SIGMA_MIN)) {
        out_pval = Real(1);
    } else {
        double z_numer = U - c.half_n1_n2;

        // Continuity correction
        double correction = (z_numer > 0.5) ? 0.5 : ((z_numer < -0.5) ? -0.5 : -z_numer);
        z_numer += correction;

        double z = z_numer / sigma;
        out_pval = pvalue::normal_two_sided(static_cast<Real>(z));
    }
}

// =============================================================================
// Compute AUROC from U statistic
// =============================================================================

SCL_FORCE_INLINE Real compute_auroc(double U, double n1, double n2) {
    double n1_n2 = n1 * n2;
    if (SCL_UNLIKELY(n1_n2 <= 0.0)) {
        return Real(0.5);
    }
    return static_cast<Real>(U / n1_n2);
}

} // namespace scl::kernel::stat::rank
