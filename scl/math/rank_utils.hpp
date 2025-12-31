#pragma once

#include "scl/core/type.hpp"
#include "scl/core/macro.hpp"
#include "scl/math/stat_base.hpp"

#include <limits>
#include <cmath>

// =============================================================================
// FILE: scl/math/rank_utils.hpp
// DESCRIPTION: Ranking utilities for non-parametric statistical tests
//              (Mann-Whitney U, Kruskal-Wallis, Kolmogorov-Smirnov, AUROC)
// =============================================================================

namespace scl::math {

// =============================================================================
// Boundary Search
// =============================================================================

/// @brief Finds the boundary between negative and non-negative values in sorted array
/// @tparam T Arithmetic type
/// @param[in] arr Sorted array (ascending order)
/// @param[in] n Array length
/// @return Index of first element >= 0, or n if all elements are negative
/// @note Uses binary search for large arrays (>= BINARY_SEARCH_THRESHOLD)
/// @note Uses unrolled linear scan for small arrays
/// @pre Array must be sorted in ascending order
template<Arithmetic T>
[[nodiscard]]
SCL_FORCE_INLINE
auto find_negative_boundary(const T* SCL_RESTRICT arr, Size n) -> Size {
    if (n == 0) [[unlikely]] {
        return 0;
    }

    // Fast path: check endpoints
    if (arr[0] >= T(0)) [[likely]] {
        return 0;
    }
    if (arr[n - 1] < T(0)) [[unlikely]] {
        return n;
    }

    // Choose strategy based on array size
    if (n >= stat_constants::BINARY_SEARCH_THRESHOLD) {
        // Binary search for first element >= 0
        Size left = 0;
        Size right = n;

        while (left < right) {
            Size mid = left + (right - left) / 2;

            if (arr[mid] < T(0)) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        return left;
    } else {
        // 4-way unrolled linear scan for small arrays
        Size k = 0;

        for (; k + 4 <= n; k += 4) {
            if (arr[k + 0] >= T(0)) return k + 0;
            if (arr[k + 1] >= T(0)) return k + 1;
            if (arr[k + 2] >= T(0)) return k + 2;
            if (arr[k + 3] >= T(0)) return k + 3;
        }

        // Handle remainder
        for (; k < n; ++k) {
            if (arr[k] >= T(0)) return k;
        }

        return n;
    }
}

// =============================================================================
// Rank Merging with Tie Handling
// =============================================================================

/// @brief Merges two sorted arrays computing rank sums with tie correction
/// @tparam T Arithmetic type
/// @param[in] a First sorted array
/// @param[in,out] pa Current position in array a
/// @param[in] pa_end End position in array a
/// @param[in] b Second sorted array
/// @param[in,out] pb Current position in array b
/// @param[in] pb_end End position in array b
/// @param[in,out] rank Current rank (1-indexed)
/// @param[in,out] R1 Rank sum for group 1 (array a)
/// @param[in,out] tie_sum Sum of (t³ - t) for tie correction
/// @note Uses prefetching for cache optimization
/// @note Handles tied values by assigning average ranks
/// @pre Both arrays must be sorted in ascending order
template<Arithmetic T>
SCL_FORCE_INLINE
auto merge_with_ties(
    const T* SCL_RESTRICT a, Size& pa, Size pa_end,
    const T* SCL_RESTRICT b, Size& pb, Size pb_end,
    Size& rank,
    double& R1,
    double& tie_sum
) -> void {
    while (pa < pa_end || pb < pb_end) {
        // Prefetch ahead for better cache utilization
        if (pa + stat_constants::PREFETCH_DISTANCE < pa_end) [[likely]] {
            SCL_PREFETCH(&a[pa + stat_constants::PREFETCH_DISTANCE], 0, 0);
        }
        if (pb + stat_constants::PREFETCH_DISTANCE < pb_end) [[likely]] {
            SCL_PREFETCH(&b[pb + stat_constants::PREFETCH_DISTANCE], 0, 0);
        }

        // Get next values (use max as sentinel)
        const T v1 = (pa < pa_end) ? a[pa] : std::numeric_limits<T>::max();
        const T v2 = (pb < pb_end) ? b[pb] : std::numeric_limits<T>::max();
        const T val = (v1 < v2) ? v1 : v2;

        // Count ties in group 1
        Size count1 = 0;
        while (pa < pa_end && a[pa] == val) {
            count1++;
            pa++;
        }

        // Count ties in group 2
        Size count2 = 0;
        while (pb < pb_end && b[pb] == val) {
            count2++;
            pb++;
        }

        // Total tie count
        const Size t = count1 + count2;

        // Average rank for this value: rank + (t-1)/2
        const double avg_rank = static_cast<double>(rank) + static_cast<double>(t - 1) * 0.5;

        // Add to rank sum for group 1
        R1 += static_cast<double>(count1) * avg_rank;

        // Update tie correction: sum of (t³ - t) = t*(t² - 1)
        if (t > 1) [[unlikely]] {
            const double td = static_cast<double>(t);
            tie_sum += td * (td * td - 1.0);
        }

        rank += t;
    }
}

// =============================================================================
// Kolmogorov-Smirnov Test Utilities
// =============================================================================

/// @brief Merges two sorted arrays computing maximum ECDF difference (for KS test)
/// @tparam T Arithmetic type
/// @param[in] a First sorted array
/// @param[in] na Length of array a
/// @param[in] b Second sorted array
/// @param[in] nb Length of array b
/// @param[in] inv_n1 Reciprocal of na (1/na)
/// @param[in] inv_n2 Reciprocal of nb (1/nb)
/// @param[out] out_D Maximum absolute difference between ECDFs
/// @note ECDF = Empirical Cumulative Distribution Function
/// @note Uses prefetching for cache optimization
/// @pre Both arrays must be sorted in ascending order
template<Arithmetic T>
SCL_FORCE_INLINE
auto merge_for_ks(
    const T* SCL_RESTRICT a, Size na,
    const T* SCL_RESTRICT b, Size nb,
    double inv_n1, double inv_n2,
    double& out_D
) -> void {
    Size pa = 0;
    Size pb = 0;
    double F1 = 0.0;
    double F2 = 0.0;
    double max_diff = 0.0;

    while (pa < na || pb < nb) {
        // Prefetch ahead
        if (pa + stat_constants::PREFETCH_DISTANCE < na) [[likely]] {
            SCL_PREFETCH(&a[pa + stat_constants::PREFETCH_DISTANCE], 0, 0);
        }
        if (pb + stat_constants::PREFETCH_DISTANCE < nb) [[likely]] {
            SCL_PREFETCH(&b[pb + stat_constants::PREFETCH_DISTANCE], 0, 0);
        }

        // Get next values (use max as sentinel)
        const T v1 = (pa < na) ? a[pa] : std::numeric_limits<T>::max();
        const T v2 = (pb < nb) ? b[pb] : std::numeric_limits<T>::max();
        const T val = (v1 < v2) ? v1 : v2;

        // Count ties in each group
        while (pa < na && a[pa] == val) {
            pa++;
        }
        while (pb < nb && b[pb] == val) {
            pb++;
        }

        // Update ECDFs
        F1 = static_cast<double>(pa) * inv_n1;
        F2 = static_cast<double>(pb) * inv_n2;

        // Track maximum difference
        const double diff = std::abs(F1 - F2);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    out_D = max_diff;
}

// =============================================================================
// Sparse Data Rank Sum
// =============================================================================

/// @brief Computes rank sum for sparse data with explicit zero handling
/// @tparam T Arithmetic type
/// @param[in] a Non-zero values from group 1 (sorted)
/// @param[in] na_nz Number of non-zero values in group 1
/// @param[in] n1_total Total size of group 1 (including zeros)
/// @param[in] b Non-zero values from group 2 (sorted)
/// @param[in] nb_nz Number of non-zero values in group 2
/// @param[in] n2_total Total size of group 2 (including zeros)
/// @param[out] out_R1 Rank sum for group 1
/// @param[out] out_tie_sum Tie correction term: sum of (t³ - t)
/// @note Zeros are treated as tied values at rank (neg_count + zero_count/2)
/// @note Efficient for sparse matrices where most values are zero
/// @pre Arrays must be sorted in ascending order
template<Arithmetic T>
SCL_FORCE_INLINE
auto compute_rank_sum_sparse(
    const T* SCL_RESTRICT a, Size na_nz, Size n1_total,
    const T* SCL_RESTRICT b, Size nb_nz, Size n2_total,
    double& out_R1,
    double& out_tie_sum
) -> void {
    double R1 = 0.0;
    double tie_sum = 0.0;

    // Count zeros in each group
    const Size a_zeros = n1_total - na_nz;
    const Size b_zeros = n2_total - nb_nz;
    const Size total_zeros = a_zeros + b_zeros;

    // Find boundary between negative and non-negative values
    const Size na_neg = find_negative_boundary(a, na_nz);
    const Size nb_neg = find_negative_boundary(b, nb_nz);

    Size rank = 1;
    Size p1 = 0;
    Size p2 = 0;

    // Phase 1: Merge negative values
    merge_with_ties(a, p1, na_neg, b, p2, nb_neg, rank, R1, tie_sum);

    // Phase 2: Handle zeros as tied values
    if (total_zeros > 0) [[unlikely]] {
        const double avg_rank = static_cast<double>(rank) +
                               static_cast<double>(total_zeros - 1) * 0.5;
        R1 += static_cast<double>(a_zeros) * avg_rank;

        // Tie correction for zeros
        if (total_zeros > 1) [[unlikely]] {
            const double tz = static_cast<double>(total_zeros);
            tie_sum += tz * (tz * tz - 1.0);
        }

        rank += total_zeros;
    }

    // Phase 3: Merge positive values
    p1 = na_neg;
    p2 = nb_neg;
    merge_with_ties(a, p1, na_nz, b, p2, nb_nz, rank, R1, tie_sum);

    out_R1 = R1;
    out_tie_sum = tie_sum;
}

// =============================================================================
// K-Group Rank Sums (for Kruskal-Wallis)
// =============================================================================

/// @brief Computes rank sums for K groups (Kruskal-Wallis H test)
/// @tparam T Arithmetic type
/// @param[in] sorted_values Combined sorted values from all groups
/// @param[in] group_assignments Group ID for each value (0 to n_groups-1)
/// @param[in] n_total Total number of values
/// @param[in] n_groups Number of groups
/// @param[out] out_rank_sums Array of rank sums for each group (size: n_groups)
/// @param[out] out_tie_sum Tie correction term: sum of (t³ - t)
/// @note Handles tied values by assigning average ranks
/// @note out_rank_sums must be pre-allocated with size >= n_groups
/// @pre sorted_values must be sorted in ascending order
/// @pre group_assignments[i] must be in range [0, n_groups)
template<Arithmetic T>
SCL_FORCE_INLINE
auto compute_rank_sums_k_groups(
    const T* SCL_RESTRICT sorted_values,
    const Size* SCL_RESTRICT group_assignments,
    Size n_total,
    Size n_groups,
    double* SCL_RESTRICT out_rank_sums,
    double& out_tie_sum
) -> void {
    // Initialize rank sums to zero
    for (Size g = 0; g < n_groups; ++g) {
        out_rank_sums[g] = 0.0;
    }

    double tie_sum = 0.0;
    Size i = 0;
    Size rank = 1;

    while (i < n_total) {
        const T val = sorted_values[i];
        const Size tie_start = i;

        // Count ties
        while (i < n_total && sorted_values[i] == val) {
            ++i;
        }

        const Size tie_count = i - tie_start;
        const double avg_rank = static_cast<double>(rank) +
                               static_cast<double>(tie_count - 1) * 0.5;

        // Assign average rank to each group
        for (Size j = tie_start; j < i; ++j) {
            const Size g = group_assignments[j];
            if (g < n_groups) [[likely]] {
                out_rank_sums[g] += avg_rank;
            }
        }

        // Update tie correction
        if (tie_count > 1) [[unlikely]] {
            const double t = static_cast<double>(tie_count);
            tie_sum += t * (t * t - 1.0);
        }

        rank += tie_count;
    }

    out_tie_sum = tie_sum;
}

// =============================================================================
// Mann-Whitney U Statistics
// =============================================================================

/// @brief Computes Mann-Whitney U statistic and two-sided p-value
/// @tparam Real Floating-point type
/// @param[in] R1 Rank sum for group 1
/// @param[in] tie_sum Tie correction term: sum of (t³ - t)
/// @param[in] c MWU precomputed constants
/// @param[out] out_u Mann-Whitney U statistic
/// @param[out] out_pval Two-sided p-value (normal approximation with continuity correction)
/// @note Uses normal approximation with tie correction
/// @note Applies continuity correction for better small-sample accuracy
/// @note Returns p=1.0 if variance is too small (degenerate case)
template<Arithmetic Real>
SCL_FORCE_INLINE
auto compute_u_and_pvalue(
    double R1,
    double tie_sum,
    const MWUConstants& c,
    Real& out_u,
    Real& out_pval
) -> void {
    // U = R1 - n1*(n1+1)/2
    const double U = R1 - c.half_n1_n1p1;

    // Tie-corrected variance: Var = n1*n2/12 * (N+1 - tie_sum/(N*(N-1)))
    const double tie_term = tie_sum * c.inv_N_Nm1;
    const double var = c.var_base * (c.N_p1 - tie_term);

    double sigma;
    if (var > 0.0) [[likely]] {
        sigma = std::sqrt(var);
    } else {
        sigma = 0.0;
    }

    out_u = static_cast<Real>(U);

    if (sigma <= stat_constants::SIGMA_MIN) [[unlikely]] {
        // Degenerate case: zero variance
        out_pval = Real(1);
    } else {
        // z-score with continuity correction
        double z_numer = U - c.half_n1_n2;

        // Continuity correction: move z towards 0 by 0.5
        const double correction = (z_numer > 0.5) ? 0.5 :
                                 ((z_numer < -0.5) ? -0.5 : -z_numer);
        z_numer += correction;

        const double z = z_numer / sigma;
        out_pval = pvalue::normal_two_sided(static_cast<Real>(z));
    }
}

// =============================================================================
// AUROC Computation
// =============================================================================

/// @brief Computes Area Under ROC Curve (AUROC) from Mann-Whitney U statistic
/// @tparam Real Floating-point type
/// @param[in] U Mann-Whitney U statistic
/// @param[in] n1 Size of group 1
/// @param[in] n2 Size of group 2
/// @return AUROC value in [0, 1], or 0.5 if n1*n2 = 0
/// @note AUROC = U / (n1 * n2)
/// @note AUROC = 0.5 indicates no discrimination
/// @note AUROC > 0.5 indicates group 1 tends to have higher values
/// @note AUROC < 0.5 indicates group 2 tends to have higher values
template<Arithmetic Real>
[[nodiscard]]
SCL_FORCE_INLINE
constexpr
auto compute_auroc(double U, double n1, double n2) -> Real {
    const double n1_n2 = n1 * n2;
    if (n1_n2 <= 0.0) [[unlikely]] {
        return Real(0.5);
    }
    return static_cast<Real>(U / n1_n2);
}

} // namespace scl::math
