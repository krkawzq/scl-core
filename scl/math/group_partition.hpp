#pragma once

#include "scl/core/type.hpp"
#include "scl/core/macro.hpp"
#include "scl/math/stat_base.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/math/group_partition.hpp
// DESCRIPTION: Optimized group partitioning utilities for statistical tests
//              Supports two-group and K-group partitioning with moment accumulation
// =============================================================================

namespace scl::math {

// =============================================================================
// Two-Group Partitioning
// =============================================================================

/// @brief Partitions values into two groups with sum accumulation
/// @tparam T Arithmetic type
/// @param[in] values Input values array
/// @param[in] indices Cell/sample indices for group lookup
/// @param[in] len Number of values
/// @param[in] group_ids Group assignment for each cell (0 or 1)
/// @param[out] buf1 Output buffer for group 0
/// @param[out] n1 Number of elements in group 0
/// @param[out] buf2 Output buffer for group 1
/// @param[out] n2 Number of elements in group 1
/// @param[out] sum1 Sum of values in group 0
/// @param[out] sum2 Sum of values in group 1
/// @note Uses 4-way loop unrolling and prefetching for cache optimization
/// @note buf1 and buf2 must be pre-allocated with size >= len
/// @pre group_ids must contain only 0, 1, or invalid group IDs
template<Arithmetic T>
SCL_FORCE_INLINE
auto partition_two_groups(
    const T* SCL_RESTRICT values,
    const Index* SCL_RESTRICT indices,
    Size len,
    const int32_t* SCL_RESTRICT group_ids,
    T* SCL_RESTRICT buf1, Size& n1,
    T* SCL_RESTRICT buf2, Size& n2,
    double& sum1, double& sum2
) -> void {
    n1 = 0;
    n2 = 0;
    sum1 = 0.0;
    sum2 = 0.0;

    Size k = 0;

    // 4-way unrolled main loop
    for (; k + 4 <= len; k += 4) {
        // Prefetch ahead for indirect access
        if (k + stat_constants::PREFETCH_DISTANCE < len) [[likely]] {
            SCL_PREFETCH(&group_ids[indices[k + stat_constants::PREFETCH_DISTANCE]], 0, 0);
            SCL_PREFETCH(&values[k + stat_constants::PREFETCH_DISTANCE], 0, 0);
        }

        // Unroll 4 iterations
        for (Size j = 0; j < 4; ++j) {
            const Index sec_idx = indices[k + j];
            const int32_t g = group_ids[sec_idx];
            const T val = values[k + j];

            if (g == 0) [[likely]] {
                buf1[n1++] = val;
                sum1 += static_cast<double>(val);
            } else if (g == 1) [[likely]] {
                buf2[n2++] = val;
                sum2 += static_cast<double>(val);
            }
            // Ignore invalid group IDs
        }
    }

    // Handle remainder
    for (; k < len; ++k) {
        const Index sec_idx = indices[k];
        const int32_t g = group_ids[sec_idx];
        const T val = values[k];

        if (g == 0) [[likely]] {
            buf1[n1++] = val;
            sum1 += static_cast<double>(val);
        } else if (g == 1) [[likely]] {
            buf2[n2++] = val;
            sum2 += static_cast<double>(val);
        }
    }
}

/// @brief Partitions values into two groups with moment accumulation (for t-test)
/// @tparam T Arithmetic type
/// @param[in] values Input values array
/// @param[in] indices Cell/sample indices for group lookup
/// @param[in] len Number of values
/// @param[in] group_ids Group assignment for each cell (0 or 1)
/// @param[out] buf1 Output buffer for group 0
/// @param[out] n1 Number of elements in group 0
/// @param[out] buf2 Output buffer for group 1
/// @param[out] n2 Number of elements in group 1
/// @param[out] sum1 Sum of values in group 0
/// @param[out] sum_sq1 Sum of squared values in group 0
/// @param[out] sum2 Sum of values in group 1
/// @param[out] sum_sq2 Sum of squared values in group 1
/// @note Computes sum and sum-of-squares for variance calculation
/// @note Uses 4-way loop unrolling and prefetching
/// @pre buf1 and buf2 must be pre-allocated with size >= len
template<Arithmetic T>
SCL_FORCE_INLINE
auto partition_two_groups_moments(
    const T* SCL_RESTRICT values,
    const Index* SCL_RESTRICT indices,
    Size len,
    const int32_t* SCL_RESTRICT group_ids,
    T* SCL_RESTRICT buf1, Size& n1,
    T* SCL_RESTRICT buf2, Size& n2,
    double& sum1, double& sum_sq1,
    double& sum2, double& sum_sq2
) -> void {
    n1 = 0;
    n2 = 0;
    sum1 = 0.0;
    sum_sq1 = 0.0;
    sum2 = 0.0;
    sum_sq2 = 0.0;

    Size k = 0;

    // 4-way unrolled main loop
    for (; k + 4 <= len; k += 4) {
        if (k + stat_constants::PREFETCH_DISTANCE < len) [[likely]] {
            SCL_PREFETCH(&group_ids[indices[k + stat_constants::PREFETCH_DISTANCE]], 0, 0);
            SCL_PREFETCH(&values[k + stat_constants::PREFETCH_DISTANCE], 0, 0);
        }

        for (Size j = 0; j < 4; ++j) {
            const Index sec_idx = indices[k + j];
            const int32_t g = group_ids[sec_idx];
            const T val = values[k + j];
            const double v = static_cast<double>(val);

            if (g == 0) [[likely]] {
                buf1[n1++] = val;
                sum1 += v;
                sum_sq1 += v * v;
            } else if (g == 1) [[likely]] {
                buf2[n2++] = val;
                sum2 += v;
                sum_sq2 += v * v;
            }
        }
    }

    // Handle remainder
    for (; k < len; ++k) {
        const Index sec_idx = indices[k];
        const int32_t g = group_ids[sec_idx];
        const T val = values[k];
        const double v = static_cast<double>(val);

        if (g == 0) [[likely]] {
            buf1[n1++] = val;
            sum1 += v;
            sum_sq1 += v * v;
        } else if (g == 1) [[likely]] {
            buf2[n2++] = val;
            sum2 += v;
            sum_sq2 += v * v;
        }
    }
}

/// @brief Simple two-group partition without accumulation
/// @tparam T Arithmetic type
/// @param[in] values Input values array
/// @param[in] indices Cell/sample indices for group lookup
/// @param[in] len Number of values
/// @param[in] group_ids Group assignment for each cell (0 or 1)
/// @param[out] buf1 Output buffer for group 0
/// @param[out] n1 Number of elements in group 0
/// @param[out] buf2 Output buffer for group 1
/// @param[out] n2 Number of elements in group 1
/// @note Fastest variant when only partitioning is needed (no statistics)
/// @note Uses 4-way loop unrolling and prefetching
/// @pre buf1 and buf2 must be pre-allocated with size >= len
template<Arithmetic T>
SCL_FORCE_INLINE
auto partition_two_groups_simple(
    const T* SCL_RESTRICT values,
    const Index* SCL_RESTRICT indices,
    Size len,
    const int32_t* SCL_RESTRICT group_ids,
    T* SCL_RESTRICT buf1, Size& n1,
    T* SCL_RESTRICT buf2, Size& n2
) -> void {
    n1 = 0;
    n2 = 0;

    Size k = 0;

    // 4-way unrolled main loop
    for (; k + 4 <= len; k += 4) {
        if (k + stat_constants::PREFETCH_DISTANCE < len) [[likely]] {
            SCL_PREFETCH(&group_ids[indices[k + stat_constants::PREFETCH_DISTANCE]], 0, 0);
            SCL_PREFETCH(&values[k + stat_constants::PREFETCH_DISTANCE], 0, 0);
        }

        for (Size j = 0; j < 4; ++j) {
            const Index sec_idx = indices[k + j];
            const int32_t g = group_ids[sec_idx];
            const T val = values[k + j];

            if (g == 0) [[likely]] {
                buf1[n1++] = val;
            } else if (g == 1) [[likely]] {
                buf2[n2++] = val;
            }
        }
    }

    // Handle remainder
    for (; k < len; ++k) {
        const Index sec_idx = indices[k];
        const int32_t g = group_ids[sec_idx];
        const T val = values[k];

        if (g == 0) [[likely]] {
            buf1[n1++] = val;
        } else if (g == 1) [[likely]] {
            buf2[n2++] = val;
        }
    }
}

// =============================================================================
// K-Group Partitioning
// =============================================================================

/// @brief Partitions values into K groups with moment accumulation (for ANOVA)
/// @tparam T Arithmetic type
/// @param[in] values Input values array
/// @param[in] indices Cell/sample indices for group lookup
/// @param[in] len Number of values
/// @param[in] group_ids Group assignment for each cell (0 to n_groups-1)
/// @param[in] n_groups Number of groups
/// @param[out] counts Number of elements in each group (size: n_groups)
/// @param[out] sums Sum of values in each group (size: n_groups)
/// @param[out] sum_sqs Sum of squared values in each group (size: n_groups)
/// @note Computes sufficient statistics for ANOVA and Kruskal-Wallis tests
/// @note Uses 4-way loop unrolling and prefetching
/// @pre counts, sums, sum_sqs must be pre-allocated with size >= n_groups
/// @pre group_ids must contain values in range [0, n_groups) or invalid IDs
template<Arithmetic T>
SCL_FORCE_INLINE
auto partition_k_groups_moments(
    const T* SCL_RESTRICT values,
    const Index* SCL_RESTRICT indices,
    Size len,
    const int32_t* SCL_RESTRICT group_ids,
    Size n_groups,
    Size* SCL_RESTRICT counts,
    double* SCL_RESTRICT sums,
    double* SCL_RESTRICT sum_sqs
) -> void {
    // Initialize arrays
    for (Size g = 0; g < n_groups; ++g) {
        counts[g] = 0;
        sums[g] = 0.0;
        sum_sqs[g] = 0.0;
    }

    Size k = 0;

    // 4-way unrolled main loop
    for (; k + 4 <= len; k += 4) {
        if (k + stat_constants::PREFETCH_DISTANCE < len) [[likely]] {
            SCL_PREFETCH(&group_ids[indices[k + stat_constants::PREFETCH_DISTANCE]], 0, 0);
            SCL_PREFETCH(&values[k + stat_constants::PREFETCH_DISTANCE], 0, 0);
        }

        for (Size j = 0; j < 4; ++j) {
            const Index sec_idx = indices[k + j];
            const int32_t g = group_ids[sec_idx];

            if (g >= 0 && static_cast<Size>(g) < n_groups) [[likely]] {
                const T val = values[k + j];
                const double v = static_cast<double>(val);

                counts[g]++;
                sums[g] += v;
                sum_sqs[g] += v * v;
            }
        }
    }

    // Handle remainder
    for (; k < len; ++k) {
        const Index sec_idx = indices[k];
        const int32_t g = group_ids[sec_idx];

        if (g >= 0 && static_cast<Size>(g) < n_groups) [[likely]] {
            const T val = values[k];
            const double v = static_cast<double>(val);

            counts[g]++;
            sums[g] += v;
            sum_sqs[g] += v * v;
        }
    }
}

/// @brief Partitions values into K groups and copies to output buffer
/// @tparam T Arithmetic type
/// @param[in] values Input values array
/// @param[in] indices Cell/sample indices for group lookup
/// @param[in] len Number of values
/// @param[in] group_ids Group assignment for each cell (0 to n_groups-1)
/// @param[in] n_groups Number of groups
/// @param[out] out_values Output values array (valid elements only)
/// @param[out] out_groups Output group IDs array (valid elements only)
/// @param[out] out_total Total number of valid elements
/// @note Filters out invalid group IDs and creates combined sorted input
/// @note Used for Kruskal-Wallis test preparation
/// @pre out_values and out_groups must be pre-allocated with size >= len
template<Arithmetic T>
SCL_FORCE_INLINE
auto partition_k_groups_to_buffer(
    const T* SCL_RESTRICT values,
    const Index* SCL_RESTRICT indices,
    Size len,
    const int32_t* SCL_RESTRICT group_ids,
    Size n_groups,
    T* SCL_RESTRICT out_values,
    Size* SCL_RESTRICT out_groups,
    Size& out_total
) -> void {
    Size total = 0;
    Size k = 0;

    // 4-way unrolled main loop
    for (; k + 4 <= len; k += 4) {
        if (k + stat_constants::PREFETCH_DISTANCE < len) [[likely]] {
            SCL_PREFETCH(&group_ids[indices[k + stat_constants::PREFETCH_DISTANCE]], 0, 0);
            SCL_PREFETCH(&values[k + stat_constants::PREFETCH_DISTANCE], 0, 0);
        }

        for (Size j = 0; j < 4; ++j) {
            const Index sec_idx = indices[k + j];
            const int32_t g = group_ids[sec_idx];

            if (g >= 0 && static_cast<Size>(g) < n_groups) [[likely]] {
                out_values[total] = values[k + j];
                out_groups[total] = static_cast<Size>(g);
                total++;
            }
        }
    }

    // Handle remainder
    for (; k < len; ++k) {
        const Index sec_idx = indices[k];
        const int32_t g = group_ids[sec_idx];

        if (g >= 0 && static_cast<Size>(g) < n_groups) [[likely]] {
            out_values[total] = values[k];
            out_groups[total] = static_cast<Size>(g);
            total++;
        }
    }

    out_total = total;
}

// =============================================================================
// Group Statistics Finalization
// =============================================================================

/// @brief Computes final group statistics from accumulated moments
/// @param[in] count Number of non-zero values observed
/// @param[in] sum Sum of observed values
/// @param[in] sum_sq Sum of squared observed values
/// @param[in] n_total Total group size (including zeros in sparse data)
/// @param[out] out_mean Group mean (including zeros)
/// @param[out] out_var Group variance (with optional degrees-of-freedom correction)
/// @param[in] ddof Degrees of freedom correction (default: 1 for Bessel's correction)
/// @note Handles sparse data by including implicit zeros in calculations
/// @note Variance is clamped to non-negative values (numerical stability)
/// @note Returns mean=0, var=0 if count=0 (empty group)
SCL_FORCE_INLINE
auto finalize_group_stats(
    Size count,
    double sum,
    double sum_sq,
    Size n_total,
    double& out_mean,
    double& out_var,
    int ddof = 1
) -> void {
    if (count == 0) [[unlikely]] {
        out_mean = 0.0;
        out_var = 0.0;
        return;
    }

    // Mean including zeros
    out_mean = sum / static_cast<double>(n_total);

    // Variance calculation with sparse data handling
    const Size effective_n = n_total;
    const double mean_from_count = sum / static_cast<double>(count);

    // Variance numerator from non-zero values
    double var_numer = sum_sq - static_cast<double>(count) * mean_from_count * mean_from_count;

    // Add contribution from zeros
    const Size n_zeros = n_total - count;
    var_numer += static_cast<double>(n_zeros) * out_mean * out_mean;

    // Apply degrees of freedom correction
    if (effective_n > static_cast<Size>(ddof)) [[likely]] {
        out_var = var_numer / static_cast<double>(effective_n - ddof);

        // Clamp negative values (numerical errors)
        if (out_var < 0.0) [[unlikely]] {
            out_var = 0.0;
        }
    } else {
        out_var = 0.0;
    }
}

} // namespace scl::math
