#pragma once

#include "scl/kernel/stat/stat_base.hpp"
#include "scl/core/type.hpp"

// =============================================================================
// FILE: scl/kernel/stat/group_partition.hpp
// BRIEF: Optimized group partitioning for statistical kernels
// =============================================================================

namespace scl::kernel::stat::partition {

// =============================================================================
// Two-Group Partition with Sum Accumulation
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void partition_two_groups(
    const T* SCL_RESTRICT values,
    const Index* SCL_RESTRICT indices,
    Size len,
    const int32_t* SCL_RESTRICT group_ids,
    T* SCL_RESTRICT buf1, Size& n1,
    T* SCL_RESTRICT buf2, Size& n2,
    double& sum1, double& sum2
) {
    n1 = 0;
    n2 = 0;
    sum1 = 0.0;
    sum2 = 0.0;

    Size k = 0;
    for (; k + 4 <= len; k += 4) {
        // Prefetch ahead for indirect access
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < len)) {
            SCL_PREFETCH_READ(&group_ids[indices[k + config::PREFETCH_DISTANCE]], 0);
            SCL_PREFETCH_READ(&values[k + config::PREFETCH_DISTANCE], 0);
        }

        SCL_UNROLL_FULL
        for (Size j = 0; j < 4; ++j) {
            Index sec_idx = indices[k + j];
            int32_t g = group_ids[sec_idx];
            T val = values[k + j];

            if (SCL_LIKELY(g == 0)) {
                buf1[n1++] = val;
                sum1 += static_cast<double>(val);
            } else if (SCL_LIKELY(g == 1)) {
                buf2[n2++] = val;
                sum2 += static_cast<double>(val);
            }
        }
    }

    for (; k < len; ++k) {
        Index sec_idx = indices[k];
        int32_t g = group_ids[sec_idx];
        T val = values[k];

        if (SCL_LIKELY(g == 0)) {
            buf1[n1++] = val;
            sum1 += static_cast<double>(val);
        } else if (SCL_LIKELY(g == 1)) {
            buf2[n2++] = val;
            sum2 += static_cast<double>(val);
        }
    }
}

// =============================================================================
// Two-Group Partition with Sum and SumSq Accumulation (for T-test)
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void partition_two_groups_moments(
    const T* SCL_RESTRICT values,
    const Index* SCL_RESTRICT indices,
    Size len,
    const int32_t* SCL_RESTRICT group_ids,
    T* SCL_RESTRICT buf1, Size& n1,
    T* SCL_RESTRICT buf2, Size& n2,
    double& sum1, double& sum_sq1,
    double& sum2, double& sum_sq2
) {
    n1 = 0;
    n2 = 0;
    sum1 = 0.0;
    sum_sq1 = 0.0;
    sum2 = 0.0;
    sum_sq2 = 0.0;

    Size k = 0;
    for (; k + 4 <= len; k += 4) {
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < len)) {
            SCL_PREFETCH_READ(&group_ids[indices[k + config::PREFETCH_DISTANCE]], 0);
            SCL_PREFETCH_READ(&values[k + config::PREFETCH_DISTANCE], 0);
        }

        SCL_UNROLL_FULL
        for (Size j = 0; j < 4; ++j) {
            Index sec_idx = indices[k + j];
            int32_t g = group_ids[sec_idx];
            T val = values[k + j];
            double v = static_cast<double>(val);

            if (SCL_LIKELY(g == 0)) {
                buf1[n1++] = val;
                sum1 += v;
                sum_sq1 += v * v;
            } else if (SCL_LIKELY(g == 1)) {
                buf2[n2++] = val;
                sum2 += v;
                sum_sq2 += v * v;
            }
        }
    }

    for (; k < len; ++k) {
        Index sec_idx = indices[k];
        int32_t g = group_ids[sec_idx];
        T val = values[k];
        double v = static_cast<double>(val);

        if (SCL_LIKELY(g == 0)) {
            buf1[n1++] = val;
            sum1 += v;
            sum_sq1 += v * v;
        } else if (SCL_LIKELY(g == 1)) {
            buf2[n2++] = val;
            sum2 += v;
            sum_sq2 += v * v;
        }
    }
}

// =============================================================================
// Simple Two-Group Partition (no accumulation)
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void partition_two_groups_simple(
    const T* SCL_RESTRICT values,
    const Index* SCL_RESTRICT indices,
    Size len,
    const int32_t* SCL_RESTRICT group_ids,
    T* SCL_RESTRICT buf1, Size& n1,
    T* SCL_RESTRICT buf2, Size& n2
) {
    n1 = 0;
    n2 = 0;

    Size k = 0;
    for (; k + 4 <= len; k += 4) {
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < len)) {
            SCL_PREFETCH_READ(&group_ids[indices[k + config::PREFETCH_DISTANCE]], 0);
            SCL_PREFETCH_READ(&values[k + config::PREFETCH_DISTANCE], 0);
        }

        SCL_UNROLL_FULL
        for (Size j = 0; j < 4; ++j) {
            Index sec_idx = indices[k + j];
            int32_t g = group_ids[sec_idx];
            T val = values[k + j];

            if (SCL_LIKELY(g == 0)) {
                buf1[n1++] = val;
            } else if (SCL_LIKELY(g == 1)) {
                buf2[n2++] = val;
            }
        }
    }

    for (; k < len; ++k) {
        Index sec_idx = indices[k];
        int32_t g = group_ids[sec_idx];
        T val = values[k];

        if (SCL_LIKELY(g == 0)) {
            buf1[n1++] = val;
        } else if (SCL_LIKELY(g == 1)) {
            buf2[n2++] = val;
        }
    }
}

// =============================================================================
// K-Group Partition with Moments (for ANOVA)
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void partition_k_groups_moments(
    const T* SCL_RESTRICT values,
    const Index* SCL_RESTRICT indices,
    Size len,
    const int32_t* SCL_RESTRICT group_ids,
    Size n_groups,
    Size* SCL_RESTRICT counts,
    double* SCL_RESTRICT sums,
    double* SCL_RESTRICT sum_sqs
) {
    // Initialize
    for (Size g = 0; g < n_groups; ++g) {
        counts[g] = 0;
        sums[g] = 0.0;
        sum_sqs[g] = 0.0;
    }

    Size k = 0;
    for (; k + 4 <= len; k += 4) {
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < len)) {
            SCL_PREFETCH_READ(&group_ids[indices[k + config::PREFETCH_DISTANCE]], 0);
            SCL_PREFETCH_READ(&values[k + config::PREFETCH_DISTANCE], 0);
        }

        SCL_UNROLL_FULL
        for (Size j = 0; j < 4; ++j) {
            Index sec_idx = indices[k + j];
            int32_t g = group_ids[sec_idx];

            if (SCL_LIKELY(g >= 0 && static_cast<Size>(g) < n_groups)) {
                T val = values[k + j];
                double v = static_cast<double>(val);

                counts[g]++;
                sums[g] += v;
                sum_sqs[g] += v * v;
            }
        }
    }

    for (; k < len; ++k) {
        Index sec_idx = indices[k];
        int32_t g = group_ids[sec_idx];

        if (SCL_LIKELY(g >= 0 && static_cast<Size>(g) < n_groups)) {
            T val = values[k];
            double v = static_cast<double>(val);

            counts[g]++;
            sums[g] += v;
            sum_sqs[g] += v * v;
        }
    }
}

// =============================================================================
// K-Group Partition to Buffers (for Kruskal-Wallis)
// =============================================================================

template <typename T>
SCL_FORCE_INLINE void partition_k_groups_to_buffer(
    const T* SCL_RESTRICT values,
    const Index* SCL_RESTRICT indices,
    Size len,
    const int32_t* SCL_RESTRICT group_ids,
    Size n_groups,
    T* SCL_RESTRICT out_values,
    Size* SCL_RESTRICT out_groups,
    Size& out_total
) {
    Size total = 0;

    Size k = 0;
    for (; k + 4 <= len; k += 4) {
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < len)) {
            SCL_PREFETCH_READ(&group_ids[indices[k + config::PREFETCH_DISTANCE]], 0);
            SCL_PREFETCH_READ(&values[k + config::PREFETCH_DISTANCE], 0);
        }

        SCL_UNROLL_FULL
        for (Size j = 0; j < 4; ++j) {
            Index sec_idx = indices[k + j];
            int32_t g = group_ids[sec_idx];

            if (SCL_LIKELY(g >= 0 && static_cast<Size>(g) < n_groups)) {
                out_values[total] = values[k + j];
                out_groups[total] = static_cast<Size>(g);
                total++;
            }
        }
    }

    for (; k < len; ++k) {
        Index sec_idx = indices[k];
        int32_t g = group_ids[sec_idx];

        if (SCL_LIKELY(g >= 0 && static_cast<Size>(g) < n_groups)) {
            out_values[total] = values[k];
            out_groups[total] = static_cast<Size>(g);
            total++;
        }
    }

    out_total = total;
}

// =============================================================================
// Compute Group Statistics from Moments
// =============================================================================

SCL_FORCE_INLINE void finalize_group_stats(
    Size count,
    double sum,
    double sum_sq,
    Size n_total,
    double& out_mean,
    double& out_var,
    int ddof = 1
) {
    if (SCL_UNLIKELY(count == 0)) {
        out_mean = 0.0;
        out_var = 0.0;
        return;
    }

    // Include zeros in mean calculation
    out_mean = sum / static_cast<double>(n_total);

    // Variance with Bessel's correction
    Size effective_n = n_total;
    double mean_from_count = sum / static_cast<double>(count);
    double var_numer = sum_sq - static_cast<double>(count) * mean_from_count * mean_from_count;

    // Adjust for zeros (zeros contribute to variance)
    Size n_zeros = n_total - count;
    var_numer += static_cast<double>(n_zeros) * out_mean * out_mean;

    if (effective_n > static_cast<Size>(ddof)) {
        out_var = var_numer / static_cast<double>(effective_n - ddof);
        if (out_var < 0.0) out_var = 0.0;  // Clamp numerical errors
    } else {
        out_var = 0.0;
    }
}

} // namespace scl::kernel::stat::partition
