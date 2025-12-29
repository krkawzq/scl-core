#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sort.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"

#include <cmath>
#include <limits>

// =============================================================================
// FILE: scl/kernel/mwu.hpp
// BRIEF: Mann-Whitney U Test with optimized algorithms
// =============================================================================

namespace scl::kernel::mwu {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr double INV_SQRT2 = 0.7071067811865475244;
    constexpr double EPS = 1e-9;
    constexpr double SIGMA_MIN = 1e-12;
    constexpr Size PREFETCH_DISTANCE = 16;
    constexpr Size BINARY_SEARCH_THRESHOLD = 32;
}

// =============================================================================
// Core Algorithm
// =============================================================================

namespace detail {

// =============================================================================
// Helper: Find negative boundary using adaptive strategy
// =============================================================================

template <typename T>
SCL_FORCE_INLINE SCL_HOT Size find_negative_boundary(const T* SCL_RESTRICT arr, Size n) {
    if (SCL_UNLIKELY(n == 0)) return 0;

    // Fast path: check endpoints first
    if (SCL_LIKELY(arr[0] >= T(0))) return 0;
    if (SCL_UNLIKELY(arr[n - 1] < T(0))) return n;

    if (n >= config::BINARY_SEARCH_THRESHOLD) {
        // Binary search for large arrays: find first element >= 0
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
// Helper: Merge with tie handling and prefetch
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
            // Optimized: t^3 - t = t * (t^2 - 1)
            tie_sum += td * (td * td - 1.0);
        }

        rank += t;
    }
}

// =============================================================================
// Main rank sum computation
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

    // Use binary search for negative boundary (O(log n) instead of O(n))
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
            // Optimized: t^3 - t = t * (t^2 - 1)
            tie_sum += tz * (tz * tz - 1.0);
        }

        rank += total_zeros;
    }

    // Continue merging positive values
    p1 = na_neg;
    p2 = nb_neg;
    merge_with_ties(a, p1, na_nz, b, p2, nb_nz, rank, R1, tie_sum);

    out_R1 = R1;
    out_tie_sum = tie_sum;
}

struct MWUConstants {
    double n1d;
    double n2d;
    double N;
    double half_n1_n1p1;
    double half_n1_n2;
    double var_base;
    double N_p1;
    double N_Nm1;
    double inv_N_Nm1;  // Precomputed reciprocal for division optimization

    MWUConstants(Size n1_total, Size n2_total)
        : n1d(static_cast<double>(n1_total))
        , n2d(static_cast<double>(n2_total))
        , N(n1d + n2d)
        , half_n1_n1p1(0.5 * n1d * (n1d + 1.0))
        , half_n1_n2(0.5 * n1d * n2d)
        , var_base(n1d * n2d / 12.0)
        , N_p1(N + 1.0)
        , N_Nm1(N * (N - 1.0))
        , inv_N_Nm1((N_Nm1 > config::EPS) ? (1.0 / N_Nm1) : 0.0)
    {}
};

SCL_FORCE_INLINE SCL_HOT void compute_u_and_pvalue(
    double R1,
    double tie_sum,
    const MWUConstants& c,
    Real& out_u,
    Real& out_pval
) {
    double U = R1 - c.half_n1_n1p1;

    // Use precomputed reciprocal: multiplication is 10-20x faster than division
    double tie_term = tie_sum * c.inv_N_Nm1;
    double var = c.var_base * (c.N_p1 - tie_term);
    double sigma = SCL_LIKELY(var > 0.0) ? std::sqrt(var) : 0.0;

    out_u = static_cast<Real>(U);

    if (SCL_UNLIKELY(sigma <= config::SIGMA_MIN)) {
        out_pval = Real(1);
    } else {
        double z_numer = U - c.half_n1_n2;

        double correction = (z_numer > 0.5) ? 0.5 : ((z_numer < -0.5) ? -0.5 : -z_numer);
        z_numer += correction;

        double z = z_numer / sigma;
        double p_val = std::erfc(std::abs(z) * config::INV_SQRT2);

        out_pval = static_cast<Real>(p_val);
    }
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
// MWU Test
// =============================================================================

template <typename T, bool IsCSR>
void mwu_test(
    const Sparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Array<Real> out_u_stats,
    Array<Real> out_p_values,
    Array<Real> out_log2_fc,
    Array<Real> out_auroc = Array<Real>()  // Optional AUROC output
) {
    const Index primary_dim = matrix.primary_dim();
    const Size N = static_cast<Size>(primary_dim);

    Size n1_total{}, n2_total{};
    count_groups(group_ids, n1_total, n2_total);

    SCL_CHECK_ARG(n1_total > 0 && n2_total > 0,
                  "MWU: Both groups must have at least one member");

    const detail::MWUConstants c(n1_total, n2_total);

    // Find max row length for buffer allocation
    Size max_len = 0;
    for (Index i = 0; i < primary_dim; ++i) {
        Size len = static_cast<Size>(matrix.primary_length_unsafe(i));
        if (len > max_len) max_len = len;
    }

    // Pre-allocate dual buffer pool for all threads
    const Size n_threads = scl::threading::get_num_threads_runtime();
    scl::threading::DualWorkspacePool<T> buf_pool;
    buf_pool.init(n_threads, max_len);

    scl::threading::parallel_for(Size(0), N, [&](size_t p, size_t thread_rank) {
        const auto idx = static_cast<Index>(p);
        const Index len = matrix.primary_length_unsafe(idx);
        const Size len_sz = static_cast<Size>(len);

        auto values = matrix.primary_values_unsafe(idx);
        auto indices = matrix.primary_indices_unsafe(idx);

        // Get pre-allocated buffers for this thread
        T* SCL_RESTRICT buf1 = buf_pool.get1(thread_rank);
        T* SCL_RESTRICT buf2 = buf_pool.get2(thread_rank);
        Size n1 = 0, n2 = 0;

        double sum1 = 0.0, sum2 = 0.0;

        // Partition values by group - 4-way unroll with prefetch
        Size k = 0;
        for (; k + 4 <= len_sz; k += 4) {
            // Prefetch ahead for indirect group_ids access
            if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < len_sz)) {
                SCL_PREFETCH_READ(&group_ids[indices.ptr[k + config::PREFETCH_DISTANCE]], 0);
                SCL_PREFETCH_READ(&values.ptr[k + config::PREFETCH_DISTANCE], 0);
            }

            SCL_UNROLL_FULL
            for (Size j = 0; j < 4; ++j) {
                Index sec_idx = indices.ptr[k + j];
                int32_t g = group_ids[sec_idx];
                T val = values.ptr[k + j];

                if (SCL_LIKELY(g == 0)) {
                    buf1[n1++] = val;
                    sum1 += static_cast<double>(val);
                } else if (SCL_LIKELY(g == 1)) {
                    buf2[n2++] = val;
                    sum2 += static_cast<double>(val);
                }
            }
        }

        for (; k < len_sz; ++k) {
            Index sec_idx = indices.ptr[k];
            int32_t g = group_ids[sec_idx];
            T val = values.ptr[k];

            if (SCL_LIKELY(g == 0)) {
                buf1[n1++] = val;
                sum1 += static_cast<double>(val);
            } else if (SCL_LIKELY(g == 1)) {
                buf2[n2++] = val;
                sum2 += static_cast<double>(val);
            }
        }

        double mean1 = sum1 / c.n1d;
        double mean2 = sum2 / c.n2d;
        out_log2_fc[static_cast<Index>(p)] = static_cast<Real>(std::log2((mean2 + config::EPS) / (mean1 + config::EPS)));

        if (SCL_UNLIKELY(n1 == 0 && n2 == 0)) {
            out_u_stats[static_cast<Index>(p)] = Real(0);
            out_p_values[static_cast<Index>(p)] = Real(1);
            return;
        }

        // Sort using Highway VQSort (SIMD-optimized, single-threaded)
        if (SCL_UNLIKELY(n1 > 1)) {
            scl::sort::sort(Array<T>(buf1, n1));
        }
        if (SCL_UNLIKELY(n2 > 1)) {
            scl::sort::sort(Array<T>(buf2, n2));
        }

        double R1{}, tie_sum{};
        detail::compute_rank_sum_sparse(
            buf1, n1, n1_total,
            buf2, n2, n2_total,
            R1, tie_sum
        );

        detail::compute_u_and_pvalue(R1, tie_sum, c, out_u_stats[static_cast<Index>(p)], out_p_values[static_cast<Index>(p)]);

        // Optionally compute AUROC
        if (out_auroc.ptr != nullptr) {
            auto U = static_cast<double>(out_u_stats[static_cast<Index>(p)]);
            double auroc_val = (c.n1d * c.n2d > 0.0) ? (U / (c.n1d * c.n2d)) : 0.5;
            out_auroc[static_cast<Index>(p)] = static_cast<Real>(auroc_val);
        }
    });
}

} // namespace scl::kernel::mwu

