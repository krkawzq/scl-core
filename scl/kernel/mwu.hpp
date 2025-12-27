#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sort.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/vectorize.hpp"
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
}

// =============================================================================
// Core Algorithm
// =============================================================================

namespace detail {

template <typename T>
SCL_FORCE_INLINE void compute_rank_sum_sparse(
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

    Size na_neg = 0;
    while (na_neg < na_nz && a[na_neg] < T(0)) na_neg++;

    Size nb_neg = 0;
    while (nb_neg < nb_nz && b[nb_neg] < T(0)) nb_neg++;

    Size rank = 1;

    Size p1 = 0, p2 = 0;

    while (p1 < na_neg || p2 < nb_neg) {
        T v1 = (p1 < na_neg) ? a[p1] : std::numeric_limits<T>::max();
        T v2 = (p2 < nb_neg) ? b[p2] : std::numeric_limits<T>::max();
        T val = (v1 < v2) ? v1 : v2;

        Size count1 = 0;
        while (p1 < na_neg && a[p1] == val) { count1++; p1++; }

        Size count2 = 0;
        while (p2 < nb_neg && b[p2] == val) { count2++; p2++; }

        Size t = count1 + count2;
        double avg_rank = static_cast<double>(rank) + static_cast<double>(t - 1) * 0.5;

        R1 += static_cast<double>(count1) * avg_rank;

        if (t > 1) {
            double td = static_cast<double>(t);
            tie_sum += (td * td * td - td);
        }

        rank += t;
    }

    if (total_zeros > 0) {
        double avg_rank = static_cast<double>(rank) + static_cast<double>(total_zeros - 1) * 0.5;
        R1 += static_cast<double>(a_zeros) * avg_rank;

        if (total_zeros > 1) {
            double tz = static_cast<double>(total_zeros);
            tie_sum += (tz * tz * tz - tz);
        }

        rank += total_zeros;
    }

    p1 = na_neg;
    p2 = nb_neg;

    while (p1 < na_nz || p2 < nb_nz) {
        T v1 = (p1 < na_nz) ? a[p1] : std::numeric_limits<T>::max();
        T v2 = (p2 < nb_nz) ? b[p2] : std::numeric_limits<T>::max();
        T val = (v1 < v2) ? v1 : v2;

        Size count1 = 0;
        while (p1 < na_nz && a[p1] == val) { count1++; p1++; }

        Size count2 = 0;
        while (p2 < nb_nz && b[p2] == val) { count2++; p2++; }

        Size t = count1 + count2;
        double avg_rank = static_cast<double>(rank) + static_cast<double>(t - 1) * 0.5;

        R1 += static_cast<double>(count1) * avg_rank;

        if (t > 1) {
            double td = static_cast<double>(t);
            tie_sum += (td * td * td - td);
        }

        rank += t;
    }

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

    MWUConstants(Size n1_total, Size n2_total)
        : n1d(static_cast<double>(n1_total))
        , n2d(static_cast<double>(n2_total))
        , N(n1d + n2d)
        , half_n1_n1p1(0.5 * n1d * (n1d + 1.0))
        , half_n1_n2(0.5 * n1d * n2d)
        , var_base(n1d * n2d / 12.0)
        , N_p1(N + 1.0)
        , N_Nm1(N * (N - 1.0))
    {}
};

SCL_FORCE_INLINE void compute_u_and_pvalue(
    double R1,
    double tie_sum,
    const MWUConstants& c,
    Real& out_u,
    Real& out_pval
) {
    double U = R1 - c.half_n1_n1p1;

    double tie_term = (c.N_Nm1 > config::EPS) ? (tie_sum / c.N_Nm1) : 0.0;
    double var = c.var_base * (c.N_p1 - tie_term);
    double sigma = (var > 0.0) ? std::sqrt(var) : 0.0;

    out_u = static_cast<Real>(U);

    if (sigma <= config::SIGMA_MIN) {
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
    Array<Real> out_log2_fc
) {
    const Index primary_dim = matrix.primary_dim();
    const Size N = static_cast<Size>(primary_dim);

    Size n1_total, n2_total;
    count_groups(group_ids, n1_total, n2_total);

    SCL_CHECK_ARG(n1_total > 0 && n2_total > 0,
                  "MWU: Both groups must have at least one member");

    const detail::MWUConstants c(n1_total, n2_total);

    // Find max row length for buffer allocation
    Size max_len = 0;
    for (Index i = 0; i < primary_dim; ++i) {
        Size len = static_cast<Size>(matrix.primary_length(i));
        if (len > max_len) max_len = len;
    }

    // Pre-allocate dual buffer pool for all threads
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    scl::threading::DualWorkspacePool<T> buf_pool;
    buf_pool.init(n_threads, max_len);

    scl::threading::parallel_for(Size(0), N, [&](size_t p, size_t thread_rank) {
        const Index idx = static_cast<Index>(p);
        const Index len = matrix.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);

        auto values = matrix.primary_values(idx);
        auto indices = matrix.primary_indices(idx);

        // Get pre-allocated buffers for this thread
        T* SCL_RESTRICT buf1 = buf_pool.get1(thread_rank);
        T* SCL_RESTRICT buf2 = buf_pool.get2(thread_rank);
        Size n1 = 0, n2 = 0;

        double sum1 = 0.0, sum2 = 0.0;

        // Partition values by group - 4-way unroll
        Size k = 0;
        for (; k + 4 <= len_sz; k += 4) {
            SCL_UNROLL_FULL
            for (Size j = 0; j < 4; ++j) {
                Index sec_idx = indices.ptr[k + j];
                int32_t g = group_ids[sec_idx];
                T val = values.ptr[k + j];

                if (g == 0) {
                    buf1[n1++] = val;
                    sum1 += static_cast<double>(val);
                } else if (g == 1) {
                    buf2[n2++] = val;
                    sum2 += static_cast<double>(val);
                }
            }
        }

        for (; k < len_sz; ++k) {
            Index sec_idx = indices.ptr[k];
            int32_t g = group_ids[sec_idx];
            T val = values.ptr[k];

            if (g == 0) {
                buf1[n1++] = val;
                sum1 += static_cast<double>(val);
            } else if (g == 1) {
                buf2[n2++] = val;
                sum2 += static_cast<double>(val);
            }
        }

        double mean1 = sum1 / c.n1d;
        double mean2 = sum2 / c.n2d;
        out_log2_fc[p] = static_cast<Real>(std::log2((mean2 + config::EPS) / (mean1 + config::EPS)));

        if (n1 == 0 && n2 == 0) {
            out_u_stats[p] = Real(0);
            out_p_values[p] = Real(1);
            return;
        }

        // Sort using Highway VQSort (SIMD-optimized, single-threaded)
        if (n1 > 1) {
            scl::sort::sort(Array<T>(buf1, n1));
        }
        if (n2 > 1) {
            scl::sort::sort(Array<T>(buf2, n2));
        }

        double R1, tie_sum;
        detail::compute_rank_sum_sparse(
            buf1, n1, n1_total,
            buf2, n2, n2_total,
            R1, tie_sum
        );

        detail::compute_u_and_pvalue(R1, tie_sum, c, out_u_stats[p], out_p_values[p]);
    });
}

} // namespace scl::kernel::mwu

