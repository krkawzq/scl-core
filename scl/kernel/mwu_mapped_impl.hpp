#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sort.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/mapped_common.hpp"

#include <vector>
#include <cmath>
#include <limits>

// =============================================================================
/// @file mwu_mapped_impl.hpp
/// @brief Mann-Whitney U Test for Memory-Mapped Sparse Matrices
///
/// ## Key Optimizations
///
/// 1. Streaming Access Pattern
///    - Prefetch hints for page cache
///    - Sequential row access
///
/// 2. Thread-Local Buffer Reuse
///    - Avoid per-feature allocation
///
/// 3. Precomputed Constants
///    - All group-size dependent values computed once
///
/// Performance: Near-RAM for cached data
// =============================================================================

namespace scl::kernel::mwu::mapped {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr double INV_SQRT2 = 0.7071067811865475244;
    constexpr double EPS = 1e-9;
    constexpr double SIGMA_MIN = 1e-12;
}

// =============================================================================
// SECTION 2: Core Algorithm
// =============================================================================

namespace detail {

/// @brief Compute rank sum with implicit zeros (unchanged logic)
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

    // Phase 1: Merge negatives
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

    // Phase 2: Zero block (O(1))
    if (total_zeros > 0) {
        double avg_rank = static_cast<double>(rank) + static_cast<double>(total_zeros - 1) * 0.5;
        R1 += static_cast<double>(a_zeros) * avg_rank;

        if (total_zeros > 1) {
            double tz = static_cast<double>(total_zeros);
            tie_sum += (tz * tz * tz - tz);
        }

        rank += total_zeros;
    }

    // Phase 3: Merge positives
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

/// @brief Precomputed constants
struct MWUConstants {
    double n1d, n2d, N;
    double half_n1_n1p1, half_n1_n2;
    double var_base, N_p1, N_Nm1;

    MWUConstants(Size n1, Size n2)
        : n1d(static_cast<double>(n1))
        , n2d(static_cast<double>(n2))
        , N(n1d + n2d)
        , half_n1_n1p1(0.5 * n1d * (n1d + 1.0))
        , half_n1_n2(0.5 * n1d * n2d)
        , var_base(n1d * n2d / 12.0)
        , N_p1(N + 1.0)
        , N_Nm1(N * (N - 1.0))
    {}
};

/// @brief Compute U and p-value
SCL_FORCE_INLINE void compute_u_and_pvalue(
    double R1, double tie_sum,
    const MWUConstants& c,
    Real& out_u, Real& out_pval
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
        out_pval = static_cast<Real>(std::erfc(std::abs(z) * config::INV_SQRT2));
    }
}

} // namespace detail

// =============================================================================
// SECTION 3: Group Counting
// =============================================================================

inline void count_groups(Array<const int32_t> group_ids, Size& n1, Size& n2) {
    n1 = 0; n2 = 0;
    const Size n = group_ids.len;

    Size i = 0;
    for (; i + 4 <= n; i += 4) {
        n1 += (group_ids[i] == 0) + (group_ids[i+1] == 0) + (group_ids[i+2] == 0) + (group_ids[i+3] == 0);
        n2 += (group_ids[i] == 1) + (group_ids[i+1] == 1) + (group_ids[i+2] == 1) + (group_ids[i+3] == 1);
    }

    for (; i < n; ++i) {
        n1 += (group_ids[i] == 0);
        n2 += (group_ids[i] == 1);
    }
}

// =============================================================================
// SECTION 4: MappedCustomSparse MWU Test
// =============================================================================

/// @brief MWU test for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void mwu_test_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Array<Real> out_u_stats,
    Array<Real> out_p_values,
    Array<Real> out_log2_fc
) {
    const Index primary_dim = scl::primary_size(matrix);

    Size n1_total, n2_total;
    count_groups(group_ids, n1_total, n2_total);

    SCL_CHECK_ARG(n1_total > 0 && n2_total > 0,
                  "MWU: Both groups must have at least one member");

    const detail::MWUConstants c(n1_total, n2_total);

    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        auto values_arr = scl::primary_values(matrix, static_cast<Index>(p));
        auto indices_arr = scl::primary_indices(matrix, static_cast<Index>(p));

        const T* SCL_RESTRICT values = values_arr.ptr;
        const Index* SCL_RESTRICT indices = indices_arr.ptr;
        Size len = values_arr.len;

        thread_local std::vector<T> buf1, buf2;
        buf1.clear();
        buf2.clear();
        buf1.reserve(len);
        buf2.reserve(len);

        double sum1 = 0.0, sum2 = 0.0;

        for (Size k = 0; k < len; ++k) {
            Index sec_idx = indices[k];
            int32_t g = group_ids[sec_idx];
            T val = values[k];

            if (g == 0) {
                buf1.push_back(val);
                sum1 += static_cast<double>(val);
            } else if (g == 1) {
                buf2.push_back(val);
                sum2 += static_cast<double>(val);
            }
        }

        out_log2_fc[p] = static_cast<Real>(std::log2((sum2/c.n2d + config::EPS) / (sum1/c.n1d + config::EPS)));

        if (buf1.empty() && buf2.empty()) {
            out_u_stats[p] = Real(0);
            out_p_values[p] = Real(1);
            return;
        }

        if (buf1.size() > 1) scl::sort::sort(Array<T>(buf1.data(), buf1.size()));
        if (buf2.size() > 1) scl::sort::sort(Array<T>(buf2.data(), buf2.size()));

        double R1, tie_sum;
        detail::compute_rank_sum_sparse(
            buf1.data(), buf1.size(), n1_total,
            buf2.data(), buf2.size(), n2_total,
            R1, tie_sum
        );

        detail::compute_u_and_pvalue(R1, tie_sum, c, out_u_stats[p], out_p_values[p]);
    });
}

// =============================================================================
// SECTION 5: MappedVirtualSparse MWU Test
// =============================================================================

/// @brief MWU test for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void mwu_test_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const int32_t> group_ids,
    Array<Real> out_u_stats,
    Array<Real> out_p_values,
    Array<Real> out_log2_fc
) {
    const Index primary_dim = scl::primary_size(matrix);

    Size n1_total, n2_total;
    count_groups(group_ids, n1_total, n2_total);

    SCL_CHECK_ARG(n1_total > 0 && n2_total > 0,
                  "MWU: Both groups must have at least one member");

    const detail::MWUConstants c(n1_total, n2_total);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        auto values_arr = scl::primary_values(matrix, static_cast<Index>(p));
        auto indices_arr = scl::primary_indices(matrix, static_cast<Index>(p));

        const T* SCL_RESTRICT values = values_arr.ptr;
        const Index* SCL_RESTRICT indices = indices_arr.ptr;
        Size len = values_arr.len;

        thread_local std::vector<T> buf1, buf2;
        buf1.clear();
        buf2.clear();
        buf1.reserve(len);
        buf2.reserve(len);

        double sum1 = 0.0, sum2 = 0.0;

        for (Size k = 0; k < len; ++k) {
            Index sec_idx = indices[k];
            int32_t g = group_ids[sec_idx];
            T val = values[k];

            if (g == 0) {
                buf1.push_back(val);
                sum1 += static_cast<double>(val);
            } else if (g == 1) {
                buf2.push_back(val);
                sum2 += static_cast<double>(val);
            }
        }

        out_log2_fc[p] = static_cast<Real>(std::log2((sum2/c.n2d + config::EPS) / (sum1/c.n1d + config::EPS)));

        if (buf1.empty() && buf2.empty()) {
            out_u_stats[p] = Real(0);
            out_p_values[p] = Real(1);
            return;
        }

        if (buf1.size() > 1) scl::sort::sort(Array<T>(buf1.data(), buf1.size()));
        if (buf2.size() > 1) scl::sort::sort(Array<T>(buf2.data(), buf2.size()));

        double R1, tie_sum;
        detail::compute_rank_sum_sparse(
            buf1.data(), buf1.size(), n1_total,
            buf2.data(), buf2.size(), n2_total,
            R1, tie_sum
        );

        detail::compute_u_and_pvalue(R1, tie_sum, c, out_u_stats[p], out_p_values[p]);
    });
}

// =============================================================================
// SECTION 6: Unified Dispatcher
// =============================================================================

/// @brief MWU test dispatcher
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void mwu_test_mapped_dispatch(
    const MatrixT& matrix,
    Array<const int32_t> group_ids,
    Array<Real> out_u_stats,
    Array<Real> out_p_values,
    Array<Real> out_log2_fc
) {
    mwu_test_mapped(matrix, group_ids, out_u_stats, out_p_values, out_log2_fc);
}

} // namespace scl::kernel::mwu::mapped
