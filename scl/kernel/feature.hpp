#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// Backend implementations
#include "scl/kernel/feature_fast_impl.hpp"

#include <cmath>

// =============================================================================
/// @file feature.hpp
/// @brief Feature Selection and Gene Statistics
///
/// ## Supported Operations
///
/// 1. Clipped Moments
///    - Robust variance with outlier clipping (Seurat V3)
///    - Formula: mu = (1/N) * sum(min(X, theta))
///
/// 2. Standard Moments
///    - Basic mean/variance with ddof support
///    - var = (1/(N-ddof)) * (sum_sq - sum*mu)
///
/// 3. Detection Rate
///    - Fraction of non-zero elements: nnz/N
///
/// 4. Dispersion
///    - Fano factor: variance/mean
///
/// ## Backend Dispatch
///
/// - MappedSparseLike -> feature_mapped_impl.hpp
/// - CustomSparseLike -> feature_fast_impl.hpp
/// - VirtualSparseLike -> feature_fast_impl.hpp
/// - Generic -> This file (fallback)
///
/// ## Performance
///
/// All operations are single-pass, O(nnz) complexity.
/// SIMD optimized with 4-way unrolling.
// =============================================================================

namespace scl::kernel::feature {

// =============================================================================
// SECTION 1: Generic Implementation (Fallback)
// =============================================================================

namespace detail {

/// @brief Generic clipped moments
template <typename MatrixT>
    requires AnySparse<MatrixT>
void clipped_moments_generic(
    const MatrixT& matrix,
    Array<const Real> clip_vals,
    Array<Real> out_means,
    Array<Real> out_vars
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Real N = static_cast<Real>(scl::secondary_size(matrix));
    const Real N_minus_1 = N - Real(1);

    SCL_CHECK_DIM(clip_vals.size() >= static_cast<Size>(primary_dim), "Clip vals mismatch");
    SCL_CHECK_DIM(out_means.size() >= static_cast<Size>(primary_dim), "Output mean mismatch");
    SCL_CHECK_DIM(out_vars.size() >= static_cast<Size>(primary_dim), "Output var mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Real clip = clip_vals[p];
        auto vals = scl::primary_values(matrix, static_cast<Index>(p));

        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();

        auto v_clip = s::Set(d, clip);
        auto v_sum = s::Zero(d);
        auto v_ssq = s::Zero(d);

        Size k = 0;
        for (; k + lanes <= vals.size(); k += lanes) {
            auto v = s::Min(s::Load(d, vals.ptr + k), v_clip);
            v_sum = s::Add(v_sum, v);
            v_ssq = s::MulAdd(v, v, v_ssq);
        }

        Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
        Real sum_sq = s::GetLane(s::SumOfLanes(d, v_ssq));

        for (; k < vals.size(); ++k) {
            Real v = std::min(static_cast<Real>(vals[k]), clip);
            sum += v;
            sum_sq += v * v;
        }

        Real mu = sum / N;
        Real var = Real(0);
        if (N > Real(1)) {
            var = (sum_sq - N * mu * mu) / N_minus_1;
        }
        if (var < Real(0)) var = Real(0);

        out_means[p] = mu;
        out_vars[p] = var;
    });
}

/// @brief Generic standard moments
template <typename MatrixT>
    requires AnySparse<MatrixT>
void standard_moments_generic(
    const MatrixT& matrix,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Real N = static_cast<Real>(scl::secondary_size(matrix));
    const Real denom = N - static_cast<Real>(ddof);

    SCL_CHECK_DIM(out_means.size() >= static_cast<Size>(primary_dim), "Output mean mismatch");
    SCL_CHECK_DIM(out_vars.size() >= static_cast<Size>(primary_dim), "Output var mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        auto vals = scl::primary_values(matrix, static_cast<Index>(p));

        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();

        auto v_sum = s::Zero(d);
        auto v_ssq = s::Zero(d);

        Size k = 0;
        for (; k + lanes <= vals.size(); k += lanes) {
            auto v = s::Load(d, vals.ptr + k);
            v_sum = s::Add(v_sum, v);
            v_ssq = s::MulAdd(v, v, v_ssq);
        }

        Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
        Real sum_sq = s::GetLane(s::SumOfLanes(d, v_ssq));

        for (; k < vals.size(); ++k) {
            Real v = vals[k];
            sum += v;
            sum_sq += v * v;
        }

        Real mu = sum / N;
        Real var = (denom > Real(0)) ? ((sum_sq - sum * mu) / denom) : Real(0);
        if (var < Real(0)) var = Real(0);

        out_means[p] = mu;
        out_vars[p] = var;
    });
}

/// @brief Generic detection rate
template <typename MatrixT>
    requires AnySparse<MatrixT>
void detection_rate_generic(
    const MatrixT& matrix,
    Array<Real> out_rates
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Real inv_N = Real(1) / static_cast<Real>(scl::secondary_size(matrix));

    SCL_CHECK_DIM(out_rates.size() >= static_cast<Size>(primary_dim), "Output rates mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index len = scl::primary_length(matrix, static_cast<Index>(p));
        out_rates[p] = static_cast<Real>(len) * inv_N;
    });
}

} // namespace detail

// =============================================================================
// SECTION 2: Public API
// =============================================================================

/// @brief Compute clipped mean and variance
///
/// For each row/column, computes statistics on values clipped at threshold.
/// Used by Seurat V3 for highly variable gene selection.
///
/// @param matrix Input sparse matrix (any backend)
/// @param clip_vals Clipping thresholds [size = primary_dim], PRE-ALLOCATED
/// @param out_means Output means [size = primary_dim], PRE-ALLOCATED
/// @param out_vars Output variances [size = primary_dim], PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void clipped_moments(
    const MatrixT& matrix,
    Array<const Real> clip_vals,
    Array<Real> out_means,
    Array<Real> out_vars
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR> ||
                  CustomSparseLike<MatrixT, IsCSR> ||
                  VirtualSparseLike<MatrixT, IsCSR>) {
        fast::clipped_moments_fast<MatrixT, IsCSR>(matrix, clip_vals, out_means, out_vars);
    } else {
        detail::clipped_moments_generic(matrix, clip_vals, out_means, out_vars);
    }
}

/// @brief Compute standard mean and variance
///
/// @param matrix Input sparse matrix (any backend)
/// @param out_means Output means [size = primary_dim], PRE-ALLOCATED
/// @param out_vars Output variances [size = primary_dim], PRE-ALLOCATED
/// @param ddof Delta degrees of freedom (default 1 for sample variance)
template <typename MatrixT>
    requires AnySparse<MatrixT>
void standard_moments(
    const MatrixT& matrix,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof = 1
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR> ||
                  CustomSparseLike<MatrixT, IsCSR> ||
                  VirtualSparseLike<MatrixT, IsCSR>) {
        fast::standard_moments_fast<MatrixT, IsCSR>(matrix, out_means, out_vars, ddof);
    } else {
        detail::standard_moments_generic(matrix, out_means, out_vars, ddof);
    }
}

/// @brief Compute detection rate (fraction of non-zero elements)
///
/// @param matrix Input sparse matrix (any backend)
/// @param out_rates Output rates [size = primary_dim], values in [0, 1], PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void detection_rate(
    const MatrixT& matrix,
    Array<Real> out_rates
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR> ||
                  CustomSparseLike<MatrixT, IsCSR> ||
                  VirtualSparseLike<MatrixT, IsCSR>) {
        fast::detection_rate_fast<MatrixT, IsCSR>(matrix, out_rates);
    } else {
        detail::detection_rate_generic(matrix, out_rates);
    }
}

/// @brief Compute dispersion (Fano factor: variance/mean)
///
/// @param means Input means, PRE-ALLOCATED
/// @param vars Input variances, PRE-ALLOCATED
/// @param out_dispersion Output dispersion values, PRE-ALLOCATED
void dispersion(
    Array<const Real> means,
    Array<const Real> vars,
    Array<Real> out_dispersion
) {
    fast::dispersion_simd(means, vars, out_dispersion);
}

} // namespace scl::kernel::feature
