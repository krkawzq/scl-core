#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/argsort.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/feature.hpp"

// Backend implementations
#include "scl/kernel/hvg_fast_impl.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

// =============================================================================
/// @file hvg.hpp
/// @brief Highly Variable Genes (HVG) Selection
///
/// ## Methods
///
/// 1. Dispersion-Based (Fano factor)
///    - dispersion = variance / mean
///    - Select top-K by dispersion
///
/// 2. Seurat V3 (VST)
///    - Clipped variance for robustness
///    - Select top-K by clipped variance
///
/// 3. Simple Variance
///    - Select top-K by raw variance
///
/// ## Optimizations
///
/// 1. Partial Sort
///    - O(n + k log k) instead of O(n log n)
///    - Uses nth_element + partial_sort
///
/// 2. SIMD Dispersion
///    - Vectorized variance/mean with safe division
///
/// 3. Feature Statistics Fast Path
///    - Uses feature_fast_impl for mean/variance
///
/// ## Backend Dispatch
///
/// - MappedSparseLike -> hvg_mapped_impl.hpp
/// - CustomSparseLike -> hvg_fast_impl.hpp
/// - VirtualSparseLike -> hvg_fast_impl.hpp
/// - Generic -> This file (fallback)
///
/// ## Complexity
///
/// Time: O(nnz) for stats + O(n + k log k) for selection
/// Space: O(n) for temporary arrays
// =============================================================================

namespace scl::kernel::hvg {

// =============================================================================
// SECTION 1: Generic Implementation (Fallback)
// =============================================================================

namespace detail {

/// @brief SIMD dispersion computation
inline void dispersion_simd(
    Array<const Real> means,
    Array<const Real> vars,
    Array<Real> out_dispersion
) {
    const Size n = means.len;

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    constexpr Real EPSILON = 1e-12;
    const auto v_eps = s::Set(d, EPSILON);
    const auto v_zero = s::Zero(d);

    Size k = 0;

    for (; k + lanes <= n; k += lanes) {
        auto v_mean = s::Load(d, means.ptr + k);
        auto v_var = s::Load(d, vars.ptr + k);

        auto mask = s::Gt(v_mean, v_eps);
        auto v_div = s::Div(v_var, v_mean);
        auto v_res = s::IfThenElse(mask, v_div, v_zero);

        s::Store(v_res, d, out_dispersion.ptr + k);
    }

    for (; k < n; ++k) {
        Real m = means[k];
        Real v = vars[k];
        out_dispersion[k] = (m > EPSILON) ? (v / m) : Real(0);
    }
}

/// @brief Partial sort for top-K (O(n + k log k))
inline void select_top_k_partial(
    Array<const Real> scores,
    Size k,
    Array<Index> out_indices,
    Array<uint8_t> out_mask
) {
    const Size n = scores.len;

    SCL_CHECK_ARG(k <= n, "HVG: k exceeds number of elements");
    SCL_CHECK_DIM(out_indices.len >= k, "HVG: Output indices too small");
    SCL_CHECK_DIM(out_mask.len >= n, "HVG: Output mask size mismatch");

    std::vector<Index> indices(n);
    for (Size i = 0; i < n; ++i) {
        indices[i] = static_cast<Index>(i);
    }

    // Partial sort: O(n + k log k)
    std::nth_element(
        indices.begin(),
        indices.begin() + static_cast<std::ptrdiff_t>(k),
        indices.end(),
        [&scores](Index a, Index b) {
            return scores[a] > scores[b];
        }
    );

    std::sort(
        indices.begin(),
        indices.begin() + static_cast<std::ptrdiff_t>(k),
        [&scores](Index a, Index b) {
            return scores[a] > scores[b];
        }
    );

    // Zero mask efficiently
    std::memset(out_mask.ptr, 0, n * sizeof(uint8_t));

    // Mark top-K
    for (Size i = 0; i < k; ++i) {
        Index idx = indices[i];
        out_indices[i] = idx;
        out_mask[idx] = 1;
    }
}

/// @brief Generic dispersion-based selection
template <typename MatrixT>
    requires AnySparse<MatrixT>
void select_by_dispersion_generic(
    const MatrixT& matrix,
    Size n_top,
    Array<Index> out_indices,
    Array<uint8_t> out_mask,
    Array<Real> out_dispersions
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size n = static_cast<Size>(primary_dim);

    SCL_CHECK_DIM(out_dispersions.len >= n, "HVG: Dispersions size mismatch");

    std::vector<Real> means(n);
    std::vector<Real> vars(n);

    scl::kernel::feature::standard_moments(
        matrix,
        Array<Real>(means.data(), n),
        Array<Real>(vars.data(), n),
        1
    );

    dispersion_simd(
        Array<const Real>(means.data(), n),
        Array<const Real>(vars.data(), n),
        out_dispersions
    );

    select_top_k_partial(
        Array<const Real>(out_dispersions.ptr, n),
        n_top,
        out_indices,
        out_mask
    );
}

/// @brief Generic VST-based selection
template <typename MatrixT>
    requires AnySparse<MatrixT>
void select_by_vst_generic(
    const MatrixT& matrix,
    Array<const Real> clip_vals,
    Size n_top,
    Array<Index> out_indices,
    Array<uint8_t> out_mask,
    Array<Real> out_variances
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size n = static_cast<Size>(primary_dim);

    SCL_CHECK_DIM(clip_vals.len >= n, "HVG: Clip vals size mismatch");
    SCL_CHECK_DIM(out_variances.len >= n, "HVG: Variances size mismatch");

    std::vector<Real> means(n);

    scl::kernel::feature::clipped_moments(
        matrix,
        clip_vals,
        Array<Real>(means.data(), n),
        out_variances
    );

    select_top_k_partial(
        Array<const Real>(out_variances.ptr, n),
        n_top,
        out_indices,
        out_mask
    );
}

/// @brief Generic variance-based selection
template <typename MatrixT>
    requires AnySparse<MatrixT>
void select_by_variance_generic(
    const MatrixT& matrix,
    Size n_top,
    Array<Index> out_indices,
    Array<uint8_t> out_mask
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size n = static_cast<Size>(primary_dim);

    std::vector<Real> means(n);
    std::vector<Real> vars(n);

    scl::kernel::feature::standard_moments(
        matrix,
        Array<Real>(means.data(), n),
        Array<Real>(vars.data(), n),
        1
    );

    select_top_k_partial(
        Array<const Real>(vars.data(), n),
        n_top,
        out_indices,
        out_mask
    );
}

} // namespace detail

// =============================================================================
// SECTION 2: Public API
// =============================================================================

/// @brief Select HVGs by dispersion (Fano factor: variance/mean)
///
/// @param matrix Input sparse matrix (any backend)
/// @param n_top Number of HVGs to select
/// @param out_indices Output indices [size >= n_top], PRE-ALLOCATED
/// @param out_mask Output mask [size = primary_dim], PRE-ALLOCATED
/// @param out_dispersions Output dispersion values [size = primary_dim], PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void select_by_dispersion(
    const MatrixT& matrix,
    Size n_top,
    Array<Index> out_indices,
    Array<uint8_t> out_mask,
    Array<Real> out_dispersions
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR> ||
                  CustomSparseLike<MatrixT, IsCSR> ||
                  VirtualSparseLike<MatrixT, IsCSR>) {
        fast::select_by_dispersion_fast<MatrixT, IsCSR>(
            matrix, n_top, out_indices, out_mask, out_dispersions
        );
    } else {
        detail::select_by_dispersion_generic(
            matrix, n_top, out_indices, out_mask, out_dispersions
        );
    }
}

/// @brief Select HVGs using Seurat V3 method (clipped variance)
///
/// @param matrix Input sparse matrix (any backend)
/// @param clip_vals Clipping thresholds [size = primary_dim]
/// @param n_top Number of HVGs to select
/// @param out_indices Output indices [size >= n_top], PRE-ALLOCATED
/// @param out_mask Output mask [size = primary_dim], PRE-ALLOCATED
/// @param out_variances Output clipped variances [size = primary_dim], PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void select_by_vst(
    const MatrixT& matrix,
    Array<const Real> clip_vals,
    Size n_top,
    Array<Index> out_indices,
    Array<uint8_t> out_mask,
    Array<Real> out_variances
) {
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR> ||
                  CustomSparseLike<MatrixT, IsCSR> ||
                  VirtualSparseLike<MatrixT, IsCSR>) {
        fast::select_by_vst_fast<MatrixT, IsCSR>(
            matrix, clip_vals, n_top, out_indices, out_mask, out_variances
        );
    } else {
        detail::select_by_vst_generic(
            matrix, clip_vals, n_top, out_indices, out_mask, out_variances
        );
    }
}

/// @brief Select HVGs by raw variance (simple method)
///
/// @param matrix Input sparse matrix (any backend)
/// @param n_top Number of HVGs to select
/// @param out_indices Output indices [size >= n_top], PRE-ALLOCATED
/// @param out_mask Output mask [size = primary_dim], PRE-ALLOCATED
template <typename MatrixT>
    requires AnySparse<MatrixT>
void select_by_variance(
    const MatrixT& matrix,
    Size n_top,
    Array<Index> out_indices,
    Array<uint8_t> out_mask
) {
    // Variance-based selection always uses generic path
    // (no special optimization needed beyond feature stats)
    detail::select_by_variance_generic(matrix, n_top, out_indices, out_mask);
}

/// @brief Helper: Select top K indices by descending scores
///
/// @param scores Input scores
/// @param k Number of top elements
/// @param out_indices Output indices [size >= k], PRE-ALLOCATED
/// @param out_mask Output binary mask [size = n], PRE-ALLOCATED
inline void select_top_k(
    Array<const Real> scores,
    Size k,
    Array<Index> out_indices,
    Array<uint8_t> out_mask
) {
    detail::select_top_k_partial(scores, k, out_indices, out_mask);
}

} // namespace scl::kernel::hvg
