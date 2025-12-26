#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/argsort.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/mapped_common.hpp"
#include "scl/kernel/feature_mapped_impl.hpp"

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstring>

// =============================================================================
/// @file hvg_mapped_impl.hpp
/// @brief HVG Selection for Memory-Mapped Sparse Matrices
///
/// ## Key Optimizations
///
/// 1. Streaming Statistics
///    - Uses feature_mapped_impl for mean/variance
///    - Sequential access for page cache efficiency
///
/// 2. SIMD Dispersion
///    - Vectorized variance/mean computation
///
/// 3. Partial Sort for Top-K
///    - O(n + k log k) instead of O(n log n)
///
/// Performance: Bounded by feature stats O(nnz) + sorting O(k log k)
// =============================================================================

namespace scl::kernel::hvg::mapped {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr Real EPSILON = 1e-12;
}

// =============================================================================
// SECTION 2: SIMD Utilities
// =============================================================================

namespace detail {

/// @brief SIMD dispersion (variance / mean)
inline void dispersion_simd(
    Array<const Real> means,
    Array<const Real> vars,
    Array<Real> out_dispersion
) {
    const Size n = means.len;

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_eps = s::Set(d, config::EPSILON);
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
        out_dispersion[k] = (m > config::EPSILON) ? (v / m) : Real(0);
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

    std::vector<Index> indices(n);
    for (Size i = 0; i < n; ++i) {
        indices[i] = static_cast<Index>(i);
    }

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

    std::memset(out_mask.ptr, 0, n * sizeof(uint8_t));

    for (Size i = 0; i < k; ++i) {
        Index idx = indices[i];
        out_indices[i] = idx;
        out_mask[idx] = 1;
    }
}

} // namespace detail

// =============================================================================
// SECTION 3: MappedCustomSparse HVG
// =============================================================================

/// @brief HVG by dispersion for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void select_by_dispersion_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Size n_top,
    Array<Index> out_indices,
    Array<uint8_t> out_mask,
    Array<Real> out_dispersions
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size n = static_cast<Size>(primary_dim);

    std::vector<Real> means(n);
    std::vector<Real> vars(n);

    // Use feature_mapped_impl for statistics
    scl::kernel::feature::mapped::standard_moments_mapped(
        matrix,
        Array<Real>(means.data(), n),
        Array<Real>(vars.data(), n),
        1  // ddof
    );

    // SIMD dispersion
    detail::dispersion_simd(
        Array<const Real>(means.data(), n),
        Array<const Real>(vars.data(), n),
        out_dispersions
    );

    // Partial sort top-K
    detail::select_top_k_partial(
        Array<const Real>(out_dispersions.ptr, n),
        n_top,
        out_indices,
        out_mask
    );
}

/// @brief HVG by variance for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void select_by_variance_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Size n_top,
    Array<Index> out_indices,
    Array<uint8_t> out_mask
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size n = static_cast<Size>(primary_dim);

    std::vector<Real> means(n);
    std::vector<Real> vars(n);

    // Use feature_mapped_impl for statistics
    scl::kernel::feature::mapped::standard_moments_mapped(
        matrix,
        Array<Real>(means.data(), n),
        Array<Real>(vars.data(), n),
        1  // ddof
    );

    // Partial sort top-K by variance
    detail::select_top_k_partial(
        Array<const Real>(vars.data(), n),
        n_top,
        out_indices,
        out_mask
    );
}

/// @brief HVG by VST for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void select_by_vst_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const Real> clip_vals,
    Size n_top,
    Array<Index> out_indices,
    Array<uint8_t> out_mask,
    Array<Real> out_variances
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size n = static_cast<Size>(primary_dim);

    std::vector<Real> means(n);

    scl::kernel::feature::mapped::clipped_moments_mapped(
        matrix,
        clip_vals,
        Array<Real>(means.data(), n),
        out_variances
    );

    detail::select_top_k_partial(
        Array<const Real>(out_variances.ptr, n),
        n_top,
        out_indices,
        out_mask
    );
}

// =============================================================================
// SECTION 4: MappedVirtualSparse HVG
// =============================================================================

/// @brief HVG by dispersion for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void select_by_dispersion_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Size n_top,
    Array<Index> out_indices,
    Array<uint8_t> out_mask,
    Array<Real> out_dispersions
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size n = static_cast<Size>(primary_dim);

    std::vector<Real> means(n);
    std::vector<Real> vars(n);

    scl::kernel::feature::mapped::standard_moments_mapped(
        matrix,
        Array<Real>(means.data(), n),
        Array<Real>(vars.data(), n),
        1
    );

    detail::dispersion_simd(
        Array<const Real>(means.data(), n),
        Array<const Real>(vars.data(), n),
        out_dispersions
    );

    detail::select_top_k_partial(
        Array<const Real>(out_dispersions.ptr, n),
        n_top,
        out_indices,
        out_mask
    );
}

/// @brief HVG by variance for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void select_by_variance_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Size n_top,
    Array<Index> out_indices,
    Array<uint8_t> out_mask
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size n = static_cast<Size>(primary_dim);

    std::vector<Real> means(n);
    std::vector<Real> vars(n);

    scl::kernel::feature::mapped::standard_moments_mapped(
        matrix,
        Array<Real>(means.data(), n),
        Array<Real>(vars.data(), n),
        1
    );

    // Partial sort top-K by variance
    detail::select_top_k_partial(
        Array<const Real>(vars.data(), n),
        n_top,
        out_indices,
        out_mask
    );
}

/// @brief HVG by VST for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void select_by_vst_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const Real> clip_vals,
    Size n_top,
    Array<Index> out_indices,
    Array<uint8_t> out_mask,
    Array<Real> out_variances
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Size n = static_cast<Size>(primary_dim);

    std::vector<Real> means(n);

    scl::kernel::feature::mapped::clipped_moments_mapped(
        matrix,
        clip_vals,
        Array<Real>(means.data(), n),
        out_variances
    );

    detail::select_top_k_partial(
        Array<const Real>(out_variances.ptr, n),
        n_top,
        out_indices,
        out_mask
    );
}

// =============================================================================
// SECTION 5: Unified Dispatchers
// =============================================================================

/// @brief Dispersion-based HVG dispatcher
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void select_by_dispersion_mapped_dispatch(
    const MatrixT& matrix,
    Size n_top,
    Array<Index> out_indices,
    Array<uint8_t> out_mask,
    Array<Real> out_dispersions
) {
    select_by_dispersion_mapped(matrix, n_top, out_indices, out_mask, out_dispersions);
}

/// @brief VST-based HVG dispatcher
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void select_by_vst_mapped_dispatch(
    const MatrixT& matrix,
    Array<const Real> clip_vals,
    Size n_top,
    Array<Index> out_indices,
    Array<uint8_t> out_mask,
    Array<Real> out_variances
) {
    select_by_vst_mapped(matrix, clip_vals, n_top, out_indices, out_mask, out_variances);
}

/// @brief Variance-based HVG dispatcher
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void select_by_variance_mapped_dispatch(
    const MatrixT& matrix,
    Size n_top,
    Array<Index> out_indices,
    Array<uint8_t> out_mask
) {
    select_by_variance_mapped(matrix, n_top, out_indices, out_mask);
}

} // namespace scl::kernel::hvg::mapped
