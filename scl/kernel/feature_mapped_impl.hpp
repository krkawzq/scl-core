#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <algorithm>

// =============================================================================
/// @file feature_mapped_impl.hpp
/// @brief Feature Statistics for Memory-Mapped Sparse Matrices
///
/// ## Key Optimizations
///
/// 1. Streaming Access Pattern
///    - Sequential row access for cache efficiency
///    - Prefetch hints for OS page cache
///
/// 2. Single-Pass Algorithms
///    - Fused sum + sum_sq accumulation
///    - No intermediate storage
///
/// 3. 4-Way Unrolled SIMD
///    - Maximizes instruction-level parallelism
///    - Works well with page-aligned data
///
/// ## Supported Operations
///
/// - standard_moments: Mean and variance
/// - clipped_moments: Mean and variance with clipping
/// - detection_rate: Fraction of non-zero elements
///
/// Performance: Near-RAM performance for cached data
// =============================================================================

namespace scl::kernel::feature::mapped {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr Size CHUNK_SIZE = 256;
}

// =============================================================================
// SECTION 2: SIMD Utilities
// =============================================================================

namespace detail {

/// @brief SIMD fused sum + sum_sq
template <typename T>
SCL_FORCE_INLINE void compute_sum_sq_simd(
    const T* SCL_RESTRICT vals,
    Size len,
    Real& out_sum,
    Real& out_sq_sum
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sq0 = s::Zero(d);
    auto v_sq1 = s::Zero(d);

    Size k = 0;

    // 4-way unrolled
    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);

        v_sum0 = s::Add(v_sum0, v0);
        v_sum1 = s::Add(v_sum1, v1);
        v_sum0 = s::Add(v_sum0, v2);
        v_sum1 = s::Add(v_sum1, v3);

        v_sq0 = s::MulAdd(v0, v0, v_sq0);
        v_sq1 = s::MulAdd(v1, v1, v_sq1);
        v_sq0 = s::MulAdd(v2, v2, v_sq0);
        v_sq1 = s::MulAdd(v3, v3, v_sq1);
    }

    auto v_sum = s::Add(v_sum0, v_sum1);
    auto v_sq = s::Add(v_sq0, v_sq1);

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Load(d, vals + k);
        v_sum = s::Add(v_sum, v);
        v_sq = s::MulAdd(v, v, v_sq);
    }

    Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
    Real sq_sum = s::GetLane(s::SumOfLanes(d, v_sq));

    for (; k < len; ++k) {
        Real v = static_cast<Real>(vals[k]);
        sum += v;
        sq_sum += v * v;
    }

    out_sum = sum;
    out_sq_sum = sq_sum;
}

/// @brief SIMD fused clipped sum + sum_sq
template <typename T>
SCL_FORCE_INLINE void compute_clipped_sum_sq_simd(
    const T* SCL_RESTRICT vals,
    Size len,
    Real clip,
    Real& out_sum,
    Real& out_sq_sum
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    auto v_clip = s::Set(d, clip);
    auto v_sum = s::Zero(d);
    auto v_sq = s::Zero(d);

    Size k = 0;

    for (; k + lanes <= len; k += lanes) {
        auto v = s::Min(s::Load(d, vals + k), v_clip);
        v_sum = s::Add(v_sum, v);
        v_sq = s::MulAdd(v, v, v_sq);
    }

    Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
    Real sq_sum = s::GetLane(s::SumOfLanes(d, v_sq));

    for (; k < len; ++k) {
        Real v = std::min(static_cast<Real>(vals[k]), clip);
        sum += v;
        sq_sum += v * v;
    }

    out_sum = sum;
    out_sq_sum = sq_sum;
}

} // namespace detail

// =============================================================================
// SECTION 3: Standard Moments
// =============================================================================

/// @brief Standard moments for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void standard_moments_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof
) {
    const Index n_primary = scl::primary_size(matrix);
    const Real N = static_cast<Real>(scl::secondary_size(matrix));
    const Real denom = N - static_cast<Real>(ddof);

    SCL_CHECK_DIM(out_means.len >= static_cast<Size>(n_primary), "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len >= static_cast<Size>(n_primary), "Vars size mismatch");

    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        auto values = scl::primary_values(matrix, p);

        Real sum, sq_sum;
        detail::compute_sum_sq_simd(values.ptr, values.len, sum, sq_sum);

        Real mu = sum / N;
        Real var = (denom > Real(0)) ? ((sq_sum - sum * mu) / denom) : Real(0);
        if (var < Real(0)) var = Real(0);

        out_means[p] = mu;
        out_vars[p] = var;
    });
}

/// @brief Standard moments for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void standard_moments_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof
) {
    const Index n_primary = scl::primary_size(matrix);
    const Real N = static_cast<Real>(scl::secondary_size(matrix));
    const Real denom = N - static_cast<Real>(ddof);

    SCL_CHECK_DIM(out_means.len >= static_cast<Size>(n_primary), "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len >= static_cast<Size>(n_primary), "Vars size mismatch");

    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        auto values = scl::primary_values(matrix, p);

        Real sum, sq_sum;
        detail::compute_sum_sq_simd(values.ptr, values.len, sum, sq_sum);

        Real mu = sum / N;
        Real var = (denom > Real(0)) ? ((sq_sum - sum * mu) / denom) : Real(0);
        if (var < Real(0)) var = Real(0);

        out_means[p] = mu;
        out_vars[p] = var;
    });
}

// =============================================================================
// SECTION 4: Clipped Moments
// =============================================================================

/// @brief Clipped moments for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void clipped_moments_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<const Real> clip_vals,
    Array<Real> out_means,
    Array<Real> out_vars
) {
    const Index n_primary = scl::primary_size(matrix);
    const Real N = static_cast<Real>(scl::secondary_size(matrix));
    const Real N_minus_1 = N - Real(1);

    SCL_CHECK_DIM(clip_vals.len >= static_cast<Size>(n_primary), "Clip vals mismatch");
    SCL_CHECK_DIM(out_means.len >= static_cast<Size>(n_primary), "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len >= static_cast<Size>(n_primary), "Vars size mismatch");

    kernel::mapped::hint_prefetch(matrix);

    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        auto values = scl::primary_values(matrix, p);
        Real clip = clip_vals[p];

        Real sum, sq_sum;
        detail::compute_clipped_sum_sq_simd(values.ptr, values.len, clip, sum, sq_sum);

        Real mu = sum / N;
        Real var = Real(0);
        if (N > Real(1)) {
            var = (sq_sum - N * mu * mu) / N_minus_1;
        }
        if (var < Real(0)) var = Real(0);

        out_means[p] = mu;
        out_vars[p] = var;
    });
}

/// @brief Clipped moments for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void clipped_moments_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<const Real> clip_vals,
    Array<Real> out_means,
    Array<Real> out_vars
) {
    const Index n_primary = scl::primary_size(matrix);
    const Real N = static_cast<Real>(scl::secondary_size(matrix));
    const Real N_minus_1 = N - Real(1);

    SCL_CHECK_DIM(clip_vals.len >= static_cast<Size>(n_primary), "Clip vals mismatch");
    SCL_CHECK_DIM(out_means.len >= static_cast<Size>(n_primary), "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len >= static_cast<Size>(n_primary), "Vars size mismatch");

    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        auto values = scl::primary_values(matrix, p);
        Real clip = clip_vals[p];

        Real sum, sq_sum;
        detail::compute_clipped_sum_sq_simd(values.ptr, values.len, clip, sum, sq_sum);

        Real mu = sum / N;
        Real var = Real(0);
        if (N > Real(1)) {
            var = (sq_sum - N * mu * mu) / N_minus_1;
        }
        if (var < Real(0)) var = Real(0);

        out_means[p] = mu;
        out_vars[p] = var;
    });
}

// =============================================================================
// SECTION 5: Detection Rate
// =============================================================================

/// @brief Detection rate for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void detection_rate_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<Real> out_rates
) {
    const Index n_primary = scl::primary_size(matrix);
    const Real inv_N = Real(1) / static_cast<Real>(scl::secondary_size(matrix));

    SCL_CHECK_DIM(out_rates.len >= static_cast<Size>(n_primary), "Rates size mismatch");

    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        Index len = scl::primary_length(matrix, p);
        out_rates[p] = static_cast<Real>(len) * inv_N;
    });
}

/// @brief Detection rate for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void detection_rate_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<Real> out_rates
) {
    const Index n_primary = scl::primary_size(matrix);
    const Real inv_N = Real(1) / static_cast<Real>(scl::secondary_size(matrix));

    SCL_CHECK_DIM(out_rates.len >= static_cast<Size>(n_primary), "Rates size mismatch");

    scl::threading::parallel_for(Index(0), n_primary, [&](Index p) {
        Index len = scl::primary_length(matrix, p);
        out_rates[p] = static_cast<Real>(len) * inv_N;
    });
}

// =============================================================================
// SECTION 6: Unified Dispatchers
// =============================================================================

/// @brief Standard moments dispatcher
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void standard_moments_mapped_dispatch(
    const MatrixT& matrix,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof
) {
    standard_moments_mapped(matrix, out_means, out_vars, ddof);
}

/// @brief Clipped moments dispatcher
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void clipped_moments_mapped_dispatch(
    const MatrixT& matrix,
    Array<const Real> clip_vals,
    Array<Real> out_means,
    Array<Real> out_vars
) {
    clipped_moments_mapped(matrix, clip_vals, out_means, out_vars);
}

/// @brief Detection rate dispatcher
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void detection_rate_mapped_dispatch(
    const MatrixT& matrix,
    Array<Real> out_rates
) {
    detection_rate_mapped(matrix, out_rates);
}

} // namespace scl::kernel::feature::mapped
