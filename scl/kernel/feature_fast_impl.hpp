#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// Mapped backend support
#include "scl/kernel/mapped_common.hpp"
#include "scl/kernel/feature_mapped_impl.hpp"

#include <cmath>
#include <algorithm>

// =============================================================================
/// @file feature_fast_impl.hpp
/// @brief Extreme Performance Feature Statistics
///
/// ## Key Optimizations
///
/// 1. 4-Way Unrolled SIMD
///    - Fused sum + sum_sq accumulation
///    - Maximizes instruction-level parallelism
///
/// 2. Direct Memory Access
///    - CustomSparse: indptr-based contiguous access
///    - VirtualSparse: Single pointer dereference
///
/// 3. Cache-Optimized Processing
///    - Sequential row access
///    - Prefetch hints for mapped data
///
/// ## Supported Operations
///
/// - standard_moments: Mean and variance (single pass)
/// - clipped_moments: Mean and variance with clipping (Seurat V3)
/// - detection_rate: Fraction of non-zero elements
/// - dispersion: Fano factor (variance/mean)
///
/// Performance Target: 2-3x faster than generic
// =============================================================================

namespace scl::kernel::feature::fast {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr Size CHUNK_SIZE = 256;     // Rows per parallel chunk
    constexpr Real EPSILON = 1e-12;      // For dispersion division
}

// =============================================================================
// SECTION 2: SIMD Utilities
// =============================================================================

namespace detail {

/// @brief SIMD fused sum + sum_sq with 4-way unroll
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

/// @brief SIMD fused clipped sum + sum_sq with 4-way unroll
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
    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sq0 = s::Zero(d);
    auto v_sq1 = s::Zero(d);

    Size k = 0;

    // 4-way unrolled with clipping
    for (; k + 4 * lanes <= len; k += 4 * lanes) {
        auto v0 = s::Min(s::Load(d, vals + k + 0 * lanes), v_clip);
        auto v1 = s::Min(s::Load(d, vals + k + 1 * lanes), v_clip);
        auto v2 = s::Min(s::Load(d, vals + k + 2 * lanes), v_clip);
        auto v3 = s::Min(s::Load(d, vals + k + 3 * lanes), v_clip);

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
// SECTION 3: Standard Moments - CustomSparse
// =============================================================================

/// @brief Standard moments for CustomSparse
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void standard_moments_custom(
    const CustomSparse<T, IsCSR>& matrix,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Real N = static_cast<Real>(scl::secondary_size(matrix));
    const Real denom = N - static_cast<Real>(ddof);

    SCL_CHECK_DIM(out_means.len >= static_cast<Size>(primary_dim), "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len >= static_cast<Size>(primary_dim), "Vars size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Size len = static_cast<Size>(end - start);

        Real sum, sq_sum;
        detail::compute_sum_sq_simd(matrix.data + start, len, sum, sq_sum);

        Real mu = sum / N;
        Real var = (denom > Real(0)) ? ((sq_sum - sum * mu) / denom) : Real(0);
        if (var < Real(0)) var = Real(0);

        out_means[p] = mu;
        out_vars[p] = var;
    });
}

/// @brief Clipped moments for CustomSparse
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void clipped_moments_custom(
    const CustomSparse<T, IsCSR>& matrix,
    Array<const Real> clip_vals,
    Array<Real> out_means,
    Array<Real> out_vars
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Real N = static_cast<Real>(scl::secondary_size(matrix));
    const Real N_minus_1 = N - Real(1);

    SCL_CHECK_DIM(clip_vals.len >= static_cast<Size>(primary_dim), "Clip vals mismatch");
    SCL_CHECK_DIM(out_means.len >= static_cast<Size>(primary_dim), "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len >= static_cast<Size>(primary_dim), "Vars size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Size len = static_cast<Size>(end - start);
        Real clip = clip_vals[p];

        Real sum, sq_sum;
        detail::compute_clipped_sum_sq_simd(matrix.data + start, len, clip, sum, sq_sum);

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

/// @brief Detection rate for CustomSparse
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void detection_rate_custom(
    const CustomSparse<T, IsCSR>& matrix,
    Array<Real> out_rates
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Real inv_N = Real(1) / static_cast<Real>(scl::secondary_size(matrix));

    SCL_CHECK_DIM(out_rates.len >= static_cast<Size>(primary_dim), "Rates size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Index len = matrix.indptr[p + 1] - matrix.indptr[p];
        out_rates[p] = static_cast<Real>(len) * inv_N;
    });
}

// =============================================================================
// SECTION 4: Standard Moments - VirtualSparse
// =============================================================================

/// @brief Standard moments for VirtualSparse
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void standard_moments_virtual(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Real N = static_cast<Real>(scl::secondary_size(matrix));
    const Real denom = N - static_cast<Real>(ddof);

    SCL_CHECK_DIM(out_means.len >= static_cast<Size>(primary_dim), "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len >= static_cast<Size>(primary_dim), "Vars size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Size len = static_cast<Size>(matrix.lengths[p]);
        const T* vals = static_cast<const T*>(matrix.data_ptrs[p]);

        Real sum, sq_sum;
        detail::compute_sum_sq_simd(vals, len, sum, sq_sum);

        Real mu = sum / N;
        Real var = (denom > Real(0)) ? ((sq_sum - sum * mu) / denom) : Real(0);
        if (var < Real(0)) var = Real(0);

        out_means[p] = mu;
        out_vars[p] = var;
    });
}

/// @brief Clipped moments for VirtualSparse
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void clipped_moments_virtual(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<const Real> clip_vals,
    Array<Real> out_means,
    Array<Real> out_vars
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Real N = static_cast<Real>(scl::secondary_size(matrix));
    const Real N_minus_1 = N - Real(1);

    SCL_CHECK_DIM(clip_vals.len >= static_cast<Size>(primary_dim), "Clip vals mismatch");
    SCL_CHECK_DIM(out_means.len >= static_cast<Size>(primary_dim), "Means size mismatch");
    SCL_CHECK_DIM(out_vars.len >= static_cast<Size>(primary_dim), "Vars size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        Size len = static_cast<Size>(matrix.lengths[p]);
        const T* vals = static_cast<const T*>(matrix.data_ptrs[p]);
        Real clip = clip_vals[p];

        Real sum, sq_sum;
        detail::compute_clipped_sum_sq_simd(vals, len, clip, sum, sq_sum);

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

/// @brief Detection rate for VirtualSparse
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void detection_rate_virtual(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<Real> out_rates
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Real inv_N = Real(1) / static_cast<Real>(scl::secondary_size(matrix));

    SCL_CHECK_DIM(out_rates.len >= static_cast<Size>(primary_dim), "Rates size mismatch");

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        out_rates[p] = static_cast<Real>(matrix.lengths[p]) * inv_N;
    });
}

// =============================================================================
// SECTION 5: Dispersion (SIMD Vectorized)
// =============================================================================

/// @brief SIMD dispersion computation
void dispersion_simd(
    Array<const Real> means,
    Array<const Real> vars,
    Array<Real> out_dispersion
) {
    const Size n = means.len;

    SCL_CHECK_DIM(vars.len >= n, "Vars size mismatch");
    SCL_CHECK_DIM(out_dispersion.len >= n, "Output size mismatch");

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_eps = s::Set(d, config::EPSILON);
    const auto v_zero = s::Zero(d);

    Size k = 0;

    // SIMD loop
    for (; k + lanes <= n; k += lanes) {
        auto v_mean = s::Load(d, means.ptr + k);
        auto v_var = s::Load(d, vars.ptr + k);

        auto mask = s::Gt(v_mean, v_eps);
        auto v_div = s::Div(v_var, v_mean);
        auto v_res = s::IfThenElse(mask, v_div, v_zero);

        s::Store(v_res, d, out_dispersion.ptr + k);
    }

    // Scalar tail
    for (; k < n; ++k) {
        Real m = means[k];
        Real v = vars[k];
        out_dispersion[k] = (m > config::EPSILON) ? (v / m) : Real(0);
    }
}

// =============================================================================
// SECTION 6: Unified Dispatchers
// =============================================================================

/// @brief Standard moments dispatcher
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void standard_moments_fast(
    const MatrixT& matrix,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof
) {
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        scl::kernel::feature::mapped::standard_moments_mapped_dispatch<MatrixT, IsCSR>(
            matrix, out_means, out_vars, ddof
        );
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        standard_moments_custom(matrix, out_means, out_vars, ddof);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        standard_moments_virtual(matrix, out_means, out_vars, ddof);
    }
}

/// @brief Clipped moments dispatcher
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void clipped_moments_fast(
    const MatrixT& matrix,
    Array<const Real> clip_vals,
    Array<Real> out_means,
    Array<Real> out_vars
) {
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        scl::kernel::feature::mapped::clipped_moments_mapped_dispatch<MatrixT, IsCSR>(
            matrix, clip_vals, out_means, out_vars
        );
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        clipped_moments_custom(matrix, clip_vals, out_means, out_vars);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        clipped_moments_virtual(matrix, clip_vals, out_means, out_vars);
    }
}

/// @brief Detection rate dispatcher
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void detection_rate_fast(
    const MatrixT& matrix,
    Array<Real> out_rates
) {
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        scl::kernel::feature::mapped::detection_rate_mapped_dispatch<MatrixT, IsCSR>(
            matrix, out_rates
        );
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        detection_rate_custom(matrix, out_rates);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        detection_rate_virtual(matrix, out_rates);
    }
}

} // namespace scl::kernel::feature::fast
