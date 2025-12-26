#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <vector>
#include <algorithm>

// =============================================================================
/// @file mmd_mapped_impl.hpp
/// @brief Extreme Performance MMD for Memory-Mapped Sparse Matrices
///
/// Streaming-optimized Maximum Mean Discrepancy for disk-backed data:
///
/// Key Optimizations:
///
/// 1. Chunk-Based Processing: L2-friendly chunks with prefetch
/// 2. Sequential Streaming: Minimize random page faults
/// 3. SIMD Exp Computation: 4-way unrolled exp(-gamma * x^2)
/// 4. Symmetry Exploitation: Self-kernel uses upper triangle
/// 5. Prefetch Pipeline: Hint OS for upcoming pages
///
/// Memory Access Pattern:
///
/// - Sequential row access within chunks
/// - Prefetch next chunk while processing current
/// - Thread-local caches to avoid repeated reads
///
/// Performance Target: Near in-memory performance for sequential access
// =============================================================================

namespace scl::kernel::mmd::mapped {

namespace detail {

// =============================================================================
// SECTION 1: SIMD Helpers (Streaming-Optimized)
// =============================================================================

/// @brief 4-way unrolled exp sum with cache write
///
/// Optimized for streaming from mapped memory.
template <typename T>
SCL_FORCE_INLINE T unary_exp_sum_streaming(
    const T* SCL_RESTRICT vals,
    Size nnz,
    T gamma,
    T* SCL_RESTRICT cache
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_gamma = s::Set(d, gamma);

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);

    size_t k = 0;

    // 4-way unrolled with prefetch
    for (; k + 4 * lanes <= nnz; k += 4 * lanes) {
        // Prefetch ahead for streaming
        SCL_PREFETCH_READ(vals + k + 8 * lanes, 0);

        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);

        auto sq0 = s::Mul(v0, v0);
        auto sq1 = s::Mul(v1, v1);
        auto sq2 = s::Mul(v2, v2);
        auto sq3 = s::Mul(v3, v3);

        auto exp0 = s::Exp(d, s::Neg(s::Mul(sq0, v_gamma)));
        auto exp1 = s::Exp(d, s::Neg(s::Mul(sq1, v_gamma)));
        auto exp2 = s::Exp(d, s::Neg(s::Mul(sq2, v_gamma)));
        auto exp3 = s::Exp(d, s::Neg(s::Mul(sq3, v_gamma)));

        s::Store(exp0, d, cache + k + 0 * lanes);
        s::Store(exp1, d, cache + k + 1 * lanes);
        s::Store(exp2, d, cache + k + 2 * lanes);
        s::Store(exp3, d, cache + k + 3 * lanes);

        v_sum0 = s::Add(v_sum0, exp0);
        v_sum1 = s::Add(v_sum1, exp1);
        v_sum2 = s::Add(v_sum2, exp2);
        v_sum3 = s::Add(v_sum3, exp3);
    }

    // Single vector tail
    for (; k + lanes <= nnz; k += lanes) {
        auto v = s::Load(d, vals + k);
        auto sq = s::Mul(v, v);
        auto exp_v = s::Exp(d, s::Neg(s::Mul(sq, v_gamma)));
        s::Store(exp_v, d, cache + k);
        v_sum0 = s::Add(v_sum0, exp_v);
    }

    auto v_total = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
    T sum = s::GetLane(s::SumOfLanes(d, v_total));

    // Scalar tail
    for (; k < nnz; ++k) {
        T val = vals[k];
        T exp_term = std::exp(-gamma * val * val);
        cache[k] = exp_term;
        sum += exp_term;
    }

    return sum;
}

/// @brief Self-kernel sum with symmetry exploitation
template <typename T>
SCL_FORCE_INLINE T self_kernel_sum_symmetric(
    const T* SCL_RESTRICT vals,
    Size nnz,
    Size N,
    T gamma,
    T sum_unary
) {
    const Size n_zeros = N - nnz;

    T sum = T(0);

    // Zero-Zero
    sum += static_cast<T>(n_zeros * n_zeros);

    // Zero-Val (symmetric)
    if (n_zeros > 0) {
        sum += T(2) * static_cast<T>(n_zeros) * sum_unary;
    }

    // Diagonal Val-Val: k(v, v) = 1
    sum += static_cast<T>(nnz);

    // Off-diagonal (upper triangle * 2)
    if (nnz <= 1) {
        return sum;
    }

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto v_gamma = s::Set(d, gamma);

    T off_diag = T(0);

    for (size_t i = 0; i < nnz - 1; ++i) {
        const T vi = vals[i];
        const auto v_vi = s::Set(d, vi);

        auto v_row_sum = s::Zero(d);
        size_t j = i + 1;

        for (; j + lanes <= nnz; j += lanes) {
            auto v_vj = s::Load(d, vals + j);
            auto diff = s::Sub(v_vi, v_vj);
            auto sq = s::Mul(diff, diff);
            v_row_sum = s::Add(v_row_sum, s::Exp(d, s::Neg(s::Mul(sq, v_gamma))));
        }

        off_diag += s::GetLane(s::SumOfLanes(d, v_row_sum));

        for (; j < nnz; ++j) {
            T diff = vi - vals[j];
            off_diag += std::exp(-gamma * diff * diff);
        }
    }

    sum += T(2) * off_diag;
    return sum;
}

/// @brief Cross-kernel sum
template <typename T>
SCL_FORCE_INLINE T cross_kernel_sum(
    const T* SCL_RESTRICT vals_x, Size nnz_x, Size N_x,
    const T* SCL_RESTRICT vals_y, Size nnz_y, Size N_y,
    T gamma,
    T sum_x_unary,
    T sum_y_unary
) {
    const Size zeros_x = N_x - nnz_x;
    const Size zeros_y = N_y - nnz_y;

    T sum = T(0);

    // Zero-Zero
    sum += static_cast<T>(zeros_x * zeros_y);

    // Zero-Val
    if (zeros_x > 0) {
        sum += static_cast<T>(zeros_x) * sum_y_unary;
    }
    if (zeros_y > 0) {
        sum += static_cast<T>(zeros_y) * sum_x_unary;
    }

    // Val-Val
    if (nnz_x == 0 || nnz_y == 0) {
        return sum;
    }

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto v_gamma = s::Set(d, gamma);

    T cross_sum = T(0);

    for (size_t i = 0; i < nnz_x; ++i) {
        const T xi = vals_x[i];
        const auto v_xi = s::Set(d, xi);

        auto v_row_sum = s::Zero(d);
        size_t j = 0;

        for (; j + lanes <= nnz_y; j += lanes) {
            auto v_yj = s::Load(d, vals_y + j);
            auto diff = s::Sub(v_xi, v_yj);
            auto sq = s::Mul(diff, diff);
            v_row_sum = s::Add(v_row_sum, s::Exp(d, s::Neg(s::Mul(sq, v_gamma))));
        }

        cross_sum += s::GetLane(s::SumOfLanes(d, v_row_sum));

        for (; j < nnz_y; ++j) {
            T diff = xi - vals_y[j];
            cross_sum += std::exp(-gamma * diff * diff);
        }
    }

    sum += cross_sum;
    return sum;
}

} // namespace detail

// =============================================================================
// SECTION 2: MappedCustomSparse MMD
// =============================================================================

/// @brief Compute MMD^2 for MappedCustomSparse
///
/// Streaming algorithm with chunk-based processing for cache efficiency.
/// Uses prefetch hints to minimize page fault latency.
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void mmd_rbf_mapped_custom(
    const scl::io::MappedCustomSparse<T, IsCSR>& mat_x,
    const scl::io::MappedCustomSparse<T, IsCSR>& mat_y,
    Array<T> output,
    T gamma = T(1)
) {
    const Index primary_dim = scl::primary_size(mat_x);
    const Size secondary_x = static_cast<Size>(scl::secondary_size(mat_x));
    const Size secondary_y = static_cast<Size>(scl::secondary_size(mat_y));

    SCL_CHECK_DIM(scl::primary_size(mat_y) == primary_dim, "MMD: Primary dimension mismatch");
    SCL_CHECK_DIM(output.len == static_cast<Size>(primary_dim), "MMD: Output size mismatch");

    // Prefetch hints
    kernel::mapped::hint_prefetch(mat_x);
    kernel::mapped::hint_prefetch(mat_y);

    const T inv_Nx2 = T(1) / static_cast<T>(secondary_x * secondary_x);
    const T inv_Ny2 = T(1) / static_cast<T>(secondary_y * secondary_y);
    const T inv_NxNy = T(1) / static_cast<T>(secondary_x * secondary_y);

    // Chunk-based processing for cache efficiency
    constexpr Size CHUNK_SIZE = 128;
    const Size n_chunks = (static_cast<Size>(primary_dim) + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Index chunk_start = static_cast<Index>(chunk_id * CHUNK_SIZE);
        Index chunk_end = std::min(chunk_start + static_cast<Index>(CHUNK_SIZE), primary_dim);

        // Prefetch next chunk
        if (chunk_id + 1 < n_chunks) {
            Index next_start = static_cast<Index>((chunk_id + 1) * CHUNK_SIZE);
            auto vals_x_next = scl::primary_values(mat_x, next_start);
            auto vals_y_next = scl::primary_values(mat_y, next_start);
            SCL_PREFETCH_READ(vals_x_next.ptr, 0);
            SCL_PREFETCH_READ(vals_y_next.ptr, 0);
        }

        // Parallel within chunk
        scl::threading::parallel_for(chunk_start, chunk_end, [&](Index p) {
            thread_local std::vector<T> x_cache;
            thread_local std::vector<T> y_cache;

            auto vals_x = scl::primary_values(mat_x, p);
            auto vals_y = scl::primary_values(mat_y, p);
            Size nnz_x = vals_x.len;
            Size nnz_y = vals_y.len;

            if (SCL_UNLIKELY(nnz_x == 0 && nnz_y == 0)) {
                output[p] = T(0);
                return;
            }

            if (x_cache.size() < nnz_x) x_cache.resize(nnz_x);
            if (y_cache.size() < nnz_y) y_cache.resize(nnz_y);

            T sum_x_unary = (nnz_x > 0)
                ? detail::unary_exp_sum_streaming(vals_x.ptr, nnz_x, gamma, x_cache.data())
                : T(0);
            T sum_y_unary = (nnz_y > 0)
                ? detail::unary_exp_sum_streaming(vals_y.ptr, nnz_y, gamma, y_cache.data())
                : T(0);

            T sum_xx = detail::self_kernel_sum_symmetric(vals_x.ptr, nnz_x, secondary_x, gamma, sum_x_unary);
            T sum_yy = detail::self_kernel_sum_symmetric(vals_y.ptr, nnz_y, secondary_y, gamma, sum_y_unary);
            T sum_xy = detail::cross_kernel_sum(
                vals_x.ptr, nnz_x, secondary_x,
                vals_y.ptr, nnz_y, secondary_y,
                gamma, sum_x_unary, sum_y_unary
            );

            T mmd2 = (sum_xx * inv_Nx2) + (sum_yy * inv_Ny2) - (T(2) * sum_xy * inv_NxNy);
            output[p] = std::max(mmd2, T(0));
        });
    }
}

// =============================================================================
// SECTION 3: MappedVirtualSparse MMD
// =============================================================================

/// @brief Compute MMD^2 for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void mmd_rbf_mapped_virtual(
    const scl::io::MappedVirtualSparse<T, IsCSR>& mat_x,
    const scl::io::MappedVirtualSparse<T, IsCSR>& mat_y,
    Array<T> output,
    T gamma = T(1)
) {
    const Index primary_dim = scl::primary_size(mat_x);
    const Size secondary_x = static_cast<Size>(scl::secondary_size(mat_x));
    const Size secondary_y = static_cast<Size>(scl::secondary_size(mat_y));

    SCL_CHECK_DIM(scl::primary_size(mat_y) == primary_dim, "MMD: Primary dimension mismatch");
    SCL_CHECK_DIM(output.len == static_cast<Size>(primary_dim), "MMD: Output size mismatch");

    const T inv_Nx2 = T(1) / static_cast<T>(secondary_x * secondary_x);
    const T inv_Ny2 = T(1) / static_cast<T>(secondary_y * secondary_y);
    const T inv_NxNy = T(1) / static_cast<T>(secondary_x * secondary_y);

    constexpr Size CHUNK_SIZE = 128;
    const Size n_chunks = (static_cast<Size>(primary_dim) + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Index chunk_start = static_cast<Index>(chunk_id * CHUNK_SIZE);
        Index chunk_end = std::min(chunk_start + static_cast<Index>(CHUNK_SIZE), primary_dim);

        scl::threading::parallel_for(chunk_start, chunk_end, [&](Index p) {
            thread_local std::vector<T> x_cache;
            thread_local std::vector<T> y_cache;

            auto vals_x = scl::primary_values(mat_x, p);
            auto vals_y = scl::primary_values(mat_y, p);
            Size nnz_x = vals_x.len;
            Size nnz_y = vals_y.len;

            if (SCL_UNLIKELY(nnz_x == 0 && nnz_y == 0)) {
                output[p] = T(0);
                return;
            }

            if (x_cache.size() < nnz_x) x_cache.resize(nnz_x);
            if (y_cache.size() < nnz_y) y_cache.resize(nnz_y);

            T sum_x_unary = (nnz_x > 0)
                ? detail::unary_exp_sum_streaming(vals_x.ptr, nnz_x, gamma, x_cache.data())
                : T(0);
            T sum_y_unary = (nnz_y > 0)
                ? detail::unary_exp_sum_streaming(vals_y.ptr, nnz_y, gamma, y_cache.data())
                : T(0);

            T sum_xx = detail::self_kernel_sum_symmetric(vals_x.ptr, nnz_x, secondary_x, gamma, sum_x_unary);
            T sum_yy = detail::self_kernel_sum_symmetric(vals_y.ptr, nnz_y, secondary_y, gamma, sum_y_unary);
            T sum_xy = detail::cross_kernel_sum(
                vals_x.ptr, nnz_x, secondary_x,
                vals_y.ptr, nnz_y, secondary_y,
                gamma, sum_x_unary, sum_y_unary
            );

            T mmd2 = (sum_xx * inv_Nx2) + (sum_yy * inv_Ny2) - (T(2) * sum_xy * inv_NxNy);
            output[p] = std::max(mmd2, T(0));
        });
    }
}

// =============================================================================
// SECTION 4: Unified Dispatcher
// =============================================================================

/// @brief Unified dispatcher for mapped MMD
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
void mmd_rbf_mapped_dispatch(
    const MatrixT& mat_x,
    const MatrixT& mat_y,
    Array<typename MatrixT::ValueType> output,
    typename MatrixT::ValueType gamma = typename MatrixT::ValueType(1)
) {
    using T = typename MatrixT::ValueType;

    if constexpr (std::is_same_v<MatrixT, scl::io::MappedCustomSparse<T, IsCSR>>) {
        mmd_rbf_mapped_custom(mat_x, mat_y, output, gamma);
    } else if constexpr (std::is_same_v<MatrixT, scl::io::MappedVirtualSparse<T, IsCSR>>) {
        mmd_rbf_mapped_virtual(mat_x, mat_y, output, gamma);
    } else {
        // Generic fallback for other MappedSparseLike types
        const Index primary_dim = scl::primary_size(mat_x);
        const Size secondary_x = static_cast<Size>(scl::secondary_size(mat_x));
        const Size secondary_y = static_cast<Size>(scl::secondary_size(mat_y));

        const T inv_Nx2 = T(1) / static_cast<T>(secondary_x * secondary_x);
        const T inv_Ny2 = T(1) / static_cast<T>(secondary_y * secondary_y);
        const T inv_NxNy = T(1) / static_cast<T>(secondary_x * secondary_y);

        scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
            thread_local std::vector<T> x_cache;
            thread_local std::vector<T> y_cache;

            auto vals_x = scl::primary_values(mat_x, p);
            auto vals_y = scl::primary_values(mat_y, p);
            Size nnz_x = vals_x.len;
            Size nnz_y = vals_y.len;

            if (SCL_UNLIKELY(nnz_x == 0 && nnz_y == 0)) {
                output[p] = T(0);
                return;
            }

            if (x_cache.size() < nnz_x) x_cache.resize(nnz_x);
            if (y_cache.size() < nnz_y) y_cache.resize(nnz_y);

            T sum_x_unary = (nnz_x > 0)
                ? detail::unary_exp_sum_streaming(vals_x.ptr, nnz_x, gamma, x_cache.data())
                : T(0);
            T sum_y_unary = (nnz_y > 0)
                ? detail::unary_exp_sum_streaming(vals_y.ptr, nnz_y, gamma, y_cache.data())
                : T(0);

            T sum_xx = detail::self_kernel_sum_symmetric(vals_x.ptr, nnz_x, secondary_x, gamma, sum_x_unary);
            T sum_yy = detail::self_kernel_sum_symmetric(vals_y.ptr, nnz_y, secondary_y, gamma, sum_y_unary);
            T sum_xy = detail::cross_kernel_sum(
                vals_x.ptr, nnz_x, secondary_x,
                vals_y.ptr, nnz_y, secondary_y,
                gamma, sum_x_unary, sum_y_unary
            );

            T mmd2 = (sum_xx * inv_Nx2) + (sum_yy * inv_Ny2) - (T(2) * sum_xy * inv_NxNy);
            output[p] = std::max(mmd2, T(0));
        });
    }
}

} // namespace scl::kernel::mmd::mapped
