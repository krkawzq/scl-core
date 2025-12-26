#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <vector>
#include <algorithm>

// =============================================================================
/// @file mmd_fast_impl.hpp
/// @brief Extreme Performance MMD for In-Memory Sparse Matrices
///
/// Ultra-optimized Maximum Mean Discrepancy with RBF kernel:
///
/// Key Optimizations:
///
/// 1. 8-way Unrolled SIMD: Maximizes instruction-level parallelism
/// 2. Fused Exp-Sum: Single pass for exp computation and accumulation
/// 3. Symmetry Exploitation: Self-kernel uses upper triangle only
/// 4. Cache-Blocked Cross: L2-friendly blocking for cross-kernel
/// 5. Prefetch Pipeline: Software prefetching for streaming access
///
/// Performance Target: 3-4x faster than generic implementation
///
/// Mathematical Background:
///
/// MMD^2(P, Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
/// RBF kernel: k(a, b) = exp(-gamma * ||a - b||^2)
///
/// Sparse Decomposition:
/// - Zero-Zero: k(0, 0) = 1
/// - Zero-Val: k(0, v) = exp(-gamma * v^2) (precomputed as unary)
/// - Val-Val: k(u, v) = exp(-gamma * (u-v)^2)
// =============================================================================

namespace scl::kernel::mmd::fast {

namespace detail {

// =============================================================================
// SECTION 1: SIMD Helpers
// =============================================================================

/// @brief 8-way unrolled exp sum with cache write
///
/// Computes exp(-gamma * v^2) for all values and accumulates sum.
/// Uses 8 accumulators for maximum instruction-level parallelism.
template <typename T>
SCL_FORCE_INLINE T unary_exp_sum_ultra(
    const T* SCL_RESTRICT vals,
    Size nnz,
    T gamma,
    T* SCL_RESTRICT cache
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_gamma = s::Set(d, gamma);

    // 8 accumulators for ILP
    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);
    auto v_sum4 = s::Zero(d);
    auto v_sum5 = s::Zero(d);
    auto v_sum6 = s::Zero(d);
    auto v_sum7 = s::Zero(d);

    size_t k = 0;

    // 8-way unrolled main loop
    for (; k + 8 * lanes <= nnz; k += 8 * lanes) {
        // Prefetch next cache line
        SCL_PREFETCH_READ(vals + k + 16 * lanes, 0);

        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);
        auto v4 = s::Load(d, vals + k + 4 * lanes);
        auto v5 = s::Load(d, vals + k + 5 * lanes);
        auto v6 = s::Load(d, vals + k + 6 * lanes);
        auto v7 = s::Load(d, vals + k + 7 * lanes);

        // Square: v^2
        auto sq0 = s::Mul(v0, v0);
        auto sq1 = s::Mul(v1, v1);
        auto sq2 = s::Mul(v2, v2);
        auto sq3 = s::Mul(v3, v3);
        auto sq4 = s::Mul(v4, v4);
        auto sq5 = s::Mul(v5, v5);
        auto sq6 = s::Mul(v6, v6);
        auto sq7 = s::Mul(v7, v7);

        // exp(-gamma * v^2)
        auto exp0 = s::Exp(d, s::Neg(s::Mul(sq0, v_gamma)));
        auto exp1 = s::Exp(d, s::Neg(s::Mul(sq1, v_gamma)));
        auto exp2 = s::Exp(d, s::Neg(s::Mul(sq2, v_gamma)));
        auto exp3 = s::Exp(d, s::Neg(s::Mul(sq3, v_gamma)));
        auto exp4 = s::Exp(d, s::Neg(s::Mul(sq4, v_gamma)));
        auto exp5 = s::Exp(d, s::Neg(s::Mul(sq5, v_gamma)));
        auto exp6 = s::Exp(d, s::Neg(s::Mul(sq6, v_gamma)));
        auto exp7 = s::Exp(d, s::Neg(s::Mul(sq7, v_gamma)));

        // Store to cache
        s::Store(exp0, d, cache + k + 0 * lanes);
        s::Store(exp1, d, cache + k + 1 * lanes);
        s::Store(exp2, d, cache + k + 2 * lanes);
        s::Store(exp3, d, cache + k + 3 * lanes);
        s::Store(exp4, d, cache + k + 4 * lanes);
        s::Store(exp5, d, cache + k + 5 * lanes);
        s::Store(exp6, d, cache + k + 6 * lanes);
        s::Store(exp7, d, cache + k + 7 * lanes);

        // Accumulate
        v_sum0 = s::Add(v_sum0, exp0);
        v_sum1 = s::Add(v_sum1, exp1);
        v_sum2 = s::Add(v_sum2, exp2);
        v_sum3 = s::Add(v_sum3, exp3);
        v_sum4 = s::Add(v_sum4, exp4);
        v_sum5 = s::Add(v_sum5, exp5);
        v_sum6 = s::Add(v_sum6, exp6);
        v_sum7 = s::Add(v_sum7, exp7);
    }

    // 4-way tail
    for (; k + 4 * lanes <= nnz; k += 4 * lanes) {
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

    // Reduce accumulators
    auto v_total = s::Add(
        s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3)),
        s::Add(s::Add(v_sum4, v_sum5), s::Add(v_sum6, v_sum7))
    );
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

/// @brief Unary exp sum without cache (when cache not needed)
template <typename T>
SCL_FORCE_INLINE T unary_exp_sum_nocache(
    const T* SCL_RESTRICT vals,
    Size nnz,
    T gamma
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

    // 4-way unrolled
    for (; k + 4 * lanes <= nnz; k += 4 * lanes) {
        auto v0 = s::Load(d, vals + k + 0 * lanes);
        auto v1 = s::Load(d, vals + k + 1 * lanes);
        auto v2 = s::Load(d, vals + k + 2 * lanes);
        auto v3 = s::Load(d, vals + k + 3 * lanes);

        auto sq0 = s::Mul(v0, v0);
        auto sq1 = s::Mul(v1, v1);
        auto sq2 = s::Mul(v2, v2);
        auto sq3 = s::Mul(v3, v3);

        v_sum0 = s::Add(v_sum0, s::Exp(d, s::Neg(s::Mul(sq0, v_gamma))));
        v_sum1 = s::Add(v_sum1, s::Exp(d, s::Neg(s::Mul(sq1, v_gamma))));
        v_sum2 = s::Add(v_sum2, s::Exp(d, s::Neg(s::Mul(sq2, v_gamma))));
        v_sum3 = s::Add(v_sum3, s::Exp(d, s::Neg(s::Mul(sq3, v_gamma))));
    }

    for (; k + lanes <= nnz; k += lanes) {
        auto v = s::Load(d, vals + k);
        auto sq = s::Mul(v, v);
        v_sum0 = s::Add(v_sum0, s::Exp(d, s::Neg(s::Mul(sq, v_gamma))));
    }

    auto v_total = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
    T sum = s::GetLane(s::SumOfLanes(d, v_total));

    for (; k < nnz; ++k) {
        T val = vals[k];
        sum += std::exp(-gamma * val * val);
    }

    return sum;
}

// =============================================================================
// SECTION 2: Self-Kernel with Symmetry
// =============================================================================

/// @brief Self-kernel sum exploiting symmetry
///
/// Computes sum of k(x_i, x_j) for all i, j.
/// Uses symmetry: k(x_i, x_j) = k(x_j, x_i) to compute only upper triangle.
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

    // Zero-Zero: n_zeros^2 terms, each = 1
    sum += static_cast<T>(n_zeros * n_zeros);

    // Zero-Val: 2 * n_zeros * sum(exp(-gamma * v^2))
    if (n_zeros > 0) {
        sum += T(2) * static_cast<T>(n_zeros) * sum_unary;
    }

    // Diagonal Val-Val: k(v, v) = 1 for all nnz values
    sum += static_cast<T>(nnz);

    // Off-diagonal Val-Val: use symmetry, compute upper triangle * 2
    if (nnz <= 1) {
        return sum;
    }

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto v_gamma = s::Set(d, gamma);

    T off_diag = T(0);

    // For each row i, compute sum over j > i
    for (size_t i = 0; i < nnz - 1; ++i) {
        const T vi = vals[i];
        const auto v_vi = s::Set(d, vi);

        auto v_row_sum0 = s::Zero(d);
        auto v_row_sum1 = s::Zero(d);

        size_t j = i + 1;

        // 2-way unrolled inner loop
        for (; j + 2 * lanes <= nnz; j += 2 * lanes) {
            auto v_vj0 = s::Load(d, vals + j + 0 * lanes);
            auto v_vj1 = s::Load(d, vals + j + 1 * lanes);

            auto diff0 = s::Sub(v_vi, v_vj0);
            auto diff1 = s::Sub(v_vi, v_vj1);

            auto sq0 = s::Mul(diff0, diff0);
            auto sq1 = s::Mul(diff1, diff1);

            v_row_sum0 = s::Add(v_row_sum0, s::Exp(d, s::Neg(s::Mul(sq0, v_gamma))));
            v_row_sum1 = s::Add(v_row_sum1, s::Exp(d, s::Neg(s::Mul(sq1, v_gamma))));
        }

        for (; j + lanes <= nnz; j += lanes) {
            auto v_vj = s::Load(d, vals + j);
            auto diff = s::Sub(v_vi, v_vj);
            auto sq = s::Mul(diff, diff);
            v_row_sum0 = s::Add(v_row_sum0, s::Exp(d, s::Neg(s::Mul(sq, v_gamma))));
        }

        off_diag += s::GetLane(s::SumOfLanes(d, s::Add(v_row_sum0, v_row_sum1)));

        // Scalar tail
        for (; j < nnz; ++j) {
            T diff = vi - vals[j];
            off_diag += std::exp(-gamma * diff * diff);
        }
    }

    // Double for symmetry
    sum += T(2) * off_diag;

    return sum;
}

// =============================================================================
// SECTION 3: Cross-Kernel with Cache Blocking
// =============================================================================

/// @brief Cache-blocked cross-kernel sum
///
/// Computes sum of k(x_i, y_j) for all i, j.
/// Uses L2-friendly blocking for better cache utilization.
template <typename T>
SCL_FORCE_INLINE T cross_kernel_sum_blocked(
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

    // Zero-Val interactions
    if (zeros_x > 0) {
        sum += static_cast<T>(zeros_x) * sum_y_unary;
    }
    if (zeros_y > 0) {
        sum += static_cast<T>(zeros_y) * sum_x_unary;
    }

    // Val-Val: blocked for L2 cache
    if (nnz_x == 0 || nnz_y == 0) {
        return sum;
    }

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto v_gamma = s::Set(d, gamma);

    // Block size tuned for L2 cache (~256KB)
    // Each block: BLOCK_X * BLOCK_Y * sizeof(T) * 2 (two vectors)
    constexpr size_t BLOCK_X = 64;
    constexpr size_t BLOCK_Y = 512;

    T cross_sum = T(0);

    // Block over x
    for (size_t bx = 0; bx < nnz_x; bx += BLOCK_X) {
        size_t bx_end = std::min(bx + BLOCK_X, static_cast<size_t>(nnz_x));

        // Block over y
        for (size_t by = 0; by < nnz_y; by += BLOCK_Y) {
            size_t by_end = std::min(by + BLOCK_Y, static_cast<size_t>(nnz_y));

            // Prefetch y block
            SCL_PREFETCH_READ(vals_y + by, 0);

            // Process block
            for (size_t i = bx; i < bx_end; ++i) {
                const T xi = vals_x[i];
                const auto v_xi = s::Set(d, xi);

                auto v_row_sum0 = s::Zero(d);
                auto v_row_sum1 = s::Zero(d);

                size_t j = by;

                // 2-way unrolled
                for (; j + 2 * lanes <= by_end; j += 2 * lanes) {
                    auto v_yj0 = s::Load(d, vals_y + j + 0 * lanes);
                    auto v_yj1 = s::Load(d, vals_y + j + 1 * lanes);

                    auto diff0 = s::Sub(v_xi, v_yj0);
                    auto diff1 = s::Sub(v_xi, v_yj1);

                    auto sq0 = s::Mul(diff0, diff0);
                    auto sq1 = s::Mul(diff1, diff1);

                    v_row_sum0 = s::Add(v_row_sum0, s::Exp(d, s::Neg(s::Mul(sq0, v_gamma))));
                    v_row_sum1 = s::Add(v_row_sum1, s::Exp(d, s::Neg(s::Mul(sq1, v_gamma))));
                }

                for (; j + lanes <= by_end; j += lanes) {
                    auto v_yj = s::Load(d, vals_y + j);
                    auto diff = s::Sub(v_xi, v_yj);
                    auto sq = s::Mul(diff, diff);
                    v_row_sum0 = s::Add(v_row_sum0, s::Exp(d, s::Neg(s::Mul(sq, v_gamma))));
                }

                cross_sum += s::GetLane(s::SumOfLanes(d, s::Add(v_row_sum0, v_row_sum1)));

                // Scalar tail
                for (; j < by_end; ++j) {
                    T diff = xi - vals_y[j];
                    cross_sum += std::exp(-gamma * diff * diff);
                }
            }
        }
    }

    sum += cross_sum;
    return sum;
}

} // namespace detail

// =============================================================================
// SECTION 4: Public API
// =============================================================================

/// @brief Compute unary exp terms with cache
///
/// @param vals Input values [nnz]
/// @param nnz Number of non-zeros
/// @param gamma RBF bandwidth
/// @param cache Output cache for exp terms [nnz], PRE-ALLOCATED
/// @param out_sum Output sum of exp terms
template <typename T>
SCL_FORCE_INLINE void unary_exp_sum_fast(
    const T* SCL_RESTRICT vals,
    Size nnz,
    T gamma,
    T* SCL_RESTRICT cache,
    T& out_sum
) {
    out_sum = detail::unary_exp_sum_ultra(vals, nnz, gamma, cache);
}

/// @brief Compute cross-kernel sum
///
/// @param vals_x X values [nnz_x]
/// @param nnz_x Number of non-zeros in X
/// @param vals_y Y values [nnz_y]
/// @param nnz_y Number of non-zeros in Y
/// @param N_x Total dimension of X
/// @param N_y Total dimension of Y
/// @param gamma RBF bandwidth
/// @param sum_x_unary Precomputed unary sum for X
/// @param sum_y_unary Precomputed unary sum for Y
/// @param out_sum Output cross-kernel sum
template <typename T>
SCL_FORCE_INLINE void cross_kernel_sum_fast(
    const T* SCL_RESTRICT vals_x, Size nnz_x,
    const T* SCL_RESTRICT vals_y, Size nnz_y,
    Size N_x, Size N_y,
    T gamma,
    T sum_x_unary,
    T sum_y_unary,
    T& out_sum
) {
    out_sum = detail::cross_kernel_sum_blocked(
        vals_x, nnz_x, N_x,
        vals_y, nnz_y, N_y,
        gamma, sum_x_unary, sum_y_unary
    );
}

/// @brief Compute self-kernel sum
///
/// @param vals Values [nnz]
/// @param nnz Number of non-zeros
/// @param N Total dimension
/// @param gamma RBF bandwidth
/// @param sum_unary Precomputed unary sum
/// @param out_sum Output self-kernel sum
template <typename T>
SCL_FORCE_INLINE void self_kernel_sum_fast(
    const T* SCL_RESTRICT vals,
    Size nnz,
    Size N,
    T gamma,
    T sum_unary,
    T& out_sum
) {
    out_sum = detail::self_kernel_sum_symmetric(vals, nnz, N, gamma, sum_unary);
}

/// @brief Full MMD computation for CustomSparse
template <typename T, bool IsCSR>
void mmd_rbf_custom_fast(
    const CustomSparse<T, IsCSR>& mat_x,
    const CustomSparse<T, IsCSR>& mat_y,
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

    // Parallel over primary dimension with thread-local caches
    scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
        thread_local std::vector<T> x_cache;
        thread_local std::vector<T> y_cache;

        auto vals_x = scl::primary_values(mat_x, p);
        auto vals_y = scl::primary_values(mat_y, p);
        Size nnz_x = static_cast<Size>(scl::primary_length(mat_x, p));
        Size nnz_y = static_cast<Size>(scl::primary_length(mat_y, p));

        // Early exit for all-zero
        if (SCL_UNLIKELY(nnz_x == 0 && nnz_y == 0)) {
            output[p] = T(0);
            return;
        }

        // Resize caches
        if (x_cache.size() < nnz_x) x_cache.resize(nnz_x);
        if (y_cache.size() < nnz_y) y_cache.resize(nnz_y);

        // Compute unary sums
        T sum_x_unary = (nnz_x > 0) 
            ? detail::unary_exp_sum_ultra(vals_x.ptr, nnz_x, gamma, x_cache.data())
            : T(0);
        T sum_y_unary = (nnz_y > 0)
            ? detail::unary_exp_sum_ultra(vals_y.ptr, nnz_y, gamma, y_cache.data())
            : T(0);

        // Compute kernel sums
        T sum_xx = detail::self_kernel_sum_symmetric(vals_x.ptr, nnz_x, secondary_x, gamma, sum_x_unary);
        T sum_yy = detail::self_kernel_sum_symmetric(vals_y.ptr, nnz_y, secondary_y, gamma, sum_y_unary);
        T sum_xy = detail::cross_kernel_sum_blocked(
            vals_x.ptr, nnz_x, secondary_x,
            vals_y.ptr, nnz_y, secondary_y,
            gamma, sum_x_unary, sum_y_unary
        );

        // MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
        T mmd2 = (sum_xx * inv_Nx2) + (sum_yy * inv_Ny2) - (T(2) * sum_xy * inv_NxNy);

        // Numerical stability: clamp to non-negative
        output[p] = std::max(mmd2, T(0));
    });
}

/// @brief Full MMD computation for VirtualSparse
template <typename T, bool IsCSR>
void mmd_rbf_virtual_fast(
    const VirtualSparse<T, IsCSR>& mat_x,
    const VirtualSparse<T, IsCSR>& mat_y,
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

    scl::threading::parallel_for(Index(0), primary_dim, [&](Index p) {
        thread_local std::vector<T> x_cache;
        thread_local std::vector<T> y_cache;

        auto vals_x = scl::primary_values(mat_x, p);
        auto vals_y = scl::primary_values(mat_y, p);
        Size nnz_x = static_cast<Size>(scl::primary_length(mat_x, p));
        Size nnz_y = static_cast<Size>(scl::primary_length(mat_y, p));

        if (SCL_UNLIKELY(nnz_x == 0 && nnz_y == 0)) {
            output[p] = T(0);
            return;
        }

        if (x_cache.size() < nnz_x) x_cache.resize(nnz_x);
        if (y_cache.size() < nnz_y) y_cache.resize(nnz_y);

        T sum_x_unary = (nnz_x > 0)
            ? detail::unary_exp_sum_ultra(vals_x.ptr, nnz_x, gamma, x_cache.data())
            : T(0);
        T sum_y_unary = (nnz_y > 0)
            ? detail::unary_exp_sum_ultra(vals_y.ptr, nnz_y, gamma, y_cache.data())
            : T(0);

        T sum_xx = detail::self_kernel_sum_symmetric(vals_x.ptr, nnz_x, secondary_x, gamma, sum_x_unary);
        T sum_yy = detail::self_kernel_sum_symmetric(vals_y.ptr, nnz_y, secondary_y, gamma, sum_y_unary);
        T sum_xy = detail::cross_kernel_sum_blocked(
            vals_x.ptr, nnz_x, secondary_x,
            vals_y.ptr, nnz_y, secondary_y,
            gamma, sum_x_unary, sum_y_unary
        );

        T mmd2 = (sum_xx * inv_Nx2) + (sum_yy * inv_Ny2) - (T(2) * sum_xy * inv_NxNy);
        output[p] = std::max(mmd2, T(0));
    });
}

// =============================================================================
// SECTION 5: Dispatcher
// =============================================================================

/// @brief Unified dispatcher for fast MMD
template <typename MatrixT>
    requires AnySparse<MatrixT>
void mmd_rbf_fast_dispatch(
    const MatrixT& mat_x,
    const MatrixT& mat_y,
    Array<typename MatrixT::ValueType> output,
    typename MatrixT::ValueType gamma = typename MatrixT::ValueType(1)
) {
    using T = typename MatrixT::ValueType;
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    if constexpr (std::is_same_v<MatrixT, CustomSparse<T, IsCSR>>) {
        mmd_rbf_custom_fast(mat_x, mat_y, output, gamma);
    } else if constexpr (std::is_same_v<MatrixT, VirtualSparse<T, IsCSR>>) {
        mmd_rbf_virtual_fast(mat_x, mat_y, output, gamma);
    } else {
        // Fallback: direct pointer access
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
            Size nnz_x = static_cast<Size>(scl::primary_length(mat_x, p));
            Size nnz_y = static_cast<Size>(scl::primary_length(mat_y, p));

            if (SCL_UNLIKELY(nnz_x == 0 && nnz_y == 0)) {
                output[p] = T(0);
                return;
            }

            if (x_cache.size() < nnz_x) x_cache.resize(nnz_x);
            if (y_cache.size() < nnz_y) y_cache.resize(nnz_y);

            T sum_x_unary = (nnz_x > 0)
                ? detail::unary_exp_sum_ultra(vals_x.ptr, nnz_x, gamma, x_cache.data())
                : T(0);
            T sum_y_unary = (nnz_y > 0)
                ? detail::unary_exp_sum_ultra(vals_y.ptr, nnz_y, gamma, y_cache.data())
                : T(0);

            T sum_xx = detail::self_kernel_sum_symmetric(vals_x.ptr, nnz_x, secondary_x, gamma, sum_x_unary);
            T sum_yy = detail::self_kernel_sum_symmetric(vals_y.ptr, nnz_y, secondary_y, gamma, sum_y_unary);
            T sum_xy = detail::cross_kernel_sum_blocked(
                vals_x.ptr, nnz_x, secondary_x,
                vals_y.ptr, nnz_y, secondary_y,
                gamma, sum_x_unary, sum_y_unary
            );

            T mmd2 = (sum_xx * inv_Nx2) + (sum_yy * inv_Ny2) - (T(2) * sum_xy * inv_NxNy);
            output[p] = std::max(mmd2, T(0));
        });
    }
}

} // namespace scl::kernel::mmd::fast
