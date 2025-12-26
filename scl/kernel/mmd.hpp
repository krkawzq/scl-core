#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/mmd_fast_impl.hpp"
#include "scl/kernel/mmd_mapped_impl.hpp"
#include "scl/kernel/mapped_common.hpp"

#include <cmath>
#include <vector>
#include <algorithm>

// =============================================================================
/// @file mmd.hpp
/// @brief Maximum Mean Discrepancy (MMD) with RBF Kernel
///
/// Unified entry point for MMD computation with automatic backend dispatch:
///
/// - CustomSparse / VirtualSparse: Uses mmd_fast_impl.hpp
/// - MappedCustomSparse / MappedVirtualSparse: Uses mmd_mapped_impl.hpp
///
/// Mathematical Background:
///
/// MMD^2(P, Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
///
/// For RBF kernel: k(a, b) = exp(-gamma * ||a - b||^2)
///
/// Sparse Decomposition (key to efficiency):
///
/// 1. Zero-Zero: k(0, 0) = 1
/// 2. Zero-Val: k(0, v) = exp(-gamma * v^2)  -- precomputed as "unary"
/// 3. Val-Val: k(u, v) = exp(-gamma * (u-v)^2)
///
/// Performance Optimizations:
///
/// 1. 8-way Unrolled SIMD: Maximizes instruction-level parallelism
/// 2. Unary Precomputation: exp(-gamma * v^2) computed once per feature
/// 3. Symmetry Exploitation: Self-kernel uses upper triangle only
/// 4. Cache-Blocked Cross: L2-friendly blocking for cross-kernel
/// 5. Chunk-Based Streaming: Optimal for memory-mapped data
///
/// Performance Targets:
///
/// - In-Memory: ~200M kernel evaluations/sec per core
/// - Mapped: ~150M kernel evaluations/sec per core (I/O bound)
// =============================================================================

namespace scl::kernel::mmd {

// =============================================================================
// SECTION 1: Detail Helpers (Generic Fallback)
// =============================================================================

namespace detail {

/// @brief Generic unary exp sum
template <typename T>
SCL_FORCE_INLINE T unary_exp_sum(
    const T* SCL_RESTRICT vals,
    Size nnz,
    T gamma,
    T* SCL_RESTRICT cache
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    const auto v_gamma = s::Set(d, gamma);
    auto v_sum = s::Zero(d);
    size_t k = 0;

    for (; k + lanes <= nnz; k += lanes) {
        auto v = s::Load(d, vals + k);
        auto sq = s::Mul(v, v);
        auto exp_v = s::Exp(d, s::Neg(s::Mul(sq, v_gamma)));
        s::Store(exp_v, d, cache + k);
        v_sum = s::Add(v_sum, exp_v);
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum));

    for (; k < nnz; ++k) {
        T val = vals[k];
        T exp_term = std::exp(-gamma * val * val);
        cache[k] = exp_term;
        sum += exp_term;
    }

    return sum;
}

/// @brief Generic self-kernel sum
template <typename T>
SCL_FORCE_INLINE T self_kernel_sum(
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

    // Diagonal Val-Val
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

/// @brief Generic cross-kernel sum
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
// SECTION 2: Generic MMD Implementation
// =============================================================================

/// @brief Generic MMD implementation for any sparse matrix type
template <typename MatrixT>
    requires AnySparse<MatrixT>
void mmd_rbf_generic(
    const MatrixT& mat_x,
    const MatrixT& mat_y,
    Array<typename MatrixT::ValueType> output,
    typename MatrixT::ValueType gamma = typename MatrixT::ValueType(1)
) {
    using T = typename MatrixT::ValueType;

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
            ? detail::unary_exp_sum(vals_x.ptr, nnz_x, gamma, x_cache.data())
            : T(0);
        T sum_y_unary = (nnz_y > 0)
            ? detail::unary_exp_sum(vals_y.ptr, nnz_y, gamma, y_cache.data())
            : T(0);

        T sum_xx = detail::self_kernel_sum(vals_x.ptr, nnz_x, secondary_x, gamma, sum_x_unary);
        T sum_yy = detail::self_kernel_sum(vals_y.ptr, nnz_y, secondary_y, gamma, sum_y_unary);
        T sum_xy = detail::cross_kernel_sum(
            vals_x.ptr, nnz_x, secondary_x,
            vals_y.ptr, nnz_y, secondary_y,
            gamma, sum_x_unary, sum_y_unary
        );

        T mmd2 = (sum_xx * inv_Nx2) + (sum_yy * inv_Ny2) - (T(2) * sum_xy * inv_NxNy);
        output[p] = std::max(mmd2, T(0));
    });
}

// =============================================================================
// SECTION 3: Unified Public API
// =============================================================================

/// @brief Compute MMD^2 between two distributions (unified dispatcher)
///
/// Automatically selects the optimal backend based on matrix type:
/// - CustomSparse / VirtualSparse: Fast in-memory implementation
/// - MappedCustomSparse / MappedVirtualSparse: Streaming mapped implementation
///
/// For CSC matrices: Compares gene expression distributions across samples
/// For CSR matrices: Compares sample feature distributions across genes
///
/// @tparam MatrixT Sparse matrix type (must satisfy AnySparse concept)
/// @param mat_x Reference matrix (distribution P)
/// @param mat_y Query matrix (distribution Q)
/// @param output Output buffer [size = primary_dim], PRE-ALLOCATED
/// @param gamma RBF kernel bandwidth (default 1.0)
template <typename MatrixT>
    requires AnySparse<MatrixT>
void mmd_rbf(
    const MatrixT& mat_x,
    const MatrixT& mat_y,
    Array<typename MatrixT::ValueType> output,
    typename MatrixT::ValueType gamma = typename MatrixT::ValueType(1)
) {
    using T = typename MatrixT::ValueType;
    constexpr bool IsCSR = std::is_same_v<typename MatrixT::Tag, TagCSR>;

    // Dispatch to optimized backend
    if constexpr (std::is_same_v<MatrixT, CustomSparse<T, IsCSR>>) {
        fast::mmd_rbf_custom_fast(mat_x, mat_y, output, gamma);
    } else if constexpr (std::is_same_v<MatrixT, VirtualSparse<T, IsCSR>>) {
        fast::mmd_rbf_virtual_fast(mat_x, mat_y, output, gamma);
    } else if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        mapped::mmd_rbf_mapped_dispatch<MatrixT, IsCSR>(mat_x, mat_y, output, gamma);
    } else {
        // Generic fallback
        mmd_rbf_generic(mat_x, mat_y, output, gamma);
    }
}

/// @brief Convenience wrapper: MMD^2 with default gamma=1.0
template <typename MatrixT>
    requires AnySparse<MatrixT>
void mmd(
    const MatrixT& mat_x,
    const MatrixT& mat_y,
    Array<typename MatrixT::ValueType> output
) {
    mmd_rbf(mat_x, mat_y, output, typename MatrixT::ValueType(1));
}

/// @brief Compute pairwise MMD^2 matrix
///
/// Computes MMD^2 for all pairs of features (columns in CSC, rows in CSR).
/// Output is a symmetric matrix: out[i,j] = MMD^2(feature_i, feature_j).
///
/// @tparam MatrixT Sparse matrix type
/// @param matrix Input sparse matrix
/// @param output Output matrix [primary_dim x primary_dim], PRE-ALLOCATED (row-major)
/// @param gamma RBF kernel bandwidth (default 1.0)
template <typename MatrixT>
    requires AnySparse<MatrixT>
void mmd_pairwise(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output,
    typename MatrixT::ValueType gamma = typename MatrixT::ValueType(1)
) {
    using T = typename MatrixT::ValueType;

    const Index n = scl::primary_size(matrix);
    const Size N = static_cast<Size>(scl::secondary_size(matrix));

    SCL_CHECK_DIM(output.len == static_cast<Size>(n * n), "MMD pairwise: Output size mismatch");

    // Precompute unary sums for all features
    std::vector<T> unary_sums(n);
    std::vector<std::vector<T>> unary_caches(n);

    scl::threading::parallel_for(Index(0), n, [&](Index p) {
        auto vals = scl::primary_values(matrix, p);
        Size nnz = static_cast<Size>(scl::primary_length(matrix, p));

        if (nnz == 0) {
            unary_sums[p] = T(0);
            return;
        }

        unary_caches[p].resize(nnz);
        unary_sums[p] = detail::unary_exp_sum(vals.ptr, nnz, gamma, unary_caches[p].data());
    });

    // Precompute self-kernel sums
    std::vector<T> self_sums(n);
    const T inv_N2 = T(1) / static_cast<T>(N * N);

    scl::threading::parallel_for(Index(0), n, [&](Index p) {
        auto vals = scl::primary_values(matrix, p);
        Size nnz = static_cast<Size>(scl::primary_length(matrix, p));
        self_sums[p] = detail::self_kernel_sum(vals.ptr, nnz, N, gamma, unary_sums[p]) * inv_N2;
    });

    // Compute upper triangle of MMD matrix
    scl::threading::parallel_for(Index(0), n, [&](Index i) {
        // Diagonal: MMD^2(i, i) = 0
        output[i * n + i] = T(0);

        auto vals_i = scl::primary_values(matrix, i);
        Size nnz_i = static_cast<Size>(scl::primary_length(matrix, i));
        T self_i = self_sums[i];
        T unary_i = unary_sums[i];

        for (Index j = i + 1; j < n; ++j) {
            auto vals_j = scl::primary_values(matrix, j);
            Size nnz_j = static_cast<Size>(scl::primary_length(matrix, j));

            T cross_sum = detail::cross_kernel_sum(
                vals_i.ptr, nnz_i, N,
                vals_j.ptr, nnz_j, N,
                gamma, unary_i, unary_sums[j]
            );

            T mmd2 = self_i + self_sums[j] - T(2) * cross_sum * inv_N2;
            mmd2 = std::max(mmd2, T(0));

            // Store symmetric
            output[i * n + j] = mmd2;
            output[j * n + i] = mmd2;
        }
    });
}

} // namespace scl::kernel::mmd
