#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <vector>

// =============================================================================
/// @file mmd.hpp
/// @brief Maximum Mean Discrepancy (MMD) with RBF Kernel
///
/// Computes MMD^2(P, Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
/// For RBF kernel: k(a, b) = exp(-gamma * ||a - b||^2)
///
/// Sparse Optimization:
/// 1. Zero-Zero: k(0, 0) = 1
/// 2. Zero-Val: k(0, v) = exp(-gamma * v^2)
/// 3. Val-Val: k(u, v) = exp(-gamma * (u-v)^2)
///
/// Performance:
/// - SIMD: 4-8x speedup for exp computation
/// - Cache: Precompute unary terms
/// - Parallel: Feature-level parallelism
/// - Throughput: ~100-200M kernel evals/sec per core
// =============================================================================

namespace scl::kernel::mmd {

namespace detail {

/// @brief Precompute exp(-gamma * v^2) for all values (SIMD)
template <typename T>
SCL_FORCE_INLINE T unary_exp_sum(
    Array<const T> vals,
    T gamma,
    T* cache
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const Size nnz = vals.size();
    
    const auto v_gamma = s::Set(d, gamma);
    auto v_sum = s::Zero(d);
    size_t k = 0;
    
    for (; k + lanes <= nnz; k += lanes) {
        auto v = s::Load(d, vals.ptr + k);
        auto v_sq = s::Mul(v, v);
        auto v_exp = s::Exp(d, s::Neg(s::Mul(v_sq, v_gamma)));
        s::Store(v_exp, d, cache + k);
        v_sum = s::Add(v_sum, v_exp);
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

/// @brief Compute self-kernel sum (SIMD optimized)
template <typename T>
SCL_FORCE_INLINE T self_kernel_sum(
    Array<const T> vals,
    Size N,
    T gamma,
    T sum_unary
) {
    const Size nnz = vals.size();
    const Size n_zeros = N - nnz;
    
    T sum = static_cast<T>(0.0);

    // Zero-Zero interactions
    sum += static_cast<T>(n_zeros * n_zeros);

    // Zero-Val interactions (symmetric)
    if (n_zeros > 0) {
        sum += static_cast<T>(2.0) * static_cast<T>(n_zeros) * sum_unary;
    }

    // Val-Val interactions
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto v_gamma = s::Set(d, gamma);
    
    // Diagonal
    sum += static_cast<T>(nnz);

    // Off-diagonal (upper triangle, then double)
    T off_diag = static_cast<T>(0.0);

    for (size_t i = 0; i < nnz; ++i) {
        const T vi = vals[i];
        const auto v_vi = s::Set(d, vi);
        
        auto v_row_sum = s::Zero(d);
        size_t j = i + 1;
        
        for (; j + lanes <= nnz; j += lanes) {
            auto v_vj = s::Load(d, vals.ptr + j);
            auto v_diff = s::Sub(v_vi, v_vj);
            auto v_sq = s::Mul(v_diff, v_diff);
            auto v_exp = s::Exp(d, s::Neg(s::Mul(v_sq, v_gamma)));
            v_row_sum = s::Add(v_row_sum, v_exp);
        }
        
        off_diag += s::GetLane(s::SumOfLanes(d, v_row_sum));

        for (; j < nnz; ++j) {
            T diff = vi - vals[j];
            off_diag += std::exp(-gamma * diff * diff);
        }
    }

    sum += static_cast<T>(2.0) * off_diag;
    return sum;
}

/// @brief Compute cross-kernel sum (SIMD optimized)
template <typename T>
SCL_FORCE_INLINE T cross_kernel_sum(
    Array<const T> vals_x, Size N_x,
    Array<const T> vals_y, Size N_y,
    T gamma,
    T sum_x_unary,
    T sum_y_unary
) {
    const Size nnz_x = vals_x.size();
    const Size nnz_y = vals_y.size();
    const Size zeros_x = N_x - nnz_x;
    const Size zeros_y = N_y - nnz_y;

    T sum = static_cast<T>(0.0);

    // Zero-Zero
    sum += static_cast<T>(zeros_x * zeros_y);

    // Zero(X) - Val(Y)
    if (zeros_x > 0) {
        sum += static_cast<T>(zeros_x) * sum_y_unary;
    }

    // Val(X) - Zero(Y)
    if (zeros_y > 0) {
        sum += static_cast<T>(zeros_y) * sum_x_unary;
    }

    // Val(X) - Val(Y)
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto v_gamma = s::Set(d, gamma);

    T cross_sum = static_cast<T>(0.0);

    for (size_t i = 0; i < nnz_x; ++i) {
        const T xi = vals_x[i];
        const auto v_xi = s::Set(d, xi);
        
        auto v_row_sum = s::Zero(d);
        size_t j = 0;

        for (; j + lanes <= nnz_y; j += lanes) {
            auto v_yj = s::Load(d, vals_y.ptr + j);
            auto v_diff = s::Sub(v_xi, v_yj);
            auto v_sq = s::Mul(v_diff, v_diff);
            auto v_exp = s::Exp(d, s::Neg(s::Mul(v_sq, v_gamma)));
            v_row_sum = s::Add(v_row_sum, v_exp);
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
// Public API
// =============================================================================

/// @brief Compute MMD^2 between two distributions (unified for CSR/CSC)
///
/// For CSC: Compares gene expression distributions
/// For CSR: Compares sample feature distributions
///
/// @param mat_x Reference matrix
/// @param mat_y Query matrix
/// @param output Output buffer [size = primary_dim]
/// @param gamma RBF kernel bandwidth (default 1.0)
template <typename MatrixT>
    requires AnySparse<MatrixT>
void mmd_rbf(
    const MatrixT& mat_x,
    const MatrixT& mat_y,
    Array<typename MatrixT::ValueType> output,
    typename MatrixT::ValueType gamma = static_cast<typename MatrixT::ValueType>(1.0)
) {
    using T = typename MatrixT::ValueType;
    
    const Index primary_dim = scl::primary_size(mat_x);
    const Size secondary_x = static_cast<Size>(scl::secondary_size(mat_x));
    const Size secondary_y = static_cast<Size>(scl::secondary_size(mat_y));
    
    SCL_CHECK_DIM(scl::primary_size(mat_y) == primary_dim, 
                  "MMD: Primary dimension mismatch");
    SCL_CHECK_DIM(output.size() == static_cast<Size>(primary_dim), 
                  "MMD: Output size mismatch");
    
    const T inv_Nx2 = static_cast<T>(1.0) / static_cast<T>(secondary_x * secondary_x);
    const T inv_Ny2 = static_cast<T>(1.0) / static_cast<T>(secondary_y * secondary_y);
    const T inv_NxNy = static_cast<T>(1.0) / static_cast<T>(secondary_x * secondary_y);

    constexpr size_t CHUNK_SIZE = 32;
    const size_t n_chunks = (static_cast<size_t>(primary_dim) + CHUNK_SIZE - 1) / CHUNK_SIZE;

    scl::threading::parallel_for(0, n_chunks, [&](size_t chunk_idx) {
        std::vector<T> x_unary_cache;
        std::vector<T> y_unary_cache;
        
        x_unary_cache.reserve(secondary_x / 20);
        y_unary_cache.reserve(secondary_y / 20);

        size_t p_start = chunk_idx * CHUNK_SIZE;
        size_t p_end = std::min(static_cast<size_t>(primary_dim), p_start + CHUNK_SIZE);

        for (size_t p = p_start; p < p_end; ++p) {
            const Index primary_idx = static_cast<Index>(p);
            
            auto vals_x = scl::primary_values(mat_x, primary_idx);
            auto vals_y = scl::primary_values(mat_y, primary_idx);
            Index len_x = scl::primary_length(mat_x, primary_idx);
            Index len_y = scl::primary_length(mat_y, primary_idx);

            if (SCL_UNLIKELY(len_x == 0 && len_y == 0)) {
                output[primary_idx] = static_cast<T>(0.0);
                continue;
            }

            if (x_unary_cache.size() < static_cast<size_t>(len_x)) {
                x_unary_cache.resize(static_cast<size_t>(len_x));
            }
            if (y_unary_cache.size() < static_cast<size_t>(len_y)) {
                y_unary_cache.resize(static_cast<size_t>(len_y));
            }

            Array<const T> span_x(vals_x.ptr, static_cast<Size>(len_x));
            Array<const T> span_y(vals_y.ptr, static_cast<Size>(len_y));
            
            T sum_x_unary = detail::unary_exp_sum(span_x, gamma, x_unary_cache.data());
            T sum_y_unary = detail::unary_exp_sum(span_y, gamma, y_unary_cache.data());

            T sum_xx = detail::self_kernel_sum(span_x, secondary_x, gamma, sum_x_unary);
            T sum_yy = detail::self_kernel_sum(span_y, secondary_y, gamma, sum_y_unary);
            T sum_xy = detail::cross_kernel_sum(span_x, secondary_x, span_y, secondary_y, 
                                               gamma, sum_x_unary, sum_y_unary);

            T mmd2 = (sum_xx * inv_Nx2) + (sum_yy * inv_Ny2) - 
                     (static_cast<T>(2.0) * sum_xy * inv_NxNy);

            if (mmd2 < static_cast<T>(0.0)) {
                mmd2 = static_cast<T>(0.0);
            }

            output[primary_idx] = mmd2;
        }
    });
}

} // namespace scl::kernel::mmd
