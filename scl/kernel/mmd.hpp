#pragma once

#include "scl/core/type.hpp"
#include "scl/core/matrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <vector>

// =============================================================================
/// @file mmd.hpp
/// @brief Maximum Mean Discrepancy (MMD) Kernel
///
/// Computes MMD statistic with RBF (Gaussian) kernel between two distributions.
///
/// Definition:
///
/// MMD^2(P, Q) = E_{x,x' ~ P}[k(x,x')] + E_{y,y' ~ Q}[k(y,y')] - 2E_{x ~ P, y ~ Q}[k(x,y)]
///
/// For RBF kernel: k(a, b) = exp(-gamma * ||a - b||^2)
///
/// Sparse Optimization:
///
/// For sparse vectors, decompose kernel sum into three parts:
///
/// 1. Zero-Zero: Implicit zeros interact with implicit zeros -> k(0, 0) = 1
/// 2. Zero-Val: Implicit zeros with explicit values -> k(0, v) = exp(-gamma * v^2)
/// 3. Val-Val: Explicit values with explicit values -> k(u, v) = exp(-gamma * (u-v)^2)
///
/// Complexity: O(n^2 + m^2 + nm) where n, m are nnz (not total dimension).
///
/// Performance:
///
/// - SIMD: 4-8x speedup for distance + exp computation
/// - Cache: Precompute unary exp terms to avoid redundancy
/// - Parallelism: Feature-level (gene-wise) parallelism
/// - Throughput: ~100-200M kernel evaluations/sec per core
///
/// Use Cases:
///
/// - Batch Effect Detection: Compare cell distributions across batches
/// - Perturbation Response: Measure treatment effect on gene expression
/// - Distribution Matching: Two-sample test for single-cell data
// =============================================================================

namespace scl::kernel::mmd {

namespace detail {

// =============================================================================
// Core Kernel Computation Units
// =============================================================================

/// @brief Precompute unary exp terms: exp(-gamma * v^2) for all values.
///
/// This is needed for efficient Zero-Val interaction computation.
///
/// @param vals Input sparse values
/// @param gamma RBF bandwidth parameter
/// @param cache Output buffer for exp values
/// @return Sum of all exp terms
template <typename T>
SCL_FORCE_INLINE T unary_exp_sum(
    Span<const T> vals,
    T gamma,
    T* cache
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const Size nnz = vals.size;
    
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

/// @brief Compute self-kernel sum: sum over i and j of k(v_i, v_j).
///
/// Includes all three interaction types (Zero-Zero, Zero-Val, Val-Val).
///
/// @param vals Sparse vector values
/// @param N Total vector dimension (including implicit zeros)
/// @param gamma RBF bandwidth
/// @param sum_unary Sum of exp(-gamma * v^2) terms
/// @return Total kernel sum
template <typename T>
SCL_FORCE_INLINE T self_kernel_sum(
    Span<const T> vals,
    Size N,
    T gamma,
    T sum_unary
) {
    const Size nnz = vals.size;
    const Size n_zeros = N - nnz;
    
    T sum = static_cast<T>(0.0);

    // 1. Zero-Zero interactions: count × 1.0
    sum += static_cast<T>(n_zeros * n_zeros);

    // 2. Zero-Val interactions (symmetric)
    if (n_zeros > 0) {
        sum += static_cast<T>(2.0) * static_cast<T>(n_zeros) * sum_unary;
    }

    // 3. Val-Val interactions
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto v_gamma = s::Set(d, gamma);
    
    // Diagonal: k(v, v) = 1
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

/// @brief Compute cross-kernel sum: sum over i and j of k(x_i, y_j).
///
/// @param vals_x First vector values
/// @param N_x Total dimension of first vector
/// @param vals_y Second vector values
/// @param N_y Total dimension of second vector
/// @param gamma RBF bandwidth
/// @param sum_x_unary Sum of exp(-gamma * x^2)
/// @param sum_y_unary Sum of exp(-gamma * y^2)
/// @return Total cross-kernel sum
template <typename T>
SCL_FORCE_INLINE T cross_kernel_sum(
    Span<const T> vals_x, Size N_x,
    Span<const T> vals_y, Size N_y,
    T gamma,
    T sum_x_unary,
    T sum_y_unary
) {
    const Size nnz_x = vals_x.size;
    const Size nnz_y = vals_y.size;
    const Size zeros_x = N_x - nnz_x;
    const Size zeros_y = N_y - nnz_y;

    T sum = static_cast<T>(0.0);

    // 1. Zero-Zero
    sum += static_cast<T>(zeros_x * zeros_y);

    // 2. Zero(X) - Val(Y)
    if (zeros_x > 0) {
        sum += static_cast<T>(zeros_x) * sum_y_unary;
    }

    // 3. Val(X) - Zero(Y)
    if (zeros_y > 0) {
        sum += static_cast<T>(zeros_y) * sum_x_unary;
    }

    // 4. Val(X) - Val(Y) (full rectangular)
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

/// @brief Compute MMD^2 statistic for each feature (gene).
///
/// Compares distribution of feature j in matrix X vs matrix Y using RBF kernel.
///
/// Input: Two CSC matrices (cells x genes), typically:
/// - X: Reference group (e.g., control cells)
/// - Y: Query group (e.g., treated cells)
///
/// Output: MMD^2 value per gene (higher = more different distributions).
///
/// @param mat_x Reference matrix (CSC format)
/// @param mat_y Query matrix (CSC format)
/// @param output Output buffer [size = n_genes]
/// @param gamma RBF kernel bandwidth (default 1.0). Smaller = more sensitive.
template <typename T>
void mmd_rbf(
    const CSCMatrix<T>& mat_x,
    const CSCMatrix<T>& mat_y,
    MutableSpan<T> output,
    T gamma = static_cast<T>(1.0)
) {
    const Index n_genes = mat_x.cols;
    
    SCL_CHECK_DIM(mat_y.cols == n_genes, "MMD: Matrix column count mismatch");
    SCL_CHECK_DIM(output.size == static_cast<Size>(n_genes), 
                  "MMD: Output size mismatch");

    const Size N_x = static_cast<Size>(mat_x.rows);
    const Size N_y = static_cast<Size>(mat_y.rows);
    
    // Normalization factors
    const T inv_Nx2 = static_cast<T>(1.0) / static_cast<T>(N_x * N_x);
    const T inv_Ny2 = static_cast<T>(1.0) / static_cast<T>(N_y * N_y);
    const T inv_NxNy = static_cast<T>(1.0) / static_cast<T>(N_x * N_y);

    // Chunk-based parallelism for workspace reuse
    constexpr size_t CHUNK_SIZE = 32;
    const size_t n_chunks = (static_cast<size_t>(n_genes) + CHUNK_SIZE - 1) / CHUNK_SIZE;

    scl::threading::parallel_for(0, n_chunks, [&](size_t chunk_idx) {
        // Thread-local buffers for unary exp caching
        std::vector<T> x_unary_cache;
        std::vector<T> y_unary_cache;
        
        // Reserve for typical sparsity (~5% density for single-cell)
        x_unary_cache.reserve(static_cast<size_t>(N_x) / 20);
        y_unary_cache.reserve(static_cast<size_t>(N_y) / 20);

        size_t j_start = chunk_idx * CHUNK_SIZE;
        size_t j_end = std::min(static_cast<size_t>(n_genes), j_start + CHUNK_SIZE);

        for (size_t j = j_start; j < j_end; ++j) {
            const Index col_idx = static_cast<Index>(j);
            
            auto vals_x = mat_x.col_values(col_idx);
            auto vals_y = mat_y.col_values(col_idx);

            // Skip if both vectors are empty
            if (SCL_UNLIKELY(vals_x.size == 0 && vals_y.size == 0)) {
                output[col_idx] = static_cast<T>(0.0);
                continue;
            }

            // ---------------------------------------------------------------
            // Step 1: Precompute Unary Exp Terms
            // ---------------------------------------------------------------
            
            // Ensure buffers are large enough
            if (x_unary_cache.size() < vals_x.size) {
                x_unary_cache.resize(vals_x.size);
            }
            if (y_unary_cache.size() < vals_y.size) {
                y_unary_cache.resize(vals_y.size);
            }

            T sum_x_unary = detail::unary_exp_sum(
                vals_x, gamma, x_unary_cache.data()
            );
            T sum_y_unary = detail::unary_exp_sum(
                vals_y, gamma, y_unary_cache.data()
            );

            // ---------------------------------------------------------------
            // Step 2: Compute Three MMD Terms
            // ---------------------------------------------------------------
            
            T sum_xx = detail::self_kernel_sum(
                vals_x, N_x, gamma, sum_x_unary
            );

            T sum_yy = detail::self_kernel_sum(
                vals_y, N_y, gamma, sum_y_unary
            );

            T sum_xy = detail::cross_kernel_sum(
                vals_x, N_x, vals_y, N_y, gamma, sum_x_unary, sum_y_unary
            );

            // ---------------------------------------------------------------
            // Step 3: Assemble MMD^2
            // ---------------------------------------------------------------
            
            T mmd2 = (sum_xx * inv_Nx2) + (sum_yy * inv_Ny2) - 
                     (static_cast<T>(2.0) * sum_xy * inv_NxNy);

            // Numerical stability: clip to non-negative (theoretical guarantee)
            if (mmd2 < static_cast<T>(0.0)) {
                mmd2 = static_cast<T>(0.0);
            }

            output[col_idx] = mmd2;
        }
    });
}

} // namespace scl::kernel::mmd

