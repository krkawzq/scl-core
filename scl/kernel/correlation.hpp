#pragma once

#include "scl/core/type.hpp"
#include "scl/core/matrix.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/simd.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/gram.hpp"

#include <cmath>
#include <algorithm>
#include <vector>

// =============================================================================
/// @file correlation.hpp
/// @brief High-Performance Pearson Correlation Kernel
///
/// Computes the Gene-Gene correlation matrix from sparse single-cell data.
///
/// Algorithm:
/// Pearson correlation r_xy is computed using the decomposition:
/// r_xy = (E[XY] - E[X]E[Y]) / (sigma_X * sigma_Y)
///
/// 1. Moments: Compute mu_i and sigma_i for each feature (column).
///    - Optimized scalar/SIMD pass over sparse values.
/// 2. Gram Matrix: Compute G = X^T * X (uncentered covariance).
///    - Uses scl::kernel::gram (Hybrid Merge/Galloping search).
/// 3. Finalize: Apply formula to dense result matrix.
///    - r_ij = (G_ij/N - mu_i * mu_j) * (1/sigma_i * 1/sigma_j)
///
/// Performance:
/// - Memory: O(N_genes) scratch space for stats.
/// - Throughput: Limits only by memory bandwidth for the dense write-back.
// =============================================================================

namespace scl::kernel::correlation {

namespace detail {

/// @brief Precomputed column statistics for Pearson correlation.
template <typename T>
struct ColStats {
    std::vector<T> means;
    std::vector<T> inv_stds; // 1.0 / std
};

/// @brief Compute Mean and 1/Std for each column in a CSC-like matrix.
///
/// Complexity: O(nnz)
template <CSCLike MatrixT>
SCL_FORCE_INLINE ColStats<typename MatrixT::ValueType> compute_col_stats(const MatrixT& matrix) {
    using T = typename MatrixT::ValueType;
    const Index C = matrix.cols;
    const Size R = static_cast<Size>(matrix.rows);
    const T inv_n = static_cast<T>(1.0) / static_cast<T>(R);

    ColStats<T> stats;
    stats.means.resize(C);
    stats.inv_stds.resize(C);

    scl::threading::parallel_for(0, C, [&](size_t j) {
        Index col_idx = static_cast<Index>(j);
        auto vals = matrix.col_values(col_idx);
        Index len = matrix.col_length(col_idx);
        
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();

        auto v_sum = s::Zero(d);
        auto v_sq_sum = s::Zero(d);
        Index k = 0;

        // SIMD Accumulation
        for (; k + static_cast<Index>(lanes) <= len; k += static_cast<Index>(lanes)) {
            auto v = s::Load(d, vals.ptr + k);
            v_sum = s::Add(v_sum, v);
            v_sq_sum = s::Add(v_sq_sum, s::Mul(v, v));
        }
        
        T sum = s::GetLane(s::SumOfLanes(d, v_sum));
        T sq_sum = s::GetLane(s::SumOfLanes(d, v_sq_sum));

        // Scalar Tail
        for (; k < len; ++k) {
            T v = vals[k];
            sum += v;
            sq_sum += v * v;
        }

        // Stats Calculation
        // Mean = Sum / N
        // Var = E[X^2] - (E[X])^2 = (SumSq / N) - Mean^2
        T mean = sum * inv_n;
        T mean_sq = mean * mean;
        T avg_sq = sq_sum * inv_n;
        T var = avg_sq - mean_sq;

        // Numerical stability: var can be -epsilon
        if (var < 0) var = 0;

        stats.means[j] = mean;
        
        // Handle constant columns (std = 0)
        if (var <= 0) {
            stats.inv_stds[j] = 0.0; 
        } else {
            stats.inv_stds[j] = static_cast<T>(1.0) / std::sqrt(var);
        }
    });

    return stats;
}

} // namespace detail

// =============================================================================
// Public API
// =============================================================================

/// @brief Compute Pearson Correlation Matrix from Sparse Input (Generic CSC-like matrices).
///
/// @tparam MatrixT Any CSC-like matrix type
/// @param matrix  Input sparse matrix (cells x genes).
/// @param output  Output dense correlation matrix (genes x genes).
///                Must be pre-allocated (size = cols * cols).
template <CSCLike MatrixT>
void pearson(const MatrixT& matrix, MutableSpan<typename MatrixT::ValueType> output) {
    using T = typename MatrixT::ValueType;
    const Index C = matrix.cols;
    const Size N_cells = static_cast<Size>(matrix.rows);
    SCL_CHECK_DIM(output.size == static_cast<Size>(C * C), "Pearson: Output size mismatch");

    // 1. Precompute Statistics (Mean, Std)
    // Runs in O(nnz), purely bandwidth bound
    auto stats = detail::compute_col_stats(matrix);

    // 2. Compute Gram Matrix (G = X^T * X)
    // The heavy lifting: sparse matrix multiplication
    scl::kernel::gram::gram(matrix, output);

    // 3. Finalize: Transform Gram to Correlation
    // corr_ij = (G_ij / N - mu_i * mu_j) * (1/sigma_i * 1/sigma_j)
    //
    // Optimization: This is a dense matrix operation.
    // We parallelize over rows of the result matrix.
    const T inv_n = static_cast<T>(1.0) / static_cast<T>(N_cells);

    scl::threading::parallel_for(0, C, [&](size_t i) {
        T* row_ptr = output.ptr + (i * C);
        T mu_i = stats.means[i];
        T inv_sig_i = stats.inv_stds[i];

        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();

        // Broadcast constants for row i
        const auto v_inv_n = s::Set(d, inv_n);
        const auto v_mu_i = s::Set(d, mu_i);
        const auto v_inv_sig_i = s::Set(d, inv_sig_i);
        const auto v_one = s::Set(d, static_cast<T>(1.0));
        const auto v_neg_one = s::Set(d, static_cast<T>(-1.0));

        size_t j = 0;
        
        // Vectorized loop over columns j
        for (; j + lanes <= static_cast<size_t>(C); j += lanes) {
            // Load G_ij
            auto v_g = s::Load(d, row_ptr + j);
            
            // Load stats for j
            auto v_mu_j = s::Load(d, stats.means.data() + j);
            auto v_inv_sig_j = s::Load(d, stats.inv_stds.data() + j);

            // Formula: (G * inv_n - mu_i * mu_j) * (inv_sig_i * inv_sig_j)
            auto v_cov = s::Sub(s::Mul(v_g, v_inv_n), s::Mul(v_mu_i, v_mu_j));
            auto v_norm = s::Mul(v_inv_sig_i, v_inv_sig_j);
            auto v_corr = s::Mul(v_cov, v_norm);

            // Clip to [-1, 1] for numerical stability
            v_corr = s::Min(v_corr, v_one);
            v_corr = s::Max(v_corr, v_neg_one);

            // Handle std=0 case (NaN protection)
            // If inv_sig is 0, result becomes 0 (or NaN if 0*Inf, but we check var<=0)
            // Implicitly handled if 0.0 is used.
            
            s::Store(v_corr, d, row_ptr + j);
        }

        // Scalar Tail
        for (; j < static_cast<size_t>(C); ++j) {
            T g_val = row_ptr[j];
            T mu_j = stats.means[j];
            T inv_sig_j = stats.inv_stds[j];

            T cov = (g_val * inv_n) - (mu_i * mu_j);
            T corr = cov * (inv_sig_i * inv_sig_j);

            // Clip
            if (corr > 1.0) corr = 1.0;
            if (corr < -1.0) corr = -1.0;
            
            // Fix for constant genes
            if (inv_sig_i == 0.0 || inv_sig_j == 0.0) corr = 0.0;

            row_ptr[j] = corr;
        }
    });
}

} // namespace scl::kernel::correlation
