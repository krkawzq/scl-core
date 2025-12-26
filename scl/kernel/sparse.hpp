#pragma once

#include "scl/core/type.hpp"
#include "scl/core/matrix.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath> // For std::sqrt

// =============================================================================
/// @file sparse.hpp
/// @brief Sparse Matrix Kernels (Statistics & Aggregations)
///
/// Implements high-performance statistical operations on CSR matrices.
/// optimized for row-wise access patterns common in single-cell genomics.
///
/// Optimizations:
/// - Multi-threading: Uses scl::threading::parallel_for over rows.
/// - Zero-Overhead: Iterates strictly over non-zero elements (NNZ).
/// - One-Pass: Computes Mean and Variance in a single pass over data.
///
// =============================================================================

namespace scl::kernel::sparse {

// =============================================================================
// 1. Basic Aggregations (Sum, Count)
// =============================================================================

/// @brief Compute sum of values for each row (Generic CSR-like matrices).
///
/// y_i = sum over j of A_{ij}
///
/// @tparam MatrixT Any CSR-like matrix type (CustomCSR, VirtualCSR, etc.)
/// @param input  Input CSR-like matrix (M x N).
/// @param output Output array of size M (sums).
template <CSRLike MatrixT>
void row_sums(const MatrixT& input, MutableSpan<typename MatrixT::ValueType> output) {
    using T = typename MatrixT::ValueType;
    SCL_CHECK_DIM(output.size == static_cast<Size>(input.rows), 
                  "Output size must match input rows");

    scl::threading::parallel_for(0, input.rows, [&](size_t i) {
        // CSR optimization: Iterate only non-zeros in the row
        Index row_idx = static_cast<Index>(i);
        auto row_vals = input.row_values(row_idx);
        Index len = input.row_length(row_idx);
        
        T sum = 0;
        // Compiler auto-vectorization friendly loop
        for (Index j = 0; j < len; ++j) {
            sum += row_vals[j];
        }
        output[i] = sum;
    });
}

/// @brief Count non-zero elements (NNZ) per row (Generic CSR-like matrices).
///
/// @tparam MatrixT Any CSR-like matrix type
/// @param input  Input CSR-like matrix.
/// @param output Output array of size M (counts).
template <CSRLike MatrixT>
void row_nnz(const MatrixT& input, MutableSpan<typename MatrixT::ValueType> output) {
    using T = typename MatrixT::ValueType;
    SCL_CHECK_DIM(output.size == static_cast<Size>(input.rows), 
                  "Output size must match input rows");

    scl::threading::parallel_for(0, input.rows, [&](size_t i) {
        // For CSR, NNZ is simply the row length
        Index row_idx = static_cast<Index>(i);
        Index len = input.row_length(row_idx);
        output[i] = static_cast<T>(len);
    });
}

// =============================================================================
// 2. Statistics (Mean, Variance)
// =============================================================================

/// @brief Compute Mean for each row (Generic CSR-like matrices).
///
/// mu_i = (1/N) * sum over j of A_{ij}
/// Note: N is the total number of columns (including zeros).
///
/// @tparam MatrixT Any CSR-like matrix type
/// @param input  Input CSR-like matrix (M x N).
/// @param output Output array of size M (means).
template <CSRLike MatrixT>
void row_means(const MatrixT& input, MutableSpan<typename MatrixT::ValueType> output) {
    using T = typename MatrixT::ValueType;
    SCL_CHECK_DIM(output.size == static_cast<Size>(input.rows), 
                  "Output size must match input rows");

    const T n_cols_inv = static_cast<T>(1.0) / static_cast<T>(input.cols);

    scl::threading::parallel_for(0, input.rows, [&](size_t i) {
        Index row_idx = static_cast<Index>(i);
        auto row_vals = input.row_values(row_idx);
        Index len = input.row_length(row_idx);
        
        T sum = 0;
        for (Index j = 0; j < len; ++j) {
            sum += row_vals[j];
        }
        output[i] = sum * n_cols_inv;
    });
}

/// @brief Compute Mean and Variance for each row efficiently (Generic CSR-like matrices).
///
/// Uses the efficient formula for sparse data (Population Variance):
/// Var(X) = E[X^2] - (E[X])^2
///
/// Why this formula?
/// - It allows iterating ONLY non-zeros.
/// - Standard Welford algorithm requires iterating zeros which is O(N) instead of O(NNZ).
/// - For high-dimensional sparse data, O(NNZ) is orders of magnitude faster.
///
/// @tparam MatrixT Any CSR-like matrix type
/// @param input    Input CSR-like matrix (M x N).
/// @param out_mean Output array for means (size M).
/// @param out_var  Output array for variances (size M).
template <CSRLike MatrixT>
void row_statistics(const MatrixT& input, 
                   MutableSpan<typename MatrixT::ValueType> out_mean, 
                   MutableSpan<typename MatrixT::ValueType> out_var) {
    using T = typename MatrixT::ValueType;
    SCL_CHECK_DIM(out_mean.size == static_cast<Size>(input.rows), "Mean output dimension mismatch");
    SCL_CHECK_DIM(out_var.size == static_cast<Size>(input.rows), "Var output dimension mismatch");

    const T n_cols = static_cast<T>(input.cols);
    const T n_cols_inv = static_cast<T>(1.0) / n_cols;

    scl::threading::parallel_for(0, input.rows, [&](size_t i) {
        Index row_idx = static_cast<Index>(i);
        auto row_vals = input.row_values(row_idx);
        Index len = input.row_length(row_idx);
        
        T sum = 0;
        T sum_sq = 0;

        // Single pass over non-zeros (Cache friendly)
        for (Index j = 0; j < len; ++j) {
            T val = row_vals[j];
            sum += val;
            sum_sq += val * val;
        }

        // E[X]
        T mean = sum * n_cols_inv;
        
        // E[X^2]
        T mean_sq = sum_sq * n_cols_inv;

        // Var = E[X^2] - (E[X])^2
        // Note: Theoretically can be negative due to precision, clamp to 0
        T var = mean_sq - (mean * mean);
        if (var < 0) var = 0;

        out_mean[i] = mean;
        out_var[i] = var;
    });
}

} // namespace scl::kernel::sparse
