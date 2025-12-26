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

namespace detail {

// =============================================================================
// Generic Implementation (Tag Dispatch)
// =============================================================================

/// @brief Generic implementation for row/column sums.
template <typename MatrixT>
SCL_FORCE_INLINE void sums_impl(
    const MatrixT& input,
    MutableSpan<typename MatrixT::ValueType> output
) {
    using T = typename MatrixT::ValueType;
    using Tag = typename MatrixT::Tag;
    
    if constexpr (std::is_same_v<Tag, TagCSR>) {
        SCL_CHECK_DIM(output.size == static_cast<Size>(input.rows), 
                      "Output size must match input rows");
        
        scl::threading::parallel_for(0, input.rows, [&](size_t i) {
            Index row_idx = static_cast<Index>(i);
            auto row_vals = input.row_values(row_idx);
            Index len = input.row_length(row_idx);
            
            T sum = 0;
            for (Index j = 0; j < len; ++j) {
                sum += row_vals[j];
            }
            output[i] = sum;
        });
    } else if constexpr (std::is_same_v<Tag, TagCSC>) {
        SCL_CHECK_DIM(output.size == static_cast<Size>(input.cols), 
                      "Output size must match input cols");
        
        scl::threading::parallel_for(0, input.cols, [&](size_t j) {
            Index col_idx = static_cast<Index>(j);
            auto col_vals = input.col_values(col_idx);
            Index len = input.col_length(col_idx);
            
            T sum = 0;
            for (Index k = 0; k < len; ++k) {
                sum += col_vals[k];
            }
            output[j] = sum;
        });
    }
}

/// @brief Generic implementation for row/column NNZ counts.
template <typename MatrixT>
SCL_FORCE_INLINE void nnz_impl(
    const MatrixT& input,
    MutableSpan<typename MatrixT::ValueType> output
) {
    using T = typename MatrixT::ValueType;
    using Tag = typename MatrixT::Tag;
    
    if constexpr (std::is_same_v<Tag, TagCSR>) {
        SCL_CHECK_DIM(output.size == static_cast<Size>(input.rows), 
                      "Output size must match input rows");
        
        scl::threading::parallel_for(0, input.rows, [&](size_t i) {
            Index row_idx = static_cast<Index>(i);
            Index len = input.row_length(row_idx);
            output[i] = static_cast<T>(len);
        });
    } else if constexpr (std::is_same_v<Tag, TagCSC>) {
        SCL_CHECK_DIM(output.size == static_cast<Size>(input.cols), 
                      "Output size must match input cols");
        
        scl::threading::parallel_for(0, input.cols, [&](size_t j) {
            Index col_idx = static_cast<Index>(j);
            Index len = input.col_length(col_idx);
            output[j] = static_cast<T>(len);
        });
    }
}

/// @brief Generic implementation for row/column means.
template <typename MatrixT>
SCL_FORCE_INLINE void means_impl(
    const MatrixT& input,
    MutableSpan<typename MatrixT::ValueType> output
) {
    using T = typename MatrixT::ValueType;
    using Tag = typename MatrixT::Tag;
    
    if constexpr (std::is_same_v<Tag, TagCSR>) {
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
    } else if constexpr (std::is_same_v<Tag, TagCSC>) {
        SCL_CHECK_DIM(output.size == static_cast<Size>(input.cols), 
                      "Output size must match input cols");
        
        const T n_rows_inv = static_cast<T>(1.0) / static_cast<T>(input.rows);
        
        scl::threading::parallel_for(0, input.cols, [&](size_t j) {
            Index col_idx = static_cast<Index>(j);
            auto col_vals = input.col_values(col_idx);
            Index len = input.col_length(col_idx);
            
            T sum = 0;
            for (Index k = 0; k < len; ++k) {
                sum += col_vals[k];
            }
            output[j] = sum * n_rows_inv;
        });
    }
}

/// @brief Generic implementation for row/column statistics.
template <typename MatrixT>
SCL_FORCE_INLINE void statistics_impl(
    const MatrixT& input,
    MutableSpan<typename MatrixT::ValueType> out_mean,
    MutableSpan<typename MatrixT::ValueType> out_var
) {
    using T = typename MatrixT::ValueType;
    using Tag = typename MatrixT::Tag;
    
    if constexpr (std::is_same_v<Tag, TagCSR>) {
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
            
            for (Index j = 0; j < len; ++j) {
                T val = row_vals[j];
                sum += val;
                sum_sq += val * val;
            }
            
            T mean = sum * n_cols_inv;
            T mean_sq = sum_sq * n_cols_inv;
            T var = mean_sq - (mean * mean);
            if (var < 0) var = 0;
            
            out_mean[i] = mean;
            out_var[i] = var;
        });
    } else if constexpr (std::is_same_v<Tag, TagCSC>) {
        SCL_CHECK_DIM(out_mean.size == static_cast<Size>(input.cols), "Mean output dimension mismatch");
        SCL_CHECK_DIM(out_var.size == static_cast<Size>(input.cols), "Var output dimension mismatch");
        
        const T n_rows = static_cast<T>(input.rows);
        const T n_rows_inv = static_cast<T>(1.0) / n_rows;
        
        scl::threading::parallel_for(0, input.cols, [&](size_t j) {
            Index col_idx = static_cast<Index>(j);
            auto col_vals = input.col_values(col_idx);
            Index len = input.col_length(col_idx);
            
            T sum = 0;
            T sum_sq = 0;
            
            for (Index k = 0; k < len; ++k) {
                T val = col_vals[k];
                sum += val;
                sum_sq += val * val;
            }
            
            T mean = sum * n_rows_inv;
            T mean_sq = sum_sq * n_rows_inv;
            T var = mean_sq - (mean * mean);
            if (var < 0) var = 0;
            
            out_mean[j] = mean;
            out_var[j] = var;
        });
    }
}

} // namespace detail

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
    detail::sums_impl(input, output);
}

/// @brief Compute sum of values for each column (Generic CSC-like matrices).
///
/// y_j = sum over i of A_{ij}
///
/// @tparam MatrixT Any CSC-like matrix type (CustomCSC, VirtualCSC, etc.)
/// @param input  Input CSC-like matrix (M x N).
/// @param output Output array of size N (sums).
template <CSCLike MatrixT>
void col_sums(const MatrixT& input, MutableSpan<typename MatrixT::ValueType> output) {
    detail::sums_impl(input, output);
}

/// @brief Count non-zero elements (NNZ) per row (Generic CSR-like matrices).
///
/// @tparam MatrixT Any CSR-like matrix type
/// @param input  Input CSR-like matrix.
/// @param output Output array of size M (counts).
template <CSRLike MatrixT>
void row_nnz(const MatrixT& input, MutableSpan<typename MatrixT::ValueType> output) {
    detail::nnz_impl(input, output);
}

/// @brief Count non-zero elements (NNZ) per column (Generic CSC-like matrices).
///
/// @tparam MatrixT Any CSC-like matrix type
/// @param input  Input CSC-like matrix.
/// @param output Output array of size N (counts).
template <CSCLike MatrixT>
void col_nnz(const MatrixT& input, MutableSpan<typename MatrixT::ValueType> output) {
    detail::nnz_impl(input, output);
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
    detail::means_impl(input, output);
}

/// @brief Compute Mean for each column (Generic CSC-like matrices).
///
/// mu_j = (1/M) * sum over i of A_{ij}
/// Note: M is the total number of rows (including zeros).
///
/// @tparam MatrixT Any CSC-like matrix type
/// @param input  Input CSC-like matrix (M x N).
/// @param output Output array of size N (means).
template <CSCLike MatrixT>
void col_means(const MatrixT& input, MutableSpan<typename MatrixT::ValueType> output) {
    detail::means_impl(input, output);
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
    detail::statistics_impl(input, out_mean, out_var);
}

/// @brief Compute Mean and Variance for each column efficiently (Generic CSC-like matrices).
///
/// Uses the efficient formula for sparse data (Population Variance):
/// Var(X) = E[X^2] - (E[X])^2
///
/// @tparam MatrixT Any CSC-like matrix type
/// @param input    Input CSC-like matrix (M x N).
/// @param out_mean Output array for means (size N).
/// @param out_var  Output array for variances (size N).
template <CSCLike MatrixT>
void col_statistics(const MatrixT& input, 
                   MutableSpan<typename MatrixT::ValueType> out_mean, 
                   MutableSpan<typename MatrixT::ValueType> out_var) {
    detail::statistics_impl(input, out_mean, out_var);
}

} // namespace scl::kernel::sparse
