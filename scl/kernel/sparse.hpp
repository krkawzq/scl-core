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

/// @brief Compute sum of values for each row.
///
/// y_i = sum over j of A_{ij}
///
/// @param input  Input CSR matrix (M x N).
/// @param output Output array of size M (sums).
template <typename T>
void row_sums(const CSRMatrix<T>& input, MutableSpan<T> output) {
    SCL_CHECK_DIM(output.size == static_cast<Size>(input.rows), 
                  "Output size must match input rows");

    scl::threading::parallel_for(0, input.rows, [&](size_t i) {
        // CSR optimization: Iterate only non-zeros in the row
        auto row_vals = input.row_values(static_cast<Index>(i));
        
        T sum = 0;
        // Compiler auto-vectorization friendly loop
        for (Size j = 0; j < row_vals.size; ++j) {
            sum += row_vals[j];
        }
        output[i] = sum;
    });
}

/// @brief Count non-zero elements (NNZ) per row.
///
/// @param input  Input CSR matrix.
/// @param output Output array of size M (counts).
template <typename T>
void row_nnz(const CSRMatrix<T>& input, MutableSpan<T> output) {
    SCL_CHECK_DIM(output.size == static_cast<Size>(input.rows), 
                  "Output size must match input rows");

    scl::threading::parallel_for(0, input.rows, [&](size_t i) {
        // For CSR, NNZ is simply the size of the row span
        // We calculate ptr[i+1] - ptr[i]
        Index start = input.indptr[i];
        Index end = input.indptr[i+1];
        output[i] = static_cast<T>(end - start);
    });
}

// =============================================================================
// 2. Statistics (Mean, Variance)
// =============================================================================

/// @brief Compute Mean for each row.
///
/// mu_i = (1/N) * sum over j of A_{ij}
/// Note: N is the total number of columns (including zeros).
///
/// @param input  Input CSR matrix (M x N).
/// @param output Output array of size M (means).
template <typename T>
void row_means(const CSRMatrix<T>& input, MutableSpan<T> output) {
    SCL_CHECK_DIM(output.size == static_cast<Size>(input.rows), 
                  "Output size must match input rows");

    const T n_cols_inv = static_cast<T>(1.0) / static_cast<T>(input.cols);

    scl::threading::parallel_for(0, input.rows, [&](size_t i) {
        auto row_vals = input.row_values(static_cast<Index>(i));
        
        T sum = 0;
        for (Size j = 0; j < row_vals.size; ++j) {
            sum += row_vals[j];
        }
        output[i] = sum * n_cols_inv;
    });
}

/// @brief Compute Mean and Variance for each row efficiently.
///
/// Uses the efficient formula for sparse data (Population Variance):
/// Var(X) = E[X^2] - (E[X])^2
///
/// Why this formula?
/// - It allows iterating ONLY non-zeros.
/// - Standard Welford algorithm requires iterating zeros which is O(N) instead of O(NNZ).
/// - For high-dimensional sparse data, O(NNZ) is orders of magnitude faster.
///
/// @param input    Input CSR matrix (M x N).
/// @param out_mean Output array for means (size M).
/// @param out_var  Output array for variances (size M).
template <typename T>
void row_statistics(const CSRMatrix<T>& input, 
                   MutableSpan<T> out_mean, 
                   MutableSpan<T> out_var) {
    SCL_CHECK_DIM(out_mean.size == static_cast<Size>(input.rows), "Mean output dimension mismatch");
    SCL_CHECK_DIM(out_var.size == static_cast<Size>(input.rows), "Var output dimension mismatch");

    const T n_cols = static_cast<T>(input.cols);
    const T n_cols_inv = static_cast<T>(1.0) / n_cols;

    scl::threading::parallel_for(0, input.rows, [&](size_t i) {
        auto row_vals = input.row_values(static_cast<Index>(i));
        
        T sum = 0;
        T sum_sq = 0;

        // Single pass over non-zeros (Cache friendly)
        for (Size j = 0; j < row_vals.size; ++j) {
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

// =============================================================================
// 3. Advanced (QC Metrics)
// =============================================================================

/// @brief Calculate QC metrics in one pass (similar to scanpy.pp.calculate_qc_metrics).
///
/// Computes:
/// 1. Sum (Total counts)
/// 2. Count > 0 (Number of genes expressed)
///
/// @param input      Input CSR matrix.
/// @param out_sums   Output array for row sums.
/// @param out_counts Output array for non-zero counts.
template <typename T>
void row_qc_metrics(const CSRMatrix<T>& input, 
                   MutableSpan<T> out_sums, 
                   MutableSpan<T> out_counts) {
    SCL_CHECK_DIM(out_sums.size == static_cast<Size>(input.rows), "Sums dim error");
    SCL_CHECK_DIM(out_counts.size == static_cast<Size>(input.rows), "Counts dim error");

    scl::threading::parallel_for(0, input.rows, [&](size_t i) {
        auto row_vals = input.row_values(static_cast<Index>(i));
        
        T sum = 0;
        // Count is simply the number of stored elements in CSR 
        // (assuming no explicit zeros are stored, which is standard)
        T count = static_cast<T>(row_vals.size); 

        for (Size j = 0; j < row_vals.size; ++j) {
            sum += row_vals[j];
        }

        out_sums[i] = sum;
        out_counts[i] = count;
    });
}

} // namespace scl::kernel::sparse
