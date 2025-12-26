#pragma once

#include "scl/core/type.hpp"
#include "scl/core/matrix.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
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

// =============================================================================
// 3. High-Performance Matrix Utilities (for Python bindings)
// =============================================================================

/// @brief Compute row/column lengths from indptr (parallel diff).
///
/// This is a fundamental operation for sparse matrix manipulation.
/// Computes: lengths[i] = indptr[i+1] - indptr[i]
///
/// @param indptr Row/column pointers [n+1]
/// @param n Number of rows/columns
/// @param lengths Output lengths [n]
template <typename T>
SCL_FORCE_INLINE void compute_lengths(
    const T* indptr,
    Size n,
    T* lengths
) {
    SCL_CHECK_ARG(indptr != nullptr, "compute_lengths: indptr is null");
    SCL_CHECK_ARG(lengths != nullptr, "compute_lengths: lengths is null");
    
    // Parallel diff operation
    scl::threading::parallel_for(0, n, [&](size_t i) {
        lengths[i] = indptr[i + 1] - indptr[i];
    });
}

/// @brief Inspect CSR row slice: compute output nnz.
///
/// This is used to pre-allocate memory before materializing a slice.
///
/// @param indptr Source indptr [src_rows+1]
/// @param row_indices Rows to keep [n_keep]
/// @param n_keep Number of rows to keep
/// @return Total nnz in output
template <typename T>
SCL_FORCE_INLINE Size inspect_slice_rows(
    const T* indptr,
    const T* row_indices,
    Size n_keep
) {
    SCL_CHECK_ARG(indptr != nullptr, "inspect_slice_rows: indptr is null");
    SCL_CHECK_ARG(row_indices != nullptr, "inspect_slice_rows: row_indices is null");
    
    Size total_nnz = 0;
    for (Size i = 0; i < n_keep; ++i) {
        T row_idx = row_indices[i];
        total_nnz += (indptr[row_idx + 1] - indptr[row_idx]);
    }
    
    return total_nnz;
}

/// @brief Materialize CSR row slice: copy selected rows.
///
/// Zero-overhead bulk copy using memcpy for contiguous data.
///
/// @param src_data Source data [src_nnz]
/// @param src_indices Source column indices [src_nnz]
/// @param src_indptr Source indptr [src_rows+1]
/// @param row_indices Rows to keep [n_keep]
/// @param n_keep Number of rows to keep
/// @param dst_data Destination data [out_nnz]
/// @param dst_indices Destination indices [out_nnz]
/// @param dst_indptr Destination indptr [n_keep+1]
template <typename T>
SCL_FORCE_INLINE void materialize_slice_rows(
    const T* src_data,
    const Index* src_indices,
    const Index* src_indptr,
    const Index* row_indices,
    Size n_keep,
    T* dst_data,
    Index* dst_indices,
    Index* dst_indptr
) {
    SCL_CHECK_ARG(src_data != nullptr, "materialize_slice_rows: src_data is null");
    SCL_CHECK_ARG(src_indices != nullptr, "materialize_slice_rows: src_indices is null");
    SCL_CHECK_ARG(src_indptr != nullptr, "materialize_slice_rows: src_indptr is null");
    SCL_CHECK_ARG(row_indices != nullptr, "materialize_slice_rows: row_indices is null");
    SCL_CHECK_ARG(dst_data != nullptr, "materialize_slice_rows: dst_data is null");
    SCL_CHECK_ARG(dst_indices != nullptr, "materialize_slice_rows: dst_indices is null");
    SCL_CHECK_ARG(dst_indptr != nullptr, "materialize_slice_rows: dst_indptr is null");
    
    dst_indptr[0] = 0;
    Index write_pos = 0;
    
    for (Size i = 0; i < n_keep; ++i) {
        Index src_row = row_indices[i];
        Index start = src_indptr[src_row];
        Index end = src_indptr[src_row + 1];
        Index length = end - start;
        
        // Bulk copy (zero-overhead for large blocks)
        scl::memory::copy_fast(
            Span<const T>(src_data + start, length),
            MutableSpan<T>(dst_data + write_pos, length)
        );
        scl::memory::copy_fast(
            Span<const Index>(src_indices + start, length),
            MutableSpan<Index>(dst_indices + write_pos, length)
        );
        
        write_pos += length;
        dst_indptr[i + 1] = write_pos;
    }
}

/// @brief Inspect CSR column filter: compute output nnz.
///
/// Counts how many non-zeros will remain after filtering columns.
///
/// @param src_indices Source column indices [src_nnz]
/// @param src_indptr Source indptr [rows+1]
/// @param rows Number of rows
/// @param col_mask Column mask [src_cols], 1=keep, 0=drop
/// @return Total nnz after filtering
SCL_FORCE_INLINE Size inspect_filter_cols(
    const Index* src_indices,
    const Index* src_indptr,
    Index rows,
    const uint8_t* col_mask
) {
    SCL_CHECK_ARG(src_indices != nullptr, "inspect_filter_cols: src_indices is null");
    SCL_CHECK_ARG(src_indptr != nullptr, "inspect_filter_cols: src_indptr is null");
    SCL_CHECK_ARG(col_mask != nullptr, "inspect_filter_cols: col_mask is null");
    
    Size total_nnz = 0;
    for (Index i = 0; i < rows; ++i) {
        Index start = src_indptr[i];
        Index end = src_indptr[i + 1];
        
        for (Index k = start; k < end; ++k) {
            if (col_mask[src_indices[k]]) {
                ++total_nnz;
            }
        }
    }
    
    return total_nnz;
}

/// @brief Materialize CSR column filter: copy filtered columns.
///
/// Filters and remaps column indices in one pass.
///
/// @param src_data Source data [src_nnz]
/// @param src_indices Source column indices [src_nnz]
/// @param src_indptr Source indptr [rows+1]
/// @param rows Number of rows
/// @param col_mask Column mask [src_cols], 1=keep, 0=drop
/// @param col_mapping New column indices [src_cols], where keep columns map to new indices
/// @param dst_data Destination data [out_nnz]
/// @param dst_indices Destination indices [out_nnz]
/// @param dst_indptr Destination indptr [rows+1]
template <typename T>
SCL_FORCE_INLINE void materialize_filter_cols(
    const T* src_data,
    const Index* src_indices,
    const Index* src_indptr,
    Index rows,
    const uint8_t* col_mask,
    const Index* col_mapping,
    T* dst_data,
    Index* dst_indices,
    Index* dst_indptr
) {
    SCL_CHECK_ARG(src_data != nullptr, "materialize_filter_cols: src_data is null");
    SCL_CHECK_ARG(src_indices != nullptr, "materialize_filter_cols: src_indices is null");
    SCL_CHECK_ARG(src_indptr != nullptr, "materialize_filter_cols: src_indptr is null");
    SCL_CHECK_ARG(col_mask != nullptr, "materialize_filter_cols: col_mask is null");
    SCL_CHECK_ARG(col_mapping != nullptr, "materialize_filter_cols: col_mapping is null");
    SCL_CHECK_ARG(dst_data != nullptr, "materialize_filter_cols: dst_data is null");
    SCL_CHECK_ARG(dst_indices != nullptr, "materialize_filter_cols: dst_indices is null");
    SCL_CHECK_ARG(dst_indptr != nullptr, "materialize_filter_cols: dst_indptr is null");
    
    dst_indptr[0] = 0;
    Index write_pos = 0;
    
    for (Index i = 0; i < rows; ++i) {
        Index start = src_indptr[i];
        Index end = src_indptr[i + 1];
        
        for (Index k = start; k < end; ++k) {
            Index old_col = src_indices[k];
            if (col_mask[old_col]) {
                dst_data[write_pos] = src_data[k];
                dst_indices[write_pos] = col_mapping[old_col];
                ++write_pos;
            }
        }
        
        dst_indptr[i + 1] = write_pos;
    }
}

/// @brief Align rows (with drop and pad support).
///
/// This is the core operation for AnnData-style row alignment.
/// - Map entries: old_to_new[i] = j means old_row[i] -> new_row[j]
/// - Drop entries: old_to_new[i] = -1 means drop old_row[i]
/// - Pad entries: If max(old_to_new) >= new_rows, result has sparse zeros
///
/// @param src_data Source data [src_nnz]
/// @param src_indices Source column indices [src_nnz]
/// @param src_indptr Source indptr [src_rows+1]
/// @param src_rows Number of source rows
/// @param old_to_new Mapping [src_rows], -1=drop
/// @param new_rows Target number of rows
/// @param dst_data Destination data [out_nnz]
/// @param dst_indices Destination indices [out_nnz]
/// @param dst_indptr Destination indptr [new_rows+1]
/// @param dst_row_lengths Destination row lengths [new_rows]
template <typename T>
SCL_FORCE_INLINE void align_rows(
    const T* src_data,
    const Index* src_indices,
    const Index* src_indptr,
    Index src_rows,
    const Index* old_to_new,
    Index new_rows,
    T* dst_data,
    Index* dst_indices,
    Index* dst_indptr,
    Index* dst_row_lengths
) {
    SCL_CHECK_ARG(src_data != nullptr, "align_rows: src_data is null");
    SCL_CHECK_ARG(src_indices != nullptr, "align_rows: src_indices is null");
    SCL_CHECK_ARG(src_indptr != nullptr, "align_rows: src_indptr is null");
    SCL_CHECK_ARG(old_to_new != nullptr, "align_rows: old_to_new is null");
    SCL_CHECK_ARG(dst_data != nullptr, "align_rows: dst_data is null");
    SCL_CHECK_ARG(dst_indices != nullptr, "align_rows: dst_indices is null");
    SCL_CHECK_ARG(dst_indptr != nullptr, "align_rows: dst_indptr is null");
    SCL_CHECK_ARG(dst_row_lengths != nullptr, "align_rows: dst_row_lengths is null");
    
    // Initialize lengths to zero
    scl::memory::zero(MutableSpan<Index>(dst_row_lengths, new_rows));
    
    // First pass: count lengths per new row
    for (Index i = 0; i < src_rows; ++i) {
        Index new_i = old_to_new[i];
        if (new_i >= 0) {
            Index length = src_indptr[i + 1] - src_indptr[i];
            dst_row_lengths[new_i] += length;
        }
    }
    
    // Build indptr from lengths
    dst_indptr[0] = 0;
    for (Index i = 0; i < new_rows; ++i) {
        dst_indptr[i + 1] = dst_indptr[i] + dst_row_lengths[i];
    }
    
    // Temp array to track write positions
    std::vector<Index> temp_pos(new_rows);
    for (Index i = 0; i < new_rows; ++i) {
        temp_pos[i] = dst_indptr[i];
    }
    
    // Second pass: copy data
    for (Index i = 0; i < src_rows; ++i) {
        Index new_i = old_to_new[i];
        if (new_i >= 0) {
            Index start = src_indptr[i];
            Index end = src_indptr[i + 1];
            Index length = end - start;
            
            Index write_pos = temp_pos[new_i];
            
            // Bulk copy
            scl::memory::copy_fast(
                Span<const T>(src_data + start, length),
                MutableSpan<T>(dst_data + write_pos, length)
            );
            scl::memory::copy_fast(
                Span<const Index>(src_indices + start, length),
                MutableSpan<Index>(dst_indices + write_pos, length)
            );
            
            temp_pos[new_i] += length;
        }
    }
}

} // namespace scl::kernel::sparse
