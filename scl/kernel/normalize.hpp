#pragma once

#include "scl/core/type.hpp"
#include "scl/core/matrix.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/memory.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/kernel/sparse.hpp" // Reuse row_sums from here

#include <cmath>
#include <algorithm>
#include <atomic>

// =============================================================================
/// @file normalize.hpp
/// @brief Normalization Kernels (Library Size, Scaling, Filtering)
///
/// Implements standard single-cell normalization routines:
/// 1. Row Scaling (inplace divide)
/// 2. Highly Expressed Gene Detection (for Seurat/Scanpy recipes)
/// 3. Masked Reductions
// =============================================================================

namespace scl::kernel::normalize {

namespace detail {

// =============================================================================
// Generic Implementation (Tag Dispatch)
// =============================================================================

/// @brief Generic implementation for row/column scaling.
template <typename MatrixT>
SCL_FORCE_INLINE void scale_impl(
    MatrixT matrix,
    Span<const Real> scales
) {
    using Tag = typename MatrixT::Tag;
    
    if constexpr (std::is_same_v<Tag, TagCSR>) {
        SCL_CHECK_DIM(scales.size == static_cast<Size>(matrix.rows), "Scales dim mismatch");
        
        scl::threading::parallel_for(0, matrix.rows, [&](size_t i) {
            Real s = scales[i];
            if (s == 1.0) return;
            
            auto vals = matrix.row_values(static_cast<Index>(i));
            
            namespace simd = scl::simd;
            const simd::Tag d;
            const size_t lanes = simd::lanes();
            const auto v_scale = simd::Set(d, s);
            
            size_t k = 0;
            for (; k + lanes <= vals.size; k += lanes) {
                auto v = simd::Load(d, vals.ptr + k);
                v = simd::Mul(v, v_scale);
                simd::Store(v, d, vals.ptr + k);
            }
            
            for (; k < vals.size; ++k) {
                vals[k] *= s;
            }
        });
    } else if constexpr (std::is_same_v<Tag, TagCSC>) {
        SCL_CHECK_DIM(scales.size == static_cast<Size>(matrix.cols), "Scales dim mismatch");
        
        scl::threading::parallel_for(0, matrix.cols, [&](size_t j) {
            Real s = scales[j];
            if (s == 1.0) return;
            
            auto vals = matrix.col_values(static_cast<Index>(j));
            
            namespace simd = scl::simd;
            const simd::Tag d;
            const size_t lanes = simd::lanes();
            const auto v_scale = simd::Set(d, s);
            
            size_t k = 0;
            for (; k + lanes <= vals.size; k += lanes) {
                auto v = simd::Load(d, vals.ptr + k);
                v = simd::Mul(v, v_scale);
                simd::Store(v, d, vals.ptr + k);
            }
            
            for (; k < vals.size; ++k) {
                vals[k] *= s;
            }
        });
    }
}

/// @brief Generic implementation for detecting highly expressed features.
template <typename MatrixT>
SCL_FORCE_INLINE void detect_highly_expressed_impl(
    const MatrixT& matrix,
    Span<const Real> feature_sums,
    Real max_fraction,
    MutableSpan<Byte> out_mask
) {
    using Tag = typename MatrixT::Tag;
    
    if constexpr (std::is_same_v<Tag, TagCSR>) {
        SCL_CHECK_DIM(feature_sums.size == static_cast<Size>(matrix.rows), "Row sums mismatch");
        SCL_CHECK_DIM(out_mask.size == static_cast<Size>(matrix.cols), "Output mask mismatch");
        
        scl::memory::zero(out_mask);
        
        scl::threading::parallel_for(0, matrix.rows, [&](size_t i) {
            Real total = feature_sums[i];
            if (total <= 0) return;
            
            Real threshold = total * max_fraction;
            
            auto indices = matrix.row_indices(static_cast<Index>(i));
            auto values  = matrix.row_values(static_cast<Index>(i));
            
            for (size_t k = 0; k < values.size; ++k) {
                if (values[k] > threshold) {
                    Index gene_idx = indices[k];
                    #ifdef _MSC_VER
                        out_mask[gene_idx] = 1;
                    #else
                        __atomic_store_n(&out_mask[gene_idx], 1, __ATOMIC_RELAXED);
                    #endif
                }
            }
        });
    } else if constexpr (std::is_same_v<Tag, TagCSC>) {
        SCL_CHECK_DIM(feature_sums.size == static_cast<Size>(matrix.cols), "Col sums mismatch");
        SCL_CHECK_DIM(out_mask.size == static_cast<Size>(matrix.rows), "Output mask mismatch");
        
        scl::memory::zero(out_mask);
        
        scl::threading::parallel_for(0, matrix.cols, [&](size_t j) {
            Real total = feature_sums[j];
            if (total <= 0) return;
            
            Real threshold = total * max_fraction;
            
            auto indices = matrix.col_indices(static_cast<Index>(j));
            auto values  = matrix.col_values(static_cast<Index>(j));
            
            for (size_t k = 0; k < values.size; ++k) {
                if (values[k] > threshold) {
                    Index cell_idx = indices[k];
                    #ifdef _MSC_VER
                        out_mask[cell_idx] = 1;
                    #else
                        __atomic_store_n(&out_mask[cell_idx], 1, __ATOMIC_RELAXED);
                    #endif
                }
            }
        });
    }
}

/// @brief Generic implementation for masked row/column sums.
template <typename MatrixT>
SCL_FORCE_INLINE void sums_masked_impl(
    const MatrixT& matrix,
    Span<const Byte> feature_mask,
    MutableSpan<Real> out_sums
) {
    using Tag = typename MatrixT::Tag;
    
    if constexpr (std::is_same_v<Tag, TagCSR>) {
        SCL_CHECK_DIM(feature_mask.size == static_cast<Size>(matrix.cols), "Gene mask mismatch");
        SCL_CHECK_DIM(out_sums.size == static_cast<Size>(matrix.rows), "Output sums mismatch");
        
        scl::threading::parallel_for(0, matrix.rows, [&](size_t i) {
            auto indices = matrix.row_indices(static_cast<Index>(i));
            auto values  = matrix.row_values(static_cast<Index>(i));
            
            Real sum = 0;
            for (size_t k = 0; k < values.size; ++k) {
                Index gene_idx = indices[k];
                if (feature_mask[gene_idx] == 0) {
                    sum += values[k];
                }
            }
            out_sums[i] = sum;
        });
    } else if constexpr (std::is_same_v<Tag, TagCSC>) {
        SCL_CHECK_DIM(feature_mask.size == static_cast<Size>(matrix.rows), "Cell mask mismatch");
        SCL_CHECK_DIM(out_sums.size == static_cast<Size>(matrix.cols), "Output sums mismatch");
        
        scl::threading::parallel_for(0, matrix.cols, [&](size_t j) {
            auto indices = matrix.col_indices(static_cast<Index>(j));
            auto values  = matrix.col_values(static_cast<Index>(j));
            
            Real sum = 0;
            for (size_t k = 0; k < values.size; ++k) {
                Index cell_idx = indices[k];
                if (feature_mask[cell_idx] == 0) {
                    sum += values[k];
                }
            }
            out_sums[j] = sum;
        });
    }
}

} // namespace detail

// =============================================================================
// 1. Row/Column Scaling (In-Place)
// =============================================================================

/// @brief Scale each row by a specific factor (Generic CSR-like matrices).
///
/// Operation: matrix[i, :] *= scales[i]
/// Used for Library Size Normalization (CPM/TPM) where scale = target_sum / current_sum.
///
/// @tparam MatrixT Any CSR-like matrix type
/// @param matrix CSR-like Matrix (modified in-place).
/// @param scales Array of scale factors (one per row).
template <CSRLike MatrixT>
SCL_FORCE_INLINE void scale_rows(
    MatrixT matrix,
    Span<const Real> scales
) {
    detail::scale_impl(matrix, scales);
}

/// @brief Scale each column by a specific factor (Generic CSC-like matrices).
///
/// Operation: matrix[:, j] *= scales[j]
/// Used for feature normalization where scale = target_sum / current_sum.
///
/// @tparam MatrixT Any CSC-like matrix type
/// @param matrix CSC-like Matrix (modified in-place).
/// @param scales Array of scale factors (one per column).
template <CSCLike MatrixT>
SCL_FORCE_INLINE void scale_cols(
    MatrixT matrix,
    Span<const Real> scales
) {
    detail::scale_impl(matrix, scales);
}

// =============================================================================
// 2. Highly Expressed Feature Detection
// =============================================================================

/// @brief Identify genes that consume a large fraction of counts in any cell (Generic CSR-like matrices).
///
/// Used to exclude genes like Hemoglobin or Mitochondria from normalization factors.
/// A gene is flagged if: `expression[cell, gene] > max_fraction * total_counts[cell]`.
///
/// @tparam MatrixT Any CSR-like matrix type
/// @param matrix CSR-like Matrix.
/// @param row_sums Pre-computed row sums (total counts per cell).
/// @param max_fraction Threshold (e.g., 0.05 for 5%).
/// @param out_mask Output boolean mask (Byte array) of size n_cols. 
///                 1 if highly expressed, 0 otherwise.
///                 Note: Use uint8_t/Byte instead of bool to avoid bit-vector races.
template <CSRLike MatrixT>
SCL_FORCE_INLINE void detect_highly_expressed_genes(
    const MatrixT& matrix,
    Span<const Real> row_sums,
    Real max_fraction,
    MutableSpan<Byte> out_mask
) {
    detail::detect_highly_expressed_impl(matrix, row_sums, max_fraction, out_mask);
}

/// @brief Identify cells that consume a large fraction of counts for any gene (Generic CSC-like matrices).
///
/// Used to detect outlier cells with unusually high expression.
/// A cell is flagged if: `expression[cell, gene] > max_fraction * total_counts[gene]`.
///
/// @tparam MatrixT Any CSC-like matrix type
/// @param matrix CSC-like Matrix.
/// @param col_sums Pre-computed column sums (total counts per gene).
/// @param max_fraction Threshold (e.g., 0.05 for 5%).
/// @param out_mask Output boolean mask (Byte array) of size n_rows. 
///                 1 if highly expressed, 0 otherwise.
template <CSCLike MatrixT>
SCL_FORCE_INLINE void detect_highly_expressed_cells(
    const MatrixT& matrix,
    Span<const Real> col_sums,
    Real max_fraction,
    MutableSpan<Byte> out_mask
) {
    detail::detect_highly_expressed_impl(matrix, col_sums, max_fraction, out_mask);
}

// =============================================================================
// 3. Masked Row/Column Sums
// =============================================================================

/// @brief Compute row sums excluding specific genes (Generic CSR-like matrices).
///
/// @tparam MatrixT Any CSR-like matrix type
/// @param matrix CSR-like Matrix.
/// @param gene_mask Byte mask (size n_cols). If gene_mask[j] != 0, ignore gene j.
/// @param out_sums Output row sums.
template <CSRLike MatrixT>
SCL_FORCE_INLINE void row_sums_masked(
    const MatrixT& matrix,
    Span<const Byte> gene_mask,
    MutableSpan<Real> out_sums
) {
    detail::sums_masked_impl(matrix, gene_mask, out_sums);
}

/// @brief Compute column sums excluding specific cells (Generic CSC-like matrices).
///
/// @tparam MatrixT Any CSC-like matrix type
/// @param matrix CSC-like Matrix.
/// @param cell_mask Byte mask (size n_rows). If cell_mask[i] != 0, ignore cell i.
/// @param out_sums Output column sums.
template <CSCLike MatrixT>
SCL_FORCE_INLINE void col_sums_masked(
    const MatrixT& matrix,
    Span<const Byte> cell_mask,
    MutableSpan<Real> out_sums
) {
    detail::sums_masked_impl(matrix, cell_mask, out_sums);
}

// =============================================================================
// 4. Utils
// =============================================================================

/// @brief Compute median of a dataset.
///
/// Requires a temporary buffer to sort.
/// This is typically used to find the default `target_sum`.
///
/// @param data Input data.
/// @param workspace Temporary buffer (size >= data.size).
/// @return Median value.
SCL_FORCE_INLINE Real median(
    Span<const Real> data,
    MutableSpan<Real> workspace
) {
    if (data.empty()) return 0.0;
    SCL_CHECK_DIM(workspace.size >= data.size, "Median workspace too small");
    
    // Copy to workspace
    scl::memory::copy(data, workspace);
    
    // Resize view to actual data size
    MutableSpan<Real> work_view(workspace.ptr, data.size);
    
    size_t n = work_view.size;
    size_t mid = n / 2;
    
    // Quickselect (nth_element) is O(N) average
    std::nth_element(work_view.begin(), work_view.begin() + mid, work_view.end());
    
    Real result = work_view[mid];
    
    // If even number of elements, median is average of two middle elements?
    // Standard definition varies. Numpy median averages.
    // Scanpy usually just takes one or averages.
    // Let's implement averaging for even N.
    if (n % 2 == 0) {
        // We need the element before mid.
        // max_element of the first half
        auto max_it = std::max_element(work_view.begin(), work_view.begin() + mid);
        result = (*max_it + result) * 0.5;
    }
    
    return result;
}

} // namespace scl::kernel::normalize
