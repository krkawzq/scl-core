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

// =============================================================================
// 1. Row Scaling (In-Place)
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
    SCL_CHECK_DIM(scales.size == static_cast<Size>(matrix.rows), "Scales dim mismatch");

    scl::threading::parallel_for(0, matrix.rows, [&](size_t i) {
        Real s = scales[i];
        
        // Optimization: If scale is 1.0, skip. If 0.0, zero out (or skip if sparsity requires?)
        // Usually we just multiply.
        if (s == 1.0) return;

        auto vals = matrix.row_values(static_cast<Index>(i));
        
        // SIMD Multiplication
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

// =============================================================================
// 2. Highly Expressed Gene Detection
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
///                 **Note**: Use uint8_t/Byte instead of bool to avoid bit-vector races.
template <CSRLike MatrixT>
SCL_FORCE_INLINE void detect_highly_expressed_genes(
    const MatrixT& matrix,
    Span<const Real> row_sums,
    Real max_fraction,
    MutableSpan<Byte> out_mask
) {
    SCL_CHECK_DIM(row_sums.size == static_cast<Size>(matrix.rows), "Row sums mismatch");
    SCL_CHECK_DIM(out_mask.size == static_cast<Size>(matrix.cols), "Output mask mismatch");

    // 1. Reset Mask
    scl::memory::zero(out_mask);

    // 2. Parallel Scan
    // Potential Race Condition: Multiple rows might try to mark the same gene (column).
    // Solution: Since we only write '1', the operation is idempotent.
    // On x86/ARM, byte-sized writes are atomic. 
    // To be strictly C++ compliant, we should use std::atomic_ref or relaxed atomic intrinsics,
    // but treating Byte* as a relaxed atomic map is standard HPC practice.
    
    scl::threading::parallel_for(0, matrix.rows, [&](size_t i) {
        Real total = row_sums[i];
        if (total <= 0) return;

        Real threshold = total * max_fraction;
        
        auto indices = matrix.row_indices(static_cast<Index>(i));
        auto values  = matrix.row_values(static_cast<Index>(i));

        for (size_t k = 0; k < values.size; ++k) {
            if (values[k] > threshold) {
                Index gene_idx = indices[k];
                // Relaxed atomic store to prevent compiler tearing (though unlikely on byte)
                #ifdef _MSC_VER
                    out_mask[gene_idx] = 1; // MSVC has strong ordering guarantees
                #else
                    __atomic_store_n(&out_mask[gene_idx], 1, __ATOMIC_RELAXED);
                #endif
            }
        }
    });
}

// =============================================================================
// 3. Masked Row Sums
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
    SCL_CHECK_DIM(gene_mask.size == static_cast<Size>(matrix.cols), "Gene mask mismatch");
    SCL_CHECK_DIM(out_sums.size == static_cast<Size>(matrix.rows), "Output sums mismatch");

    scl::threading::parallel_for(0, matrix.rows, [&](size_t i) {
        auto indices = matrix.row_indices(static_cast<Index>(i));
        auto values  = matrix.row_values(static_cast<Index>(i));
        
        Real sum = 0;
        
        // SIMD is hard here because of indirect lookup into gene_mask[indices[k]].
        // Scalar loop is preferred.
        // Unrolling helps.
        size_t k = 0;
        for (; k < values.size; ++k) {
            Index gene_idx = indices[k];
            // Branchless accumulation: sum += val * (mask == 0)
            // But mask check is likely sparse or dense? 
            // Standard if check is fine.
            if (gene_mask[gene_idx] == 0) {
                sum += values[k];
            }
        }
        out_sums[i] = sum;
    });
}

// =============================================================================
// 4. Utils
// =============================================================================

/// @brief Compute median of a dataset.
///
/// **Requires** a temporary buffer to sort.
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
