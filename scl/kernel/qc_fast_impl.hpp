#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/sparse.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
/// @file qc_fast_impl.hpp
/// @brief Extreme Performance QC Metrics
///
/// Separate optimizations:
/// - CustomSparse: Direct data access + SIMD reduction
/// - VirtualSparse: Row-wise SIMD with minimal indirection
///
/// Ultra-optimized quality control with:
/// - Batch SIMD summation
/// - 4-way unrolling
/// - Prefetching
///
/// Performance Target: 2-3x faster than generic
// =============================================================================

namespace scl::kernel::qc::fast {

// =============================================================================
// CustomSparse Fast Path
// =============================================================================

/// @brief Ultra-fast basic QC (CustomSparse)
///
/// Optimization: Direct data access + SIMD reduction
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void compute_basic_qc_custom_fast(
    const CustomSparse<T, IsCSR>& matrix,
    Array<Index> out_n_genes,
    Array<Real> out_total_counts
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(out_n_genes.len == static_cast<Size>(primary_dim), "n_genes size mismatch");
    SCL_CHECK_DIM(out_total_counts.len == static_cast<Size>(primary_dim), "total_counts size mismatch");

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Index len = end - start;
        
        out_n_genes[p] = len;
        
        if (len == 0) {
            out_total_counts[p] = 0.0;
            return;
        }
        
        const T* SCL_RESTRICT vals = matrix.data + start;
        
        // 4-way unrolled SIMD accumulation
        auto v_sum0 = s::Zero(d);
        auto v_sum1 = s::Zero(d);
        auto v_sum2 = s::Zero(d);
        auto v_sum3 = s::Zero(d);
        
        Index k = 0;
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            v_sum0 = s::Add(v_sum0, s::Load(d, vals + k + 0 * lanes));
            v_sum1 = s::Add(v_sum1, s::Load(d, vals + k + 1 * lanes));
            v_sum2 = s::Add(v_sum2, s::Load(d, vals + k + 2 * lanes));
            v_sum3 = s::Add(v_sum3, s::Load(d, vals + k + 3 * lanes));
        }
        
        auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
        
        for (; k + lanes <= len; k += lanes) {
            v_sum = s::Add(v_sum, s::Load(d, vals + k));
        }
        
        Real sum = static_cast<Real>(s::GetLane(s::SumOfLanes(d, v_sum)));
        
        for (; k < len; ++k) {
            sum += static_cast<Real>(vals[k]);
        }
        
        out_total_counts[p] = sum;
    });
}

// =============================================================================
// VirtualSparse Fast Path
// =============================================================================

/// @brief Ultra-fast basic QC (VirtualSparse)
///
/// Optimization: Row-wise SIMD with minimal pointer dereference
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void compute_basic_qc_virtual_fast(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<Index> out_n_genes,
    Array<Real> out_total_counts
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(out_n_genes.len == static_cast<Size>(primary_dim), "n_genes size mismatch");
    SCL_CHECK_DIM(out_total_counts.len == static_cast<Size>(primary_dim), "total_counts size mismatch");

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = matrix.lengths[p];
        
        out_n_genes[p] = len;
        
        if (len == 0) {
            out_total_counts[p] = 0.0;
            return;
        }
        
        // Single pointer dereference
        const T* SCL_RESTRICT vals = static_cast<const T*>(matrix.data_ptrs[p]);
        
        // 4-way unrolled SIMD accumulation
        auto v_sum0 = s::Zero(d);
        auto v_sum1 = s::Zero(d);
        auto v_sum2 = s::Zero(d);
        auto v_sum3 = s::Zero(d);
        
        Index k = 0;
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            v_sum0 = s::Add(v_sum0, s::Load(d, vals + k + 0 * lanes));
            v_sum1 = s::Add(v_sum1, s::Load(d, vals + k + 1 * lanes));
            v_sum2 = s::Add(v_sum2, s::Load(d, vals + k + 2 * lanes));
            v_sum3 = s::Add(v_sum3, s::Load(d, vals + k + 3 * lanes));
        }
        
        auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
        
        for (; k + lanes <= len; k += lanes) {
            v_sum = s::Add(v_sum, s::Load(d, vals + k));
        }
        
        Real sum = static_cast<Real>(s::GetLane(s::SumOfLanes(d, v_sum)));
        
        for (; k < len; ++k) {
            sum += static_cast<Real>(vals[k]);
        }
        
        out_total_counts[p] = sum;
    });
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void compute_basic_qc_fast(
    const MatrixT& matrix,
    Array<Index> out_n_genes,
    Array<Real> out_total_counts
) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        compute_basic_qc_custom_fast(matrix, out_n_genes, out_total_counts);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        compute_basic_qc_virtual_fast(matrix, out_n_genes, out_total_counts);
    }
}

} // namespace scl::kernel::qc::fast

