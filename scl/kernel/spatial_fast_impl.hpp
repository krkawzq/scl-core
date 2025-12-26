#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
/// @file spatial_fast_impl.hpp
/// @brief Extreme Performance Spatial Statistics
///
/// Separate optimizations:
/// - CustomSparse: Batch SIMD on entire data array
/// - VirtualSparse: Row-wise SIMD with minimal indirection
///
/// Ultra-optimized Moran's I with:
/// - Batch SIMD weight summation
/// - Cache-optimized centered value materialization
/// - Vectorized variance computation
///
/// Performance Target: 2-3x faster than generic
// =============================================================================

namespace scl::kernel::spatial::fast {

// =============================================================================
// CustomSparse Fast Path
// =============================================================================

/// @brief Ultra-fast weight sum (CustomSparse)
///
/// Optimization: Batch SIMD on entire data array
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void weight_sum_custom_fast(
    const CustomSparse<T, IsCSR>& graph,
    T& out_sum
) {
    const Index primary_dim = scl::primary_size(graph);
    const Index total_nnz = graph.indptr[primary_dim];
    
    if (total_nnz == 0) {
        out_sum = static_cast<T>(0);
        return;
    }
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    const T* SCL_RESTRICT data = graph.data;
    
    // 4-way unrolled SIMD accumulation
    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);
    
    size_t i = 0;
    for (; i + 4 * lanes <= total_nnz; i += 4 * lanes) {
        v_sum0 = s::Add(v_sum0, s::Load(d, data + i + 0 * lanes));
        v_sum1 = s::Add(v_sum1, s::Load(d, data + i + 1 * lanes));
        v_sum2 = s::Add(v_sum2, s::Load(d, data + i + 2 * lanes));
        v_sum3 = s::Add(v_sum3, s::Load(d, data + i + 3 * lanes));
    }
    
    auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
    
    for (; i + lanes <= total_nnz; i += lanes) {
        v_sum = s::Add(v_sum, s::Load(d, data + i));
    }
    
    T sum = s::GetLane(s::SumOfLanes(d, v_sum));
    
    for (; i < total_nnz; ++i) {
        sum += data[i];
    }
    
    out_sum = sum;
}

// =============================================================================
// VirtualSparse Fast Path
// =============================================================================

/// @brief Ultra-fast weight sum (VirtualSparse)
///
/// Optimization: Row-wise SIMD with minimal pointer dereference
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void weight_sum_virtual_fast(
    const VirtualSparse<T, IsCSR>& graph,
    Array<T> workspace,
    T& out_sum
) {
    const Index primary_dim = scl::primary_size(graph);
    
    SCL_CHECK_DIM(workspace.len >= static_cast<Size>(primary_dim), "Workspace too small");
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    // Parallel reduction across rows
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = graph.lengths[p];
        
        if (len == 0) {
            workspace[p] = static_cast<T>(0);
            return;
        }
        
        // Single pointer dereference
        const T* SCL_RESTRICT vals = static_cast<const T*>(graph.data_ptrs[p]);
        
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
        
        T sum = s::GetLane(s::SumOfLanes(d, v_sum));
        
        for (; k < len; ++k) {
            sum += vals[k];
        }
        
        workspace[p] = sum;
    });
    
    // Final reduction
    T total_sum = static_cast<T>(0);
    for (Index p = 0; p < primary_dim; ++p) {
        total_sum += workspace[p];
    }
    
    out_sum = total_sum;
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void weight_sum_fast(
    const MatrixT& graph,
    Array<typename MatrixT::ValueType> workspace,
    typename MatrixT::ValueType& out_sum
) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        weight_sum_custom_fast(graph, out_sum);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        weight_sum_virtual_fast(graph, workspace, out_sum);
    }
}

} // namespace scl::kernel::spatial::fast

