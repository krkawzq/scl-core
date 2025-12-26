#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
/// @file gram_fast_impl.hpp
/// @brief Extreme Performance Gram Matrix
///
/// Separate optimizations:
/// - CustomSparse: Direct pointer access + optimized dot product
/// - VirtualSparse: Row-wise with minimal pointer dereference
///
/// Ultra-optimized sparse dot products with:
/// - Prefetching for both index arrays
/// - 4-way unrolled merge
/// - Branch prediction hints
///
/// Performance Target: 1.5-2x faster than generic
// =============================================================================

namespace scl::kernel::gram::fast {

namespace detail {

constexpr size_t PREFETCH_DISTANCE = 32;

/// @brief Ultra-fast linear merge dot product
///
/// Optimization: 4-way unrolling + dual prefetch
template <typename T>
SCL_FORCE_INLINE void dot_linear_ultra(
    const Index* SCL_RESTRICT idx1, const T* SCL_RESTRICT val1, Size n1,
    const Index* SCL_RESTRICT idx2, const T* SCL_RESTRICT val2, Size n2,
    T& out_dot
) {
    // 4 independent accumulators for ILP
    T sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    Size i = 0, j = 0;
    Size matches = 0;
    
    while (i < n1 && j < n2) {
        // Prefetch ahead
        if (i + PREFETCH_DISTANCE < n1) {
            SCL_PREFETCH_READ(&idx1[i + PREFETCH_DISTANCE], 0);
        }
        if (j + PREFETCH_DISTANCE < n2) {
            SCL_PREFETCH_READ(&idx2[j + PREFETCH_DISTANCE], 0);
        }
        
        Index r1 = idx1[i];
        Index r2 = idx2[j];
        
        if (SCL_LIKELY(r1 == r2)) {
            // Distribute to different accumulators based on position
            T prod = val1[i] * val2[j];
            switch (matches % 4) {
                case 0: sum0 += prod; break;
                case 1: sum1 += prod; break;
                case 2: sum2 += prod; break;
                case 3: sum3 += prod; break;
            }
            matches++;
            ++i; ++j;
        } else if (r1 < r2) {
            ++i;
        } else {
            ++j;
        }
    }
    
    out_dot = (sum0 + sum1) + (sum2 + sum3);
}

} // namespace detail

// =============================================================================
// CustomSparse Fast Path
// =============================================================================

/// @brief Ultra-fast Gram matrix (CustomSparse)
///
/// Optimization: Direct pointer access + optimized dot product
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void gram_custom_fast(
    const CustomSparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index N = scl::primary_size(matrix);
    const Size N_size = static_cast<Size>(N);
    
    SCL_CHECK_DIM(output.len == N_size * N_size, "Gram: Output size mismatch");

    scl::threading::parallel_for(0, N_size, [&](size_t i) {
        Index start_i = matrix.indptr[i];
        Index end_i = matrix.indptr[i + 1];
        Index len_i = end_i - start_i;
        
        const Index* SCL_RESTRICT idx_i = matrix.indices + start_i;
        const T* SCL_RESTRICT val_i = matrix.data + start_i;
        
        T* row_ptr = output.ptr + (i * N_size);

        // Diagonal: SIMD self dot product
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        auto v_sum = s::Zero(d);
        Index k = 0;
        
        for (; k + lanes <= len_i; k += lanes) {
            auto v = s::Load(d, val_i + k);
            v_sum = s::MulAdd(v, v, v_sum);
        }
        
        T self_dot = s::GetLane(s::SumOfLanes(d, v_sum));
        for (; k < len_i; ++k) {
            self_dot += val_i[k] * val_i[k];
        }
        row_ptr[i] = self_dot;

        // Upper triangle with optimized dot product
        for (Size j = i + 1; j < N_size; ++j) {
            Index start_j = matrix.indptr[j];
            Index end_j = matrix.indptr[j + 1];
            Index len_j = end_j - start_j;
            
            const Index* SCL_RESTRICT idx_j = matrix.indices + start_j;
            const T* SCL_RESTRICT val_j = matrix.data + start_j;

            T dot;
            detail::dot_linear_ultra(
                idx_i, val_i, static_cast<Size>(len_i),
                idx_j, val_j, static_cast<Size>(len_j),
                dot
            );

            row_ptr[j] = dot;
            output.ptr[j * N_size + i] = dot;  // Mirror
        }
    });
}

// =============================================================================
// VirtualSparse Fast Path
// =============================================================================

/// @brief Ultra-fast Gram matrix (VirtualSparse)
///
/// Optimization: Row-wise with minimal pointer dereference
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void gram_virtual_fast(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index N = scl::primary_size(matrix);
    const Size N_size = static_cast<Size>(N);
    
    SCL_CHECK_DIM(output.len == N_size * N_size, "Gram: Output size mismatch");

    scl::threading::parallel_for(0, N_size, [&](size_t i) {
        Index len_i = matrix.lengths[i];
        
        // Single pointer dereference
        const Index* SCL_RESTRICT idx_i = static_cast<const Index*>(matrix.indices_ptrs[i]);
        const T* SCL_RESTRICT val_i = static_cast<const T*>(matrix.data_ptrs[i]);
        
        T* row_ptr = output.ptr + (i * N_size);

        // Diagonal: SIMD self dot product
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        auto v_sum = s::Zero(d);
        Index k = 0;
        
        for (; k + lanes <= len_i; k += lanes) {
            auto v = s::Load(d, val_i + k);
            v_sum = s::MulAdd(v, v, v_sum);
        }
        
        T self_dot = s::GetLane(s::SumOfLanes(d, v_sum));
        for (; k < len_i; ++k) {
            self_dot += val_i[k] * val_i[k];
        }
        row_ptr[i] = self_dot;

        // Upper triangle with optimized dot product
        for (Size j = i + 1; j < N_size; ++j) {
            Index len_j = matrix.lengths[j];
            
            // Single pointer dereference
            const Index* SCL_RESTRICT idx_j = static_cast<const Index*>(matrix.indices_ptrs[j]);
            const T* SCL_RESTRICT val_j = static_cast<const T*>(matrix.data_ptrs[j]);

            T dot;
            detail::dot_linear_ultra(
                idx_i, val_i, static_cast<Size>(len_i),
                idx_j, val_j, static_cast<Size>(len_j),
                dot
            );

            row_ptr[j] = dot;
            output.ptr[j * N_size + i] = dot;  // Mirror
        }
    });
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void gram_fast(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output
) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        gram_custom_fast(matrix, output);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        gram_virtual_fast(matrix, output);
    }
}

} // namespace scl::kernel::gram::fast

