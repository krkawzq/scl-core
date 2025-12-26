#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
/// @file normalize_fast_impl.hpp
/// @brief Extreme Performance Normalization
///
/// Separate optimizations for CustomSparse and VirtualSparse:
/// - CustomSparse: Batch SIMD on entire data array
/// - VirtualSparse: Row-wise SIMD with minimal indirection
///
/// Performance Target: 3-5x faster than generic
// =============================================================================

namespace scl::kernel::normalize::fast {

namespace detail {
constexpr size_t PREFETCH_DISTANCE = 128;
constexpr size_t STREAM_THRESHOLD = 1024 * 1024;
}

// =============================================================================
// CustomSparse: Batch Processing on Entire Data Array
// =============================================================================

/// @brief Ultra-fast scaling (CustomSparse - batch mode)
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void scale_primary_custom_fast(
    CustomSparse<T, IsCSR>& matrix,
    Array<const Real> scales
) {
    using ValueType = typename CustomSparse<T, IsCSR>::ValueType;
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(scales.len == static_cast<Size>(primary_dim), "Scales dim mismatch");
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Real scale = scales[p];
        if (scale == 1.0) return;
        
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Index len = end - start;
        
        if (len == 0) return;
        
        ValueType* SCL_RESTRICT data_ptr = matrix.data + start;
        const auto v_scale = s::Set(d, scale);
        
        Index k = 0;
        
        // 4-way unrolled SIMD
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, data_ptr + k + 0 * lanes);
            auto v1 = s::Load(d, data_ptr + k + 1 * lanes);
            auto v2 = s::Load(d, data_ptr + k + 2 * lanes);
            auto v3 = s::Load(d, data_ptr + k + 3 * lanes);
            
            s::Store(s::Mul(v0, v_scale), d, data_ptr + k + 0 * lanes);
            s::Store(s::Mul(v1, v_scale), d, data_ptr + k + 1 * lanes);
            s::Store(s::Mul(v2, v_scale), d, data_ptr + k + 2 * lanes);
            s::Store(s::Mul(v3, v_scale), d, data_ptr + k + 3 * lanes);
        }
        
        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, data_ptr + k);
            s::Store(s::Mul(v, v_scale), d, data_ptr + k);
        }
        
        for (; k < len; ++k) {
            data_ptr[k] *= scale;
        }
    });
}

// =============================================================================
// VirtualSparse: Row-wise Processing with Minimal Indirection
// =============================================================================

/// @brief Ultra-fast scaling (VirtualSparse - row-wise)
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void scale_primary_virtual_fast(
    VirtualSparse<T, IsCSR>& matrix,
    Array<const Real> scales
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(scales.len == static_cast<Size>(primary_dim), "Scales dim mismatch");
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Real scale = scales[p];
        if (scale == 1.0) return;
        
        Index len = matrix.lengths[p];
        if (len == 0) return;
        
        // Single pointer dereference
        T* SCL_RESTRICT vals = static_cast<T*>(matrix.data_ptrs[p]);
        const auto v_scale = s::Set(d, scale);
        
        Index k = 0;
        
        // 4-way unrolled SIMD (行内处理)
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals + k + 0 * lanes);
            auto v1 = s::Load(d, vals + k + 1 * lanes);
            auto v2 = s::Load(d, vals + k + 2 * lanes);
            auto v3 = s::Load(d, vals + k + 3 * lanes);
            
            s::Store(s::Mul(v0, v_scale), d, vals + k + 0 * lanes);
            s::Store(s::Mul(v1, v_scale), d, vals + k + 1 * lanes);
            s::Store(s::Mul(v2, v_scale), d, vals + k + 2 * lanes);
            s::Store(s::Mul(v3, v_scale), d, vals + k + 3 * lanes);
        }
        
        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            s::Store(s::Mul(v, v_scale), d, vals + k);
        }
        
        for (; k < len; ++k) {
            vals[k] *= scale;
        }
    });
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void scale_primary_fast(
    MatrixT& matrix,
    Array<const Real> scales
) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        scale_primary_custom_fast(matrix, scales);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        scale_primary_virtual_fast(matrix, scales);
    }
}

} // namespace scl::kernel::normalize::fast
