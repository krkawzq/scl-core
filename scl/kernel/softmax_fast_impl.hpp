#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
/// @file softmax_fast_impl.hpp
/// @brief Extreme Performance Softmax
///
/// Separate optimizations:
/// - CustomSparse: Direct data access + fused exp/normalize
/// - VirtualSparse: Row-wise SIMD with minimal indirection
///
/// Ultra-optimized softmax with:
/// - Two-pass SIMD (max finding + exp + normalization)
/// - 4-way unrolling
/// - Fused operations
///
/// Performance Target: 2-3x faster than generic
// =============================================================================

namespace scl::kernel::softmax::fast {

// =============================================================================
// CustomSparse Fast Path
// =============================================================================

/// @brief Ultra-fast softmax (CustomSparse)
///
/// Optimization: Direct data access + fused exp/normalize
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void softmax_inplace_custom_fast(CustomSparse<T, IsCSR>& matrix) {
    const Index primary_dim = scl::primary_size(matrix);
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Index len = end - start;
        
        if (len == 0) return;
        
        T* SCL_RESTRICT vals = matrix.data + start;
        
        // Pass 1: Find max
        T max_val = vals[0];
        for (Index k = 1; k < len; ++k) {
            if (vals[k] > max_val) max_val = vals[k];
        }
        
        const auto v_max = s::Set(d, max_val);
        
        // Pass 2: Compute exp(x - max) and sum (fused)
        auto v_sum = s::Zero(d);
        Index k = 0;
        
        // 4-way unrolled
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals + k + 0 * lanes);
            auto v1 = s::Load(d, vals + k + 1 * lanes);
            auto v2 = s::Load(d, vals + k + 2 * lanes);
            auto v3 = s::Load(d, vals + k + 3 * lanes);
            
            v0 = s::Exp(d, s::Sub(v0, v_max));
            v1 = s::Exp(d, s::Sub(v1, v_max));
            v2 = s::Exp(d, s::Sub(v2, v_max));
            v3 = s::Exp(d, s::Sub(v3, v_max));
            
            s::Store(v0, d, vals + k + 0 * lanes);
            s::Store(v1, d, vals + k + 1 * lanes);
            s::Store(v2, d, vals + k + 2 * lanes);
            s::Store(v3, d, vals + k + 3 * lanes);
            
            v_sum = s::Add(v_sum, v0);
            v_sum = s::Add(v_sum, v1);
            v_sum = s::Add(v_sum, v2);
            v_sum = s::Add(v_sum, v3);
        }
        
        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            v = s::Exp(d, s::Sub(v, v_max));
            s::Store(v, d, vals + k);
            v_sum = s::Add(v_sum, v);
        }
        
        T sum = s::GetLane(s::SumOfLanes(d, v_sum));
        
        for (; k < len; ++k) {
            T v = std::exp(vals[k] - max_val);
            vals[k] = v;
            sum += v;
        }
        
        // Pass 3: Normalize
        if (sum > 0) {
            T inv_sum = 1.0 / sum;
            const auto v_inv_sum = s::Set(d, inv_sum);
            
            k = 0;
            for (; k + 4 * lanes <= len; k += 4 * lanes) {
                auto v0 = s::Load(d, vals + k + 0 * lanes);
                auto v1 = s::Load(d, vals + k + 1 * lanes);
                auto v2 = s::Load(d, vals + k + 2 * lanes);
                auto v3 = s::Load(d, vals + k + 3 * lanes);
                
                s::Store(s::Mul(v0, v_inv_sum), d, vals + k + 0 * lanes);
                s::Store(s::Mul(v1, v_inv_sum), d, vals + k + 1 * lanes);
                s::Store(s::Mul(v2, v_inv_sum), d, vals + k + 2 * lanes);
                s::Store(s::Mul(v3, v_inv_sum), d, vals + k + 3 * lanes);
            }
            
            for (; k + lanes <= len; k += lanes) {
                auto v = s::Load(d, vals + k);
                s::Store(s::Mul(v, v_inv_sum), d, vals + k);
            }
            
            for (; k < len; ++k) {
                vals[k] *= inv_sum;
            }
        }
    });
}

// =============================================================================
// VirtualSparse Fast Path
// =============================================================================

/// @brief Ultra-fast softmax (VirtualSparse)
///
/// Optimization: Row-wise SIMD with minimal pointer dereference
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void softmax_inplace_virtual_fast(VirtualSparse<T, IsCSR>& matrix) {
    const Index primary_dim = scl::primary_size(matrix);
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = matrix.lengths[p];
        
        if (len == 0) return;
        
        // Single pointer dereference
        T* SCL_RESTRICT vals = static_cast<T*>(matrix.data_ptrs[p]);
        
        // Pass 1: Find max
        T max_val = vals[0];
        for (Index k = 1; k < len; ++k) {
            if (vals[k] > max_val) max_val = vals[k];
        }
        
        const auto v_max = s::Set(d, max_val);
        
        // Pass 2: Compute exp(x - max) and sum (fused)
        auto v_sum = s::Zero(d);
        Index k = 0;
        
        // 4-way unrolled
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals + k + 0 * lanes);
            auto v1 = s::Load(d, vals + k + 1 * lanes);
            auto v2 = s::Load(d, vals + k + 2 * lanes);
            auto v3 = s::Load(d, vals + k + 3 * lanes);
            
            v0 = s::Exp(d, s::Sub(v0, v_max));
            v1 = s::Exp(d, s::Sub(v1, v_max));
            v2 = s::Exp(d, s::Sub(v2, v_max));
            v3 = s::Exp(d, s::Sub(v3, v_max));
            
            s::Store(v0, d, vals + k + 0 * lanes);
            s::Store(v1, d, vals + k + 1 * lanes);
            s::Store(v2, d, vals + k + 2 * lanes);
            s::Store(v3, d, vals + k + 3 * lanes);
            
            v_sum = s::Add(v_sum, v0);
            v_sum = s::Add(v_sum, v1);
            v_sum = s::Add(v_sum, v2);
            v_sum = s::Add(v_sum, v3);
        }
        
        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            v = s::Exp(d, s::Sub(v, v_max));
            s::Store(v, d, vals + k);
            v_sum = s::Add(v_sum, v);
        }
        
        T sum = s::GetLane(s::SumOfLanes(d, v_sum));
        
        for (; k < len; ++k) {
            T v = std::exp(vals[k] - max_val);
            vals[k] = v;
            sum += v;
        }
        
        // Pass 3: Normalize
        if (sum > 0) {
            T inv_sum = 1.0 / sum;
            const auto v_inv_sum = s::Set(d, inv_sum);
            
            k = 0;
            for (; k + 4 * lanes <= len; k += 4 * lanes) {
                auto v0 = s::Load(d, vals + k + 0 * lanes);
                auto v1 = s::Load(d, vals + k + 1 * lanes);
                auto v2 = s::Load(d, vals + k + 2 * lanes);
                auto v3 = s::Load(d, vals + k + 3 * lanes);
                
                s::Store(s::Mul(v0, v_inv_sum), d, vals + k + 0 * lanes);
                s::Store(s::Mul(v1, v_inv_sum), d, vals + k + 1 * lanes);
                s::Store(s::Mul(v2, v_inv_sum), d, vals + k + 2 * lanes);
                s::Store(s::Mul(v3, v_inv_sum), d, vals + k + 3 * lanes);
            }
            
            for (; k + lanes <= len; k += lanes) {
                auto v = s::Load(d, vals + k);
                s::Store(s::Mul(v, v_inv_sum), d, vals + k);
            }
            
            for (; k < len; ++k) {
                vals[k] *= inv_sum;
            }
        }
    });
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void softmax_inplace_fast(MatrixT& matrix) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        softmax_inplace_custom_fast(matrix);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        softmax_inplace_virtual_fast(matrix);
    }
}

} // namespace scl::kernel::softmax::fast

