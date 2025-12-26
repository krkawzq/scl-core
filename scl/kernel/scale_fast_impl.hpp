#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/sparse.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
/// @file scale_fast_impl.hpp
/// @brief Extreme Performance Standardization
///
/// Separate optimizations:
/// - CustomSparse: Batch SIMD processing
/// - VirtualSparse: Row-wise SIMD with minimal indirection
///
/// Ultra-optimized (X - mean) / std transformation with:
/// - FMA for (x - mu) * inv_sigma
/// - Optional SIMD clipping
///
/// Performance Target: 2-3x faster than generic
// =============================================================================

namespace scl::kernel::scale::fast {

// =============================================================================
// CustomSparse Fast Path
// =============================================================================

/// @brief Ultra-fast standardization (CustomSparse)
///
/// Optimization: Direct data access + FMA + SIMD clipping
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void standardize_custom_fast(
    CustomSparse<T, IsCSR>& matrix,
    Array<const Real> means,
    Array<const Real> stds,
    Real max_value,
    bool zero_center
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(means.len == static_cast<Size>(primary_dim), "Means dim mismatch");
    SCL_CHECK_DIM(stds.len == static_cast<Size>(primary_dim), "Stds dim mismatch");
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Real mu = means[p];
        Real sigma = stds[p];
        Real inv_sigma = (sigma != 0.0) ? (1.0 / sigma) : 0.0;
        
        if (sigma == 0.0) return;

        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Index len = end - start;
        
        T* SCL_RESTRICT vals = matrix.data + start;
        
        const auto v_mu = s::Set(d, mu);
        const auto v_inv_sigma = s::Set(d, inv_sigma);
        const auto v_max = s::Set(d, max_value);
        const auto v_min = s::Set(d, -max_value);
        const bool do_clip = (max_value > 0.0);

        Index k = 0;
        
        // 4-way unrolled SIMD loop with FMA
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals + k + 0 * lanes);
            auto v1 = s::Load(d, vals + k + 1 * lanes);
            auto v2 = s::Load(d, vals + k + 2 * lanes);
            auto v3 = s::Load(d, vals + k + 3 * lanes);
            
            if (zero_center) {
                v0 = s::Sub(v0, v_mu);
                v1 = s::Sub(v1, v_mu);
                v2 = s::Sub(v2, v_mu);
                v3 = s::Sub(v3, v_mu);
            }
            
            v0 = s::Mul(v0, v_inv_sigma);
            v1 = s::Mul(v1, v_inv_sigma);
            v2 = s::Mul(v2, v_inv_sigma);
            v3 = s::Mul(v3, v_inv_sigma);
            
            if (do_clip) {
                v0 = s::Min(s::Max(v0, v_min), v_max);
                v1 = s::Min(s::Max(v1, v_min), v_max);
                v2 = s::Min(s::Max(v2, v_min), v_max);
                v3 = s::Min(s::Max(v3, v_min), v_max);
            }
            
            s::Store(v0, d, vals + k + 0 * lanes);
            s::Store(v1, d, vals + k + 1 * lanes);
            s::Store(v2, d, vals + k + 2 * lanes);
            s::Store(v3, d, vals + k + 3 * lanes);
        }
        
        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            
            if (zero_center) {
                v = s::Sub(v, v_mu);
            }
            
            v = s::Mul(v, v_inv_sigma);
            
            if (do_clip) {
                v = s::Min(s::Max(v, v_min), v_max);
            }
            
            s::Store(v, d, vals + k);
        }
        
        for (; k < len; ++k) {
            Real v = static_cast<Real>(vals[k]);
            
            if (zero_center) {
                v -= mu;
            }
            
            v *= inv_sigma;
            
            if (do_clip) {
                if (v > max_value) v = max_value;
                if (v < -max_value) v = -max_value;
            }
            
            vals[k] = static_cast<T>(v);
        }
    });
}

// =============================================================================
// VirtualSparse Fast Path
// =============================================================================

/// @brief Ultra-fast standardization (VirtualSparse)
///
/// Optimization: Row-wise SIMD with minimal pointer dereference
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void standardize_virtual_fast(
    VirtualSparse<T, IsCSR>& matrix,
    Array<const Real> means,
    Array<const Real> stds,
    Real max_value,
    bool zero_center
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(means.len == static_cast<Size>(primary_dim), "Means dim mismatch");
    SCL_CHECK_DIM(stds.len == static_cast<Size>(primary_dim), "Stds dim mismatch");
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Real mu = means[p];
        Real sigma = stds[p];
        Real inv_sigma = (sigma != 0.0) ? (1.0 / sigma) : 0.0;
        
        if (sigma == 0.0) return;

        Index len = matrix.lengths[p];
        if (len == 0) return;
        
        // Single pointer dereference
        T* SCL_RESTRICT vals = static_cast<T*>(matrix.data_ptrs[p]);
        
        const auto v_mu = s::Set(d, mu);
        const auto v_inv_sigma = s::Set(d, inv_sigma);
        const auto v_max = s::Set(d, max_value);
        const auto v_min = s::Set(d, -max_value);
        const bool do_clip = (max_value > 0.0);

        Index k = 0;
        
        // 4-way unrolled SIMD loop with FMA
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals + k + 0 * lanes);
            auto v1 = s::Load(d, vals + k + 1 * lanes);
            auto v2 = s::Load(d, vals + k + 2 * lanes);
            auto v3 = s::Load(d, vals + k + 3 * lanes);
            
            if (zero_center) {
                v0 = s::Sub(v0, v_mu);
                v1 = s::Sub(v1, v_mu);
                v2 = s::Sub(v2, v_mu);
                v3 = s::Sub(v3, v_mu);
            }
            
            v0 = s::Mul(v0, v_inv_sigma);
            v1 = s::Mul(v1, v_inv_sigma);
            v2 = s::Mul(v2, v_inv_sigma);
            v3 = s::Mul(v3, v_inv_sigma);
            
            if (do_clip) {
                v0 = s::Min(s::Max(v0, v_min), v_max);
                v1 = s::Min(s::Max(v1, v_min), v_max);
                v2 = s::Min(s::Max(v2, v_min), v_max);
                v3 = s::Min(s::Max(v3, v_min), v_max);
            }
            
            s::Store(v0, d, vals + k + 0 * lanes);
            s::Store(v1, d, vals + k + 1 * lanes);
            s::Store(v2, d, vals + k + 2 * lanes);
            s::Store(v3, d, vals + k + 3 * lanes);
        }
        
        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            
            if (zero_center) {
                v = s::Sub(v, v_mu);
            }
            
            v = s::Mul(v, v_inv_sigma);
            
            if (do_clip) {
                v = s::Min(s::Max(v, v_min), v_max);
            }
            
            s::Store(v, d, vals + k);
        }
        
        for (; k < len; ++k) {
            Real v = static_cast<Real>(vals[k]);
            
            if (zero_center) {
                v -= mu;
            }
            
            v *= inv_sigma;
            
            if (do_clip) {
                if (v > max_value) v = max_value;
                if (v < -max_value) v = -max_value;
            }
            
            vals[k] = static_cast<T>(v);
        }
    });
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void standardize_fast(
    MatrixT& matrix,
    Array<const Real> means,
    Array<const Real> stds,
    Real max_value,
    bool zero_center
) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        standardize_custom_fast(matrix, means, stds, max_value, zero_center);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        standardize_virtual_fast(matrix, means, stds, max_value, zero_center);
    }
}

} // namespace scl::kernel::scale::fast

