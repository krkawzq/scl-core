#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
/// @file log1p_fast_impl.hpp
/// @brief Extreme Performance Logarithmic Transforms
///
/// Fast path for SparseLike matrices with:
/// - 4-way unrolled SIMD
/// - Batch processing in cache-friendly chunks
/// - Works for both CustomSparse and VirtualSparse
///
/// Compiler automatically optimizes based on actual type:
/// - CustomSparse: Generates batch code on contiguous data
/// - VirtualSparse: Generates row-wise code with pointer dereference
///
/// Performance Target: 2-3x faster than generic
// =============================================================================

namespace scl::kernel::fast {

namespace detail {
constexpr size_t CHUNK_SIZE = 8192;
}

// =============================================================================
// Unified Fast Path (SparseLike)
// =============================================================================

/// @brief Ultra-fast log1p (unified for Custom and Virtual)
///
/// Uses SparseLike concept - compiler optimizes based on actual type
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void log1p_inplace_fast(MatrixT& mat) {
    const Index primary_dim = scl::primary_size(mat);
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index idx = static_cast<Index>(p);
        
        // Use unified accessor - compiler inlines appropriately
        // CustomSparse: Direct pointer arithmetic
        // VirtualSparse: Single pointer dereference
        auto vals_view = scl::primary_values(mat, idx);
        Index len = scl::primary_length(mat, idx);
        
        if (len == 0) return;
        
        // Access underlying pointer for SIMD
        // Both Custom and Virtual return Array with .ptr
        typename MatrixT::ValueType* SCL_RESTRICT vals = 
            const_cast<typename MatrixT::ValueType*>(vals_view.ptr);
        
        Index k = 0;
        
        // 4-way unrolled SIMD (works for both types)
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals + k + 0 * lanes);
            auto v1 = s::Load(d, vals + k + 1 * lanes);
            auto v2 = s::Load(d, vals + k + 2 * lanes);
            auto v3 = s::Load(d, vals + k + 3 * lanes);
            
            s::Store(s::Log1p(d, v0), d, vals + k + 0 * lanes);
            s::Store(s::Log1p(d, v1), d, vals + k + 1 * lanes);
            s::Store(s::Log1p(d, v2), d, vals + k + 2 * lanes);
            s::Store(s::Log1p(d, v3), d, vals + k + 3 * lanes);
        }
        
        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            s::Store(s::Log1p(d, v), d, vals + k);
        }
        
        for (; k < len; ++k) {
            vals[k] = std::log1p(vals[k]);
        }
    });
}

/// @brief Ultra-fast log2p1 (unified)
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void log2p1_inplace_fast(MatrixT& mat) {
    constexpr double INV_LN2 = 1.44269504088896340736;
    const Index primary_dim = scl::primary_size(mat);
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto v_inv_ln2 = s::Set(d, INV_LN2);
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index idx = static_cast<Index>(p);
        auto vals_view = scl::primary_values(mat, idx);
        Index len = scl::primary_length(mat, idx);
        
        if (len == 0) return;
        
        typename MatrixT::ValueType* SCL_RESTRICT vals = 
            const_cast<typename MatrixT::ValueType*>(vals_view.ptr);
        
        Index k = 0;
        
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals + k + 0 * lanes);
            auto v1 = s::Load(d, vals + k + 1 * lanes);
            auto v2 = s::Load(d, vals + k + 2 * lanes);
            auto v3 = s::Load(d, vals + k + 3 * lanes);
            
            s::Store(s::Mul(s::Log1p(d, v0), v_inv_ln2), d, vals + k + 0 * lanes);
            s::Store(s::Mul(s::Log1p(d, v1), v_inv_ln2), d, vals + k + 1 * lanes);
            s::Store(s::Mul(s::Log1p(d, v2), v_inv_ln2), d, vals + k + 2 * lanes);
            s::Store(s::Mul(s::Log1p(d, v3), v_inv_ln2), d, vals + k + 3 * lanes);
        }
        
        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            s::Store(s::Mul(s::Log1p(d, v), v_inv_ln2), d, vals + k);
        }
        
        for (; k < len; ++k) {
            vals[k] = std::log1p(vals[k]) * INV_LN2;
        }
    });
}

/// @brief Ultra-fast expm1 (unified)
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void expm1_inplace_fast(MatrixT& mat) {
    const Index primary_dim = scl::primary_size(mat);
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index idx = static_cast<Index>(p);
        auto vals_view = scl::primary_values(mat, idx);
        Index len = scl::primary_length(mat, idx);
        
        if (len == 0) return;
        
        typename MatrixT::ValueType* SCL_RESTRICT vals = 
            const_cast<typename MatrixT::ValueType*>(vals_view.ptr);
        
        Index k = 0;
        
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals + k + 0 * lanes);
            auto v1 = s::Load(d, vals + k + 1 * lanes);
            auto v2 = s::Load(d, vals + k + 2 * lanes);
            auto v3 = s::Load(d, vals + k + 3 * lanes);
            
            s::Store(s::Expm1(d, v0), d, vals + k + 0 * lanes);
            s::Store(s::Expm1(d, v1), d, vals + k + 1 * lanes);
            s::Store(s::Expm1(d, v2), d, vals + k + 2 * lanes);
            s::Store(s::Expm1(d, v3), d, vals + k + 3 * lanes);
        }
        
        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            s::Store(s::Expm1(d, v), d, vals + k);
        }
        
        for (; k < len; ++k) {
            vals[k] = std::expm1(vals[k]);
        }
    });
}

} // namespace scl::kernel::fast
