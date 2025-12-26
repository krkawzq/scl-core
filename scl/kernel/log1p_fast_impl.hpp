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
/// Separate optimizations:
/// - CustomSparse: Batch SIMD on entire data array
/// - VirtualSparse: Row-wise SIMD with minimal indirection
///
/// Performance Target: 2-3x faster than generic
// =============================================================================

namespace scl::kernel::fast {

namespace detail {
constexpr size_t CHUNK_SIZE = 8192;
}

// =============================================================================
// CustomSparse Fast Path
// =============================================================================

/// @brief Ultra-fast log1p (CustomSparse)
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void log1p_inplace_custom_fast(CustomSparse<T, IsCSR>& mat) {
    const Index primary_dim = scl::primary_size(mat);
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index start = mat.indptr[p];
        Index end = mat.indptr[p + 1];
        Index len = end - start;
        
        if (len == 0) return;
        
        T* SCL_RESTRICT vals = mat.data + start;
        
        Index k = 0;
        
        // 4-way unrolled SIMD
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

/// @brief Ultra-fast log2p1 (CustomSparse)
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void log2p1_inplace_custom_fast(CustomSparse<T, IsCSR>& mat) {
    constexpr double INV_LN2 = 1.44269504088896340736;
    const Index primary_dim = scl::primary_size(mat);
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto v_inv_ln2 = s::Set(d, INV_LN2);
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index start = mat.indptr[p];
        Index end = mat.indptr[p + 1];
        Index len = end - start;
        
        if (len == 0) return;
        
        T* SCL_RESTRICT vals = mat.data + start;
        
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

/// @brief Ultra-fast expm1 (CustomSparse)
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void expm1_inplace_custom_fast(CustomSparse<T, IsCSR>& mat) {
    const Index primary_dim = scl::primary_size(mat);
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index start = mat.indptr[p];
        Index end = mat.indptr[p + 1];
        Index len = end - start;
        
        if (len == 0) return;
        
        T* SCL_RESTRICT vals = mat.data + start;
        
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

// =============================================================================
// VirtualSparse Fast Path
// =============================================================================

/// @brief Ultra-fast log1p (VirtualSparse)
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void log1p_inplace_virtual_fast(VirtualSparse<T, IsCSR>& mat) {
    const Index primary_dim = scl::primary_size(mat);
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = mat.lengths[p];
        
        if (len == 0) return;
        
        // Single pointer dereference
        T* SCL_RESTRICT vals = static_cast<T*>(mat.data_ptrs[p]);
        
        Index k = 0;
        
        // 4-way unrolled SIMD
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

/// @brief Ultra-fast log2p1 (VirtualSparse)
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void log2p1_inplace_virtual_fast(VirtualSparse<T, IsCSR>& mat) {
    constexpr double INV_LN2 = 1.44269504088896340736;
    const Index primary_dim = scl::primary_size(mat);
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto v_inv_ln2 = s::Set(d, INV_LN2);
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = mat.lengths[p];
        
        if (len == 0) return;
        
        // Single pointer dereference
        T* SCL_RESTRICT vals = static_cast<T*>(mat.data_ptrs[p]);
        
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

/// @brief Ultra-fast expm1 (VirtualSparse)
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void expm1_inplace_virtual_fast(VirtualSparse<T, IsCSR>& mat) {
    const Index primary_dim = scl::primary_size(mat);
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = mat.lengths[p];
        
        if (len == 0) return;
        
        // Single pointer dereference
        T* SCL_RESTRICT vals = static_cast<T*>(mat.data_ptrs[p]);
        
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

// =============================================================================
// Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void log1p_inplace_fast(MatrixT& mat) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        log1p_inplace_custom_fast(mat);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        log1p_inplace_virtual_fast(mat);
    }
}

/// @brief Auto-dispatch to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void log2p1_inplace_fast(MatrixT& mat) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        log2p1_inplace_custom_fast(mat);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        log2p1_inplace_virtual_fast(mat);
    }
}

/// @brief Auto-dispatch to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void expm1_inplace_fast(MatrixT& mat) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        expm1_inplace_custom_fast(mat);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        expm1_inplace_virtual_fast(mat);
    }
}

} // namespace scl::kernel::fast
