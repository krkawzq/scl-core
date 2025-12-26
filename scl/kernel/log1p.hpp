#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
/// @file log1p.hpp
/// @brief Logarithmic Transformations with SIMD
///
/// Functions:
/// - log1p(x): ln(1 + x)
/// - log2p1(x): log2(1 + x)
/// - expm1(x): e^x - 1
///
/// Why Specialized:
/// 1. Numerical Stability: Avoids catastrophic cancellation
/// 2. Sparsity Preservation: ln(1 + 0) = 0
/// 3. SIMD: 4-8x faster than scalar std::log
///
/// Performance: ~1.5 GB/s per core
// =============================================================================

namespace scl::kernel {

namespace detail {
    constexpr double INV_LN2 = 1.44269504088896340736;
}

// =============================================================================
// Array Operations
// =============================================================================

/// @brief Apply ln(1 + x) to array (SIMD)
SCL_FORCE_INLINE void log1p_inplace(Array<Real> vals) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    size_t k = 0;
    for (; k + lanes <= vals.size(); k += lanes) {
        auto v = s::Load(d, vals.ptr + k);
        v = s::Log1p(d, v);
        s::Store(v, d, vals.ptr + k);
    }
    
    for (; k < vals.size(); ++k) {
        vals[k] = std::log1p(vals[k]);
    }
}

/// @brief Apply log2(1 + x) to array (SIMD)
SCL_FORCE_INLINE void log2p1_inplace(Array<Real> vals) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const auto inv_ln2 = s::Set(d, detail::INV_LN2);
    
    size_t k = 0;
    for (; k + lanes <= vals.size(); k += lanes) {
        auto v = s::Load(d, vals.ptr + k);
        v = s::Log1p(d, v);
        v = s::Mul(v, inv_ln2);
        s::Store(v, d, vals.ptr + k);
    }
    
    for (; k < vals.size(); ++k) {
        vals[k] = std::log1p(vals[k]) * detail::INV_LN2;
    }
}

/// @brief Apply e^x - 1 to array (SIMD)
SCL_FORCE_INLINE void expm1_inplace(Array<Real> vals) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    size_t k = 0;
    for (; k + lanes <= vals.size(); k += lanes) {
        auto v = s::Load(d, vals.ptr + k);
        v = s::Expm1(d, v);
        s::Store(v, d, vals.ptr + k);
    }
    
    for (; k < vals.size(); ++k) {
        vals[k] = std::expm1(vals[k]);
    }
}

// =============================================================================
// Sparse Matrix Operations (Unified for CSR/CSC)
// =============================================================================

/// @brief Apply ln(1 + x) to sparse matrix values
template <typename MatrixT>
    requires AnySparse<MatrixT>
SCL_FORCE_INLINE void log1p_inplace(MatrixT& mat) {
    const Index primary_dim = scl::primary_size(mat);
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index idx = static_cast<Index>(p);
        auto vals = scl::primary_values(mat, idx);
        
        if (vals.size() > 0) {
            Array<typename MatrixT::ValueType> mutable_vals(
                const_cast<typename MatrixT::ValueType*>(vals.ptr), vals.size());
            log1p_inplace(mutable_vals);
        }
    });
}

/// @brief Apply log2(1 + x) to sparse matrix values
template <typename MatrixT>
    requires AnySparse<MatrixT>
SCL_FORCE_INLINE void log2p1_inplace(MatrixT& mat) {
    const Index primary_dim = scl::primary_size(mat);
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index idx = static_cast<Index>(p);
        auto vals = scl::primary_values(mat, idx);
        
        if (vals.size() > 0) {
            Array<typename MatrixT::ValueType> mutable_vals(
                const_cast<typename MatrixT::ValueType*>(vals.ptr), vals.size());
            log2p1_inplace(mutable_vals);
        }
    });
}

/// @brief Apply e^x - 1 to sparse matrix values
template <typename MatrixT>
    requires AnySparse<MatrixT>
SCL_FORCE_INLINE void expm1_inplace(MatrixT& mat) {
    const Index primary_dim = scl::primary_size(mat);
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index idx = static_cast<Index>(p);
        auto vals = scl::primary_values(mat, idx);
        
        if (vals.size() > 0) {
            Array<typename MatrixT::ValueType> mutable_vals(
                const_cast<typename MatrixT::ValueType*>(vals.ptr), vals.size());
            expm1_inplace(mutable_vals);
        }
    });
}

} // namespace scl::kernel
