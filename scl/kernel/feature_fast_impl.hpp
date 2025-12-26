#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// Mapped backend support
#include "scl/kernel/mapped_common.hpp"
#include "scl/kernel/feature_mapped_impl.hpp"

// =============================================================================
/// @file feature_fast_impl.hpp
/// @brief Extreme Performance Feature Statistics
///
/// Separate optimizations:
/// - CustomSparse: Direct data access + 4-way unrolled SIMD
/// - VirtualSparse: Row-wise SIMD with minimal indirection
///
/// Performance Target: 2-3x faster than generic
// =============================================================================

namespace scl::kernel::feature::fast {

// =============================================================================
// CustomSparse Fast Path
// =============================================================================

/// @brief Ultra-fast standard moments (CustomSparse)
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void standard_moments_custom_fast(
    const CustomSparse<T, IsCSR>& matrix,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Real N = static_cast<Real>(scl::secondary_size(matrix));
    const Real denom = N - static_cast<Real>(ddof);
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Index len = end - start;
        
        const T* SCL_RESTRICT vals = matrix.data + start;
        
        // 4-way unrolled SIMD
        auto v_sum0 = s::Zero(d);
        auto v_sum1 = s::Zero(d);
        auto v_sum2 = s::Zero(d);
        auto v_sum3 = s::Zero(d);
        
        auto v_ssq0 = s::Zero(d);
        auto v_ssq1 = s::Zero(d);
        auto v_ssq2 = s::Zero(d);
        auto v_ssq3 = s::Zero(d);
        
        Index k = 0;
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals + k + 0 * lanes);
            auto v1 = s::Load(d, vals + k + 1 * lanes);
            auto v2 = s::Load(d, vals + k + 2 * lanes);
            auto v3 = s::Load(d, vals + k + 3 * lanes);
            
            v_sum0 = s::Add(v_sum0, v0);
            v_sum1 = s::Add(v_sum1, v1);
            v_sum2 = s::Add(v_sum2, v2);
            v_sum3 = s::Add(v_sum3, v3);
            
            v_ssq0 = s::MulAdd(v0, v0, v_ssq0);
            v_ssq1 = s::MulAdd(v1, v1, v_ssq1);
            v_ssq2 = s::MulAdd(v2, v2, v_ssq2);
            v_ssq3 = s::MulAdd(v3, v3, v_ssq3);
        }
        
        auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
        auto v_ssq = s::Add(s::Add(v_ssq0, v_ssq1), s::Add(v_ssq2, v_ssq3));
        
        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            v_sum = s::Add(v_sum, v);
            v_ssq = s::MulAdd(v, v, v_ssq);
        }
        
        Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
        Real sum_sq = s::GetLane(s::SumOfLanes(d, v_ssq));
        
        for (; k < len; ++k) {
            Real v = static_cast<Real>(vals[k]);
            sum += v;
            sum_sq += v * v;
        }
        
        Real mu = sum / N;
        Real var = (denom > 0) ? ((sum_sq - sum * mu) / denom) : 0.0;
        if (var < 0) var = 0.0;
        
        out_means[p] = mu;
        out_vars[p] = var;
    });
}

// =============================================================================
// VirtualSparse Fast Path
// =============================================================================

/// @brief Ultra-fast standard moments (VirtualSparse)
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void standard_moments_virtual_fast(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Real N = static_cast<Real>(scl::secondary_size(matrix));
    const Real denom = N - static_cast<Real>(ddof);
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = matrix.lengths[p];
        
        // Single pointer dereference
        const T* SCL_RESTRICT vals = static_cast<const T*>(matrix.data_ptrs[p]);
        
        // 4-way unrolled SIMD
        auto v_sum0 = s::Zero(d);
        auto v_sum1 = s::Zero(d);
        auto v_sum2 = s::Zero(d);
        auto v_sum3 = s::Zero(d);
        
        auto v_ssq0 = s::Zero(d);
        auto v_ssq1 = s::Zero(d);
        auto v_ssq2 = s::Zero(d);
        auto v_ssq3 = s::Zero(d);
        
        Index k = 0;
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals + k + 0 * lanes);
            auto v1 = s::Load(d, vals + k + 1 * lanes);
            auto v2 = s::Load(d, vals + k + 2 * lanes);
            auto v3 = s::Load(d, vals + k + 3 * lanes);
            
            v_sum0 = s::Add(v_sum0, v0);
            v_sum1 = s::Add(v_sum1, v1);
            v_sum2 = s::Add(v_sum2, v2);
            v_sum3 = s::Add(v_sum3, v3);
            
            v_ssq0 = s::MulAdd(v0, v0, v_ssq0);
            v_ssq1 = s::MulAdd(v1, v1, v_ssq1);
            v_ssq2 = s::MulAdd(v2, v2, v_ssq2);
            v_ssq3 = s::MulAdd(v3, v3, v_ssq3);
        }
        
        auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
        auto v_ssq = s::Add(s::Add(v_ssq0, v_ssq1), s::Add(v_ssq2, v_ssq3));
        
        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            v_sum = s::Add(v_sum, v);
            v_ssq = s::MulAdd(v, v, v_ssq);
        }
        
        Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
        Real sum_sq = s::GetLane(s::SumOfLanes(d, v_ssq));
        
        for (; k < len; ++k) {
            Real v = static_cast<Real>(vals[k]);
            sum += v;
            sum_sq += v * v;
        }
        
        Real mu = sum / N;
        Real var = (denom > 0) ? ((sum_sq - sum * mu) / denom) : 0.0;
        if (var < 0) var = 0.0;
        
        out_means[p] = mu;
        out_vars[p] = var;
    });
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void standard_moments_fast(
    const MatrixT& matrix,
    Array<Real> out_means,
    Array<Real> out_vars,
    int ddof
) {
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        scl::kernel::feature::mapped::standard_moments_mapped_dispatch<MatrixT, IsCSR>(matrix, out_means, out_vars, ddof);
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        standard_moments_custom_fast(matrix, out_means, out_vars, ddof);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        standard_moments_virtual_fast(matrix, out_means, out_vars, ddof);
    }
}

} // namespace scl::kernel::feature::fast
