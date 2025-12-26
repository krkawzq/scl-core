#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/sparse.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// Mapped backend support
#include "scl/kernel/mapped_common.hpp"
#include "scl/kernel/correlation_mapped_impl.hpp"

// =============================================================================
/// @file correlation_fast_impl.hpp
/// @brief Extreme Performance Correlation
///
/// Separate optimizations:
/// - CustomSparse: Direct data access + 4-way unrolled SIMD
/// - VirtualSparse: Row-wise SIMD with minimal indirection
///
/// Ultra-optimized statistics computation with:
/// - Fused mean/variance calculation
/// - Cache-optimized memory access
///
/// Performance Target: 2-3x faster than generic
// =============================================================================

namespace scl::kernel::correlation::fast {

// =============================================================================
// CustomSparse Fast Path
// =============================================================================

/// @brief Ultra-fast statistics (CustomSparse)
///
/// Optimization: Direct data access + 4-way unrolled SIMD
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void compute_stats_custom_fast(
    const CustomSparse<T, IsCSR>& matrix,
    Array<Real> out_means,
    Array<Real> out_inv_stds
) {
    using ValueType = typename CustomSparse<T, IsCSR>::ValueType;
    const Index primary_dim = scl::primary_size(matrix);
    const Size secondary_dim = static_cast<Size>(scl::secondary_size(matrix));
    const Real inv_n = static_cast<Real>(1.0) / static_cast<Real>(secondary_dim);
    
    SCL_CHECK_DIM(out_means.len == static_cast<Size>(primary_dim), "Means size mismatch");
    SCL_CHECK_DIM(out_inv_stds.len == static_cast<Size>(primary_dim), "Inv_stds size mismatch");

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Index len = end - start;
        
        const ValueType* SCL_RESTRICT vals = matrix.data + start;
        
        // Fused SIMD accumulation
        auto v_sum = s::Zero(d);
        auto v_sq_sum = s::Zero(d);
        
        Index k = 0;
        
        // 4-way unrolled
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals + k + 0 * lanes);
            auto v1 = s::Load(d, vals + k + 1 * lanes);
            auto v2 = s::Load(d, vals + k + 2 * lanes);
            auto v3 = s::Load(d, vals + k + 3 * lanes);
            
            v_sum = s::Add(v_sum, v0);
            v_sum = s::Add(v_sum, v1);
            v_sum = s::Add(v_sum, v2);
            v_sum = s::Add(v_sum, v3);
            
            v_sq_sum = s::MulAdd(v0, v0, v_sq_sum);
            v_sq_sum = s::MulAdd(v1, v1, v_sq_sum);
            v_sq_sum = s::MulAdd(v2, v2, v_sq_sum);
            v_sq_sum = s::MulAdd(v3, v3, v_sq_sum);
        }
        
        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            v_sum = s::Add(v_sum, v);
            v_sq_sum = s::MulAdd(v, v, v_sq_sum);
        }
        
        Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
        Real sq_sum = s::GetLane(s::SumOfLanes(d, v_sq_sum));

        for (; k < len; ++k) {
            Real v = static_cast<Real>(vals[k]);
            sum += v;
            sq_sum += v * v;
        }

        Real mean = sum * inv_n;
        Real var = (sq_sum * inv_n) - (mean * mean);
        if (var < 0) var = 0;

        out_means[p] = mean;
        out_inv_stds[p] = (var > 0) ? (1.0 / std::sqrt(var)) : 0.0;
    });
}

// =============================================================================
// VirtualSparse Fast Path
// =============================================================================

/// @brief Ultra-fast statistics (VirtualSparse)
///
/// Optimization: Row-wise SIMD with minimal pointer dereference
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void compute_stats_virtual_fast(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<Real> out_means,
    Array<Real> out_inv_stds
) {
    using ValueType = typename VirtualSparse<T, IsCSR>::ValueType;
    const Index primary_dim = scl::primary_size(matrix);
    const Size secondary_dim = static_cast<Size>(scl::secondary_size(matrix));
    const Real inv_n = static_cast<Real>(1.0) / static_cast<Real>(secondary_dim);
    
    SCL_CHECK_DIM(out_means.len == static_cast<Size>(primary_dim), "Means size mismatch");
    SCL_CHECK_DIM(out_inv_stds.len == static_cast<Size>(primary_dim), "Inv_stds size mismatch");

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = matrix.lengths[p];
        
        // Single pointer dereference
        const ValueType* SCL_RESTRICT vals = static_cast<const ValueType*>(matrix.data_ptrs[p]);
        
        // Fused SIMD accumulation
        auto v_sum = s::Zero(d);
        auto v_sq_sum = s::Zero(d);
        
        Index k = 0;
        
        // 4-way unrolled
        for (; k + 4 * lanes <= len; k += 4 * lanes) {
            auto v0 = s::Load(d, vals + k + 0 * lanes);
            auto v1 = s::Load(d, vals + k + 1 * lanes);
            auto v2 = s::Load(d, vals + k + 2 * lanes);
            auto v3 = s::Load(d, vals + k + 3 * lanes);
            
            v_sum = s::Add(v_sum, v0);
            v_sum = s::Add(v_sum, v1);
            v_sum = s::Add(v_sum, v2);
            v_sum = s::Add(v_sum, v3);
            
            v_sq_sum = s::MulAdd(v0, v0, v_sq_sum);
            v_sq_sum = s::MulAdd(v1, v1, v_sq_sum);
            v_sq_sum = s::MulAdd(v2, v2, v_sq_sum);
            v_sq_sum = s::MulAdd(v3, v3, v_sq_sum);
        }
        
        for (; k + lanes <= len; k += lanes) {
            auto v = s::Load(d, vals + k);
            v_sum = s::Add(v_sum, v);
            v_sq_sum = s::MulAdd(v, v, v_sq_sum);
        }
        
        Real sum = s::GetLane(s::SumOfLanes(d, v_sum));
        Real sq_sum = s::GetLane(s::SumOfLanes(d, v_sq_sum));

        for (; k < len; ++k) {
            Real v = static_cast<Real>(vals[k]);
            sum += v;
            sq_sum += v * v;
        }

        Real mean = sum * inv_n;
        Real var = (sq_sum * inv_n) - (mean * mean);
        if (var < 0) var = 0;

        out_means[p] = mean;
        out_inv_stds[p] = (var > 0) ? (1.0 / std::sqrt(var)) : 0.0;
    });
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void compute_stats_fast(
    const MatrixT& matrix,
    Array<Real> out_means,
    Array<Real> out_inv_stds
) {
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        scl::kernel::correlation::mapped::compute_stats_mapped_dispatch<MatrixT, IsCSR>(matrix, out_means, out_inv_stds);
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        compute_stats_custom_fast(matrix, out_means, out_inv_stds);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        compute_stats_virtual_fast(matrix, out_means, out_inv_stds);
    }
}

} // namespace scl::kernel::correlation::fast

