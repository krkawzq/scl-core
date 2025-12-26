#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/sparse.hpp"
#include "scl/threading/parallel_for.hpp"

// Mapped backend support
#include "scl/kernel/mapped_common.hpp"
#include "scl/kernel/sparse_mapped_impl.hpp"

// =============================================================================
/// @file sparse_fast_impl.hpp
/// @brief Extreme Performance Sparse Statistics
///
/// Separate optimizations:
/// - CustomSparse: Batch SIMD on entire data array
/// - VirtualSparse: Row-wise SIMD with minimal indirection
///
/// Performance Target: 3-4x faster than generic
/// Bandwidth: 15-20 GB/s per core
// =============================================================================

namespace scl::kernel::sparse::fast {

namespace detail {

/// @brief Cache-aligned accumulator
template <typename T>
struct alignas(64) Accumulator {
    T value;
    char padding[64 - sizeof(T)];
    Accumulator() : value(0) {}
};

} // namespace detail

// =============================================================================
// CustomSparse Fast Path
// =============================================================================

/// @brief Ultra-fast primary sums (CustomSparse - batch mode)
///
/// Optimization: Process entire data array, then distribute to outputs
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void primary_sums_custom_fast(
    const CustomSparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(output.len == static_cast<Size>(primary_dim), "Output size mismatch");
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    // Parallel over primary dimension
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Index len = end - start;
        
        if (len == 0) {
            output[p] = static_cast<T>(0);
            return;
        }
        
        const T* SCL_RESTRICT vals = matrix.data + start;
        
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
        
        // Combine accumulators
        auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
        
        for (; k + lanes <= len; k += lanes) {
            v_sum = s::Add(v_sum, s::Load(d, vals + k));
        }
        
        T sum = s::GetLane(s::SumOfLanes(d, v_sum));
        
        for (; k < len; ++k) {
            sum += vals[k];
        }
        
        output[p] = sum;
    });
}

/// @brief Ultra-fast primary variances (CustomSparse - fused)
///
/// Optimization: Fused sum/sum_sq accumulation in single pass
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void primary_variances_custom_fast(
    const CustomSparse<T, IsCSR>& matrix,
    Array<typename CustomSparse<T, IsCSR>::ValueType> output,
    int ddof
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Real N = static_cast<Real>(scl::secondary_size(matrix));
    const Real denom = N - static_cast<Real>(ddof);
    
    SCL_CHECK_DIM(output.len == static_cast<Size>(primary_dim), "Output size mismatch");
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index start = matrix.indptr[p];
        Index end = matrix.indptr[p + 1];
        Index len = end - start;
        
        const T* SCL_RESTRICT vals = matrix.data + start;
        
        auto v_sum = s::Zero(d);
        auto v_ssq = s::Zero(d);
        
        Index k = 0;
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
        Real var = 0.0;
        
        if (denom > 0) {
            var = (sum_sq - sum * mu) / denom;
        }
        
        if (var < 0) var = 0.0;
        
        output[p] = var;
    });
}

// =============================================================================
// VirtualSparse Fast Path
// =============================================================================

/// @brief Ultra-fast primary sums (VirtualSparse - row-wise)
///
/// Optimization: Row-wise SIMD with minimal pointer dereference
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void primary_sums_virtual_fast(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index primary_dim = scl::primary_size(matrix);
    
    SCL_CHECK_DIM(output.len == static_cast<Size>(primary_dim), "Output size mismatch");
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = matrix.lengths[p];
        
        if (len == 0) {
            output[p] = static_cast<T>(0);
            return;
        }
        
        // Single pointer dereference
        const T* SCL_RESTRICT vals = static_cast<const T*>(matrix.data_ptrs[p]);
        
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
        
        // Combine accumulators
        auto v_sum = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));
        
        for (; k + lanes <= len; k += lanes) {
            v_sum = s::Add(v_sum, s::Load(d, vals + k));
        }
        
        T sum = s::GetLane(s::SumOfLanes(d, v_sum));
        
        for (; k < len; ++k) {
            sum += vals[k];
        }
        
        output[p] = sum;
    });
}

/// @brief Ultra-fast primary variances (VirtualSparse - fused)
///
/// Optimization: Fused sum/sum_sq accumulation in single pass
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void primary_variances_virtual_fast(
    const VirtualSparse<T, IsCSR>& matrix,
    Array<typename VirtualSparse<T, IsCSR>::ValueType> output,
    int ddof
) {
    const Index primary_dim = scl::primary_size(matrix);
    const Real N = static_cast<Real>(scl::secondary_size(matrix));
    const Real denom = N - static_cast<Real>(ddof);
    
    SCL_CHECK_DIM(output.len == static_cast<Size>(primary_dim), "Output size mismatch");
    
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = matrix.lengths[p];
        
        // Single pointer dereference
        const T* SCL_RESTRICT vals = static_cast<const T*>(matrix.data_ptrs[p]);
        
        auto v_sum = s::Zero(d);
        auto v_ssq = s::Zero(d);
        
        Index k = 0;
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
        Real var = 0.0;
        
        if (denom > 0) {
            var = (sum_sq - sum * mu) / denom;
        }
        
        if (var < 0) var = 0.0;
        
        output[p] = var;
    });
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void primary_sums_fast(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output
) {
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        scl::kernel::sparse::mapped::primary_sums_mapped_dispatch<MatrixT, IsCSR>(matrix, output);
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        primary_sums_custom_fast(matrix, output);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        primary_sums_virtual_fast(matrix, output);
    }
}

/// @brief Auto-dispatch to appropriate fast path
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void primary_variances_fast(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output,
    int ddof
) {
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        scl::kernel::sparse::mapped::primary_variances_mapped_dispatch<MatrixT, IsCSR>(matrix, output, ddof);
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        primary_variances_custom_fast(matrix, output, ddof);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        primary_variances_virtual_fast(matrix, output, ddof);
    }
}

} // namespace scl::kernel::sparse::fast

