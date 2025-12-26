#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
/// @file algebra_fast_impl.hpp
/// @brief Extreme Performance SpMV
///
/// Separate optimizations:
/// - CustomSparse: 8-way unrolling + prefetch on contiguous data
/// - VirtualSparse: Row-wise with minimal pointer dereference
///
/// Performance Target: 1.5-2x faster than generic
// =============================================================================

namespace scl::kernel::algebra::fast {

namespace detail {

constexpr size_t PREFETCH_DISTANCE = 64;
constexpr size_t UNROLL_FACTOR = 8;

/// @brief Ultra-optimized sparse-dense dot (8-way unroll)
template <typename T>
SCL_FORCE_INLINE T sparse_dot_ultra(
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Size nnz,
    const T* SCL_RESTRICT x
) {
    T sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    T sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;
    
    Size k = 0;
    for (; k + 8 <= nnz; k += 8) {
        if (k + PREFETCH_DISTANCE < nnz) {
            Index future_idx = indices[k + PREFETCH_DISTANCE];
            SCL_PREFETCH_READ(&x[future_idx], 0);
        }
        
        sum0 += values[k + 0] * x[indices[k + 0]];
        sum1 += values[k + 1] * x[indices[k + 1]];
        sum2 += values[k + 2] * x[indices[k + 2]];
        sum3 += values[k + 3] * x[indices[k + 3]];
        sum4 += values[k + 4] * x[indices[k + 4]];
        sum5 += values[k + 5] * x[indices[k + 5]];
        sum6 += values[k + 6] * x[indices[k + 6]];
        sum7 += values[k + 7] * x[indices[k + 7]];
    }
    
    T sum = (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7);
    
    for (; k < nnz; ++k) {
        sum += values[k] * x[indices[k]];
    }
    
    return sum;
}

} // namespace detail

// =============================================================================
// CustomSparse Fast Path
// =============================================================================

template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void spmv_custom_fast(
    const CustomSparse<T, IsCSR>& A,
    Array<const T> x,
    Array<T> y,
    T alpha,
    T beta
) {
    const Index primary_dim = scl::primary_size(A);
    
    // Beta scaling (same as before)
    if (beta != static_cast<T>(1.0)) {
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        if (beta == static_cast<T>(0.0)) {
            const auto v_zero = s::Zero(d);
            size_t i = 0;
            for (; i + lanes <= y.len; i += lanes) {
                s::Store(v_zero, d, y.ptr + i);
            }
            for (; i < y.len; ++i) {
                y[i] = static_cast<T>(0.0);
            }
        } else {
            const auto v_beta = s::Set(d, beta);
            size_t i = 0;
            for (; i + lanes <= y.len; i += lanes) {
                auto v = s::Load(d, y.ptr + i);
                s::Store(s::Mul(v, v_beta), d, y.ptr + i);
            }
            for (; i < y.len; ++i) {
                y[i] *= beta;
            }
        }
    }

    // Parallel SpMV
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index start = A.indptr[p];
        Index end = A.indptr[p + 1];
        Index len = end - start;
        
        if (len == 0) return;

        T dot = detail::sparse_dot_ultra(
            A.indices + start,
            A.data + start,
            static_cast<Size>(len),
            x.ptr
        );

        y[p] += alpha * dot;
    });
}

// =============================================================================
// VirtualSparse Fast Path
// =============================================================================

template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
SCL_FORCE_INLINE void spmv_virtual_fast(
    const VirtualSparse<T, IsCSR>& A,
    Array<const T> x,
    Array<T> y,
    T alpha,
    T beta
) {
    const Index primary_dim = scl::primary_size(A);
    
    // Beta scaling (same as Custom)
    if (beta != static_cast<T>(1.0)) {
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();
        
        if (beta == static_cast<T>(0.0)) {
            const auto v_zero = s::Zero(d);
            size_t i = 0;
            for (; i + lanes <= y.len; i += lanes) {
                s::Store(v_zero, d, y.ptr + i);
            }
            for (; i < y.len; ++i) {
                y[i] = static_cast<T>(0.0);
            }
        } else {
            const auto v_beta = s::Set(d, beta);
            size_t i = 0;
            for (; i + lanes <= y.len; i += lanes) {
                auto v = s::Load(d, y.ptr + i);
                s::Store(s::Mul(v, v_beta), d, y.ptr + i);
            }
            for (; i < y.len; ++i) {
                y[i] *= beta;
            }
        }
    }

    // Parallel SpMV (minimal pointer dereference)
    scl::threading::parallel_for(0, static_cast<size_t>(primary_dim), [&](size_t p) {
        Index len = A.lengths[p];
        
        if (len == 0) return;

        // Single pointer dereference per row
        const T* SCL_RESTRICT vals = static_cast<const T*>(A.data_ptrs[p]);
        const Index* SCL_RESTRICT inds = static_cast<const Index*>(A.indices_ptrs[p]);

        T dot = detail::sparse_dot_ultra(
            inds,
            vals,
            static_cast<Size>(len),
            x.ptr
        );

        y[p] += alpha * dot;
    });
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void spmv_fast(
    const MatrixT& A,
    Array<const typename MatrixT::ValueType> x,
    Array<typename MatrixT::ValueType> y,
    typename MatrixT::ValueType alpha,
    typename MatrixT::ValueType beta
) {
    if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        spmv_custom_fast(A, x, y, alpha, beta);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        spmv_virtual_fast(A, x, y, alpha, beta);
    }
}

} // namespace scl::kernel::algebra::fast
