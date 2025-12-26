#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// Generic SpMV for fallback
#include "scl/kernel/algebra.hpp"

// Mapped backend support
#include "scl/kernel/mapped_common.hpp"
#include "scl/kernel/algebra_mapped_impl.hpp"

// =============================================================================
/// @file algebra_fast_impl.hpp
/// @brief Extreme Performance SpMV - Backend-Specific Optimizations
///
/// ## Interface Contract
///
/// All functions follow the unified signature:
///
///     void spmv_xxx(A, x, y, alpha, beta)
///
/// - VOID RETURN: No allocation, no ownership transfer
/// - PRE-ALLOCATED: y buffer provided by caller
/// - ZERO OVERHEAD: Compile-time dispatch, no virtual calls
///
/// ## Backend Hierarchy
///
/// 1. Mapped (MappedCustomSparse, MappedVirtualSparse)
///    - Memory-mapped files, OS page cache
///    - Load-balanced parallelism
///
/// 2. Custom (CustomSparse)
///    - Contiguous arrays in RAM
///    - 8-way unroll + prefetch
///
/// 3. Virtual (VirtualSparse)
///    - Pointer arrays (row-wise storage)
///    - Minimal indirection overhead
///
/// ## Performance Targets
///
/// - CustomSparse: 1.5-2x faster than generic
/// - VirtualSparse: 1.3-1.5x faster than generic
/// - MappedSparse: Near RAM performance for cached data
// =============================================================================

namespace scl::kernel::algebra::fast {

// =============================================================================
// SECTION 1: Configuration
// =============================================================================

namespace config {
    constexpr size_t PREFETCH_DISTANCE = 64;
    constexpr size_t SHORT_ROW_THRESHOLD = 8;
    constexpr size_t MEDIUM_ROW_THRESHOLD = 64;
}

// =============================================================================
// SECTION 2: Common Utilities
// =============================================================================

namespace detail {

/// @brief SIMD-optimized beta scaling
template <typename T>
void scale_output(T* SCL_RESTRICT y, Size n, T beta) noexcept {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    if (beta == T(0)) {
        const auto v_zero = s::Zero(d);
        size_t i = 0;
        for (; i + lanes <= n; i += lanes) {
            s::Store(v_zero, d, y + i);
        }
        for (; i < n; ++i) {
            y[i] = T(0);
        }
    } else if (beta == T(1)) {
        return;
    } else {
        const auto v_beta = s::Set(d, beta);
        size_t i = 0;
        for (; i + lanes <= n; i += lanes) {
            auto v = s::Load(d, y + i);
            s::Store(s::Mul(v, v_beta), d, y + i);
        }
        for (; i < n; ++i) {
            y[i] *= beta;
        }
    }
}

/// @brief Horizontal sum of 8 accumulators
template <typename T>
SCL_FORCE_INLINE T horizontal_sum_8(T s0, T s1, T s2, T s3, T s4, T s5, T s6, T s7) noexcept {
    return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
}

/// @brief Short row dot (nnz < 8): scalar
template <typename T>
SCL_FORCE_INLINE T sparse_dot_short(
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Size nnz,
    const T* SCL_RESTRICT x
) noexcept {
    T sum = T(0);
    for (Size k = 0; k < nnz; ++k) {
        sum += values[k] * x[indices[k]];
    }
    return sum;
}

/// @brief Medium row dot (8-64): 4-way unroll
template <typename T>
SCL_FORCE_INLINE T sparse_dot_medium(
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Size nnz,
    const T* SCL_RESTRICT x
) noexcept {
    T sum0 = T(0), sum1 = T(0), sum2 = T(0), sum3 = T(0);

    Size k = 0;
    for (; k + 4 <= nnz; k += 4) {
        sum0 += values[k + 0] * x[indices[k + 0]];
        sum1 += values[k + 1] * x[indices[k + 1]];
        sum2 += values[k + 2] * x[indices[k + 2]];
        sum3 += values[k + 3] * x[indices[k + 3]];
    }

    T sum = (sum0 + sum1) + (sum2 + sum3);
    for (; k < nnz; ++k) {
        sum += values[k] * x[indices[k]];
    }
    return sum;
}

/// @brief Long row dot (>= 64): 8-way unroll + prefetch
template <typename T>
SCL_FORCE_INLINE T sparse_dot_long(
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Size nnz,
    const T* SCL_RESTRICT x
) noexcept {
    T sum0 = T(0), sum1 = T(0), sum2 = T(0), sum3 = T(0);
    T sum4 = T(0), sum5 = T(0), sum6 = T(0), sum7 = T(0);

    Size k = 0;
    for (; k + 8 <= nnz; k += 8) {
        if (k + config::PREFETCH_DISTANCE < nnz) {
            SCL_PREFETCH_READ(&values[k + config::PREFETCH_DISTANCE], 0);
            SCL_PREFETCH_READ(&indices[k + config::PREFETCH_DISTANCE], 0);
            SCL_PREFETCH_READ(&x[indices[k + config::PREFETCH_DISTANCE]], 0);
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

    T sum = horizontal_sum_8(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7);
    for (; k < nnz; ++k) {
        sum += values[k] * x[indices[k]];
    }
    return sum;
}

/// @brief Adaptive dispatcher
template <typename T>
SCL_FORCE_INLINE T sparse_dot_adaptive(
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Size nnz,
    const T* SCL_RESTRICT x
) noexcept {
    if (nnz < config::SHORT_ROW_THRESHOLD) {
        return sparse_dot_short(indices, values, nnz, x);
    } else if (nnz < config::MEDIUM_ROW_THRESHOLD) {
        return sparse_dot_medium(indices, values, nnz, x);
    } else {
        return sparse_dot_long(indices, values, nnz, x);
    }
}

} // namespace detail

// =============================================================================
// SECTION 3: CustomSparse Fast Path
// =============================================================================

/// @brief SpMV for CustomSparse: y = alpha * A * x + beta * y
///
/// Optimized for contiguous array storage:
/// - Direct pointer arithmetic
/// - 8-way unroll + prefetch for long rows
/// - Row-parallel processing
///
/// @param A CustomSparse matrix
/// @param x Input vector
/// @param y Output vector, PRE-ALLOCATED
/// @param alpha Scalar for A*x
/// @param beta Scalar for y
template <typename T, bool IsCSR>
    requires CustomSparseLike<CustomSparse<T, IsCSR>, IsCSR>
void spmv_custom(
    const CustomSparse<T, IsCSR>& A,
    Array<const T> x,
    Array<T> y,
    T alpha = T(1),
    T beta = T(0)
) {
    const Index primary_dim = scl::primary_size(A);

    SCL_CHECK_DIM(y.len >= static_cast<Size>(primary_dim),
        "SpMV: output vector too small");

    // Beta scaling
    detail::scale_output(y.ptr, static_cast<Size>(primary_dim), beta);

    // Early exit
    if (alpha == T(0)) return;

    // Parallel SpMV
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index start = A.indptr[p];
        const Index end = A.indptr[p + 1];
        const Size len = static_cast<Size>(end - start);

        if (len == 0) return;

        T dot = detail::sparse_dot_adaptive(
            A.indices + start,
            A.data + start,
            len,
            x.ptr
        );

        y[p] += alpha * dot;
    });
}

// =============================================================================
// SECTION 4: VirtualSparse Fast Path
// =============================================================================

/// @brief SpMV for VirtualSparse: y = alpha * A * x + beta * y
///
/// Optimized for pointer-based row storage:
/// - Single pointer dereference per row
/// - Adaptive dot product strategy
///
/// @param A VirtualSparse matrix
/// @param x Input vector
/// @param y Output vector, PRE-ALLOCATED
/// @param alpha Scalar for A*x
/// @param beta Scalar for y
template <typename T, bool IsCSR>
    requires VirtualSparseLike<VirtualSparse<T, IsCSR>, IsCSR>
void spmv_virtual(
    const VirtualSparse<T, IsCSR>& A,
    Array<const T> x,
    Array<T> y,
    T alpha = T(1),
    T beta = T(0)
) {
    const Index primary_dim = scl::primary_size(A);

    SCL_CHECK_DIM(y.len >= static_cast<Size>(primary_dim),
        "SpMV: output vector too small");

    // Beta scaling
    detail::scale_output(y.ptr, static_cast<Size>(primary_dim), beta);

    // Early exit
    if (alpha == T(0)) return;

    // Parallel SpMV with minimal indirection
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Size len = static_cast<Size>(A.lengths[p]);

        if (len == 0) return;

        // Single dereference per row
        const T* SCL_RESTRICT vals = static_cast<const T*>(A.data_ptrs[p]);
        const Index* SCL_RESTRICT inds = static_cast<const Index*>(A.indices_ptrs[p]);

        T dot = detail::sparse_dot_adaptive(inds, vals, len, x.ptr);

        y[p] += alpha * dot;
    });
}

// =============================================================================
// SECTION 5: Unified Dispatcher
// =============================================================================

/// @brief Unified SpMV dispatcher with compile-time backend selection
///
/// Automatically selects the optimal implementation:
/// - MappedSparseLike -> mapped::spmv_mapped
/// - CustomSparseLike -> spmv_custom
/// - VirtualSparseLike -> spmv_virtual
///
/// @param A Sparse matrix (any backend)
/// @param x Input vector
/// @param y Output vector, PRE-ALLOCATED
/// @param alpha Scalar for A*x
/// @param beta Scalar for y
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void spmv_fast(
    const MatrixT& A,
    Array<const typename MatrixT::ValueType> x,
    Array<typename MatrixT::ValueType> y,
    typename MatrixT::ValueType alpha = typename MatrixT::ValueType(1),
    typename MatrixT::ValueType beta = typename MatrixT::ValueType(0)
) {
    if constexpr (kernel::mapped::MappedSparseLike<MatrixT, IsCSR>) {
        scl::kernel::algebra::mapped::spmv_mapped(A, x, y, alpha, beta);
    } else if constexpr (CustomSparseLike<MatrixT, IsCSR>) {
        spmv_custom(A, x, y, alpha, beta);
    } else if constexpr (VirtualSparseLike<MatrixT, IsCSR>) {
        spmv_virtual(A, x, y, alpha, beta);
    } else {
        // Fallback to generic SpMV
        scl::kernel::algebra::spmv(A, x, y, alpha, beta);
    }
}

// =============================================================================
// SECTION 6: Convenience Wrappers
// =============================================================================

/// @brief y = A * x (simplified)
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void spmv(
    const MatrixT& A,
    Array<const typename MatrixT::ValueType> x,
    Array<typename MatrixT::ValueType> y
) {
    using T = typename MatrixT::ValueType;
    spmv_fast<MatrixT, IsCSR>(A, x, y, T(1), T(0));
}

/// @brief y += A * x (accumulate)
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void spmv_add(
    const MatrixT& A,
    Array<const typename MatrixT::ValueType> x,
    Array<typename MatrixT::ValueType> y
) {
    using T = typename MatrixT::ValueType;
    spmv_fast<MatrixT, IsCSR>(A, x, y, T(1), T(1));
}

/// @brief y = alpha * A * x (scaled)
template <typename MatrixT, bool IsCSR>
    requires SparseLike<MatrixT, IsCSR>
void spmv_scaled(
    const MatrixT& A,
    Array<const typename MatrixT::ValueType> x,
    Array<typename MatrixT::ValueType> y,
    typename MatrixT::ValueType alpha
) {
    using T = typename MatrixT::ValueType;
    spmv_fast<MatrixT, IsCSR>(A, x, y, alpha, T(0));
}

} // namespace scl::kernel::algebra::fast
