#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
/// @file algebra.hpp
/// @brief Sparse Linear Algebra - Generic Implementation
///
/// ## Interface Contract
///
/// All SpMV functions follow the signature:
///
///     void spmv(A, x, y, alpha, beta)
///
/// Computes: y = alpha * A * x + beta * y
///
/// Parameters:
/// - A: Sparse matrix (any SparseLike type)
/// - x: Input vector (Array<const T>), size = secondary_dim(A)
/// - y: Output vector (Array<T>), size = primary_dim(A), PRE-ALLOCATED
/// - alpha: Scalar for A*x (default: 1)
/// - beta: Scalar for y (default: 0)
///
/// Design Principles:
/// - VOID RETURN: No memory allocation, no ownership transfer
/// - PRE-ALLOCATED: Caller provides output buffer
/// - GENERIC: Works with any SparseLike type
///
/// ## Performance Strategy
///
/// - Generic Path: Works for all sparse types via SparseLike concept
/// - Fast Path: Use algebra_fast_impl.hpp for optimized backends
///
/// Bandwidth: ~10-15 GB/s per core (memory bound)
// =============================================================================

namespace scl::kernel::algebra {

namespace detail {

/// @brief Scale vector: y = beta * y (SIMD optimized)
///
/// Handles special cases efficiently:
/// - beta = 0: zero fill
/// - beta = 1: no-op
/// - other: vectorized scaling
template <typename T>
SCL_FORCE_INLINE void scale_vector(T* y, Size n, T beta) noexcept {
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
    } else if (beta != T(1)) {
        const auto v_beta = s::Set(d, beta);
        size_t i = 0;

        for (; i + lanes <= n; i += lanes) {
            auto v = s::Load(d, y + i);
            v = s::Mul(v, v_beta);
            s::Store(v, d, y + i);
        }

        for (; i < n; ++i) {
            y[i] *= beta;
        }
    }
    // beta == 1: no-op
}

/// @brief Sparse-dense dot product with 4-way unrolling
///
/// Breaks dependency chains for better ILP on modern CPUs.
template <typename T>
SCL_FORCE_INLINE T sparse_dot_dense(
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

    // Scalar tail
    for (; k < nnz; ++k) {
        sum += values[k] * x[indices[k]];
    }

    return sum;
}

} // namespace detail

// =============================================================================
// Public API: Generic SpMV
// =============================================================================

/// @brief Sparse Matrix-Vector Multiplication (generic)
///
/// Computes: y = alpha * A * x + beta * y
///
/// Works with any type satisfying SparseLike concept.
/// For optimized backends (Custom, Virtual, Mapped), use algebra_fast_impl.hpp.
///
/// @param A Sparse matrix (SparseLike)
/// @param x Input vector, size = secondary_dim(A)
/// @param y Output vector, size = primary_dim(A), PRE-ALLOCATED
/// @param alpha Scalar multiplier for A*x (default: 1)
/// @param beta Scalar multiplier for y (default: 0)
template <typename MatrixT>
    requires AnySparse<MatrixT>
void spmv(
    const MatrixT& A,
    Array<const typename MatrixT::ValueType> x,
    Array<typename MatrixT::ValueType> y,
    typename MatrixT::ValueType alpha = typename MatrixT::ValueType(1),
    typename MatrixT::ValueType beta = typename MatrixT::ValueType(0)
) {
    using T = typename MatrixT::ValueType;
    const Index primary_dim = scl::primary_size(A);
    const Index secondary_dim = scl::secondary_size(A);

    SCL_CHECK_DIM(x.size() >= static_cast<Size>(secondary_dim),
        "SpMV: x dimension too small");
    SCL_CHECK_DIM(y.size() >= static_cast<Size>(primary_dim),
        "SpMV: y dimension too small");

    // Handle beta scaling
    detail::scale_vector(y.ptr, static_cast<Size>(primary_dim), beta);

    // Early exit
    if (alpha == T(0)) return;

    // Fast path for CustomSparseLike (direct pointer access)
    if constexpr (CustomSparseLike<MatrixT, true> || CustomSparseLike<MatrixT, false>) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
            const Index start = A.indptr[p];
            const Index end = A.indptr[p + 1];
            const Size len = static_cast<Size>(end - start);

            if (len == 0) return;

            T dot = detail::sparse_dot_dense(
                A.indices + start,
                A.data + start,
                len,
                x.ptr
            );

            y[p] += alpha * dot;
        });
    } else {
        // Generic path via SparseLike interface
        scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
            const Index primary_idx = static_cast<Index>(p);

            auto vals = scl::primary_values(A, primary_idx);
            auto inds = scl::primary_indices(A, primary_idx);

            if (vals.size() == 0) return;

            T dot = detail::sparse_dot_dense(inds.ptr, vals.ptr, vals.size(), x.ptr);
            y[p] += alpha * dot;
        });
    }
}

/// @brief y = A * x (simplified interface)
template <typename MatrixT>
    requires AnySparse<MatrixT>
void spmv_simple(
    const MatrixT& A,
    Array<const typename MatrixT::ValueType> x,
    Array<typename MatrixT::ValueType> y
) {
    using T = typename MatrixT::ValueType;
    spmv(A, x, y, T(1), T(0));
}

/// @brief y += A * x (accumulate)
template <typename MatrixT>
    requires AnySparse<MatrixT>
void spmv_add(
    const MatrixT& A,
    Array<const typename MatrixT::ValueType> x,
    Array<typename MatrixT::ValueType> y
) {
    using T = typename MatrixT::ValueType;
    spmv(A, x, y, T(1), T(1));
}

// =============================================================================
// Public API: SpMM (Sparse-Dense Matrix Multiply)
// =============================================================================

/// @brief Sparse Matrix-Dense Matrix Multiplication
///
/// Computes: Y = alpha * A * X + beta * Y
/// Where X and Y are dense matrices (column-major).
///
/// @param A Sparse matrix (M x K)
/// @param X Dense input matrix (K x N), column-major
/// @param Y Dense output matrix (M x N), column-major, PRE-ALLOCATED
/// @param n_vectors Number of columns in X and Y
/// @param alpha Scalar multiplier for A*X (default: 1)
/// @param beta Scalar multiplier for Y (default: 0)
template <typename MatrixT>
    requires AnySparse<MatrixT>
void spmm(
    const MatrixT& A,
    const typename MatrixT::ValueType* X,
    typename MatrixT::ValueType* Y,
    Size n_vectors,
    typename MatrixT::ValueType alpha = typename MatrixT::ValueType(1),
    typename MatrixT::ValueType beta = typename MatrixT::ValueType(0)
) {
    using T = typename MatrixT::ValueType;
    const Index primary_dim = scl::primary_size(A);
    const Index secondary_dim = scl::secondary_size(A);

    // Process each column of X/Y independently
    scl::threading::parallel_for(Size(0), n_vectors, [&](size_t vec_idx) {
        const T* x_col = X + vec_idx * static_cast<Size>(secondary_dim);
        T* y_col = Y + vec_idx * static_cast<Size>(primary_dim);

        spmv(
            A,
            Array<const T>(x_col, static_cast<Size>(secondary_dim)),
            Array<T>(y_col, static_cast<Size>(primary_dim)),
            alpha,
            beta
        );
    });
}

} // namespace scl::kernel::algebra
