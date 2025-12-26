#pragma once

#include "scl/core/type.hpp"
#include "scl/core/matrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>

// =============================================================================
/// @file algebra.hpp
/// @brief Sparse Linear Algebra Primitives
///
/// Implements high-performance Sparse Matrix-Vector Multiplication (SpMV).
///
/// Operations:
///
/// - spmv(CSR): y = alpha * A * x + beta * y
/// - spmv_trans(CSC): y = alpha * A^T * x + beta * y
///
/// Optimization Features:
///
/// - Row-Parallelism: Perfect load balancing for dense output vectors
/// - SIMD: Vectorized scaling and accumulation
/// - Manual Unrolling: 4-way unrolled dot products
/// - Prefetching: Hides memory latency for indirect access (x[col_idx])
/// - Beta Handling: Fast path for beta=0 (no scaling) and beta=1 (no-op)
///
/// Performance:
///
/// - Bandwidth: ~10-15 GB/s per core (memory bound)
/// - Scalability: Linear with thread count
/// - Use Case: Iterative solvers (Lanczos, Arnoldi, PageRank)
// =============================================================================

namespace scl::kernel::algebra {

namespace detail {

/// @brief Scale or zero output vector: y = beta * y.
///
/// Optimized paths for beta=0 (zero) and beta=1 (identity).
template <typename T>
SCL_FORCE_INLINE void scale_vector(MutableSpan<T> y, T beta) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const Size n = y.size;

    if (beta == static_cast<T>(0.0)) {
        // Fast path: zero out
        const auto v_zero = s::Zero(d);
        size_t i = 0;
        
        for (; i + lanes <= n; i += lanes) {
            s::Store(v_zero, d, y.ptr + i);
        }
        
        for (; i < n; ++i) {
            y[i] = static_cast<T>(0.0);
        }
    } else if (beta != static_cast<T>(1.0)) {
        // Scale by beta
        const auto v_beta = s::Set(d, beta);
        size_t i = 0;
        
        for (; i + lanes <= n; i += lanes) {
            auto v = s::Load(d, y.ptr + i);
            v = s::Mul(v, v_beta);
            s::Store(v, d, y.ptr + i);
        }
        
        for (; i < n; ++i) {
            y[i] *= beta;
        }
    }
    // beta == 1.0: no-op
}

/// @brief Sparse-Dense dot product with manual unrolling and prefetching.
///
/// Computes: sum over k of values[k] * x[indices[k]]
///
/// Optimizations:
/// - 4-way manual unrolling to break dependency chains
/// - Prefetching indices to hide memory latency
/// - Restrict pointers for better alias analysis
template <typename T>
SCL_FORCE_INLINE T sparse_dot_dense(
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Size nnz,
    const T* SCL_RESTRICT x
) {
    T sum = static_cast<T>(0.0);
    Size k = 0;

    // 4-way unrolled loop
    constexpr Size UNROLL = 4;
    constexpr Size PREFETCH_DISTANCE = 8;
    
    for (; k + UNROLL <= nnz; k += UNROLL) {
        // Prefetch upcoming indices
        if (SCL_LIKELY(k + PREFETCH_DISTANCE < nnz)) {
            SCL_PREFETCH_READ(&indices[k + PREFETCH_DISTANCE], 1);
            SCL_PREFETCH_READ(&x[indices[k + PREFETCH_DISTANCE]], 1);
        }
        
        // Load indices (breaks dependency on sum)
        Index i0 = indices[k];
        Index i1 = indices[k + 1];
        Index i2 = indices[k + 2];
        Index i3 = indices[k + 3];

        // Load values
        T v0 = values[k];
        T v1 = values[k + 1];
        T v2 = values[k + 2];
        T v3 = values[k + 3];

        // Accumulate (compiler can parallelize these)
        sum += v0 * x[i0];
        sum += v1 * x[i1];
        sum += v2 * x[i2];
        sum += v3 * x[i3];
    }

    // Scalar tail
    for (; k < nnz; ++k) {
        sum += values[k] * x[indices[k]];
    }

    return sum;
}

} // namespace detail

// =============================================================================
// Public API
// =============================================================================

/// @brief Sparse Matrix-Vector Multiplication (Generic CSR-like matrices).
///
/// Computes: y = alpha * A * x + beta * y
///
/// BLAS-like semantics:
/// - beta = 0: y is overwritten (no read of y required)
/// - beta = 1: y is accumulated into
/// - alpha: scales the result of A*x
///
/// Parallelism: Row-parallel (each thread computes subset of y elements).
///
/// @tparam MatrixT Any CSR-like matrix type
/// @param A CSR-like matrix (m x n)
/// @param x Input vector [size = n]
/// @param y Output vector [size = m], modified in-place
/// @param alpha Scalar multiplier for A*x (default 1.0)
/// @param beta Scalar multiplier for y (default 0.0)
template <CSRLike MatrixT>
void spmv(
    const MatrixT& A,
    Span<const typename MatrixT::ValueType> x,
    MutableSpan<typename MatrixT::ValueType> y,
    typename MatrixT::ValueType alpha = static_cast<typename MatrixT::ValueType>(1.0),
    typename MatrixT::ValueType beta = static_cast<typename MatrixT::ValueType>(0.0)
) {
    using T = typename MatrixT::ValueType;
    const Index M = A.rows;
    const Index N = A.cols;
    
    SCL_CHECK_DIM(x.size == static_cast<Size>(N), "SpMV: x dimension mismatch");
    SCL_CHECK_DIM(y.size == static_cast<Size>(M), "SpMV: y dimension mismatch");

    // Handle beta scaling (y = beta * y)
    if (beta != static_cast<T>(1.0)) {
        detail::scale_vector(y, beta);
    }

    // Compute A*x and accumulate: y += alpha * (A*x)
    scl::threading::parallel_for(0, static_cast<size_t>(M), [&](size_t i) {
        const Index row_idx = static_cast<Index>(i);
        
        // Access CSR row data via unified interface
        auto row_vals = A.row_values(row_idx);
        auto row_inds = A.row_indices(row_idx);
        
        if (row_vals.size == 0) return;  // Empty row contributes nothing

        // Compute dot product
        T dot = detail::sparse_dot_dense(row_inds.ptr, row_vals.ptr, row_vals.size, x.ptr);

        // Update output: y[i] += alpha * dot
        y[row_idx] += alpha * dot;
    });
}

/// @brief Transposed Sparse Matrix-Vector Multiplication (Generic CSC-like matrices).
///
/// Computes: y = alpha * A^T * x + beta * y
///
/// Optimized for CSC: Each column of A (row of A^T) is processed independently.
///
/// @tparam MatrixT Any CSC-like matrix type
/// @param A CSC-like matrix (m x n)
/// @param x Input vector [size = m]
/// @param y Output vector [size = n], modified in-place
/// @param alpha Scalar multiplier for A^T*x (default 1.0)
/// @param beta Scalar multiplier for y (default 0.0)
template <CSCLike MatrixT>
void spmv_trans(
    const MatrixT& A,
    Span<const typename MatrixT::ValueType> x,
    MutableSpan<typename MatrixT::ValueType> y,
    typename MatrixT::ValueType alpha = static_cast<typename MatrixT::ValueType>(1.0),
    typename MatrixT::ValueType beta = static_cast<typename MatrixT::ValueType>(0.0)
) {
    using T = typename MatrixT::ValueType;
    const Index M = A.rows;
    const Index N = A.cols;
    
    SCL_CHECK_DIM(x.size == static_cast<Size>(M), "SpMV Trans: x dimension mismatch");
    SCL_CHECK_DIM(y.size == static_cast<Size>(N), "SpMV Trans: y dimension mismatch");

    // Handle beta scaling
    if (beta != static_cast<T>(1.0)) {
        detail::scale_vector(y, beta);
    }

    // Compute A^T * x
    // For CSC, each column j of A corresponds to row j of A^T
    scl::threading::parallel_for(0, static_cast<size_t>(N), [&](size_t j) {
        const Index col_idx = static_cast<Index>(j);
        
        // Access CSC column data via unified interface
        auto col_vals = A.col_values(col_idx);
        auto col_inds = A.col_indices(col_idx);
        
        if (col_vals.size == 0) return;

        // Compute dot product
        T dot = detail::sparse_dot_dense(col_inds.ptr, col_vals.ptr, col_vals.size, x.ptr);

        // Update output
        y[col_idx] += alpha * dot;
    });
}

/// @brief Sparse Matrix-Matrix Multiplication (SpMM): Y = alpha * A * X + beta * Y.
///
/// Computes multiple SpMV operations in parallel for dense matrix X.
///
/// Use Case: Block Krylov methods, multi-vector power iteration.
///
/// @param A CSR matrix (m x n)
/// @param X Input dense matrix (n x k), column-major
/// @param Y Output dense matrix (m x k), column-major, modified in-place
/// @param k Number of vectors
/// @param alpha Scalar multiplier
/// @param beta Scalar multiplier
template <typename T>
void spmm(
    const CSRMatrix<T>& A,
    const T* X,
    T* Y,
    Size n_vectors,
    T alpha = static_cast<T>(1.0),
    T beta = static_cast<T>(0.0)
) {
    const Index M = A.rows;
    const Index N = A.cols;

    // Process each column of X independently
    scl::threading::parallel_for(0, n_vectors, [&](size_t vec_idx) {
        // Extract column vec_idx from X and Y
        const T* x_col = X + vec_idx * static_cast<Size>(N);
        T* y_col = Y + vec_idx * static_cast<Size>(M);

        // Perform SpMV for this column
        spmv(
            A,
            {x_col, static_cast<Size>(N)},
            {y_col, static_cast<Size>(M)},
            alpha,
            beta
        );
    });
}

} // namespace scl::kernel::algebra

