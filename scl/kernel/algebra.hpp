#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
// FILE: scl/kernel/algebra.hpp
// BRIEF: High-Performance Sparse Linear Algebra Kernels
// =============================================================================

namespace scl::kernel::algebra {

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

// SIMD-optimized beta scaling
template <typename T>
void scale_output(T* SCL_RESTRICT y, Size n, T beta) noexcept {
    if (beta == T(0)) {
        scl::memory::zero(Array<T>(y, n));
    } else if (beta == T(1)) {
        return;
    } else {
        namespace s = scl::simd;
        const s::Tag d;
        const size_t lanes = s::lanes();

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

// Horizontal sum of 8 accumulators
template <typename T>
SCL_FORCE_INLINE T horizontal_sum_8(T s0, T s1, T s2, T s3, T s4, T s5, T s6, T s7) noexcept {
    return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
}

// Short row dot (nnz < 8): scalar
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

// Medium row dot (8-64): 4-way unroll
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

// Long row dot (>= 64): 8-way unroll + prefetch
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

// Adaptive dispatcher
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
// SECTION 3: Sparse Matrix-Vector Multiplication
// =============================================================================

// SpMV for Sparse matrices: y = alpha * A * x + beta * y
// Optimized for pointer-based row storage with adaptive dot product strategy
template <typename T, bool IsCSR>
void spmv(
    const Sparse<T, IsCSR>& A,
    Array<const T> x,
    Array<T> y,
    T alpha = T(1),
    T beta = T(0)
) {
    const Index primary_dim = A.primary_dim();

    SCL_CHECK_DIM(y.len >= static_cast<Size>(primary_dim),
        "SpMV: output vector too small");

    // Beta scaling
    detail::scale_output(y.ptr, static_cast<Size>(primary_dim), beta);

    // Early exit
    if (alpha == T(0)) return;

    // Parallel SpMV
    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index primary_idx = static_cast<Index>(p);
        const Size len = static_cast<Size>(A.primary_length(primary_idx));

        if (len == 0) return;

        // Single dereference per row/column
        auto vals = A.primary_values(primary_idx);
        auto inds = A.primary_indices(primary_idx);

        T dot = detail::sparse_dot_adaptive(inds.ptr, vals.ptr, len, x.ptr);

        y[p] += alpha * dot;
    });
}

// =============================================================================
// SECTION 4: Convenience Wrappers
// =============================================================================

// y = A * x (simplified)
template <typename T, bool IsCSR>
void spmv_simple(
    const Sparse<T, IsCSR>& A,
    Array<const T> x,
    Array<T> y
) {
    spmv(A, x, y, T(1), T(0));
}

// y += A * x (accumulate)
template <typename T, bool IsCSR>
void spmv_add(
    const Sparse<T, IsCSR>& A,
    Array<const T> x,
    Array<T> y
) {
    spmv(A, x, y, T(1), T(1));
}

// y = alpha * A * x (scaled)
template <typename T, bool IsCSR>
void spmv_scaled(
    const Sparse<T, IsCSR>& A,
    Array<const T> x,
    Array<T> y,
    T alpha
) {
    spmv(A, x, y, alpha, T(0));
}

} // namespace scl::kernel::algebra

