#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"
#include "scl/threading/scheduler.hpp"

#include <atomic>
#include <cstring>
#include <cmath>
#include <new>

// =============================================================================
// FILE: scl/kernel/algebra.hpp
// BRIEF: High-Performance Sparse Linear Algebra Kernels
// =============================================================================

namespace scl::kernel::algebra {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr size_t PREFETCH_DISTANCE = 64;
    constexpr size_t SHORT_ROW_THRESHOLD = 8;
    constexpr size_t MEDIUM_ROW_THRESHOLD = 64;
    constexpr size_t LONG_ROW_THRESHOLD = 256;
    constexpr size_t CONSECUTIVE_CHECK_THRESHOLD = 16;
    constexpr size_t SPMM_BLOCK_COLS = 64;
    constexpr size_t SPMM_BLOCK_ROWS = 32;
    constexpr Size PARALLEL_THRESHOLD = 128;
}

// =============================================================================
// Internal Optimized Operations
// =============================================================================

namespace detail {

// =============================================================================
// SIMD Scale with 4-way Unrolling + Prefetch
// =============================================================================

template <typename T>
SCL_HOT SCL_FORCE_INLINE void scale_output(T* SCL_RESTRICT y, Size n, T beta) noexcept {
    if (SCL_UNLIKELY(beta == T(0))) {
        scl::memory::zero(Array<T>(y, n));
        return;
    }
    if (SCL_LIKELY(beta == T(1))) {
        return;
    }

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);
    const auto v_beta = s::Set(d, beta);

    Size i = 0;

    // 4-way SIMD unroll with prefetch
    for (; i + 4 * lanes <= n; i += 4 * lanes) {
        if (SCL_LIKELY(i + 4 * lanes + config::PREFETCH_DISTANCE < n)) {
            SCL_PREFETCH_WRITE(y + i + config::PREFETCH_DISTANCE, 0);
        }

        auto v0 = s::Load(d, y + i);
        auto v1 = s::Load(d, y + i + lanes);
        auto v2 = s::Load(d, y + i + 2 * lanes);
        auto v3 = s::Load(d, y + i + 3 * lanes);

        s::Store(s::Mul(v0, v_beta), d, y + i);
        s::Store(s::Mul(v1, v_beta), d, y + i + lanes);
        s::Store(s::Mul(v2, v_beta), d, y + i + 2 * lanes);
        s::Store(s::Mul(v3, v_beta), d, y + i + 3 * lanes);
    }

    // Cleanup with single SIMD
    for (; i + lanes <= n; i += lanes) {
        auto v = s::Load(d, y + i);
        s::Store(s::Mul(v, v_beta), d, y + i);
    }

    // Scalar cleanup
    for (; i < n; ++i) {
        y[i] *= beta;
    }
}

// =============================================================================
// SIMD AXPY: y += alpha * x
// =============================================================================

template <typename T>
SCL_HOT SCL_FORCE_INLINE void axpy(T alpha, const T* SCL_RESTRICT x, T* SCL_RESTRICT y, Size n) noexcept {
    if (SCL_UNLIKELY(alpha == T(0))) return;

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);
    const auto v_alpha = s::Set(d, alpha);

    Size i = 0;

    // 2-way unroll with FMA
    for (; i + 2 * lanes <= n; i += 2 * lanes) {
        auto vx0 = s::Load(d, x + i);
        auto vx1 = s::Load(d, x + i + lanes);
        auto vy0 = s::Load(d, y + i);
        auto vy1 = s::Load(d, y + i + lanes);

        s::Store(s::MulAdd(v_alpha, vx0, vy0), d, y + i);
        s::Store(s::MulAdd(v_alpha, vx1, vy1), d, y + i + lanes);
    }

    for (; i + lanes <= n; i += lanes) {
        auto vx = s::Load(d, x + i);
        auto vy = s::Load(d, y + i);
        s::Store(s::MulAdd(v_alpha, vx, vy), d, y + i);
    }

    for (; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

// =============================================================================
// Horizontal Sum Helpers
// =============================================================================

template <typename T>
SCL_FORCE_INLINE T horizontal_sum_4(T s0, T s1, T s2, T s3) noexcept {
    return (s0 + s1) + (s2 + s3);
}

template <typename T>
SCL_FORCE_INLINE T horizontal_sum_8(T s0, T s1, T s2, T s3, T s4, T s5, T s6, T s7) noexcept {
    return ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
}

// =============================================================================
// Check if Indices are Consecutive
// =============================================================================

SCL_FORCE_INLINE bool is_consecutive(const Index* indices, Size nnz) noexcept {
    if (SCL_UNLIKELY(nnz < 2)) return true;

    // Only check for small-to-medium arrays (cost amortized)
    if (SCL_UNLIKELY(nnz > config::CONSECUTIVE_CHECK_THRESHOLD)) return false;

    Index expected = indices[0];
    
    // 4-way unrolled check for small arrays
    Size k = 0;
    for (; k + 4 <= nnz; k += 4) {
        Index base = expected + static_cast<Index>(k);
        if (SCL_UNLIKELY(indices[k + 0] != base + 0 ||
                         indices[k + 1] != base + 1 ||
                         indices[k + 2] != base + 2 ||
                         indices[k + 3] != base + 3)) {
            return false;
        }
    }

    // Scalar cleanup
    for (; k < nnz; ++k) {
        if (SCL_UNLIKELY(indices[k] != expected + static_cast<Index>(k))) {
            return false;
        }
    }
    return true;
}

// =============================================================================
// Dense Dot Product (for consecutive indices)
// =============================================================================

template <typename T>
SCL_HOT SCL_FORCE_INLINE T dense_dot(
    const T* SCL_RESTRICT a,
    const T* SCL_RESTRICT b,
    Size n
) noexcept {
    if (SCL_UNLIKELY(n == 0)) return T(0);

    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<T>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);
    auto v_sum2 = s::Zero(d);
    auto v_sum3 = s::Zero(d);

    Size k = 0;
    for (; k + 4 * lanes <= n; k += 4 * lanes) {
        v_sum0 = s::MulAdd(s::Load(d, a + k), s::Load(d, b + k), v_sum0);
        v_sum1 = s::MulAdd(s::Load(d, a + k + lanes), s::Load(d, b + k + lanes), v_sum1);
        v_sum2 = s::MulAdd(s::Load(d, a + k + 2*lanes), s::Load(d, b + k + 2*lanes), v_sum2);
        v_sum3 = s::MulAdd(s::Load(d, a + k + 3*lanes), s::Load(d, b + k + 3*lanes), v_sum3);
    }

    v_sum0 = s::Add(s::Add(v_sum0, v_sum1), s::Add(v_sum2, v_sum3));

    for (; k + lanes <= n; k += lanes) {
        v_sum0 = s::MulAdd(s::Load(d, a + k), s::Load(d, b + k), v_sum0);
    }

    T sum = s::GetLane(s::SumOfLanes(d, v_sum0));

    for (; k < n; ++k) {
        sum += a[k] * b[k];
    }

    return sum;
}

// =============================================================================
// Sparse Dot Products (Tiered)
// =============================================================================

// Short row (nnz < 8): scalar
template <typename T>
SCL_FORCE_INLINE T sparse_dot_short(
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Size nnz,
    const T* SCL_RESTRICT x
) noexcept {
    T sum = T(0);
    // Unroll small loops for better ILP
    Size k = 0;
    for (; k + 2 <= nnz; k += 2) {
        sum += values[k] * x[indices[k]] + values[k+1] * x[indices[k+1]];
    }
    for (; k < nnz; ++k) {
        sum += values[k] * x[indices[k]];
    }
    return sum;
}

// Medium row (8-64): 4-way unroll
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
        // Prefetch ahead for better cache behavior
        if (SCL_LIKELY(k + 8 < nnz)) {
            SCL_PREFETCH_READ(&values[k + 8], 0);
            SCL_PREFETCH_READ(&indices[k + 8], 0);
        }

        Index idx0 = indices[k + 0], idx1 = indices[k + 1];
        Index idx2 = indices[k + 2], idx3 = indices[k + 3];
        T val0 = values[k + 0], val1 = values[k + 1];
        T val2 = values[k + 2], val3 = values[k + 3];

        sum0 += val0 * x[idx0];
        sum1 += val1 * x[idx1];
        sum2 += val2 * x[idx2];
        sum3 += val3 * x[idx3];
    }

    T sum = horizontal_sum_4(sum0, sum1, sum2, sum3);
    for (; k < nnz; ++k) {
        sum += values[k] * x[indices[k]];
    }
    return sum;
}

// Long row (64-256): 8-way unroll + prefetch
template <typename T>
SCL_FORCE_INLINE SCL_HOT T sparse_dot_long(
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Size nnz,
    const T* SCL_RESTRICT x
) noexcept {
    T sum0 = T(0), sum1 = T(0), sum2 = T(0), sum3 = T(0);
    T sum4 = T(0), sum5 = T(0), sum6 = T(0), sum7 = T(0);

    Size k = 0;
    for (; k + 8 <= nnz; k += 8) {
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < nnz)) {
            SCL_PREFETCH_READ(&values[k + config::PREFETCH_DISTANCE], 0);
            SCL_PREFETCH_READ(&indices[k + config::PREFETCH_DISTANCE], 0);
        }

        // Extract to reduce memory pressure
        Index idx0 = indices[k + 0], idx1 = indices[k + 1];
        Index idx2 = indices[k + 2], idx3 = indices[k + 3];
        Index idx4 = indices[k + 4], idx5 = indices[k + 5];
        Index idx6 = indices[k + 6], idx7 = indices[k + 7];
        T val0 = values[k + 0], val1 = values[k + 1];
        T val2 = values[k + 2], val3 = values[k + 3];
        T val4 = values[k + 4], val5 = values[k + 5];
        T val6 = values[k + 6], val7 = values[k + 7];

        sum0 += val0 * x[idx0];
        sum1 += val1 * x[idx1];
        sum2 += val2 * x[idx2];
        sum3 += val3 * x[idx3];
        sum4 += val4 * x[idx4];
        sum5 += val5 * x[idx5];
        sum6 += val6 * x[idx6];
        sum7 += val7 * x[idx7];
    }

    T sum = horizontal_sum_8(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7);
    for (; k < nnz; ++k) {
        sum += values[k] * x[indices[k]];
    }
    return sum;
}

// Very long row (>= 256): 8-way + aggressive prefetch + x vector prefetch
template <typename T>
SCL_HOT SCL_FORCE_INLINE T sparse_dot_very_long(
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Size nnz,
    const T* SCL_RESTRICT x
) noexcept {
    T sum0 = T(0), sum1 = T(0), sum2 = T(0), sum3 = T(0);
    T sum4 = T(0), sum5 = T(0), sum6 = T(0), sum7 = T(0);

    Size k = 0;
    for (; k + 8 <= nnz; k += 8) {
        // Prefetch indices and values ahead
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < nnz)) {
            SCL_PREFETCH_READ(&values[k + config::PREFETCH_DISTANCE], 0);
            SCL_PREFETCH_READ(&indices[k + config::PREFETCH_DISTANCE], 0);
        }

        // Prefetch x vector at future indices (reduce cache misses)
        if (SCL_LIKELY(k + 16 < nnz)) {
            SCL_PREFETCH_READ(&x[indices[k + 8]], 0);
            SCL_PREFETCH_READ(&x[indices[k + 12]], 0);
        }

        // Extract indices and values for better register allocation
        Index idx0 = indices[k + 0], idx1 = indices[k + 1];
        Index idx2 = indices[k + 2], idx3 = indices[k + 3];
        Index idx4 = indices[k + 4], idx5 = indices[k + 5];
        Index idx6 = indices[k + 6], idx7 = indices[k + 7];
        T val0 = values[k + 0], val1 = values[k + 1];
        T val2 = values[k + 2], val3 = values[k + 3];
        T val4 = values[k + 4], val5 = values[k + 5];
        T val6 = values[k + 6], val7 = values[k + 7];

        sum0 += val0 * x[idx0];
        sum1 += val1 * x[idx1];
        sum2 += val2 * x[idx2];
        sum3 += val3 * x[idx3];
        sum4 += val4 * x[idx4];
        sum5 += val5 * x[idx5];
        sum6 += val6 * x[idx6];
        sum7 += val7 * x[idx7];
    }

    T sum = horizontal_sum_8(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7);
    for (; k < nnz; ++k) {
        sum += values[k] * x[indices[k]];
    }
    return sum;
}

// Adaptive dispatcher with consecutive detection
template <typename T>
SCL_FORCE_INLINE T sparse_dot_adaptive(
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Size nnz,
    const T* SCL_RESTRICT x
) noexcept {
    if (SCL_UNLIKELY(nnz == 0)) return T(0);

    // Check for consecutive indices (enables dense SIMD path)
    if (nnz <= config::CONSECUTIVE_CHECK_THRESHOLD && is_consecutive(indices, nnz)) {
        return dense_dot(values, x + indices[0], nnz);
    }

    if (nnz < config::SHORT_ROW_THRESHOLD) {
        return sparse_dot_short(indices, values, nnz, x);
    } else if (nnz < config::MEDIUM_ROW_THRESHOLD) {
        return sparse_dot_medium(indices, values, nnz, x);
    } else if (nnz < config::LONG_ROW_THRESHOLD) {
        return sparse_dot_long(indices, values, nnz, x);
    } else {
        return sparse_dot_very_long(indices, values, nnz, x);
    }
}

} // namespace detail

// =============================================================================
// SpMV: y = alpha * A * x + beta * y
// =============================================================================

template <typename T, bool IsCSR>
void spmv(
    const Sparse<T, IsCSR>& A,
    Array<const T> x,
    Array<T> y,
    T alpha = T(1),
    T beta = T(0)
) {
    const Index primary_dim = A.primary_dim();
    const Size N = static_cast<Size>(primary_dim);

    SCL_CHECK_DIM(y.len >= N, "SpMV: output vector too small");

    // Beta scaling
    detail::scale_output(y.ptr, N, beta);

    if (SCL_UNLIKELY(alpha == T(0))) return;

    // Parallel SpMV
    scl::threading::parallel_for(Size(0), N, [&](size_t p) {
        const Index primary_idx = static_cast<Index>(p);
        const Size len = static_cast<Size>(A.primary_length(primary_idx));

        if (SCL_UNLIKELY(len == 0)) return;

        auto vals = A.primary_values(primary_idx);
        auto inds = A.primary_indices(primary_idx);

        T dot = detail::sparse_dot_adaptive(inds.ptr, vals.ptr, len, x.ptr);
        y[p] += alpha * dot;
    });
}

// =============================================================================
// SpTSpV: y = alpha * A^T * x + beta * y (Transposed SpMV)
// =============================================================================

template <typename T, bool IsCSR>
void spmv_transpose(
    const Sparse<T, IsCSR>& A,
    Array<const T> x,
    Array<T> y,
    T alpha = T(1),
    T beta = T(0)
) {
    const Index primary_dim = A.primary_dim();
    const Index secondary_dim = A.secondary_dim();
    const Size M = static_cast<Size>(primary_dim);
    const Size N = static_cast<Size>(secondary_dim);

    SCL_CHECK_DIM(y.len >= N, "SpMV^T: output vector too small");

    // Beta scaling
    detail::scale_output(y.ptr, N, beta);

    if (SCL_UNLIKELY(alpha == T(0))) return;

    // Use atomic accumulation for parallel safety
    // Allocate raw aligned memory for placement new of atomic objects
    // Note: std::atomic<T> may not be trivially constructible, so we use Byte allocation
    Byte* raw_mem = scl::memory::aligned_alloc<Byte>(sizeof(std::atomic<T>) * N, SCL_ALIGNMENT);
    
    SCL_ASSERT(raw_mem != nullptr, "spmv_transpose: Failed to allocate atomic array");
    
    std::atomic<T>* atomic_y = reinterpret_cast<std::atomic<T>*>(raw_mem);

    // Initialize atomic array from y using placement new
    for (Size i = 0; i < N; ++i) {
        new (&atomic_y[i]) std::atomic<T>(y[i]);
    }

    scl::threading::parallel_for(Size(0), M, [&](size_t p) {
        const Index primary_idx = static_cast<Index>(p);
        const Size len = static_cast<Size>(A.primary_length(primary_idx));

        if (SCL_UNLIKELY(len == 0)) return;

        auto vals = A.primary_values(primary_idx);
        auto inds = A.primary_indices(primary_idx);

        T x_p = x[p] * alpha;

        // 4-way unrolled accumulation with prefetch
        Size k = 0;
        for (; k + 4 <= len; k += 4) {
            // Prefetch ahead
            if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < len)) {
                SCL_PREFETCH_READ(&vals[k + config::PREFETCH_DISTANCE], 0);
                SCL_PREFETCH_READ(&inds[k + config::PREFETCH_DISTANCE], 0);
            }

            Index col0 = inds[k+0], col1 = inds[k+1], col2 = inds[k+2], col3 = inds[k+3];
            T val0 = vals[k+0], val1 = vals[k+1], val2 = vals[k+2], val3 = vals[k+3];

            atomic_y[col0].fetch_add(val0 * x_p, std::memory_order_relaxed);
            atomic_y[col1].fetch_add(val1 * x_p, std::memory_order_relaxed);
            atomic_y[col2].fetch_add(val2 * x_p, std::memory_order_relaxed);
            atomic_y[col3].fetch_add(val3 * x_p, std::memory_order_relaxed);
        }

        // Scalar cleanup
        for (; k < len; ++k) {
            Index col = inds[k];
            atomic_y[col].fetch_add(vals[k] * x_p, std::memory_order_relaxed);
        }
    });

    // Convert back and destroy atomic objects
    for (Size i = 0; i < N; ++i) {
        y[i] = atomic_y[i].load(std::memory_order_relaxed);
        atomic_y[i].~atomic();
    }

    // Free aligned memory using scl::memory
    scl::memory::aligned_free(raw_mem, SCL_ALIGNMENT);
}

// =============================================================================
// SpMM: Y = alpha * A * X + beta * Y (Sparse-Dense Matrix Multiply)
// =============================================================================

template <typename T, bool IsCSR>
void spmm(
    const Sparse<T, IsCSR>& A,
    const T* X,           // secondary_dim x n_cols, row-major
    Index n_cols,
    T* Y,                 // primary_dim x n_cols, row-major
    T alpha = T(1),
    T beta = T(0)
) {
    const Index primary_dim = A.primary_dim();
    const Size M = static_cast<Size>(primary_dim);
    const Size K = static_cast<Size>(n_cols);

    // Beta scaling - parallelize for large matrices
    if (static_cast<Size>(M) >= config::PARALLEL_THRESHOLD) {
        scl::threading::parallel_for(Size(0), M, [&](size_t i) {
            detail::scale_output(Y + i * K, K, beta);
        });
    } else {
        for (Size i = 0; i < M; ++i) {
            detail::scale_output(Y + i * K, K, beta);
        }
    }

    if (SCL_UNLIKELY(alpha == T(0))) return;

    // Block-tiled SpMM for cache efficiency
    const Size BLOCK_COLS = config::SPMM_BLOCK_COLS;
    for (Size col_start = 0; col_start < K; col_start += BLOCK_COLS) {
        Size col_end = scl::algo::min2(col_start + BLOCK_COLS, K);
        Size block_cols = col_end - col_start;

        scl::threading::parallel_for(Size(0), M, [&](size_t p) {
            const Index primary_idx = static_cast<Index>(p);
            const Size len = static_cast<Size>(A.primary_length(primary_idx));

            if (SCL_UNLIKELY(len == 0)) return;

            auto vals = A.primary_values(primary_idx);
            auto inds = A.primary_indices(primary_idx);

            T* y_row = Y + p * K + col_start;

            for (Size k = 0; k < len; ++k) {
                // Prefetch ahead for next iteration
                if (SCL_LIKELY(k + 1 < len)) {
                    Index next_col = inds[k + 1];
                    SCL_PREFETCH_READ(X + static_cast<Size>(next_col) * K + col_start, 0);
                }

                Index col_idx = inds[k];
                T val = vals[k] * alpha;
                const T* x_row = X + static_cast<Size>(col_idx) * K + col_start;

                // SIMD accumulation
                detail::axpy(val, x_row, y_row, block_cols);
            }
        });
    }
}

// =============================================================================
// Fused SpMV: y = alpha * A * x + beta * A * z + gamma * y
// =============================================================================

template <typename T, bool IsCSR>
void spmv_fused_linear(
    const Sparse<T, IsCSR>& A,
    Array<const T> x,
    Array<const T> z,
    Array<T> y,
    T alpha,
    T beta,
    T gamma
) {
    const Index primary_dim = A.primary_dim();
    const Size N = static_cast<Size>(primary_dim);

    SCL_CHECK_DIM(y.len >= N, "Fused SpMV: output vector too small");
    SCL_CHECK_DIM(x.len >= static_cast<Size>(A.secondary_dim()), "Fused SpMV: x too small");
    SCL_CHECK_DIM(z.len >= static_cast<Size>(A.secondary_dim()), "Fused SpMV: z too small");

    // Gamma scaling
    detail::scale_output(y.ptr, N, gamma);

    if (SCL_UNLIKELY(alpha == T(0) && beta == T(0))) return;

    scl::threading::parallel_for(Size(0), N, [&](size_t p) {
        const Index primary_idx = static_cast<Index>(p);
        const Size len = static_cast<Size>(A.primary_length(primary_idx));

        if (SCL_UNLIKELY(len == 0)) return;

        auto vals = A.primary_values(primary_idx);
        auto inds = A.primary_indices(primary_idx);

        T dot_x = T(0), dot_z = T(0);

        // Fused computation with 4-way unroll + prefetch
        Size k = 0;
        for (; k + 4 <= len; k += 4) {
            // Prefetch ahead for indirect access
            if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < len)) {
                SCL_PREFETCH_READ(&vals[k + config::PREFETCH_DISTANCE], 0);
                SCL_PREFETCH_READ(&inds[k + config::PREFETCH_DISTANCE], 0);
            }

            Index idx0 = inds[k+0], idx1 = inds[k+1], idx2 = inds[k+2], idx3 = inds[k+3];
            T val0 = vals[k+0], val1 = vals[k+1], val2 = vals[k+2], val3 = vals[k+3];

            dot_x += val0 * x[idx0] + val1 * x[idx1] + val2 * x[idx2] + val3 * x[idx3];
            dot_z += val0 * z[idx0] + val1 * z[idx1] + val2 * z[idx2] + val3 * z[idx3];
        }

        // Scalar cleanup
        for (; k < len; ++k) {
            Index idx = inds[k];
            T val = vals[k];
            dot_x += val * x[idx];
            dot_z += val * z[idx];
        }

        y[p] += alpha * dot_x + beta * dot_z;
    });
}

// =============================================================================
// Row Norms: Compute L2 norm of each row
// =============================================================================

template <typename T, bool IsCSR>
void row_norms(
    const Sparse<T, IsCSR>& A,
    Array<T> norms
) {
    const Index primary_dim = A.primary_dim();
    const Size N = static_cast<Size>(primary_dim);

    SCL_CHECK_DIM(norms.len >= N, "row_norms: output buffer too small");

    scl::threading::parallel_for(Size(0), N, [&](size_t p) {
        const Index primary_idx = static_cast<Index>(p);
        const Size len = static_cast<Size>(A.primary_length(primary_idx));

        if (SCL_UNLIKELY(len == 0)) {
            norms[p] = T(0);
            return;
        }

        auto vals = A.primary_values(primary_idx);

        // Use vectorize::sum_squared for SIMD-optimized sum of squares
        T sum_sq = scl::vectorize::sum_squared(Array<const T>(vals.ptr, len));
        norms[p] = std::sqrt(sum_sq);
    });
}

// =============================================================================
// Row Sums
// =============================================================================

template <typename T, bool IsCSR>
void row_sums(
    const Sparse<T, IsCSR>& A,
    Array<T> sums
) {
    const Index primary_dim = A.primary_dim();
    const Size N = static_cast<Size>(primary_dim);

    SCL_CHECK_DIM(sums.len >= N, "row_sums: output buffer too small");

    scl::threading::parallel_for(Size(0), N, [&](size_t p) {
        const Index primary_idx = static_cast<Index>(p);
        const Size len = static_cast<Size>(A.primary_length(primary_idx));

        if (SCL_UNLIKELY(len == 0)) {
            sums[p] = T(0);
            return;
        }

        auto vals = A.primary_values(primary_idx);

        // Use vectorize::sum for SIMD-optimized sum
        sums[p] = scl::vectorize::sum(Array<const T>(vals.ptr, len));
    });
}

// =============================================================================
// Diagonal Extraction
// =============================================================================

template <typename T, bool IsCSR>
void extract_diagonal(
    const Sparse<T, IsCSR>& A,
    Array<T> diag
) {
    const Index n = scl::algo::min2(A.rows(), A.cols());
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(diag.len >= N, "extract_diagonal: output buffer too small");

    scl::algo::zero(diag.ptr, N);

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        const Index idx = static_cast<Index>(i);
        const Size len = static_cast<Size>(A.primary_length(idx));

        if (SCL_UNLIKELY(len == 0)) return;

        auto vals = A.primary_values(idx);
        auto inds = A.primary_indices(idx);

        // Use algo::lower_bound for binary search
        const Index* pos = scl::algo::lower_bound(inds.ptr, inds.ptr + len, idx);

        if (SCL_LIKELY(pos < inds.ptr + len && *pos == idx)) {
            Size offset = static_cast<Size>(pos - inds.ptr);
            diag[i] = vals[offset];
        }
    });
}

// =============================================================================
// Scale Rows: A_i = diag[i] * A_i
// =============================================================================

template <typename T, bool IsCSR>
void scale_rows(
    Sparse<T, IsCSR>& A,
    Array<const T> scale_factors
) {
    const Index primary_dim = A.primary_dim();
    const Size N = static_cast<Size>(primary_dim);

    SCL_CHECK_DIM(scale_factors.len >= N, "scale_rows: scale_factors too small");

    scl::threading::parallel_for(Size(0), N, [&](size_t p) {
        const Index primary_idx = static_cast<Index>(p);
        const Size len = static_cast<Size>(A.primary_length(primary_idx));

        if (SCL_UNLIKELY(len == 0)) return;

        T* vals = const_cast<T*>(A.primary_values(primary_idx).ptr);
        T s = scale_factors[p];

        if (SCL_UNLIKELY(s == T(0))) {
            scl::memory::zero(Array<T>(vals, len));
            return;
        }
        if (SCL_LIKELY(s == T(1))) return;

        // Use memory::scale or SIMD scale
        detail::scale_output(vals, len, s);
    });
}

// =============================================================================
// Convenience Wrappers
// =============================================================================

// y = A * x
template <typename T, bool IsCSR>
void spmv_simple(
    const Sparse<T, IsCSR>& A,
    Array<const T> x,
    Array<T> y
) {
    spmv(A, x, y, T(1), T(0));
}

// y += A * x
template <typename T, bool IsCSR>
void spmv_add(
    const Sparse<T, IsCSR>& A,
    Array<const T> x,
    Array<T> y
) {
    spmv(A, x, y, T(1), T(1));
}

// y = alpha * A * x
template <typename T, bool IsCSR>
void spmv_scaled(
    const Sparse<T, IsCSR>& A,
    Array<const T> x,
    Array<T> y,
    T alpha
) {
    spmv(A, x, y, alpha, T(0));
}

// y = A^T * x
template <typename T, bool IsCSR>
void spmv_transpose_simple(
    const Sparse<T, IsCSR>& A,
    Array<const T> x,
    Array<T> y
) {
    spmv_transpose(A, x, y, T(1), T(0));
}

} // namespace scl::kernel::algebra
