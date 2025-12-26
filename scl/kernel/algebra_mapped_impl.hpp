#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
/// @file algebra_mapped_impl.hpp
/// @brief Mapped Backend SpMV Operations
///
/// SpMV (sparse matrix-vector multiply) is a read-only operation that can
/// stream directly from mapped data. No materialization needed.
///
/// Key optimizations:
/// - Chunk-based processing for cache efficiency
/// - 8-way unrolling with prefetch
/// - Streaming reads from mapped memory
// =============================================================================

namespace scl::kernel::algebra::mapped {

namespace detail {

constexpr size_t PREFETCH_DISTANCE = 64;

/// @brief Ultra-optimized sparse-dense dot (8-way unroll)
template <typename T>
SCL_FORCE_INLINE void sparse_dot_ultra(
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Size nnz,
    const T* SCL_RESTRICT x,
    T& out_dot
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

    out_dot = sum;
}

} // namespace detail

// =============================================================================
// MappedCustomSparse SpMV
// =============================================================================

/// @brief SpMV for MappedCustomSparse: y = alpha * A * x + beta * y
///
/// Streaming algorithm - reads matrix data once in sequential chunks.
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void spmv_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& A,
    Array<const T> x,
    Array<T> y,
    T alpha,
    T beta
) {
    const Index n_primary = scl::primary_size(A);

    SCL_CHECK_DIM(y.len >= static_cast<Size>(n_primary), "Output size mismatch");

    // Beta scaling
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

    // Prefetch hint
    kernel::mapped::hint_prefetch(A);

    // Process in chunks for cache efficiency
    constexpr Size CHUNK_SIZE = 256;
    const Size n_chunks = (n_primary + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Index chunk_start = static_cast<Index>(chunk_id * CHUNK_SIZE);
        Index chunk_end = std::min(chunk_start + static_cast<Index>(CHUNK_SIZE), n_primary);

        scl::threading::parallel_for(chunk_start, chunk_end, [&](Index p) {
            auto values = scl::primary_values(A, p);
            auto indices = scl::primary_indices(A, p);

            if (values.len == 0) return;

            T dot;
            detail::sparse_dot_ultra(
                indices.ptr,
                values.ptr,
                values.len,
                x.ptr,
                dot
            );

            y[p] += alpha * dot;
        });
    }
}

/// @brief SpMV with output allocation for MappedCustomSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
std::vector<T> spmv_alloc_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& A,
    Array<const T> x,
    T alpha = 1.0
) {
    const Index n_primary = scl::primary_size(A);
    std::vector<T> y(n_primary, static_cast<T>(0.0));

    spmv_mapped(A, x, Array<T>(y.data(), y.size()), alpha, static_cast<T>(0.0));

    return y;
}

// =============================================================================
// MappedVirtualSparse SpMV
// =============================================================================

/// @brief SpMV for MappedVirtualSparse: y = alpha * A * x + beta * y
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void spmv_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& A,
    Array<const T> x,
    Array<T> y,
    T alpha,
    T beta
) {
    const Index n_primary = scl::primary_size(A);

    SCL_CHECK_DIM(y.len >= static_cast<Size>(n_primary), "Output size mismatch");

    // Beta scaling
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

    // Process in chunks
    constexpr Size CHUNK_SIZE = 256;
    const Size n_chunks = (n_primary + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Index chunk_start = static_cast<Index>(chunk_id * CHUNK_SIZE);
        Index chunk_end = std::min(chunk_start + static_cast<Index>(CHUNK_SIZE), n_primary);

        scl::threading::parallel_for(chunk_start, chunk_end, [&](Index p) {
            auto values = scl::primary_values(A, p);
            auto indices = scl::primary_indices(A, p);

            if (values.len == 0) return;

            T dot;
            detail::sparse_dot_ultra(
                indices.ptr,
                values.ptr,
                values.len,
                x.ptr,
                dot
            );

            y[p] += alpha * dot;
        });
    }
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate mapped fast path
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void spmv_mapped_dispatch(
    const MatrixT& A,
    Array<const typename MatrixT::ValueType> x,
    Array<typename MatrixT::ValueType> y,
    typename MatrixT::ValueType alpha,
    typename MatrixT::ValueType beta
) {
    spmv_mapped(A, x, y, alpha, beta);
}

} // namespace scl::kernel::algebra::mapped
