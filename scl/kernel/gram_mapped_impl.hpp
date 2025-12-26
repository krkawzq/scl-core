#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

// =============================================================================
/// @file gram_mapped_impl.hpp
/// @brief Mapped Backend Gram Matrix Operations
///
/// Gram matrix computation is read-only, can stream directly from mapped data.
/// No materialization needed.
///
/// Key optimizations:
/// - Chunk-based processing for cache efficiency
/// - Optimized sparse dot product with prefetch
// =============================================================================

namespace scl::kernel::gram::mapped {

namespace detail {

constexpr size_t PREFETCH_DISTANCE = 32;

/// @brief Ultra-fast linear merge dot product
template <typename T>
SCL_FORCE_INLINE void dot_linear_ultra(
    const Index* SCL_RESTRICT idx1, const T* SCL_RESTRICT val1, Size n1,
    const Index* SCL_RESTRICT idx2, const T* SCL_RESTRICT val2, Size n2,
    T& out_dot
) {
    T sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    Size i = 0, j = 0;
    Size matches = 0;

    while (i < n1 && j < n2) {
        if (i + PREFETCH_DISTANCE < n1) {
            SCL_PREFETCH_READ(&idx1[i + PREFETCH_DISTANCE], 0);
        }
        if (j + PREFETCH_DISTANCE < n2) {
            SCL_PREFETCH_READ(&idx2[j + PREFETCH_DISTANCE], 0);
        }

        Index r1 = idx1[i];
        Index r2 = idx2[j];

        if (SCL_LIKELY(r1 == r2)) {
            T prod = val1[i] * val2[j];
            switch (matches % 4) {
                case 0: sum0 += prod; break;
                case 1: sum1 += prod; break;
                case 2: sum2 += prod; break;
                case 3: sum3 += prod; break;
            }
            matches++;
            ++i; ++j;
        } else if (r1 < r2) {
            ++i;
        } else {
            ++j;
        }
    }

    out_dot = (sum0 + sum1) + (sum2 + sum3);
}

} // namespace detail

// =============================================================================
// MappedCustomSparse Gram Matrix
// =============================================================================

/// @brief Compute Gram matrix for MappedCustomSparse
///
/// Streaming algorithm with chunk-based processing.
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void gram_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index N = scl::primary_size(matrix);
    const Size N_size = static_cast<Size>(N);

    SCL_CHECK_DIM(output.len == N_size * N_size, "Gram: Output size mismatch");

    // Prefetch hint
    kernel::mapped::hint_prefetch(matrix);

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    // Process in chunks for cache efficiency
    constexpr Size CHUNK_SIZE = 64;  // Smaller chunks for Gram (O(N^2) accesses)
    const Size n_chunks = (N_size + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Size chunk_start = chunk_id * CHUNK_SIZE;
        Size chunk_end = std::min(chunk_start + CHUNK_SIZE, N_size);

        scl::threading::parallel_for(chunk_start, chunk_end, [&](size_t i) {
            auto values_i = scl::primary_values(matrix, static_cast<Index>(i));
            auto indices_i = scl::primary_indices(matrix, static_cast<Index>(i));
            Index len_i = static_cast<Index>(values_i.len);

            const Index* SCL_RESTRICT idx_i = indices_i.ptr;
            const T* SCL_RESTRICT val_i = values_i.ptr;

            T* row_ptr = output.ptr + (i * N_size);

            // Diagonal: SIMD self dot product
            auto v_sum = s::Zero(d);
            Index k = 0;

            for (; k + lanes <= len_i; k += lanes) {
                auto v = s::Load(d, val_i + k);
                v_sum = s::MulAdd(v, v, v_sum);
            }

            T self_dot = s::GetLane(s::SumOfLanes(d, v_sum));
            for (; k < len_i; ++k) {
                self_dot += val_i[k] * val_i[k];
            }
            row_ptr[i] = self_dot;

            // Upper triangle with optimized dot product
            for (Size j = i + 1; j < N_size; ++j) {
                auto values_j = scl::primary_values(matrix, static_cast<Index>(j));
                auto indices_j = scl::primary_indices(matrix, static_cast<Index>(j));
                Index len_j = static_cast<Index>(values_j.len);

                const Index* SCL_RESTRICT idx_j = indices_j.ptr;
                const T* SCL_RESTRICT val_j = values_j.ptr;

                T dot;
                detail::dot_linear_ultra(
                    idx_i, val_i, static_cast<Size>(len_i),
                    idx_j, val_j, static_cast<Size>(len_j),
                    dot
                );

                row_ptr[j] = dot;
                output.ptr[j * N_size + i] = dot;  // Mirror
            }
        });
    }
}

// =============================================================================
// MappedVirtualSparse Gram Matrix
// =============================================================================

/// @brief Compute Gram matrix for MappedVirtualSparse
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void gram_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& matrix,
    Array<T> output
) {
    const Index N = scl::primary_size(matrix);
    const Size N_size = static_cast<Size>(N);

    SCL_CHECK_DIM(output.len == N_size * N_size, "Gram: Output size mismatch");

    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    constexpr Size CHUNK_SIZE = 64;
    const Size n_chunks = (N_size + CHUNK_SIZE - 1) / CHUNK_SIZE;

    for (Size chunk_id = 0; chunk_id < n_chunks; ++chunk_id) {
        Size chunk_start = chunk_id * CHUNK_SIZE;
        Size chunk_end = std::min(chunk_start + CHUNK_SIZE, N_size);

        scl::threading::parallel_for(chunk_start, chunk_end, [&](size_t i) {
            auto values_i = scl::primary_values(matrix, static_cast<Index>(i));
            auto indices_i = scl::primary_indices(matrix, static_cast<Index>(i));
            Index len_i = static_cast<Index>(values_i.len);

            const Index* SCL_RESTRICT idx_i = indices_i.ptr;
            const T* SCL_RESTRICT val_i = values_i.ptr;

            T* row_ptr = output.ptr + (i * N_size);

            // Diagonal
            auto v_sum = s::Zero(d);
            Index k = 0;

            for (; k + lanes <= len_i; k += lanes) {
                auto v = s::Load(d, val_i + k);
                v_sum = s::MulAdd(v, v, v_sum);
            }

            T self_dot = s::GetLane(s::SumOfLanes(d, v_sum));
            for (; k < len_i; ++k) {
                self_dot += val_i[k] * val_i[k];
            }
            row_ptr[i] = self_dot;

            // Upper triangle
            for (Size j = i + 1; j < N_size; ++j) {
                auto values_j = scl::primary_values(matrix, static_cast<Index>(j));
                auto indices_j = scl::primary_indices(matrix, static_cast<Index>(j));
                Index len_j = static_cast<Index>(values_j.len);

                const Index* SCL_RESTRICT idx_j = indices_j.ptr;
                const T* SCL_RESTRICT val_j = values_j.ptr;

                T dot;
                detail::dot_linear_ultra(
                    idx_i, val_i, static_cast<Size>(len_i),
                    idx_j, val_j, static_cast<Size>(len_j),
                    dot
                );

                row_ptr[j] = dot;
                output.ptr[j * N_size + i] = dot;
            }
        });
    }
}

// =============================================================================
// Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate mapped fast path
template <typename MatrixT, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<MatrixT, IsCSR>
SCL_FORCE_INLINE void gram_mapped_dispatch(
    const MatrixT& matrix,
    Array<typename MatrixT::ValueType> output
) {
    gram_mapped(matrix, output);
}

} // namespace scl::kernel::gram::mapped
