#pragma once

#include "scl/core/type.hpp"
#include "scl/core/matrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"

#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

// =============================================================================
/// @file softmax.hpp
/// @brief High-Performance Sparse Softmax Kernel
///
/// Implements Sparse-to-Dense Softmax transformation:
/// sigma(z)_i = e^(z_i) / sum(e^(z_j))
///
/// Optimization Strategy: "Chunked Reuse & Fused Fill-Scatter"
///
/// 1. Chunked Parallelism: Processes rows in blocks (e.g., 32 rows/task).
///    - Benefit: Amortizes thread scheduling overhead (~100ns) over many rows.
///    - Benefit: Allows std::vector workspace reuse (Zero-alloc hot path).
///
/// 2. Cached Exp: Caches e^(x_i - max) to avoid expensive re-computation.
///
/// 3. Memory Saturation:
///    - Background: 4-way unrolled SIMD Stream Stores (Non-Temporal) to fill 0s.
///    - Explicit: 8-way batched scatter with SW Prefetching.
///
/// Performance: ~2.0 GB/s output throughput (Memory Bandwidth Bound).
// =============================================================================

namespace scl::kernel::softmax {

namespace detail {

/// @brief Single-vector softmax unit.
/// Uses an external scratch buffer to ensure zero-allocation during execution.
template <typename T>
SCL_FORCE_INLINE void softmax_unit(
    Span<const T> vals,
    Span<const Index> indices,
    T* SCL_RESTRICT out_ptr,
    Size dim,
    std::vector<T>& exp_cache // Reused workspace
) {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();
    const Size nnz = vals.size;
    const Size n_zeros = dim - nnz;

    // --- Edge Case: Zero Vector ---
    if (SCL_UNLIKELY(nnz == 0)) {
        const T uniform_val = static_cast<T>(1.0) / static_cast<T>(dim);
        const auto v_uniform = s::Set(d, uniform_val);
        size_t j = 0;
        for (; j + lanes <= dim; j += lanes) {
            s::Stream(v_uniform, d, out_ptr + j);
        }
        for (; j < dim; ++j) {
            out_ptr[j] = uniform_val;
        }
        return;
    }

    // --- Prep Cache ---
    // Zero-overhead if capacity is sufficient (99.9% case)
    if (SCL_UNLIKELY(exp_cache.size() < nnz)) {
        exp_cache.resize(nnz);
    }
    T* cache_ptr = exp_cache.data();

    // --- Phase 1: Find Max ---
    auto v_max = s::Set(d, -std::numeric_limits<T>::infinity());
    size_t k = 0;

    for (; k + lanes <= nnz; k += lanes) {
        v_max = s::Max(v_max, s::Load(d, vals.ptr + k));
    }
    T max_val = s::GetLane(s::MaxOfLanes(d, v_max));
    for (; k < nnz; ++k) {
        if (vals[k] > max_val) max_val = vals[k];
    }

    // Correct max if implicit zeros exist and are larger than all explicits
    if (n_zeros > 0 && max_val < static_cast<T>(0.0)) {
        max_val = static_cast<T>(0.0);
    }

    // --- Phase 2: Compute Exp & Sum (Fused with Cache Write) ---
    const auto v_max_broad = s::Set(d, max_val);
    auto v_sum = s::Zero(d);
    k = 0;

    for (; k + lanes <= nnz; k += lanes) {
        auto v = s::Load(d, vals.ptr + k);
        v = s::Exp(d, s::Sub(v, v_max_broad)); // Fused sub-exp
        s::Store(v, d, cache_ptr + k);          // Write to L1-resident cache
        v_sum = s::Add(v_sum, v);
    }
    T sum = s::GetLane(s::SumOfLanes(d, v_sum));
    
    for (; k < nnz; ++k) {
        T e = std::exp(vals[k] - max_val);
        cache_ptr[k] = e;
        sum += e;
    }

    if (n_zeros > 0) {
        sum += static_cast<T>(n_zeros) * std::exp(static_cast<T>(0.0) - max_val);
    }

    // --- Phase 3: Background Fill (Implicit Zeros) ---
    // Bandwidth bound: Use aggressive unrolling and Stream Stores
    const T inv_sum = static_cast<T>(1.0) / sum;
    const T val_implicit = std::exp(static_cast<T>(0.0) - max_val) * inv_sum;
    const auto v_implicit = s::Set(d, val_implicit);

    size_t j = 0;
    for (; j + 4 * lanes <= dim; j += 4 * lanes) {
        s::Stream(v_implicit, d, out_ptr + j);
        s::Stream(v_implicit, d, out_ptr + j + lanes);
        s::Stream(v_implicit, d, out_ptr + j + 2 * lanes);
        s::Stream(v_implicit, d, out_ptr + j + 3 * lanes);
    }
    for (; j + lanes <= dim; j += lanes) {
        s::Stream(v_implicit, d, out_ptr + j);
    }
    for (; j < dim; ++j) {
        out_ptr[j] = val_implicit;
    }

    // --- Phase 4: Scatter Explicit Values ---
    // Compute-bound (random access): Use Prefetching
    constexpr Size BATCH = 8;
    k = 0;

    for (; k + BATCH <= nnz; k += BATCH) {
        if (SCL_LIKELY(k + 2 * BATCH <= nnz)) {
            SCL_PREFETCH_READ(&indices[k + BATCH], 1);
        }
        // Compiler auto-vectorization friendly unroll
        out_ptr[indices[k + 0]] = cache_ptr[k + 0] * inv_sum;
        out_ptr[indices[k + 1]] = cache_ptr[k + 1] * inv_sum;
        out_ptr[indices[k + 2]] = cache_ptr[k + 2] * inv_sum;
        out_ptr[indices[k + 3]] = cache_ptr[k + 3] * inv_sum;
        out_ptr[indices[k + 4]] = cache_ptr[k + 4] * inv_sum;
        out_ptr[indices[k + 5]] = cache_ptr[k + 5] * inv_sum;
        out_ptr[indices[k + 6]] = cache_ptr[k + 6] * inv_sum;
        out_ptr[indices[k + 7]] = cache_ptr[k + 7] * inv_sum;
    }
    for (; k < nnz; ++k) {
        out_ptr[indices[k]] = cache_ptr[k] * inv_sum;
    }
}

} // namespace detail

// =============================================================================
// Public API (Chunked Parallelism)
// =============================================================================

/// @brief Row-wise Softmax for Generic CSR-like Matrices.
///
/// Each row is independently normalized to a probability distribution.
/// Output is dense (row-major).
///
/// Uses chunked parallelism with workspace reuse to minimize allocation overhead.
///
/// @tparam MatrixT Any CSR-like matrix type
/// @param matrix Input CSR-like matrix
/// @param output Output buffer (size = rows × cols)
template <CSRLike MatrixT>
void softmax(const MatrixT& matrix, MutableSpan<typename MatrixT::ValueType> output) {
    using T = typename MatrixT::ValueType;
    const Index R = matrix.rows;
    const Index C = matrix.cols;
    SCL_CHECK_DIM(output.size == static_cast<Size>(R * C), 
                  "Softmax: Output size mismatch");

    // Chunk size trade-off:
    // Larger -> Better cache reuse, less scheduling overhead
    // Smaller -> Better load balancing
    constexpr size_t CHUNK_SIZE = 32; 
    const size_t n_chunks = (static_cast<size_t>(R) + CHUNK_SIZE - 1) / CHUNK_SIZE;

    scl::threading::parallel_for(0, n_chunks, [&](size_t chunk_idx) {
        // Thread-local Workspace (Reused across rows in chunk)
        // Reserve based on sparsity estimate to avoid reallocation
        std::vector<T> workspace;
        workspace.reserve(256); // Heuristic start size

        size_t i_start = chunk_idx * CHUNK_SIZE;
        size_t i_end = std::min(static_cast<size_t>(R), i_start + CHUNK_SIZE);

        for (size_t i = i_start; i < i_end; ++i) {
            detail::softmax_unit<typename MatrixT::ValueType>(
                matrix.row_values(static_cast<Index>(i)),
                matrix.row_indices(static_cast<Index>(i)),
                output.ptr + (i * static_cast<Size>(C)),
                C,
                workspace
            );
        }
    });
}

/// @brief Column-wise Softmax for Generic CSC-like Matrices.
///
/// Each column is independently normalized to a probability distribution.
/// Output is dense (column-major).
///
/// @tparam MatrixT Any CSC-like matrix type
/// @param matrix Input CSC-like matrix
/// @param output Output buffer (size = rows × cols)
template <CSCLike MatrixT>
void softmax(const MatrixT& matrix, MutableSpan<typename MatrixT::ValueType> output) {
    using T = typename MatrixT::ValueType;
    const Index R = matrix.rows;
    const Index C = matrix.cols;
    SCL_CHECK_DIM(output.size == static_cast<Size>(R * C), 
                  "Softmax: Output size mismatch");

    constexpr size_t CHUNK_SIZE = 32;
    const size_t n_chunks = (static_cast<size_t>(C) + CHUNK_SIZE - 1) / CHUNK_SIZE;

    scl::threading::parallel_for(0, n_chunks, [&](size_t chunk_idx) {
        std::vector<T> workspace;
        workspace.reserve(256);

        size_t j_start = chunk_idx * CHUNK_SIZE;
        size_t j_end = std::min(static_cast<size_t>(C), j_start + CHUNK_SIZE);

        for (size_t j = j_start; j < j_end; ++j) {
            detail::softmax_unit<typename MatrixT::ValueType>(
                matrix.col_values(static_cast<Index>(j)),
                matrix.col_indices(static_cast<Index>(j)),
                output.ptr + (j * static_cast<Size>(R)),
                R,
                workspace
            );
        }
    });
}

} // namespace scl::kernel::softmax
