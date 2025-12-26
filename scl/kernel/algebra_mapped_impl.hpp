#pragma once

#include "scl/kernel/mapped_common.hpp"
#include "scl/io/mmatrix.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/error.hpp"
#include "scl/threading/parallel_for.hpp"

#include <vector>
#include <numeric>
#include <algorithm>
#include <cstring>

// =============================================================================
/// @file algebra_mapped_impl.hpp
/// @brief Ultra High-Performance Mapped Backend SpMV Operations
///
/// Design Philosophy:
///
/// 1. ZERO ALLOCATION - All memory is pre-allocated by caller
/// 2. VOID RETURN - No ownership transfer, no lifetime conflicts
/// 3. MMAP-NATIVE - No extra cache layer (OS page cache is sufficient)
///
/// ## Key Optimizations
///
/// 1. Load-Balanced Parallelism
///    - Partition by nnz, not by rows
///    - Each thread processes similar work
///    - Handles extreme row-length variance
///
/// 2. Multi-Level Prefetch
///    - Prefetch values, indices, AND x vector
///    - Software pipelining with configurable distance
///    - Reduces memory stall cycles
///
/// 3. Adaptive Row Strategy
///    - Short rows (nnz < 8): scalar, no overhead
///    - Medium rows (8-64): 4-way unroll
///    - Long rows (>= 64): 8-way unroll + prefetch
///
/// 4. SIMD Beta Scaling
///    - Vectorized output scaling
///    - Special fast paths for beta=0, beta=1
///
/// ## Interface Contract
///
/// All functions follow the pattern:
///
///     void spmv_xxx(
///         const MatrixT& A,           // Input sparse matrix
///         Array<const T> x,           // Input vector (read-only)
///         Array<T> y,                 // Output vector (pre-allocated)
///         T alpha = 1,                // Scalar for A*x
///         T beta = 0                  // Scalar for y
///     );
///
/// Computes: y = alpha * A * x + beta * y
///
/// ## Performance Targets
///
/// - 10M rows, 30K cols, 1B nnz: target < 50ms per SpMV
/// - Memory bandwidth utilization: > 80%
/// - Load imbalance ratio: < 1.2x
// =============================================================================

namespace scl::kernel::algebra::mapped {

// =============================================================================
// SECTION 1: Configuration Constants
// =============================================================================

namespace config {

/// Prefetch distance for values/indices (elements ahead)
constexpr Size PREFETCH_DISTANCE = 64;

/// Prefetch distance for x vector random access (elements ahead)
constexpr Size PREFETCH_X_DISTANCE = 32;

/// Threshold: rows with nnz < this use scalar loop
constexpr Size SHORT_ROW_THRESHOLD = 8;

/// Threshold: rows with nnz < this use 4-way unroll
constexpr Size MEDIUM_ROW_THRESHOLD = 64;

/// Minimum nnz per thread for load balancing to be worthwhile
constexpr Size MIN_NNZ_PER_THREAD = 4096;

/// Serial threshold: matrices smaller than this use serial SpMV
constexpr Index SERIAL_ROW_THRESHOLD = 1000;

/// Balanced threshold: matrices smaller than this use simple parallel
constexpr Index BALANCED_ROW_THRESHOLD = 100000;

} // namespace config

// =============================================================================
// SECTION 2: SIMD Utilities
// =============================================================================

namespace detail {

/// @brief SIMD-optimized horizontal sum of 8 accumulators
///
/// Uses pairwise reduction to minimize dependency chains.
/// Final result has same precision as sequential sum.
template <typename T>
SCL_FORCE_INLINE T horizontal_sum_8(T s0, T s1, T s2, T s3, T s4, T s5, T s6, T s7) noexcept {
    T a = s0 + s1;
    T b = s2 + s3;
    T c = s4 + s5;
    T d = s6 + s7;
    T e = a + b;
    T f = c + d;
    return e + f;
}

/// @brief SIMD-optimized beta scaling: y = beta * y
///
/// Handles three cases with minimal branching:
/// - beta = 0: vectorized zero fill (most common)
/// - beta = 1: no-op (skip entirely)
/// - other: vectorized scaling
template <typename T>
void scale_output(T* SCL_RESTRICT y, Size len, T beta) noexcept {
    namespace s = scl::simd;
    const s::Tag d;
    const size_t lanes = s::lanes();

    if (beta == T(0)) {
        // Zero fill (common case for y = A*x)
        const auto v_zero = s::Zero(d);
        size_t i = 0;
        for (; i + lanes <= len; i += lanes) {
            s::Store(v_zero, d, y + i);
        }
        for (; i < len; ++i) {
            y[i] = T(0);
        }
    } else if (beta == T(1)) {
        // Identity (common case for y += A*x)
        return;
    } else {
        // General scaling
        const auto v_beta = s::Set(d, beta);
        size_t i = 0;
        for (; i + lanes <= len; i += lanes) {
            auto v = s::Load(d, y + i);
            s::Store(s::Mul(v, v_beta), d, y + i);
        }
        for (; i < len; ++i) {
            y[i] *= beta;
        }
    }
}

// =============================================================================
// SECTION 3: Adaptive Sparse Dot Products
// =============================================================================

/// @brief Short row dot product (nnz < 8): minimal overhead
///
/// Simple scalar loop, no unrolling. Best for very short rows where
/// loop overhead dominates.
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

/// @brief Medium row dot product (8 <= nnz < 64): 4-way unroll
///
/// 4-way unrolling breaks dependency chains, allowing ILP.
/// No prefetch (data likely already in cache).
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

    // Cleanup tail
    for (; k < nnz; ++k) {
        sum += values[k] * x[indices[k]];
    }

    return sum;
}

/// @brief Long row dot product (nnz >= 64): 8-way unroll + prefetch
///
/// Maximum ILP with 8 independent accumulators.
/// Software prefetch for values, indices, and x vector.
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

    // Main loop with prefetch
    for (; k + 8 <= nnz; k += 8) {
        // Prefetch values and indices ahead
        if (k + config::PREFETCH_DISTANCE < nnz) {
            SCL_PREFETCH_READ(&values[k + config::PREFETCH_DISTANCE], 0);
            SCL_PREFETCH_READ(&indices[k + config::PREFETCH_DISTANCE], 0);
        }

        // Prefetch x vector (random access)
        if (k + config::PREFETCH_X_DISTANCE < nnz) {
            SCL_PREFETCH_READ(&x[indices[k + config::PREFETCH_X_DISTANCE]], 0);
        }

        // 8-way unrolled computation
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

    // Cleanup tail
    for (; k < nnz; ++k) {
        sum += values[k] * x[indices[k]];
    }

    return sum;
}

/// @brief Adaptive dispatcher based on row length
///
/// Selects optimal strategy at runtime based on nnz.
/// Branch predictor will learn pattern for uniform matrices.
template <typename T>
SCL_FORCE_INLINE T sparse_dot_adaptive(
    const Index* SCL_RESTRICT indices,
    const T* SCL_RESTRICT values,
    Size nnz,
    const T* SCL_RESTRICT x
) noexcept {
    if (SCL_LIKELY(nnz < config::SHORT_ROW_THRESHOLD)) {
        return sparse_dot_short(indices, values, nnz, x);
    } else if (nnz < config::MEDIUM_ROW_THRESHOLD) {
        return sparse_dot_medium(indices, values, nnz, x);
    } else {
        return sparse_dot_long(indices, values, nnz, x);
    }
}

// =============================================================================
// SECTION 4: Load-Balanced Row Ranges
// =============================================================================

/// @brief Row range for load-balanced processing
struct RowRange {
    Index start;
    Index end;
};

/// @brief Compute load-balanced row ranges based on nnz distribution
///
/// Instead of dividing rows evenly (which causes load imbalance when
/// row lengths vary), we divide work (nnz) evenly.
///
/// Algorithm:
/// 1. Compute prefix sum of row lengths -> total nnz per prefix
/// 2. For each thread, binary search for row where cumulative nnz
///    reaches (thread_id + 1) * (total_nnz / num_threads)
///
/// Complexity: O(n_rows) for prefix sum, O(num_threads * log(n_rows)) for search
template <typename MatrixT>
std::vector<RowRange> compute_balanced_ranges(
    const MatrixT& A,
    Index n_rows,
    Size num_threads
) {
    std::vector<RowRange> ranges(num_threads);

    if (n_rows == 0 || num_threads == 0) {
        return ranges;
    }

    // Build prefix sum of row lengths
    std::vector<Index> prefix_nnz(static_cast<Size>(n_rows) + 1);
    prefix_nnz[0] = 0;

    for (Index i = 0; i < n_rows; ++i) {
        prefix_nnz[static_cast<Size>(i) + 1] =
            prefix_nnz[static_cast<Size>(i)] + scl::primary_length(A, i);
    }

    Index total_nnz = prefix_nnz.back();
    Index nnz_per_thread = (total_nnz + static_cast<Index>(num_threads) - 1)
                         / static_cast<Index>(num_threads);

    // Assign ranges via binary search
    Index current_row = 0;
    for (Size t = 0; t < num_threads; ++t) {
        ranges[t].start = current_row;

        Index target_nnz = static_cast<Index>(t + 1) * nnz_per_thread;
        target_nnz = std::min(target_nnz, total_nnz);

        // Binary search for end row
        auto it = std::upper_bound(
            prefix_nnz.begin() + current_row,
            prefix_nnz.end(),
            target_nnz
        );

        Index end_row = static_cast<Index>(it - prefix_nnz.begin());
        end_row = std::min(end_row, n_rows);

        ranges[t].end = end_row;
        current_row = end_row;
    }

    // Ensure last range covers all remaining rows
    ranges.back().end = n_rows;

    return ranges;
}

// =============================================================================
// SECTION 5: Core SpMV Kernels
// =============================================================================

/// @brief SpMV kernel for a contiguous range of rows
///
/// Processes rows [start_row, end_row) without any thread overhead.
/// Used as building block for both serial and parallel SpMV.
template <typename MatrixT, typename T>
SCL_FORCE_INLINE void spmv_kernel_range(
    const MatrixT& A,
    const T* SCL_RESTRICT x,
    T* SCL_RESTRICT y,
    T alpha,
    Index start_row,
    Index end_row
) noexcept {
    for (Index i = start_row; i < end_row; ++i) {
        auto values = scl::primary_values(A, i);
        auto indices = scl::primary_indices(A, i);

        if (values.len == 0) {
            continue;
        }

        T dot = sparse_dot_adaptive(
            indices.ptr,
            values.ptr,
            values.len,
            x
        );

        y[i] += alpha * dot;
    }
}

/// @brief Serial SpMV (for small matrices)
///
/// No thread overhead, optimal for matrices with < 1K rows.
template <typename MatrixT, typename T>
void spmv_serial(
    const MatrixT& A,
    const T* SCL_RESTRICT x,
    T* SCL_RESTRICT y,
    T alpha,
    Index n_rows
) noexcept {
    spmv_kernel_range(A, x, y, alpha, Index(0), n_rows);
}

/// @brief Simple row-parallel SpMV
///
/// Each thread processes a contiguous range of rows.
/// Good for uniform matrices where row lengths don't vary much.
template <typename MatrixT, typename T>
void spmv_parallel_simple(
    const MatrixT& A,
    const T* SCL_RESTRICT x,
    T* SCL_RESTRICT y,
    T alpha,
    Index n_rows
) {
    scl::threading::parallel_for(Index(0), n_rows, [&](Index i) {
        auto values = scl::primary_values(A, i);
        auto indices = scl::primary_indices(A, i);

        if (values.len == 0) return;

        T dot = sparse_dot_adaptive(
            indices.ptr,
            values.ptr,
            values.len,
            x
        );

        y[i] += alpha * dot;
    });
}

/// @brief Load-balanced parallel SpMV
///
/// Partitions work by nnz, not rows. Each thread processes
/// approximately equal total nnz, regardless of row length distribution.
///
/// Optimal for highly non-uniform matrices (power-law degree distribution).
template <typename MatrixT, typename T>
void spmv_parallel_balanced(
    const MatrixT& A,
    const T* SCL_RESTRICT x,
    T* SCL_RESTRICT y,
    T alpha,
    Index n_rows,
    Index total_nnz
) {
    // Estimate optimal thread count
    Size num_threads = std::max(Size(1),
        std::min(
            static_cast<Size>(scl::threading::Scheduler::get_num_threads()),
            static_cast<Size>(total_nnz / config::MIN_NNZ_PER_THREAD)
        )
    );

    if (num_threads <= 1) {
        spmv_serial(A, x, y, alpha, n_rows);
        return;
    }

    // Compute balanced row ranges
    auto ranges = compute_balanced_ranges(A, n_rows, num_threads);

    // Parallel execution with balanced ranges
    scl::threading::parallel_for(Size(0), num_threads, [&](Size t) {
        const auto& range = ranges[t];
        if (range.start < range.end) {
            spmv_kernel_range(A, x, y, alpha, range.start, range.end);
        }
    });
}

} // namespace detail

// =============================================================================
// SECTION 6: Public API - MappedCustomSparse
// =============================================================================

/// @brief SpMV for MappedCustomSparse: y = alpha * A * x + beta * y
///
/// Ultra-optimized implementation for memory-mapped sparse matrices.
/// No extra caching layer - relies on OS page cache which is sufficient
/// for sequential access patterns.
///
/// Strategy selection:
/// - rows < 1K: serial (no thread overhead)
/// - rows < 100K: simple parallel (good for uniform matrices)
/// - rows >= 100K: load-balanced parallel (handles variance)
///
/// @param A Input sparse matrix (memory-mapped)
/// @param x Input vector, size = secondary_dim(A)
/// @param y Output vector, size = primary_dim(A), pre-allocated
/// @param alpha Scalar multiplier for A*x (default: 1)
/// @param beta Scalar multiplier for y (default: 0)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedCustomSparse<T, IsCSR>, IsCSR>
void spmv_mapped(
    const scl::io::MappedCustomSparse<T, IsCSR>& A,
    Array<const T> x,
    Array<T> y,
    T alpha = T(1),
    T beta = T(0)
) {
    const Index n_primary = scl::primary_size(A);
    const Index nnz = A.nnz();

    SCL_CHECK_DIM(y.len >= static_cast<Size>(n_primary),
        "SpMV: output vector too small");

    // Step 1: Beta scaling
    detail::scale_output(y.ptr, static_cast<Size>(n_primary), beta);

    // Step 2: Early exit
    if (alpha == T(0) || nnz == 0) return;

    // Step 3: Issue prefetch hint (advise kernel)
    kernel::mapped::hint_prefetch(A);

    // Step 4: Strategy selection
    if (n_primary < config::SERIAL_ROW_THRESHOLD) {
        detail::spmv_serial(A, x.ptr, y.ptr, alpha, n_primary);
    } else if (n_primary < config::BALANCED_ROW_THRESHOLD) {
        detail::spmv_parallel_simple(A, x.ptr, y.ptr, alpha, n_primary);
    } else {
        detail::spmv_parallel_balanced(A, x.ptr, y.ptr, alpha, n_primary, nnz);
    }
}

// =============================================================================
// SECTION 7: Public API - MappedVirtualSparse
// =============================================================================

/// @brief SpMV for MappedVirtualSparse: y = alpha * A * x + beta * y
///
/// Virtual sparse has indirection overhead (row mapping), so we use
/// simpler parallelism to maintain cache locality.
///
/// @param A Input virtual sparse matrix (with row indirection)
/// @param x Input vector
/// @param y Output vector, pre-allocated
/// @param alpha Scalar multiplier for A*x (default: 1)
/// @param beta Scalar multiplier for y (default: 0)
template <typename T, bool IsCSR>
    requires kernel::mapped::MappedSparseLike<scl::io::MappedVirtualSparse<T, IsCSR>, IsCSR>
void spmv_mapped(
    const scl::io::MappedVirtualSparse<T, IsCSR>& A,
    Array<const T> x,
    Array<T> y,
    T alpha = T(1),
    T beta = T(0)
) {
    const Index n_primary = scl::primary_size(A);
    const Index nnz = A.nnz();

    SCL_CHECK_DIM(y.len >= static_cast<Size>(n_primary),
        "SpMV: output vector too small");

    // Step 1: Beta scaling
    detail::scale_output(y.ptr, static_cast<Size>(n_primary), beta);

    // Step 2: Early exit
    if (alpha == T(0) || nnz == 0) return;

    // Step 3: Strategy selection
    // Virtual sparse benefits from simpler parallelism due to indirection
    constexpr Index VIRTUAL_SERIAL_THRESHOLD = 500;

    if (n_primary < VIRTUAL_SERIAL_THRESHOLD) {
        detail::spmv_serial(A, x.ptr, y.ptr, alpha, n_primary);
    } else {
        detail::spmv_parallel_simple(A, x.ptr, y.ptr, alpha, n_primary);
    }
}

// =============================================================================
// SECTION 8: Convenience Wrappers (void return, no allocation)
// =============================================================================

/// @brief y = A * x (simplified interface)
///
/// Equivalent to spmv_mapped(A, x, y, 1, 0)
template <typename MatrixT>
    requires kernel::mapped::MappedSparseLike<MatrixT, MatrixT::is_csr>
void spmv(
    const MatrixT& A,
    Array<const typename MatrixT::ValueType> x,
    Array<typename MatrixT::ValueType> y
) {
    using T = typename MatrixT::ValueType;
    spmv_mapped(A, x, y, T(1), T(0));
}

/// @brief y += A * x (accumulate)
///
/// Equivalent to spmv_mapped(A, x, y, 1, 1)
template <typename MatrixT>
    requires kernel::mapped::MappedSparseLike<MatrixT, MatrixT::is_csr>
void spmv_add(
    const MatrixT& A,
    Array<const typename MatrixT::ValueType> x,
    Array<typename MatrixT::ValueType> y
) {
    using T = typename MatrixT::ValueType;
    spmv_mapped(A, x, y, T(1), T(1));
}

/// @brief y = alpha * A * x (scaled)
///
/// Equivalent to spmv_mapped(A, x, y, alpha, 0)
template <typename MatrixT>
    requires kernel::mapped::MappedSparseLike<MatrixT, MatrixT::is_csr>
void spmv_scaled(
    const MatrixT& A,
    Array<const typename MatrixT::ValueType> x,
    Array<typename MatrixT::ValueType> y,
    typename MatrixT::ValueType alpha
) {
    using T = typename MatrixT::ValueType;
    spmv_mapped(A, x, y, alpha, T(0));
}

// =============================================================================
// SECTION 9: Unified Dispatcher
// =============================================================================

/// @brief Auto-dispatch to appropriate mapped fast path
///
/// Enables generic code to use mapped optimizations without
/// knowing the specific matrix type.
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
