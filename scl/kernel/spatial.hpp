#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/macros.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"
#include "scl/threading/scheduler.hpp"

#include <cmath>

// =============================================================================
// FILE: scl/kernel/spatial.hpp
// BRIEF: Spatial statistics with SIMD optimization
// =============================================================================

namespace scl::kernel::spatial {

namespace config {
    constexpr Size PREFETCH_DISTANCE = 8;
    constexpr Size SIMD_GATHER_THRESHOLD = 16;
    constexpr Size PARALLEL_CELL_THRESHOLD = 1024;
    constexpr Size CELL_BLOCK_SIZE = 256;
}

namespace detail {

// Optimized 8-way unrolled weighted neighbor sum with aggressive prefetching
template <typename T>
SCL_FORCE_INLINE SCL_HOT T compute_weighted_neighbor_sum_simd(
    const T* SCL_RESTRICT weights,
    const Index* SCL_RESTRICT indices,
    Size len,
    const T* SCL_RESTRICT z
) {
    // Multi-accumulator pattern to hide memory latency
    T sum0 = T(0), sum1 = T(0), sum2 = T(0), sum3 = T(0);
    T sum4 = T(0), sum5 = T(0), sum6 = T(0), sum7 = T(0);

    Size k = 0;

    // 8-way unrolled loop with aggressive prefetching
    for (; k + 8 <= len; k += 8) {
        // Prefetch ahead for indirect z access
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < len)) {
            SCL_PREFETCH_READ(&z[indices[k + config::PREFETCH_DISTANCE]], 0);
            SCL_PREFETCH_READ(&z[indices[k + config::PREFETCH_DISTANCE + 4]], 0);
            SCL_PREFETCH_READ(&weights[k + config::PREFETCH_DISTANCE], 0);
        }

        // Load weights and gather z values
        T w0 = weights[k + 0], w1 = weights[k + 1], w2 = weights[k + 2], w3 = weights[k + 3];
        T w4 = weights[k + 4], w5 = weights[k + 5], w6 = weights[k + 6], w7 = weights[k + 7];

        T z0 = z[indices[k + 0]], z1 = z[indices[k + 1]], z2 = z[indices[k + 2]], z3 = z[indices[k + 3]];
        T z4 = z[indices[k + 4]], z5 = z[indices[k + 5]], z6 = z[indices[k + 6]], z7 = z[indices[k + 7]];

        // Fused multiply-add
        sum0 += w0 * z0;
        sum1 += w1 * z1;
        sum2 += w2 * z2;
        sum3 += w3 * z3;
        sum4 += w4 * z4;
        sum5 += w5 * z5;
        sum6 += w6 * z6;
        sum7 += w7 * z7;
    }

    T sum = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;

    // Scalar cleanup
    for (; k < len; ++k) {
        sum += weights[k] * z[indices[k]];
    }

    return sum;
}

// Scalar fallback for short arrays
template <typename T>
SCL_FORCE_INLINE T compute_weighted_neighbor_sum_scalar(
    const T* SCL_RESTRICT weights,
    const Index* SCL_RESTRICT indices,
    Size len,
    const T* SCL_RESTRICT z
) {
    T sum = T(0);

    Size k = 0;
    for (; k + 4 <= len; k += 4) {
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < len)) {
            SCL_PREFETCH_READ(&z[indices[k + config::PREFETCH_DISTANCE]], 0);
        }

        T w0 = weights[k + 0];
        T w1 = weights[k + 1];
        T w2 = weights[k + 2];
        T w3 = weights[k + 3];

        T zj0 = z[indices[k + 0]];
        T zj1 = z[indices[k + 1]];
        T zj2 = z[indices[k + 2]];
        T zj3 = z[indices[k + 3]];

        sum += w0 * zj0 + w1 * zj1 + w2 * zj2 + w3 * zj3;
    }

    for (; k < len; ++k) {
        sum += weights[k] * z[indices[k]];
    }

    return sum;
}

// Adaptive dispatch: SIMD for long arrays, scalar for short
template <typename T>
SCL_FORCE_INLINE T compute_weighted_neighbor_sum(
    const T* SCL_RESTRICT weights,
    const Index* SCL_RESTRICT indices,
    Size len,
    const T* SCL_RESTRICT z
) {
    if (SCL_UNLIKELY(len == 0)) return T(0);

    if (len >= config::SIMD_GATHER_THRESHOLD) {
        return compute_weighted_neighbor_sum_simd(weights, indices, len, z);
    } else {
        return compute_weighted_neighbor_sum_scalar(weights, indices, len, z);
    }
}

} // namespace detail

// =============================================================================
// Transform Functions
// =============================================================================

template <typename T, bool GraphCSR>
void weight_sum(
    const Sparse<T, GraphCSR>& graph,
    T& out_sum
) {
    const Index primary_dim = graph.primary_dim();
    const Index total_nnz = graph.nnz();

    if (SCL_UNLIKELY(total_nnz == 0)) {
        out_sum = T(0);
        return;
    }

    const Size n_threads = scl::threading::Scheduler::get_num_threads();
    scl::threading::WorkspacePool<T> partial_sums;
    partial_sums.init(n_threads, 1);
    partial_sums.zero_all();

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p, size_t thread_rank) {
        const auto idx = static_cast<Index>(p);
        const Index len = graph.primary_length_unsafe(idx);
        const Size len_sz = static_cast<Size>(len);

        if (SCL_UNLIKELY(len_sz == 0)) return;

        auto values = graph.primary_values_unsafe(idx);
        T* local_sum = partial_sums.get(thread_rank);
        *local_sum += scl::vectorize::sum(Array<const T>(values.ptr, len_sz));
    });

    // Reduce partial sums
    out_sum = T(0);
    for (Size t = 0; t < n_threads; ++t) {
        out_sum += *partial_sums.get(t);
    }
}

namespace detail {

// Block-wise computation for parallel reduction
template <typename T, bool GraphCSR>
SCL_FORCE_INLINE Real compute_moran_numer_block(
    const Sparse<T, GraphCSR>& graph,
    const Real* SCL_RESTRICT z,
    Index start_cell,
    Index end_cell
) {
    Real numer = Real(0);

    for (Index i = start_cell; i < end_cell; ++i) {
        const Index g_len = graph.primary_length_unsafe(i);
        const Size g_len_sz = static_cast<Size>(g_len);

        if (SCL_UNLIKELY(g_len_sz == 0)) continue;

        // Prefetch next cell's graph data
        if (SCL_LIKELY(i + 1 < end_cell)) {
            const Index next_len = graph.primary_length_unsafe(i + 1);
            if (next_len > 0) {
                auto next_weights = graph.primary_values_unsafe(i + 1);
                auto next_indices = graph.primary_indices_unsafe(i + 1);
                SCL_PREFETCH_READ(next_weights.ptr, 1);
                SCL_PREFETCH_READ(next_indices.ptr, 1);
            }
        }

        auto g_weights = graph.primary_values_unsafe(i);
        auto g_indices = graph.primary_indices_unsafe(i);

        Real z_i = z[i];
        Real neighbor_sum = compute_weighted_neighbor_sum(
            g_weights.ptr, g_indices.ptr, g_len_sz,
            reinterpret_cast<const T*>(z)
        );
        numer += z_i * neighbor_sum;
    }

    return numer;
}

} // namespace detail

template <typename T, bool GraphCSR, bool FeatCSR>
void morans_i(
    const Sparse<T, GraphCSR>& graph,
    const Sparse<T, FeatCSR>& features,
    Array<Real> output
) {
    const Index n_cells = graph.primary_dim();
    const Index n_features = features.primary_dim();

    SCL_CHECK_DIM(graph.secondary_dim() == n_cells, "Graph must be square");
    SCL_CHECK_DIM(features.secondary_dim() == n_cells, "Features dim mismatch");
    SCL_CHECK_DIM(output.len == static_cast<Size>(n_features), "Output size mismatch");

    T W_sum;
    weight_sum(graph, W_sum);

    if (SCL_UNLIKELY(W_sum <= T(0))) {
        scl::memory::fill(output, Real(0));
        return;
    }

    const Real N = static_cast<Real>(n_cells);
    const Real N_over_W = N / static_cast<Real>(W_sum);

    const Size n_threads = scl::threading::Scheduler::get_num_threads();
    const bool use_nested_parallel = (n_features == 1 && static_cast<Size>(n_cells) >= config::PARALLEL_CELL_THRESHOLD);

    // Pre-allocate z buffers for all threads
    scl::threading::WorkspacePool<Real> z_pool;
    z_pool.init(n_threads, static_cast<Size>(n_cells));

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_features), [&](size_t f, size_t thread_rank) {
        const auto f_idx = static_cast<Index>(f);
        const Index len = features.primary_length_unsafe(f_idx);
        const Size len_sz = static_cast<Size>(len);

        auto feat_values = features.primary_values_unsafe(f_idx);
        auto feat_indices = features.primary_indices_unsafe(f_idx);

        Real sum = SCL_LIKELY(len_sz > 0)
            ? static_cast<Real>(scl::vectorize::sum(Array<const T>(feat_values.ptr, len_sz)))
            : Real(0);
        Real mean = sum / N;

        Real* SCL_RESTRICT z = z_pool.get(thread_rank);

        scl::memory::fill(Array<Real>(z, static_cast<Size>(n_cells)), -mean);

        // 4-way unrolled z initialization
        Size k = 0;
        for (; k + 4 <= len_sz; k += 4) {
            z[feat_indices[k + 0]] = static_cast<Real>(feat_values[k + 0]) - mean;
            z[feat_indices[k + 1]] = static_cast<Real>(feat_values[k + 1]) - mean;
            z[feat_indices[k + 2]] = static_cast<Real>(feat_values[k + 2]) - mean;
            z[feat_indices[k + 3]] = static_cast<Real>(feat_values[k + 3]) - mean;
        }

        for (; k < len_sz; ++k) {
            z[feat_indices[k]] = static_cast<Real>(feat_values[k]) - mean;
        }

        Real denom = scl::vectorize::sum_squared(Array<const Real>(z, static_cast<Size>(n_cells)));

        if (SCL_UNLIKELY(denom <= Real(0))) {
            output[static_cast<Index>(f)] = Real(0);
            return;
        }

        Real numer = Real(0);

        // Parallel reduction for single feature + large cell count
        if (use_nested_parallel) {
            const Size n_blocks = (static_cast<Size>(n_cells) + config::CELL_BLOCK_SIZE - 1) / config::CELL_BLOCK_SIZE;
            auto block_results_ptr = scl::memory::aligned_alloc<Real>(n_blocks, SCL_ALIGNMENT);

            Real* block_results = block_results_ptr.release();

            scl::threading::parallel_for(Size(0), n_blocks, [&](size_t b) {
                auto start = static_cast<Index>(b * config::CELL_BLOCK_SIZE);
                Index end = scl::algo::min2(start + static_cast<Index>(config::CELL_BLOCK_SIZE), n_cells);
                block_results[b] = detail::compute_moran_numer_block(graph, z, start, end);
            });

            numer = scl::vectorize::sum(Array<const Real>(block_results, n_blocks));
            scl::memory::aligned_free(block_results, SCL_ALIGNMENT);
        } else {
            // Sequential loop for multiple features or small cell count
            numer = detail::compute_moran_numer_block(graph, z, Index(0), n_cells);
        }

        output[static_cast<Index>(f)] = N_over_W * (numer / denom);
    });
}

// =============================================================================
// Geary's C Statistic
// =============================================================================

namespace detail {

// Block-wise Geary's C numerator computation with 8-way unroll
template <typename T, bool GraphCSR>
SCL_FORCE_INLINE Real compute_geary_numer_block(
    const Sparse<T, GraphCSR>& graph,
    const Real* SCL_RESTRICT z,
    Index start_cell,
    Index end_cell
) {
    Real numer = Real(0);

    for (Index i = start_cell; i < end_cell; ++i) {
        const Index g_len = graph.primary_length_unsafe(i);
        const Size g_len_sz = static_cast<Size>(g_len);

        if (SCL_UNLIKELY(g_len_sz == 0)) continue;

        // Prefetch next cell's graph data
        if (SCL_LIKELY(i + 1 < end_cell)) {
            const Index next_len = graph.primary_length_unsafe(i + 1);
            if (next_len > 0) {
                auto next_weights = graph.primary_values_unsafe(i + 1);
                auto next_indices = graph.primary_indices_unsafe(i + 1);
                SCL_PREFETCH_READ(next_weights.ptr, 1);
                SCL_PREFETCH_READ(next_indices.ptr, 1);
            }
        }

        auto g_weights = graph.primary_values_unsafe(i);
        auto g_indices = graph.primary_indices_unsafe(i);

        Real z_i = z[i];

        // 8-way unrolled with multi-accumulator pattern
        Size k = 0;
        Real acc0 = Real(0), acc1 = Real(0), acc2 = Real(0), acc3 = Real(0);
        Real acc4 = Real(0), acc5 = Real(0), acc6 = Real(0), acc7 = Real(0);

        for (; k + 8 <= g_len_sz; k += 8) {
            // Prefetch ahead for indirect z access
            if (SCL_LIKELY(k + config::PREFETCH_DISTANCE < g_len_sz)) {
                SCL_PREFETCH_READ(&z[g_indices[k + config::PREFETCH_DISTANCE]], 0);
                SCL_PREFETCH_READ(&z[g_indices[k + config::PREFETCH_DISTANCE + 4]], 0);
            }

            Real diff0 = z_i - z[g_indices[k + 0]];
            Real diff1 = z_i - z[g_indices[k + 1]];
            Real diff2 = z_i - z[g_indices[k + 2]];
            Real diff3 = z_i - z[g_indices[k + 3]];
            Real diff4 = z_i - z[g_indices[k + 4]];
            Real diff5 = z_i - z[g_indices[k + 5]];
            Real diff6 = z_i - z[g_indices[k + 6]];
            Real diff7 = z_i - z[g_indices[k + 7]];

            acc0 += g_weights[k + 0] * diff0 * diff0;
            acc1 += g_weights[k + 1] * diff1 * diff1;
            acc2 += g_weights[k + 2] * diff2 * diff2;
            acc3 += g_weights[k + 3] * diff3 * diff3;
            acc4 += g_weights[k + 4] * diff4 * diff4;
            acc5 += g_weights[k + 5] * diff5 * diff5;
            acc6 += g_weights[k + 6] * diff6 * diff6;
            acc7 += g_weights[k + 7] * diff7 * diff7;
        }

        Real local_sum = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;

        // 4-way cleanup
        for (; k + 4 <= g_len_sz; k += 4) {
            Real diff0 = z_i - z[g_indices[k + 0]];
            Real diff1 = z_i - z[g_indices[k + 1]];
            Real diff2 = z_i - z[g_indices[k + 2]];
            Real diff3 = z_i - z[g_indices[k + 3]];

            local_sum += g_weights[k + 0] * diff0 * diff0;
            local_sum += g_weights[k + 1] * diff1 * diff1;
            local_sum += g_weights[k + 2] * diff2 * diff2;
            local_sum += g_weights[k + 3] * diff3 * diff3;
        }

        // Scalar cleanup
        for (; k < g_len_sz; ++k) {
            Real diff = z_i - z[g_indices[k]];
            local_sum += g_weights[k] * diff * diff;
        }

        numer += local_sum;
    }

    return numer;
}

} // namespace detail

template <typename T, bool GraphCSR, bool FeatCSR>
void gearys_c(
    const Sparse<T, GraphCSR>& graph,
    const Sparse<T, FeatCSR>& features,
    Array<Real> output
) {
    const Index n_cells = graph.primary_dim();
    const Index n_features = features.primary_dim();

    SCL_CHECK_DIM(graph.secondary_dim() == n_cells, "Graph must be square");
    SCL_CHECK_DIM(features.secondary_dim() == n_cells, "Features dim mismatch");
    SCL_CHECK_DIM(output.len == static_cast<Size>(n_features), "Output size mismatch");

    T W_sum;
    weight_sum(graph, W_sum);

    if (SCL_UNLIKELY(W_sum <= T(0))) {
        scl::memory::fill(output, Real(0));
        return;
    }

    const Real N = static_cast<Real>(n_cells);
    const Real N_minus_1 = N - Real(1);
    const Real scale = N_minus_1 / (static_cast<Real>(W_sum) * Real(2));

    const Size n_threads = scl::threading::Scheduler::get_num_threads();
    const bool use_nested_parallel = (n_features == 1 && static_cast<Size>(n_cells) >= config::PARALLEL_CELL_THRESHOLD);

    scl::threading::WorkspacePool<Real> z_pool;
    z_pool.init(n_threads, static_cast<Size>(n_cells));

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_features), [&](size_t f, size_t thread_rank) {
        const auto f_idx = static_cast<Index>(f);
        const Index len = features.primary_length_unsafe(f_idx);
        const Size len_sz = static_cast<Size>(len);

        auto feat_values = features.primary_values_unsafe(f_idx);
        auto feat_indices = features.primary_indices_unsafe(f_idx);

        Real sum = SCL_LIKELY(len_sz > 0)
            ? static_cast<Real>(scl::vectorize::sum(Array<const T>(feat_values.ptr, len_sz)))
            : Real(0);
        Real mean = sum / N;

        Real* SCL_RESTRICT z = z_pool.get(thread_rank);

        scl::memory::fill(Array<Real>(z, static_cast<Size>(n_cells)), -mean);

        // 4-way unrolled z initialization
        Size k = 0;
        for (; k + 4 <= len_sz; k += 4) {
            z[feat_indices[k + 0]] = static_cast<Real>(feat_values[k + 0]) - mean;
            z[feat_indices[k + 1]] = static_cast<Real>(feat_values[k + 1]) - mean;
            z[feat_indices[k + 2]] = static_cast<Real>(feat_values[k + 2]) - mean;
            z[feat_indices[k + 3]] = static_cast<Real>(feat_values[k + 3]) - mean;
        }

        for (; k < len_sz; ++k) {
            z[feat_indices[k]] = static_cast<Real>(feat_values[k]) - mean;
        }

        Real denom = scl::vectorize::sum_squared(Array<const Real>(z, static_cast<Size>(n_cells)));

        if (SCL_UNLIKELY(denom <= Real(0))) {
            output[static_cast<Index>(f)] = Real(0);
            return;
        }

        Real numer = Real(0);

        if (use_nested_parallel) {
            const Size n_blocks = (static_cast<Size>(n_cells) + config::CELL_BLOCK_SIZE - 1) / config::CELL_BLOCK_SIZE;
            auto block_results_ptr = scl::memory::aligned_alloc<Real>(n_blocks, SCL_ALIGNMENT);

            Real* block_results = block_results_ptr.release();

            scl::threading::parallel_for(Size(0), n_blocks, [&](size_t b) {
                auto start = static_cast<Index>(b * config::CELL_BLOCK_SIZE);
                Index end = scl::algo::min2(start + static_cast<Index>(config::CELL_BLOCK_SIZE), n_cells);
                block_results[b] = detail::compute_geary_numer_block(graph, z, start, end);
            });

            numer = scl::vectorize::sum(Array<const Real>(block_results, n_blocks));
            scl::memory::aligned_free(block_results, SCL_ALIGNMENT);
        } else {
            numer = detail::compute_geary_numer_block(graph, z, Index(0), n_cells);
        }

        output[static_cast<Index>(f)] = scale * (numer / denom);
    });
}

} // namespace scl::kernel::spatial

