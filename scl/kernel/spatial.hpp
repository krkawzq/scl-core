#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/core/memory.hpp"
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
}

namespace detail {

// Compute sum(w_j * z_j) for neighbors - z_i factored out
template <typename T>
SCL_FORCE_INLINE T compute_weighted_neighbor_sum(
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

        // Factored: z_i multiplied once outside
        sum += w0 * zj0 + w1 * zj1 + w2 * zj2 + w3 * zj3;
    }

    for (; k < len; ++k) {
        sum += weights[k] * z[indices[k]];
    }

    return sum;
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

    if (total_nnz == 0) {
        out_sum = T(0);
        return;
    }

    // Pre-allocate workspace for partial sums
    T* workspace = scl::memory::aligned_alloc<T>(static_cast<Size>(primary_dim), SCL_ALIGNMENT);

    scl::threading::parallel_for(Size(0), static_cast<Size>(primary_dim), [&](size_t p) {
        const Index idx = static_cast<Index>(p);
        const Index len = graph.primary_length(idx);
        const Size len_sz = static_cast<Size>(len);

        if (len_sz == 0) {
            workspace[p] = T(0);
            return;
        }

        auto values = graph.primary_values(idx);
        workspace[p] = scl::vectorize::sum(Array<const T>(values.ptr, len_sz));
    });

    out_sum = scl::vectorize::sum(Array<const T>(workspace, static_cast<Size>(primary_dim)));

    scl::memory::aligned_free(workspace, SCL_ALIGNMENT);
}

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

    if (W_sum <= T(0)) {
        scl::memory::fill(output, Real(0));
        return;
    }

    const Real N = static_cast<Real>(n_cells);
    const Real N_over_W = N / static_cast<Real>(W_sum);

    // Pre-allocate z buffers for all threads
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    scl::threading::WorkspacePool<Real> z_pool;
    z_pool.init(n_threads, static_cast<Size>(n_cells));

    scl::threading::parallel_for(Size(0), static_cast<Size>(n_features), [&](size_t f, size_t thread_rank) {
        const Index f_idx = static_cast<Index>(f);
        const Index len = features.primary_length(f_idx);
        const Size len_sz = static_cast<Size>(len);

        auto feat_values = features.primary_values(f_idx);
        auto feat_indices = features.primary_indices(f_idx);

        Real sum = (len_sz > 0)
            ? static_cast<Real>(scl::vectorize::sum(Array<const T>(feat_values.ptr, len_sz)))
            : Real(0);
        Real mean = sum / N;

        Real* SCL_RESTRICT z = z_pool.get(thread_rank);

        scl::memory::fill(Array<Real>(z, static_cast<Size>(n_cells)), -mean);

        for (Size k = 0; k < len_sz; ++k) {
            z[feat_indices[k]] = static_cast<Real>(feat_values[k]) - mean;
        }

        Real denom = scl::vectorize::sum_squared(Array<const Real>(z, static_cast<Size>(n_cells)));

        if (denom <= Real(0)) {
            output[f] = Real(0);
            return;
        }

        Real numer = Real(0);

        for (Index i = 0; i < n_cells; ++i) {
            const Index g_len = graph.primary_length(i);
            const Size g_len_sz = static_cast<Size>(g_len);

            if (g_len_sz == 0) continue;

            auto g_weights = graph.primary_values(i);
            auto g_indices = graph.primary_indices(i);

            // z_i factored out: numer += z_i * sum(w_ij * z_j)
            Real z_i = z[i];
            Real neighbor_sum = detail::compute_weighted_neighbor_sum(
                g_weights.ptr, g_indices.ptr, g_len_sz,
                reinterpret_cast<const T*>(z)
            );
            numer += z_i * neighbor_sum;
        }

        output[f] = N_over_W * (numer / denom);
    });
}

} // namespace scl::kernel::spatial

