#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"
#include "scl/threading/scheduler.hpp"

#include <cmath>
#include <cstring>

// =============================================================================
// FILE: scl/kernel/pseudotime.hpp
// BRIEF: High-performance pseudotime inference for trajectory analysis
//
// Optimizations applied:
// - 4-ary heap for faster Dijkstra
// - Parallel multi-source shortest paths
// - SIMD-accelerated Gram-Schmidt orthogonalization
// - Parallel power iteration for diffusion components
// - Fused correlation computation
// - Cache-aligned workspace structures
// =============================================================================

namespace scl::kernel::pseudotime {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Index DEFAULT_N_DCS = 10;
    constexpr Index DEFAULT_N_ITERATIONS = 100;
    constexpr Real DEFAULT_THRESHOLD = Real(0.1);
    constexpr Real DEFAULT_DAMPING = Real(0.85);
    constexpr Real CONVERGENCE_TOL = Real(1e-6);
    constexpr Real INF_DISTANCE = Real(1e30);
    constexpr Size PARALLEL_THRESHOLD = 256;
    constexpr Size SIMD_THRESHOLD = 16;
    constexpr Index HEAP_ARITY = 4;  // 4-ary heap
}

// =============================================================================
// Trajectory Methods
// =============================================================================

enum class PseudotimeMethod {
    DiffusionPseudotime,
    ShortestPath,
    GraphDistance,
    WatershedDescent
};

// =============================================================================
// Internal Optimized Operations
// =============================================================================

namespace detail {

// =============================================================================
// SIMD Vector Operations
// =============================================================================

SCL_HOT SCL_FORCE_INLINE Real dot_simd(
    const Real* SCL_RESTRICT a,
    const Real* SCL_RESTRICT b,
    Size n
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    auto v_sum0 = s::Zero(d);
    auto v_sum1 = s::Zero(d);

    Size k = 0;
    for (; k + 2 * lanes <= n; k += 2 * lanes) {
        v_sum0 = s::MulAdd(s::Load(d, a + k), s::Load(d, b + k), v_sum0);
        v_sum1 = s::MulAdd(s::Load(d, a + k + lanes), s::Load(d, b + k + lanes), v_sum1);
    }

    Real result = s::GetLane(s::SumOfLanes(d, s::Add(v_sum0, v_sum1)));

    for (; k < n; ++k) {
        result += a[k] * b[k];
    }

    return result;
}

SCL_HOT SCL_FORCE_INLINE Real norm_squared_simd(const Real* SCL_RESTRICT a, Size n) noexcept {
    return dot_simd(a, a, n);
}

SCL_HOT SCL_FORCE_INLINE void axpy_simd(
    Real alpha,
    const Real* SCL_RESTRICT x,
    Real* SCL_RESTRICT y,
    Size n
) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    auto v_alpha = s::Set(d, alpha);

    Size k = 0;
    for (; k + 2 * lanes <= n; k += 2 * lanes) {
        auto y0 = s::Load(d, y + k);
        auto y1 = s::Load(d, y + k + lanes);
        y0 = s::MulAdd(v_alpha, s::Load(d, x + k), y0);
        y1 = s::MulAdd(v_alpha, s::Load(d, x + k + lanes), y1);
        s::Store(y0, d, y + k);
        s::Store(y1, d, y + k + lanes);
    }

    for (; k < n; ++k) {
        y[k] += alpha * x[k];
    }
}

SCL_HOT SCL_FORCE_INLINE void scale_simd(Real* SCL_RESTRICT x, Real alpha, Size n) noexcept {
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const size_t lanes = s::Lanes(d);

    auto v_alpha = s::Set(d, alpha);

    Size k = 0;
    for (; k + 2 * lanes <= n; k += 2 * lanes) {
        s::Store(s::Mul(v_alpha, s::Load(d, x + k)), d, x + k);
        s::Store(s::Mul(v_alpha, s::Load(d, x + k + lanes)), d, x + k + lanes);
    }

    for (; k < n; ++k) {
        x[k] *= alpha;
    }
}

// =============================================================================
// Fast PRNG (Xoshiro128+)
// =============================================================================

struct alignas(16) FastRNG {
    uint32_t s[4];

    SCL_FORCE_INLINE explicit FastRNG(uint64_t seed) noexcept {
        uint64_t z = seed;
        for (int i = 0; i < 4; ++i) {
            z += 0x9e3779b97f4a7c15ULL;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            s[i] = static_cast<uint32_t>(z >> 32);
        }
    }

    SCL_FORCE_INLINE uint32_t next() noexcept {
        uint32_t t = s[3];
        uint32_t const x = s[0];
        s[3] = s[2];
        s[2] = s[1];
        s[1] = x;
        t ^= t >> 11;
        t ^= t << 8;
        s[0] = t ^ x ^ (x << 19);
        return s[0];
    }

    SCL_FORCE_INLINE Real uniform() noexcept {
        return static_cast<Real>(next()) * Real(2.3283064365386963e-10);
    }
};

// =============================================================================
// 4-ary Min Heap (Faster than binary for Dijkstra)
// =============================================================================

struct alignas(64) FastMinHeap {
    static constexpr Index ARITY = config::HEAP_ARITY;

    Index* indices;
    Real* keys;
    Index* positions;
    Index size;
    Index capacity;

    void init(Index cap) {
        capacity = cap;
        size = 0;
        indices = scl::memory::aligned_alloc<Index>(cap, SCL_ALIGNMENT);
        keys = scl::memory::aligned_alloc<Real>(cap, SCL_ALIGNMENT);
        positions = scl::memory::aligned_alloc<Index>(cap, SCL_ALIGNMENT);
        for (Index i = 0; i < cap; ++i) {
            positions[i] = -1;
        }
    }

    void destroy() {
        scl::memory::aligned_free(positions, SCL_ALIGNMENT);
        scl::memory::aligned_free(keys, SCL_ALIGNMENT);
        scl::memory::aligned_free(indices, SCL_ALIGNMENT);
    }

    void clear() noexcept {
        for (Index i = 0; i < size; ++i) {
            positions[indices[i]] = -1;
        }
        size = 0;
    }

    SCL_FORCE_INLINE void push(Index idx, Real key) noexcept {
        Index i = size++;
        indices[i] = idx;
        keys[i] = key;
        positions[idx] = i;
        sift_up(i);
    }

    SCL_FORCE_INLINE void decrease_key(Index idx, Real new_key) noexcept {
        Index i = positions[idx];
        if (i >= 0 && i < size && new_key < keys[i]) {
            keys[i] = new_key;
            sift_up(i);
        }
    }

    SCL_FORCE_INLINE Index pop_min() noexcept {
        if (size == 0) return -1;

        Index min_idx = indices[0];
        positions[min_idx] = -1;

        --size;
        if (size > 0) {
            indices[0] = indices[size];
            keys[0] = keys[size];
            positions[indices[0]] = 0;
            sift_down(0);
        }

        return min_idx;
    }

    SCL_FORCE_INLINE bool empty() const noexcept { return size == 0; }

    SCL_FORCE_INLINE bool contains(Index idx) const noexcept {
        return idx >= 0 && idx < capacity && positions[idx] >= 0 && positions[idx] < size;
    }

private:
    SCL_FORCE_INLINE void sift_up(Index i) noexcept {
        Index idx = indices[i];
        Real key = keys[i];

        while (i > 0) {
            Index parent = (i - 1) / ARITY;
            if (key < keys[parent]) {
                indices[i] = indices[parent];
                keys[i] = keys[parent];
                positions[indices[i]] = i;
                i = parent;
            } else {
                break;
            }
        }

        indices[i] = idx;
        keys[i] = key;
        positions[idx] = i;
    }

    SCL_FORCE_INLINE void sift_down(Index i) noexcept {
        Index idx = indices[i];
        Real key = keys[i];

        while (true) {
            Index first_child = ARITY * i + 1;
            if (first_child >= size) break;

            // Find minimum child
            Index min_child = first_child;
            Real min_key = keys[first_child];
            Index last_child = scl::algo::min2(first_child + ARITY, size);

            for (Index c = first_child + 1; c < last_child; ++c) {
                if (keys[c] < min_key) {
                    min_child = c;
                    min_key = keys[c];
                }
            }

            if (min_key < key) {
                indices[i] = indices[min_child];
                keys[i] = keys[min_child];
                positions[indices[i]] = i;
                i = min_child;
            } else {
                break;
            }
        }

        indices[i] = idx;
        keys[i] = key;
        positions[idx] = i;
    }
};

// =============================================================================
// Parallel SpMV for transition matrix
// =============================================================================

template <typename T, bool IsCSR>
SCL_HOT void spmv_parallel(
    const Sparse<T, IsCSR>& mat,
    const Real* SCL_RESTRICT x,
    Real* SCL_RESTRICT y,
    Size n
) {
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    if (n >= config::PARALLEL_THRESHOLD && n_threads > 1) {
        scl::threading::parallel_for(Size(0), n, [&](size_t i) {
            const Index idx = static_cast<Index>(i);
            auto indices = mat.primary_indices_unsafe(idx);
            auto values = mat.primary_values_unsafe(idx);
            const Index len = mat.primary_length_unsafe(idx);

            Real sum = Real(0);
            Index k = 0;
            for (; k + 4 <= len; k += 4) {
                sum += static_cast<Real>(values[k + 0]) * x[indices[k + 0]];
                sum += static_cast<Real>(values[k + 1]) * x[indices[k + 1]];
                sum += static_cast<Real>(values[k + 2]) * x[indices[k + 2]];
                sum += static_cast<Real>(values[k + 3]) * x[indices[k + 3]];
            }
            for (; k < len; ++k) {
                sum += static_cast<Real>(values[k]) * x[indices[k]];
            }
            y[i] = sum;
        });
    } else {
        for (Size i = 0; i < n; ++i) {
            const Index idx = static_cast<Index>(i);
            auto indices = mat.primary_indices_unsafe(idx);
            auto values = mat.primary_values_unsafe(idx);
            const Index len = mat.primary_length_unsafe(idx);

            Real sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                sum += static_cast<Real>(values[k]) * x[indices[k]];
            }
            y[i] = sum;
        }
    }
}

// =============================================================================
// Block SpMM for diffusion components
// =============================================================================

template <typename T, bool IsCSR>
SCL_HOT void spmm_block(
    const Sparse<T, IsCSR>& mat,
    const Real* SCL_RESTRICT X,  // n x n_cols, row-major (column c at X[i*n_cols + c])
    Real* SCL_RESTRICT Y,
    Size n,
    Index n_cols
) {
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    scl::threading::parallel_for(Size(0), n, [&](size_t i) {
        const Index idx = static_cast<Index>(i);
        auto indices = mat.primary_indices_unsafe(idx);
        auto values = mat.primary_values_unsafe(idx);
        const Index len = mat.primary_length_unsafe(idx);

        Real* Yi = Y + i * n_cols;

        // Initialize output row
        for (Index c = 0; c < n_cols; ++c) {
            Yi[c] = Real(0);
        }

        // Accumulate neighbor contributions
        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            Real w = static_cast<Real>(values[k]);
            const Real* Xj = X + static_cast<Size>(j) * n_cols;
            // SIMD accumulation
            axpy_simd(w, Xj, Yi, static_cast<Size>(n_cols));
        }
    });
}

// =============================================================================
// Modified Gram-Schmidt with SIMD
// =============================================================================

void orthonormalize_columns(
    Real* Q,       // n x n_cols, row-major
    Size n,
    Index n_cols,
    Real* workspace  // size n for column extraction
) {
    for (Index c = 0; c < n_cols; ++c) {
        // Extract column c to contiguous workspace
        for (Size i = 0; i < n; ++i) {
            workspace[i] = Q[i * n_cols + c];
        }

        // Orthogonalize against previous columns
        for (Index p = 0; p < c; ++p) {
            // Compute dot product with column p
            Real dot = Real(0);
            Real norm_p = Real(0);
            for (Size i = 0; i < n; ++i) {
                Real qp_i = Q[i * n_cols + p];
                dot += workspace[i] * qp_i;
                norm_p += qp_i * qp_i;
            }

            if (norm_p > Real(1e-15)) {
                Real coeff = dot / norm_p;
                for (Size i = 0; i < n; ++i) {
                    workspace[i] -= coeff * Q[i * n_cols + p];
                }
            }
        }

        // Normalize
        Real norm = Real(0);
        for (Size i = 0; i < n; ++i) {
            norm += workspace[i] * workspace[i];
        }

        if (norm > Real(1e-15)) {
            Real inv_norm = Real(1) / std::sqrt(norm);
            for (Size i = 0; i < n; ++i) {
                Q[i * n_cols + c] = workspace[i] * inv_norm;
            }
        } else {
            // Degenerate: use random direction
            FastRNG rng(static_cast<uint64_t>(c) * 12345);
            norm = Real(0);
            for (Size i = 0; i < n; ++i) {
                workspace[i] = rng.uniform() - Real(0.5);
                norm += workspace[i] * workspace[i];
            }
            Real inv_norm = Real(1) / std::sqrt(norm);
            for (Size i = 0; i < n; ++i) {
                Q[i * n_cols + c] = workspace[i] * inv_norm;
            }
        }
    }
}

} // namespace detail

// =============================================================================
// Dijkstra's Shortest Path (Optimized)
// =============================================================================

template <typename T, bool IsCSR>
void dijkstra_shortest_path(
    const Sparse<T, IsCSR>& adjacency,
    Index source,
    Array<Real> distances
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(distances.len >= N, "Pseudotime: distances buffer too small");

    // Initialize
    scl::algo::fill(distances.ptr, N, config::INF_DISTANCE);
    distances[source] = Real(0);

    detail::FastMinHeap heap;
    heap.init(n);
    heap.push(source, Real(0));

    while (!heap.empty()) {
        Index u = heap.pop_min();
        Real dist_u = distances[u];

        auto indices = adjacency.primary_indices_unsafe(u);
        auto values = adjacency.primary_values_unsafe(u);
        const Index len = adjacency.primary_length_unsafe(u);

        for (Index k = 0; k < len; ++k) {
            Index v = indices[k];
            Real weight = static_cast<Real>(values[k]);
            if (weight <= Real(0)) weight = Real(1);

            Real new_dist = dist_u + weight;

            if (new_dist < distances[v]) {
                distances[v] = new_dist;
                if (heap.contains(v)) {
                    heap.decrease_key(v, new_dist);
                } else {
                    heap.push(v, new_dist);
                }
            }
        }
    }

    heap.destroy();
}

// =============================================================================
// Multi-Source Dijkstra (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void dijkstra_multi_source(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Index> sources,
    Real* distances  // n_sources x n, row-major
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);
    const Size n_sources = sources.len;

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // Per-thread heaps
    detail::FastMinHeap* heaps = static_cast<detail::FastMinHeap*>(
        scl::memory::aligned_alloc<detail::FastMinHeap>(n_threads, SCL_ALIGNMENT));
    
    for (size_t t = 0; t < n_threads; ++t) {
        heaps[t].init(n);
    }

    scl::threading::parallel_for(Size(0), n_sources, [&](size_t s, size_t thread_rank) {
        Index source = sources[s];
        Real* dist = distances + s * N;

        scl::algo::fill(dist, N, config::INF_DISTANCE);
        if (source < 0 || source >= n) return;

        dist[source] = Real(0);

        detail::FastMinHeap& heap = heaps[thread_rank];
        heap.clear();
        heap.push(source, Real(0));

        while (!heap.empty()) {
            Index u = heap.pop_min();
            Real dist_u = dist[u];

            auto indices = adjacency.primary_indices_unsafe(u);
            auto values = adjacency.primary_values_unsafe(u);
            const Index len = adjacency.primary_length_unsafe(u);

            for (Index k = 0; k < len; ++k) {
                Index v = indices[k];
                Real weight = static_cast<Real>(values[k]);
                if (weight <= Real(0)) weight = Real(1);

                Real new_dist = dist_u + weight;

                if (new_dist < dist[v]) {
                    dist[v] = new_dist;
                    if (heap.contains(v)) {
                        heap.decrease_key(v, new_dist);
                    } else {
                        heap.push(v, new_dist);
                    }
                }
            }
        }
    });

    for (size_t t = 0; t < n_threads; ++t) {
        heaps[t].destroy();
    }
    scl::memory::aligned_free(heaps, SCL_ALIGNMENT);
}

// =============================================================================
// Graph-Based Pseudotime
// =============================================================================

template <typename T, bool IsCSR>
void graph_pseudotime(
    const Sparse<T, IsCSR>& adjacency,
    Index root_cell,
    Array<Real> pseudotime
) {
    dijkstra_shortest_path(adjacency, root_cell, pseudotime);

    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);

    // Normalize to [0, 1]
    Real max_pt = Real(0);
    for (Size i = 0; i < N; ++i) {
        if (pseudotime[i] < config::INF_DISTANCE) {
            max_pt = scl::algo::max2(max_pt, pseudotime[i]);
        }
    }

    if (max_pt > Real(0)) {
        Real inv_max = Real(1) / max_pt;
        for (Size i = 0; i < N; ++i) {
            if (pseudotime[i] < config::INF_DISTANCE) {
                pseudotime[i] *= inv_max;
            } else {
                pseudotime[i] = Real(1);
            }
        }
    }
}

// =============================================================================
// Diffusion Pseudotime (Optimized)
// =============================================================================

template <typename T, bool IsCSR>
void diffusion_pseudotime(
    const Sparse<T, IsCSR>& transition_matrix,
    Index root_cell,
    Array<Real> pseudotime,
    Index n_dcs = config::DEFAULT_N_DCS,
    Index n_iterations = config::DEFAULT_N_ITERATIONS
) {
    const Index n = transition_matrix.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(pseudotime.len >= N, "Pseudotime: output buffer too small");

    // Allocate diffusion components
    Size dc_size = N * static_cast<Size>(n_dcs);
    Real* dc = scl::memory::aligned_alloc<Real>(dc_size, SCL_ALIGNMENT);
    Real* dc_new = scl::memory::aligned_alloc<Real>(dc_size, SCL_ALIGNMENT);
    Real* workspace = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    // Initialize with random values
    detail::FastRNG rng(42);
    for (Size i = 0; i < dc_size; ++i) {
        dc[i] = rng.uniform() - Real(0.5);
    }

    // Initial orthonormalization
    detail::orthonormalize_columns(dc, N, n_dcs, workspace);

    // Power iteration
    for (Index iter = 0; iter < n_iterations; ++iter) {
        // Apply transition matrix to all components
        detail::spmm_block(transition_matrix, dc, dc_new, N, n_dcs);

        // Orthonormalize
        detail::orthonormalize_columns(dc_new, N, n_dcs, workspace);

        // Swap
        std::swap(dc, dc_new);
    }

    // Compute DPT: distance from root in diffusion space
    const Real* root_dc = dc + static_cast<Size>(root_cell) * n_dcs;

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        const Real* cell_dc = dc + i * n_dcs;

        Real dist_sq = Real(0);
        for (Index c = 0; c < n_dcs; ++c) {
            Real d = cell_dc[c] - root_dc[c];
            dist_sq += d * d;
        }

        pseudotime[i] = std::sqrt(dist_sq);
    });

    // Normalize to [0, 1]
    Real max_pt = pseudotime[0];
    for (Size i = 1; i < N; ++i) {
        max_pt = scl::algo::max2(max_pt, pseudotime[i]);
    }

    if (max_pt > Real(0)) {
        detail::scale_simd(pseudotime.ptr, Real(1) / max_pt, N);
    }

    scl::memory::aligned_free(workspace, SCL_ALIGNMENT);
    scl::memory::aligned_free(dc_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(dc, SCL_ALIGNMENT);
}

// =============================================================================
// Select Root Cell
// =============================================================================

template <typename T, bool IsCSR>
Index select_root_cell(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> marker_expression
) {
    const Index n = adjacency.primary_dim();
    Index root = 0;
    Real min_expr = marker_expression[0];

    for (Index i = 1; i < n; ++i) {
        if (marker_expression[i] < min_expr) {
            min_expr = marker_expression[i];
            root = i;
        }
    }

    return root;
}

// =============================================================================
// Select Root by Peripherality
// =============================================================================

template <typename T, bool IsCSR>
Index select_root_peripheral(
    const Sparse<T, IsCSR>& adjacency
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);

    Real* avg_dist = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        auto values = adjacency.primary_values_unsafe(static_cast<Index>(i));
        const Index len = adjacency.primary_length_unsafe(static_cast<Index>(i));

        if (len == 0) {
            avg_dist[i] = Real(0);
            return;
        }

        Real sum = Real(0);
        for (Index k = 0; k < len; ++k) {
            sum += static_cast<Real>(values[k]);
        }

        avg_dist[i] = sum / static_cast<Real>(len);
    });

    Index root = 0;
    Real max_dist = avg_dist[0];

    for (Size i = 1; i < N; ++i) {
        if (avg_dist[i] > max_dist) {
            max_dist = avg_dist[i];
            root = static_cast<Index>(i);
        }
    }

    scl::memory::aligned_free(avg_dist, SCL_ALIGNMENT);
    return root;
}

// =============================================================================
// Detect Branch Points (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
Index detect_branch_points(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> pseudotime,
    Array<Index> branch_points,
    Real threshold = config::DEFAULT_THRESHOLD
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(branch_points.len >= N, "Pseudotime: branch_points buffer too small");

    // Parallel detection with thread-local counts
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    
    Index* local_counts = scl::memory::aligned_alloc<Index>(n_threads, SCL_ALIGNMENT);
    Index** local_branches = static_cast<Index**>(
        scl::memory::aligned_alloc<Index*>(n_threads, SCL_ALIGNMENT));
    
    for (size_t t = 0; t < n_threads; ++t) {
        local_counts[t] = 0;
        local_branches[t] = scl::memory::aligned_alloc<Index>(N, SCL_ALIGNMENT);
    }

    scl::threading::parallel_for(Size(0), N, [&](size_t i, size_t thread_rank) {
        auto indices = adjacency.primary_indices_unsafe(static_cast<Index>(i));
        const Index len = adjacency.primary_length_unsafe(static_cast<Index>(i));

        if (len < 3) return;

        Index n_earlier = 0;
        Index n_later = 0;
        Real pt_i = pseudotime[i];

        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            Real pt_j = pseudotime[j];

            if (pt_j < pt_i - threshold) ++n_earlier;
            if (pt_j > pt_i + threshold) ++n_later;
        }

        bool is_branch = (n_earlier >= 1 && n_later >= 2) ||
                         (n_earlier >= 2 && n_later >= 1);

        if (is_branch) {
            local_branches[thread_rank][local_counts[thread_rank]++] = static_cast<Index>(i);
        }
    });

    // Merge results
    Index total = 0;
    for (size_t t = 0; t < n_threads; ++t) {
        for (Index i = 0; i < local_counts[t] && total < static_cast<Index>(branch_points.len); ++i) {
            branch_points[total++] = local_branches[t][i];
        }
        scl::memory::aligned_free(local_branches[t], SCL_ALIGNMENT);
    }

    scl::memory::aligned_free(local_branches, SCL_ALIGNMENT);
    scl::memory::aligned_free(local_counts, SCL_ALIGNMENT);

    return total;
}

// =============================================================================
// Segment Trajectory
// =============================================================================

template <typename T, bool IsCSR>
void segment_trajectory(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> pseudotime,
    Array<const Index> branch_points,
    Index n_branch_points,
    Array<Index> segment_labels
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(segment_labels.len >= N, "Pseudotime: segment_labels buffer too small");

    if (n_branch_points == 0) {
        scl::algo::zero(segment_labels.ptr, N);
        return;
    }

    // Sort branch points by pseudotime
    Index* sorted_bp = scl::memory::aligned_alloc<Index>(n_branch_points, SCL_ALIGNMENT);
    Real* bp_times = scl::memory::aligned_alloc<Real>(n_branch_points, SCL_ALIGNMENT);

    for (Index i = 0; i < n_branch_points; ++i) {
        sorted_bp[i] = branch_points[i];
        bp_times[i] = pseudotime[branch_points[i]];
    }

    // Insertion sort (n_branch_points usually small)
    for (Index i = 1; i < n_branch_points; ++i) {
        Index idx = sorted_bp[i];
        Real time = bp_times[i];
        Index j = i;

        while (j > 0 && bp_times[j - 1] > time) {
            sorted_bp[j] = sorted_bp[j - 1];
            bp_times[j] = bp_times[j - 1];
            --j;
        }

        sorted_bp[j] = idx;
        bp_times[j] = time;
    }

    // Parallel segment assignment
    scl::threading::parallel_for(Size(0), N, [&](size_t i) {
        Real pt = pseudotime[i];
        Index seg = 0;

        for (Index b = 0; b < n_branch_points; ++b) {
            if (pt >= bp_times[b]) {
                seg = b + 1;
            }
        }

        segment_labels[i] = seg;
    });

    scl::memory::aligned_free(bp_times, SCL_ALIGNMENT);
    scl::memory::aligned_free(sorted_bp, SCL_ALIGNMENT);
}

// =============================================================================
// Smooth Pseudotime (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void smooth_pseudotime(
    const Sparse<T, IsCSR>& adjacency,
    Array<Real> pseudotime,
    Index n_iterations = 10,
    Real alpha = Real(0.5)
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);

    Real* temp = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    Real one_minus_alpha = Real(1) - alpha;

    for (Index iter = 0; iter < n_iterations; ++iter) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            auto indices = adjacency.primary_indices_unsafe(static_cast<Index>(i));
            auto values = adjacency.primary_values_unsafe(static_cast<Index>(i));
            const Index len = adjacency.primary_length_unsafe(static_cast<Index>(i));

            if (len == 0) {
                temp[i] = pseudotime[i];
                return;
            }

            Real neighbor_sum = Real(0);
            Real weight_sum = Real(0);

            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                Real w = static_cast<Real>(values[k]);
                if (w <= Real(0)) w = Real(1);

                neighbor_sum += w * pseudotime[j];
                weight_sum += w;
            }

            Real neighbor_avg = neighbor_sum / weight_sum;
            temp[i] = one_minus_alpha * pseudotime[i] + alpha * neighbor_avg;
        });

        std::memcpy(pseudotime.ptr, temp, N * sizeof(Real));
    }

    scl::memory::aligned_free(temp, SCL_ALIGNMENT);
}

// =============================================================================
// Pseudotime Correlation (Parallel + SIMD)
// =============================================================================

template <typename T, bool IsCSR>
void pseudotime_correlation(
    const Sparse<T, IsCSR>& X,
    Array<const Real> pseudotime,
    Index n_cells,
    Index n_genes,
    Array<Real> correlations
) {
    const Size N = static_cast<Size>(n_cells);
    const Size G = static_cast<Size>(n_genes);

    SCL_CHECK_DIM(correlations.len >= G, "Pseudotime: correlations buffer too small");

    // Compute pseudotime statistics
    Real pt_mean = Real(0);
    for (Size i = 0; i < N; ++i) {
        pt_mean += pseudotime[i];
    }
    pt_mean /= static_cast<Real>(n_cells);

    Real pt_var = Real(0);
    for (Size i = 0; i < N; ++i) {
        Real d = pseudotime[i] - pt_mean;
        pt_var += d * d;
    }

    // Gene statistics
    Real* gene_means = scl::memory::aligned_alloc<Real>(G, SCL_ALIGNMENT);
    Real* gene_vars = scl::memory::aligned_alloc<Real>(G, SCL_ALIGNMENT);
    Real* covariances = scl::memory::aligned_alloc<Real>(G, SCL_ALIGNMENT);
    Index* gene_nnz = scl::memory::aligned_alloc<Index>(G, SCL_ALIGNMENT);

    scl::algo::zero(gene_means, G);
    scl::algo::zero(gene_vars, G);
    scl::algo::zero(covariances, G);
    scl::algo::zero(gene_nnz, G);

    // First pass: compute means and covariances
    for (Index c = 0; c < n_cells; ++c) {
        Real pt_dev = pseudotime[c] - pt_mean;

        auto indices = X.row_indices_unsafe(c);
        auto values = X.row_values_unsafe(c);
        const Index len = X.row_length_unsafe(c);

        for (Index k = 0; k < len; ++k) {
            Index gene = indices[k];
            if (gene < n_genes) {
                Real expr = static_cast<Real>(values[k]);
                gene_means[gene] += expr;
                covariances[gene] += pt_dev * expr;
                ++gene_nnz[gene];
            }
        }
    }

    for (Size g = 0; g < G; ++g) {
        gene_means[g] /= static_cast<Real>(n_cells);
    }

    // Second pass: compute variances
    for (Index c = 0; c < n_cells; ++c) {
        auto indices = X.row_indices_unsafe(c);
        auto values = X.row_values_unsafe(c);
        const Index len = X.row_length_unsafe(c);

        for (Index k = 0; k < len; ++k) {
            Index gene = indices[k];
            if (gene < n_genes) {
                Real expr = static_cast<Real>(values[k]);
                Real gene_dev = expr - gene_means[gene];
                gene_vars[gene] += gene_dev * gene_dev;
            }
        }
    }

    // Add contribution from zeros
    for (Size g = 0; g < G; ++g) {
        Index n_zeros = n_cells - gene_nnz[g];
        gene_vars[g] += static_cast<Real>(n_zeros) * gene_means[g] * gene_means[g];
        covariances[g] -= static_cast<Real>(n_zeros) * pt_mean * gene_means[g];
    }

    // Compute correlations in parallel
    scl::threading::parallel_for(Size(0), G, [&](size_t g) {
        Real denom = std::sqrt(pt_var * gene_vars[g]);
        correlations[g] = (denom > Real(1e-15)) ? covariances[g] / denom : Real(0);
    });

    scl::memory::aligned_free(gene_nnz, SCL_ALIGNMENT);
    scl::memory::aligned_free(covariances, SCL_ALIGNMENT);
    scl::memory::aligned_free(gene_vars, SCL_ALIGNMENT);
    scl::memory::aligned_free(gene_means, SCL_ALIGNMENT);
}

// =============================================================================
// Velocity-Weighted Pseudotime (Parallel)
// =============================================================================

template <typename T, bool IsCSR>
void velocity_weighted_pseudotime(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> initial_pseudotime,
    Array<const Real> velocity_field,
    Array<Real> refined_pseudotime,
    Index n_iterations = 20
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(refined_pseudotime.len >= N, "Pseudotime: refined buffer too small");

    std::memcpy(refined_pseudotime.ptr, initial_pseudotime.ptr, N * sizeof(Real));

    Real* temp = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    for (Index iter = 0; iter < n_iterations; ++iter) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            auto indices = adjacency.primary_indices_unsafe(static_cast<Index>(i));
            auto values = adjacency.primary_values_unsafe(static_cast<Index>(i));
            const Index len = adjacency.primary_length_unsafe(static_cast<Index>(i));

            if (len == 0) {
                temp[i] = refined_pseudotime[i];
                return;
            }

            Real weighted_sum = Real(0);
            Real weight_sum = Real(0);
            Real vel_i = velocity_field[i];

            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                Real edge_weight = static_cast<Real>(values[k]);
                if (edge_weight <= Real(0)) edge_weight = Real(1);

                Real vel_weight = Real(1) + vel_i;
                if (refined_pseudotime[j] < refined_pseudotime[i]) {
                    vel_weight = Real(1) / (Real(1) + vel_i);
                }

                Real w = edge_weight * vel_weight;
                weighted_sum += w * refined_pseudotime[j];
                weight_sum += w;
            }

            temp[i] = Real(0.5) * refined_pseudotime[i] +
                      Real(0.5) * weighted_sum / weight_sum;
        });

        std::memcpy(refined_pseudotime.ptr, temp, N * sizeof(Real));
    }

    // Renormalize
    Real min_pt = refined_pseudotime[0], max_pt = refined_pseudotime[0];
    for (Size i = 1; i < N; ++i) {
        min_pt = scl::algo::min2(min_pt, refined_pseudotime[i]);
        max_pt = scl::algo::max2(max_pt, refined_pseudotime[i]);
    }

    Real range = max_pt - min_pt;
    if (range > Real(1e-15)) {
        scl::threading::parallel_for(Size(0), N, [&](size_t i) {
            refined_pseudotime[i] = (refined_pseudotime[i] - min_pt) / range;
        });
    }

    scl::memory::aligned_free(temp, SCL_ALIGNMENT);
}

// =============================================================================
// Find Terminal States
// =============================================================================

template <typename T, bool IsCSR>
Index find_terminal_states(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> pseudotime,
    Array<Index> terminal_cells,
    Real percentile = Real(0.95)
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);

    // Find threshold via partial sort
    Real* sorted = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);
    std::memcpy(sorted, pseudotime.ptr, N * sizeof(Real));

    // Quick select for percentile
    Index threshold_idx = static_cast<Index>(percentile * static_cast<Real>(n - 1));
    
    // Simple approach: full sort for correctness
    for (Size i = 1; i < N; ++i) {
        Real val = sorted[i];
        Size j = i;
        while (j > 0 && sorted[j - 1] > val) {
            sorted[j] = sorted[j - 1];
            --j;
        }
        sorted[j] = val;
    }

    Real threshold = sorted[threshold_idx];
    scl::memory::aligned_free(sorted, SCL_ALIGNMENT);

    // Find cells above threshold
    Index count = 0;
    for (Index i = 0; i < n; ++i) {
        if (pseudotime[i] >= threshold && count < static_cast<Index>(terminal_cells.len)) {
            terminal_cells[count++] = i;
        }
    }

    return count;
}

// =============================================================================
// Compute Trajectory Backbone
// =============================================================================

template <typename T, bool IsCSR>
Index compute_backbone(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Real> pseudotime,
    Index n_backbone_cells,
    Array<Index> backbone_indices
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(backbone_indices.len >= static_cast<Size>(n_backbone_cells),
                  "Pseudotime: backbone_indices buffer too small");

    // Sort indices by pseudotime
    Index* sorted_idx = scl::memory::aligned_alloc<Index>(N, SCL_ALIGNMENT);
    
    for (Size i = 0; i < N; ++i) {
        sorted_idx[i] = static_cast<Index>(i);
    }

    // Sort using pseudotime as key
    for (Size i = 1; i < N; ++i) {
        Index idx = sorted_idx[i];
        Real pt = pseudotime[idx];
        Size j = i;

        while (j > 0 && pseudotime[sorted_idx[j - 1]] > pt) {
            sorted_idx[j] = sorted_idx[j - 1];
            --j;
        }

        sorted_idx[j] = idx;
    }

    // Select uniformly spaced cells
    Index actual_n = scl::algo::min2(n_backbone_cells, n);
    
    if (actual_n == 1) {
        backbone_indices[0] = sorted_idx[N / 2];
    } else {
        for (Index i = 0; i < actual_n; ++i) {
            Size sample_idx = static_cast<Size>(i) * (N - 1) / static_cast<Size>(actual_n - 1);
            backbone_indices[i] = sorted_idx[sample_idx];
        }
    }

    scl::memory::aligned_free(sorted_idx, SCL_ALIGNMENT);
    return actual_n;
}

// =============================================================================
// Generic Pseudotime Computation
// =============================================================================

template <typename T, bool IsCSR>
void compute_pseudotime(
    const Sparse<T, IsCSR>& adjacency,
    Index root_cell,
    Array<Real> pseudotime,
    PseudotimeMethod method = PseudotimeMethod::DiffusionPseudotime,
    Index n_dcs = config::DEFAULT_N_DCS
) {
    switch (method) {
        case PseudotimeMethod::DiffusionPseudotime:
            diffusion_pseudotime(adjacency, root_cell, pseudotime, n_dcs);
            break;
        case PseudotimeMethod::ShortestPath:
        case PseudotimeMethod::GraphDistance:
        default:
            graph_pseudotime(adjacency, root_cell, pseudotime);
            break;
    }
}

} // namespace scl::kernel::pseudotime
