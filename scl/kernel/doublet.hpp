#pragma once

#include "scl/core/type.hpp"
#include "scl/core/simd.hpp"
#include "scl/core/sparse.hpp"
#include "scl/core/error.hpp"
#include "scl/core/macros.hpp"
#include "scl/core/memory.hpp"
#include "scl/core/algo.hpp"
#include "scl/core/vectorize.hpp"
#include "scl/core/sort.hpp"
#include "scl/threading/parallel_for.hpp"
#include "scl/threading/workspace.hpp"
#include "scl/threading/scheduler.hpp"

#include <cmath>
#include <cstring>

// =============================================================================
// FILE: scl/kernel/doublet.hpp
// BRIEF: Doublet detection for single-cell RNA-seq data (Scrublet/DoubletFinder style)
// =============================================================================

namespace scl::kernel::doublet {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_DOUBLET_RATE = Real(0.06);
    constexpr Real DEFAULT_THRESHOLD = Real(0.5);
    constexpr Index DEFAULT_N_NEIGHBORS = 30;
    constexpr Index DEFAULT_N_SIMULATED = 0;  // 0 = auto (2x n_cells)
    constexpr Real MIN_SCORE = Real(1e-10);
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size PREFETCH_DISTANCE = 16;
}

// =============================================================================
// Doublet Detection Methods
// =============================================================================

enum class DoubletMethod {
    Scrublet,
    DoubletFinder,
    Hybrid
};

// =============================================================================
// Internal Helpers
// =============================================================================

namespace detail {

// Xoshiro256++ PRNG (higher quality, consistent with enrichment.hpp)
struct Xoshiro256pp {
    alignas(32) std::array<uint64_t, 4> s{};

    [[nodiscard]] SCL_FORCE_INLINE explicit Xoshiro256pp(uint64_t seed) noexcept {
        uint64_t z = seed;
        for (int i = 0; i < 4; ++i) {
            z += 0x9e3779b97f4a7c15ULL;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            s[i] = z ^ (z >> 31);
        }
    }

    [[nodiscard]] SCL_FORCE_INLINE uint64_t rotl(uint64_t x, int k) const noexcept {
        return (x << k) | (x >> (64 - k));
    }

    SCL_FORCE_INLINE uint64_t next() noexcept {
        const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
        const uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        return result;
    }

    // Lemire's nearly divisionless bounded random
    SCL_FORCE_INLINE Size bounded(Size n) noexcept {
        uint64_t x = next();
#if defined(__SIZEOF_INT128__) && defined(__GNUC__)
        auto m = static_cast<uint64_t>((static_cast<__uint128_t>(x) * static_cast<__uint128_t>(n)) >> 64);
        while (static_cast<__uint128_t>(m) * static_cast<__uint128_t>(n) < static_cast<__uint128_t>(x)) {
            x = next();
            m = static_cast<uint64_t>((static_cast<__uint128_t>(x) * static_cast<__uint128_t>(n)) >> 64);
        }
        return static_cast<Size>(m);
#else
        return static_cast<Size>(x % static_cast<uint64_t>(n));
#endif
    }

    SCL_FORCE_INLINE Real uniform() noexcept {
        return static_cast<Real>(next() >> 11) * Real(0x1.0p-53);
    }
};

// Backward compatibility alias
using FastRNG = Xoshiro256pp;

// SIMD-optimized squared Euclidean distance with 4-way unrolling
SCL_FORCE_INLINE SCL_HOT Real squared_distance(
    const Real* SCL_RESTRICT a,
    const Real* SCL_RESTRICT b,
    Index dim
) {
    namespace s = scl::simd;
    const auto d = s::SimdTagFor<Real>();
    const size_t lanes = s::Lanes(d);

    // Multi-accumulator pattern for FMA latency hiding
    auto acc0 = s::Zero(d);
    auto acc1 = s::Zero(d);
    auto acc2 = s::Zero(d);
    auto acc3 = s::Zero(d);

    Size k = 0;
    const Size dim_sz = static_cast<Size>(dim);

    // 4-way unrolled SIMD loop with prefetch
    for (; k + 4 * lanes <= dim_sz; k += 4 * lanes) {
        if (SCL_LIKELY(k + config::PREFETCH_DISTANCE * lanes < dim_sz)) {
            SCL_PREFETCH_READ(a + k + config::PREFETCH_DISTANCE * lanes, 0);
            SCL_PREFETCH_READ(b + k + config::PREFETCH_DISTANCE * lanes, 0);
        }

        auto v_a0 = s::Load(d, a + k);
        auto v_b0 = s::Load(d, b + k);
        auto v_d0 = s::Sub(v_a0, v_b0);
        acc0 = s::MulAdd(v_d0, v_d0, acc0);

        auto v_a1 = s::Load(d, a + k + lanes);
        auto v_b1 = s::Load(d, b + k + lanes);
        auto v_d1 = s::Sub(v_a1, v_b1);
        acc1 = s::MulAdd(v_d1, v_d1, acc1);

        auto v_a2 = s::Load(d, a + k + 2 * lanes);
        auto v_b2 = s::Load(d, b + k + 2 * lanes);
        auto v_d2 = s::Sub(v_a2, v_b2);
        acc2 = s::MulAdd(v_d2, v_d2, acc2);

        auto v_a3 = s::Load(d, a + k + 3 * lanes);
        auto v_b3 = s::Load(d, b + k + 3 * lanes);
        auto v_d3 = s::Sub(v_a3, v_b3);
        acc3 = s::MulAdd(v_d3, v_d3, acc3);
    }

    // Combine accumulators
    acc0 = s::Add(acc0, acc1);
    acc2 = s::Add(acc2, acc3);
    acc0 = s::Add(acc0, acc2);

    // Single-lane SIMD cleanup
    for (; k + lanes <= dim_sz; k += lanes) {
        auto v_a = s::Load(d, a + k);
        auto v_b = s::Load(d, b + k);
        auto v_d = s::Sub(v_a, v_b);
        acc0 = s::MulAdd(v_d, v_d, acc0);
    }

    Real dist_sq = s::GetLane(s::SumOfLanes(d, acc0));

    // Scalar cleanup
    for (; k < dim_sz; ++k) {
        Real diff = a[k] - b[k];
        dist_sq += diff * diff;
    }

    return dist_sq;
}

// Heap-based top-k selection (O(n log k) vs O(n*k))
struct HeapElement {
    Real distance;
    Index index;

    SCL_FORCE_INLINE bool operator>(const HeapElement& other) const noexcept {
        return distance > other.distance;
    }
};

// Max-heap sift down
SCL_FORCE_INLINE void heap_sift_down(HeapElement* heap, Index heap_size, Index i) {
    while (true) {
        Index largest = i;
        Index left = 2 * i + 1;
        Index right = 2 * i + 2;

        if (left < heap_size && heap[left] > heap[largest]) {
            largest = left;
        }
        if (right < heap_size && heap[right] > heap[largest]) {
            largest = right;
        }
        if (largest == i) break;

        HeapElement tmp = heap[i];
        heap[i] = heap[largest];
        heap[largest] = tmp;
        i = largest;
    }
}

// Build max-heap
SCL_FORCE_INLINE void build_max_heap(HeapElement* heap, Index k) {
    for (Index i = k / 2 - 1; i >= 0; --i) {
        heap_sift_down(heap, k, i);
    }
}

// Heap-based partial sort for k smallest elements
SCL_HOT void partial_sort_k_smallest(
    const Real* SCL_RESTRICT distances,
    Index n,
    Index k,
    Index* SCL_RESTRICT indices
) {
    if (SCL_UNLIKELY(k == 0 || n == 0)) return;
    k = scl::algo::min2(k, n);

    // PERFORMANCE: RAII memory management with unique_ptr
    // Using aligned_alloc returns unique_ptr for automatic cleanup
    auto heap_ptr = scl::memory::aligned_alloc<HeapElement>(static_cast<Size>(k), SCL_ALIGNMENT);
    HeapElement* heap = heap_ptr.get();

    // Initialize heap with first k elements
    for (Index i = 0; i < k; ++i) {
        heap[i].distance = distances[i];
        heap[i].index = i;
    }
    build_max_heap(heap, k);

    // Process remaining elements
    for (Index i = k; i < n; ++i) {
        if (SCL_LIKELY(distances[i] < heap[0].distance)) {
            heap[0].distance = distances[i];
            heap[0].index = i;
            heap_sift_down(heap, k, 0);
        }
    }

    // Sort heap for consistent output order
    for (Index i = k - 1; i > 0; --i) {
        HeapElement tmp = heap[0];
        heap[0] = heap[i];
        heap[i] = tmp;
        heap_sift_down(heap, i, 0);
    }

    // Extract indices
    for (Index i = 0; i < k; ++i) {
        indices[i] = heap[i].index;
    }

    // unique_ptr automatically frees memory when going out of scope
}

} // namespace detail

// =============================================================================
// Simulate Doublets by Averaging Random Cell Pairs
// =============================================================================

template <typename T, bool IsCSR>
void simulate_doublets(
    const Sparse<T, IsCSR>& X,
    Index n_cells,
    Index n_genes,
    Index n_doublets,
    Real* doublet_profiles,  // n_doublets x n_genes, row-major
    uint64_t seed = 42
) {
    SCL_CHECK_ARG(IsCSR, "Doublet: requires CSR format (cells x genes)");

    const bool use_parallel = n_doublets >= static_cast<Index>(config::PARALLEL_THRESHOLD);

    // Initialize doublet profiles to zero
    Size total = static_cast<Size>(n_doublets) * static_cast<Size>(n_genes);
    scl::memory::zero(Array<Real>(doublet_profiles, total));

    auto simulate_one = [&](Index d, uint64_t local_seed) {
        detail::Xoshiro256pp rng(local_seed);

        // Select two random cells
        auto cell1 = static_cast<Index>(rng.bounded(static_cast<Size>(n_cells)));
        auto cell2 = static_cast<Index>(rng.bounded(static_cast<Size>(n_cells)));
        while (SCL_UNLIKELY(cell2 == cell1 && n_cells > 1)) {
            cell2 = static_cast<Index>(rng.bounded(static_cast<Size>(n_cells)));
        }

        Real* profile = doublet_profiles + static_cast<Size>(d) * n_genes;

        // Add cell1's expression (4-way unrolled)
        if constexpr (IsCSR) {
            auto indices1 = X.row_indices_unsafe(cell1);
            auto values1 = X.row_values_unsafe(cell1);
            Index len1 = X.row_length_unsafe(cell1);

            Index k = 0;
            for (; k + 4 <= len1; k += 4) {
                Index g0 = indices1.ptr[k];
                Index g1 = indices1.ptr[k + 1];
                Index g2 = indices1.ptr[k + 2];
                Index g3 = indices1.ptr[k + 3];
                if (SCL_LIKELY(g0 < n_genes)) profile[g0] += static_cast<Real>(values1.ptr[k]) * Real(0.5);
                if (SCL_LIKELY(g1 < n_genes)) profile[g1] += static_cast<Real>(values1.ptr[k + 1]) * Real(0.5);
                if (SCL_LIKELY(g2 < n_genes)) profile[g2] += static_cast<Real>(values1.ptr[k + 2]) * Real(0.5);
                if (SCL_LIKELY(g3 < n_genes)) profile[g3] += static_cast<Real>(values1.ptr[k + 3]) * Real(0.5);
            }
            for (; k < len1; ++k) {
                Index gene = indices1.ptr[k];
                if (SCL_LIKELY(gene < n_genes)) {
                    profile[gene] += static_cast<Real>(values1.ptr[k]) * Real(0.5);
                }
            }
        }

        // Add cell2's expression (4-way unrolled)
        if constexpr (IsCSR) {
            auto indices2 = X.row_indices_unsafe(cell2);
            auto values2 = X.row_values_unsafe(cell2);
            Index len2 = X.row_length_unsafe(cell2);

            Index k = 0;
            for (; k + 4 <= len2; k += 4) {
                Index g0 = indices2.ptr[k];
                Index g1 = indices2.ptr[k + 1];
                Index g2 = indices2.ptr[k + 2];
                Index g3 = indices2.ptr[k + 3];
                if (SCL_LIKELY(g0 < n_genes)) profile[g0] += static_cast<Real>(values2.ptr[k]) * Real(0.5);
                if (SCL_LIKELY(g1 < n_genes)) profile[g1] += static_cast<Real>(values2.ptr[k + 1]) * Real(0.5);
                if (SCL_LIKELY(g2 < n_genes)) profile[g2] += static_cast<Real>(values2.ptr[k + 2]) * Real(0.5);
                if (SCL_LIKELY(g3 < n_genes)) profile[g3] += static_cast<Real>(values2.ptr[k + 3]) * Real(0.5);
            }
            for (; k < len2; ++k) {
                Index gene = indices2.ptr[k];
                if (SCL_LIKELY(gene < n_genes)) {
                    profile[gene] += static_cast<Real>(values2.ptr[k]) * Real(0.5);
                }
            }
        }
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_doublets),
            [&](auto d) {
                auto local_seed = seed + static_cast<uint64_t>(d) * 0x9e3779b97f4a7c15ULL;
                simulate_one(static_cast<Index>(d), local_seed);
            });
    } else {
        for (Index d = 0; d < n_doublets; ++d) {
            simulate_one(d, seed + static_cast<uint64_t>(d) * 0x9e3779b97f4a7c15ULL);
        }
    }
}

// =============================================================================
// Convert Sparse Cell to Dense Vector
// =============================================================================

template <typename T, bool IsCSR>
SCL_FORCE_INLINE void sparse_to_dense_row(
    const Sparse<T, IsCSR>& X,
    Index cell,
    Index n_genes,
    Real* dense
) {
    scl::memory::zero(Array<Real>(dense, static_cast<Size>(n_genes)));

    if constexpr (IsCSR) {
        auto indices = X.row_indices_unsafe(cell);
        auto values = X.row_values_unsafe(cell);
        Index len = X.row_length_unsafe(cell);

        // 4-way unrolled scatter
        Index k = 0;
        for (; k + 4 <= len; k += 4) {
            Index g0 = indices[k];
            Index g1 = indices[k + 1];
            Index g2 = indices[k + 2];
            Index g3 = indices[k + 3];
            if (SCL_LIKELY(g0 < n_genes)) dense[g0] = static_cast<Real>(values[k]);
            if (SCL_LIKELY(g1 < n_genes)) dense[g1] = static_cast<Real>(values[k + 1]);
            if (SCL_LIKELY(g2 < n_genes)) dense[g2] = static_cast<Real>(values[k + 2]);
            if (SCL_LIKELY(g3 < n_genes)) dense[g3] = static_cast<Real>(values[k + 3]);
        }

        for (; k < len; ++k) {
            Index gene = indices[k];
            if (SCL_LIKELY(gene < n_genes)) {
                dense[gene] = static_cast<Real>(values[k]);
            }
        }
    }
}

// =============================================================================
// Compute k-NN for Each Cell Against Observed + Simulated (Parallel with WorkspacePool)
// =============================================================================

template <typename T, bool IsCSR>
void compute_knn_doublet_scores(
    const Sparse<T, IsCSR>& X,
    Index n_cells,
    Index n_genes,
    const Real* doublet_profiles,
    Index n_doublets,
    Index k_neighbors,
    Array<Real> doublet_scores
) {
    SCL_CHECK_DIM(doublet_scores.len >= static_cast<Size>(n_cells),
                  "Doublet: scores buffer too small");

    Index n_total = n_cells + n_doublets;
    const bool use_parallel = n_cells >= static_cast<Index>(config::PARALLEL_THRESHOLD);
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    // Workspace size per thread: query + neighbor + distances + knn_indices
    const Size workspace_per_thread =
        static_cast<Size>(n_genes) * sizeof(Real) +          // query
        static_cast<Size>(n_genes) * sizeof(Real) +          // neighbor
        static_cast<Size>(n_total) * sizeof(Real) +          // distances
        static_cast<Size>(k_neighbors) * sizeof(Index);      // knn_indices

    scl::threading::WorkspacePool<char> workspace_pool;
    if (use_parallel) {
        workspace_pool.init(n_threads, workspace_per_thread);
    } else {
        workspace_pool.init(1, workspace_per_thread);
    }

    auto process_cell = [&](Index i, void* workspace) {
        char* ws = static_cast<char*>(workspace);
        Real* query = reinterpret_cast<Real*>(ws);
        Real* neighbor = query + n_genes;
        Real* distances = neighbor + n_genes;
        auto knn_indices = reinterpret_cast<Index*>(distances + n_total);

        // Convert query cell to dense
        sparse_to_dense_row(X, i, n_genes, query);

        // Compute distances to all observed cells
        for (Index j = 0; j < n_cells; ++j) {
            if (SCL_UNLIKELY(j == i)) {
                distances[j] = Real(1e30);  // Exclude self
            } else {
                sparse_to_dense_row(X, j, n_genes, neighbor);
                distances[j] = detail::squared_distance(query, neighbor, n_genes);
            }
        }

        // Compute distances to all simulated doublets
        for (Index j = 0; j < n_doublets; ++j) {
            const Real* doublet = doublet_profiles + static_cast<Size>(j) * n_genes;
            distances[n_cells + j] = detail::squared_distance(query, doublet, n_genes);
        }

        // Find k nearest neighbors using heap-based selection
        detail::partial_sort_k_smallest(distances, n_total, k_neighbors, knn_indices);

        // Count doublet neighbors
        Index doublet_neighbors = 0;
        for (Index k = 0; k < k_neighbors; ++k) {
            if (SCL_LIKELY(knn_indices[k] >= n_cells)) {
                ++doublet_neighbors;
            }
        }

        doublet_scores[i] = static_cast<Real>(doublet_neighbors) /
                           static_cast<Real>(k_neighbors);
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_cells),
            [&](size_t i, size_t thread_rank) {
                process_cell(static_cast<Index>(i), workspace_pool.get(thread_rank));
            });
    } else {
        for (Index i = 0; i < n_cells; ++i) {
            process_cell(i, workspace_pool.get(0));
        }
    }
}

// =============================================================================
// Compute Doublet Scores on PCA-Reduced Data (Parallel with WorkspacePool)
// =============================================================================

inline void compute_knn_doublet_scores_pca(
    const Real* cell_embeddings,     // n_cells x n_dims
    Index n_cells,
    Index n_dims,
    const Real* doublet_embeddings,  // n_doublets x n_dims
    Index n_doublets,
    Index k_neighbors,
    Array<Real> doublet_scores
) {
    SCL_CHECK_DIM(doublet_scores.len >= static_cast<Size>(n_cells),
                  "Doublet: scores buffer too small");

    Index n_total = n_cells + n_doublets;
    const bool use_parallel = n_cells >= static_cast<Index>(config::PARALLEL_THRESHOLD);
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();

    const Size workspace_per_thread =
        static_cast<Size>(n_total) * sizeof(Real) +
        static_cast<Size>(k_neighbors) * sizeof(Index);

    scl::threading::WorkspacePool<char> workspace_pool;
    if (use_parallel) {
        workspace_pool.init(n_threads, workspace_per_thread);
    } else {
        workspace_pool.init(1, workspace_per_thread);
    }

    auto process_cell = [&](Index i, void* workspace) {
        char* ws = static_cast<char*>(workspace);
        Real* distances = reinterpret_cast<Real*>(ws);
        auto knn_indices = reinterpret_cast<Index*>(distances + n_total);

        const Real* query = cell_embeddings + static_cast<Size>(i) * n_dims;

        // Distances to observed cells
        for (Index j = 0; j < n_cells; ++j) {
            if (SCL_UNLIKELY(j == i)) {
                distances[j] = Real(1e30);
            } else {
                const Real* other = cell_embeddings + static_cast<Size>(j) * n_dims;
                distances[j] = detail::squared_distance(query, other, n_dims);
            }
        }

        // Distances to simulated doublets
        for (Index j = 0; j < n_doublets; ++j) {
            const Real* doublet = doublet_embeddings + static_cast<Size>(j) * n_dims;
            distances[n_cells + j] = detail::squared_distance(query, doublet, n_dims);
        }

        // Find k-NN
        detail::partial_sort_k_smallest(distances, n_total, k_neighbors, knn_indices);

        // Count doublet neighbors
        Index doublet_neighbors = 0;
        for (Index k = 0; k < k_neighbors; ++k) {
            if (SCL_LIKELY(knn_indices[k] >= n_cells)) {
                ++doublet_neighbors;
            }
        }

        doublet_scores[i] = static_cast<Real>(doublet_neighbors) /
                           static_cast<Real>(k_neighbors);
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_cells),
            [&](size_t i, size_t thread_rank) {
                process_cell(static_cast<Index>(i), workspace_pool.get(thread_rank));
            });
    } else {
        for (Index i = 0; i < n_cells; ++i) {
            process_cell(i, workspace_pool.get(0));
        }
    }
}

// =============================================================================
// Scrublet-Style Doublet Detection
// =============================================================================

template <typename T, bool IsCSR>
void scrublet_scores(
    const Sparse<T, IsCSR>& X,
    Index n_cells,
    Index n_genes,
    Array<Real> scores,
    Index n_simulated = 0,
    Index k_neighbors = config::DEFAULT_N_NEIGHBORS,
    uint64_t seed = 42
) {
    SCL_CHECK_DIM(scores.len >= static_cast<Size>(n_cells),
                  "Scrublet: scores buffer too small");

    // Auto-determine number of simulated doublets
    Index n_doublets = (n_simulated > 0) ? n_simulated : n_cells * 2;

    // Simulate doublets
    Size doublet_size = static_cast<Size>(n_doublets) * static_cast<Size>(n_genes);
    // PERFORMANCE: RAII memory management with unique_ptr
    auto doublet_profiles_ptr = scl::memory::aligned_alloc<Real>(doublet_size, SCL_ALIGNMENT);
    Real* doublet_profiles = doublet_profiles_ptr.get();

    simulate_doublets(X, n_cells, n_genes, n_doublets, doublet_profiles, seed);

    // Compute k-NN based scores
    compute_knn_doublet_scores(
        X, n_cells, n_genes, doublet_profiles, n_doublets, k_neighbors, scores
    );

    // unique_ptr automatically frees memory when going out of scope
}

// =============================================================================
// DoubletFinder-Style pANN Score
// =============================================================================

inline void doubletfinder_pann(
    const Real* cell_embeddings,
    Index n_cells,
    Index n_dims,
    const Real* doublet_embeddings,
    Index n_doublets,
    Real pK,  // proportion of k to use
    Array<Real> pann_scores
) {
    SCL_CHECK_DIM(pann_scores.len >= static_cast<Size>(n_cells),
                  "DoubletFinder: pann_scores buffer too small");

    // Number of neighbors based on pK
    auto k = static_cast<Index>(std::ceil(pK * static_cast<Real>(n_cells + n_doublets)));
    k = scl::algo::max2(k, Index(1));

    compute_knn_doublet_scores_pca(
        cell_embeddings, n_cells, n_dims,
        doublet_embeddings, n_doublets, k, pann_scores
    );
}

// =============================================================================
// Estimate Threshold from Score Distribution (O(n log n) sort)
// =============================================================================

inline Real estimate_threshold(
    Array<const Real> scores,
    Real expected_doublet_rate
) {
    Size n = scores.len;
    if (SCL_UNLIKELY(n == 0)) return config::DEFAULT_THRESHOLD;

    // Copy and sort using efficient SIMD sort
    // PERFORMANCE: RAII memory management with unique_ptr
    auto sorted_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* sorted = sorted_ptr.get();
    scl::memory::copy_fast(
        Array<const Real>(scores.ptr, n),
        Array<Real>(sorted, n)
    );

    // Use scl::sort::sort for O(n log n) sorting
    scl::sort::sort(Array<Real>(sorted, n));

    // Find threshold at (1 - expected_doublet_rate) percentile
    Size threshold_idx = static_cast<Size>(
        (Real(1) - expected_doublet_rate) * static_cast<Real>(n - 1)
    );
    threshold_idx = scl::algo::min2(threshold_idx, n - 1);

    Real threshold = sorted[threshold_idx];

    // unique_ptr automatically frees memory when going out of scope
    return threshold;
}

// =============================================================================
// Call Doublets Based on Score Threshold
// =============================================================================

inline Index call_doublets(
    Array<const Real> scores,
    Real threshold,
    Array<bool> is_doublet
) {
    SCL_CHECK_DIM(is_doublet.len >= scores.len, "Doublet: output buffer too small");

    Index count = 0;
    Size n = scores.len;

    // 4-way unrolled for better ILP
    Size k = 0;
    for (; k + 4 <= n; k += 4) {
        bool d0 = scores[static_cast<Index>(k)] > threshold;
        bool d1 = scores[static_cast<Index>(k + 1)] > threshold;
        bool d2 = scores[static_cast<Index>(k + 2)] > threshold;
        bool d3 = scores[static_cast<Index>(k + 3)] > threshold;
        is_doublet[static_cast<Index>(k)] = d0;
        is_doublet[static_cast<Index>(k + 1)] = d1;
        is_doublet[static_cast<Index>(k + 2)] = d2;
        is_doublet[static_cast<Index>(k + 3)] = d3;
        count += static_cast<Index>(d0) + static_cast<Index>(d1) +
                 static_cast<Index>(d2) + static_cast<Index>(d3);
    }

    for (; k < n; ++k) {
        is_doublet[static_cast<Index>(k)] = scores[static_cast<Index>(k)] > threshold;
        if (is_doublet[static_cast<Index>(k)]) ++count;
    }

    return count;
}

// =============================================================================
// Bimodal Score Detection (Histogram-Based)
// =============================================================================

inline Real detect_bimodal_threshold(
    Array<const Real> scores,
    Index n_bins = 50
) {
    Size n = scores.len;
    if (SCL_UNLIKELY(n < 10)) return config::DEFAULT_THRESHOLD;

    // Find score range with SIMD min/max
    Real min_score = Real(0);
    Real max_score = Real(0);
    namespace s = scl::simd;
    const auto d = s::SimdTagFor<Real>();
    const auto lanes = s::Lanes(d);

    if (n >= lanes) {
        auto v_min = s::Set(d, scores[0]);
        auto v_max = s::Set(d, scores[0]);
        Size i = 0;

        for (; i + lanes <= n; i += lanes) {
            auto v = s::Load(d, scores.ptr + i);
            v_min = s::Min(v_min, v);
            v_max = s::Max(v_max, v);
        }

        min_score = s::GetLane(s::MinOfLanes(d, v_min));
        max_score = s::GetLane(s::MaxOfLanes(d, v_max));

        for (; i < n; ++i) {
            min_score = scl::algo::min2(min_score, scores[static_cast<Index>(i)]);
            max_score = scl::algo::max2(max_score, scores[static_cast<Index>(i)]);
        }
    } else {
        min_score = scores[0];
        max_score = scores[0];
        for (Size i = 1; i < n; ++i) {
            min_score = scl::algo::min2(min_score, scores[static_cast<Index>(i)]);
            max_score = scl::algo::max2(max_score, scores[static_cast<Index>(i)]);
        }
    }

    Real range = max_score - min_score;
    if (range < config::MIN_SCORE) return config::DEFAULT_THRESHOLD;

    // Build histogram
    // PERFORMANCE: RAII memory management with unique_ptr
    auto hist_ptr = scl::memory::aligned_alloc<Index>(static_cast<Size>(n_bins), SCL_ALIGNMENT);
    Index* hist = hist_ptr.get();
    scl::algo::zero(hist, static_cast<Size>(n_bins));

    Real bin_width = range / static_cast<Real>(n_bins);
    Real inv_bin_width = Real(1) / bin_width;

    // 4-way unrolled histogram building
    Size i = 0;
    for (; i + 4 <= n; i += 4) {
        auto b0 = static_cast<Index>((scores[static_cast<Index>(i)] - min_score) * inv_bin_width);
        auto b1 = static_cast<Index>((scores[static_cast<Index>(i + 1)] - min_score) * inv_bin_width);
        auto b2 = static_cast<Index>((scores[static_cast<Index>(i + 2)] - min_score) * inv_bin_width);
        auto b3 = static_cast<Index>((scores[static_cast<Index>(i + 3)] - min_score) * inv_bin_width);
        b0 = scl::algo::min2(b0, n_bins - 1);
        b1 = scl::algo::min2(b1, n_bins - 1);
        b2 = scl::algo::min2(b2, n_bins - 1);
        b3 = scl::algo::min2(b3, n_bins - 1);
        ++hist[b0];
        ++hist[b1];
        ++hist[b2];
        ++hist[b3];
    }

    for (; i < n; ++i) {
        auto bin = static_cast<Index>((scores[static_cast<Index>(i)] - min_score) * inv_bin_width);
        bin = scl::algo::min2(bin, n_bins - 1);
        ++hist[bin];
    }

    // Find valley between two peaks (simple approach)
    // Look for local minimum after first peak
    Index first_peak = 0;
    Index first_peak_val = hist[0];
    for (Index b = 1; b < n_bins / 2; ++b) {
        if (hist[b] > first_peak_val) {
            first_peak = b;
            first_peak_val = hist[b];
        }
    }

    // Find minimum after first peak
    Index valley = first_peak;
    Index valley_val = hist[first_peak];
    for (Index b = first_peak + 1; b < n_bins; ++b) {
        if (hist[b] < valley_val) {
            valley = b;
            valley_val = hist[b];
        } else if (hist[b] > valley_val * 2) {
            // Found second peak, stop at valley
            break;
        }
    }

    Real threshold = min_score + (static_cast<Real>(valley) + Real(0.5)) * bin_width;

    // unique_ptr automatically frees memory when going out of scope
    return threshold;
}

// =============================================================================
// Expected Number of Doublets
// =============================================================================

inline Index expected_doublets(
    Index n_cells,
    Real doublet_rate = config::DEFAULT_DOUBLET_RATE
) {
    return static_cast<Index>(std::round(static_cast<Real>(n_cells) * doublet_rate));
}

// =============================================================================
// Doublet Rate Estimation from Loading
// =============================================================================

inline Real estimate_doublet_rate(
    Index n_cells_loaded,
    Real cells_per_droplet_mean = Real(0.05)
) {
    // Simple Poisson-based estimate
    Real lambda = static_cast<Real>(n_cells_loaded) / Real(10000);
    return Real(2) * lambda * cells_per_droplet_mean;
}

// =============================================================================
// Heterotypic vs Homotypic Doublet Classification
// =============================================================================

inline void classify_doublet_types(
    Array<const Index> cluster_labels,
    Array<const bool> is_doublet,
    Array<Index> doublet_type  // 0=singlet, 1=heterotypic, 2=homotypic (heuristic)
) {
    Size n = cluster_labels.len;
    SCL_CHECK_DIM(is_doublet.len >= n, "Doublet: is_doublet buffer too small");
    SCL_CHECK_DIM(doublet_type.len >= n, "Doublet: doublet_type buffer too small");

    for (Size i = 0; i < n; ++i) {
        if (!is_doublet[static_cast<Index>(i)]) {
            doublet_type[static_cast<Index>(i)] = 0;  // Singlet
        } else {
            // Heuristic: mark all as potentially heterotypic
            doublet_type[static_cast<Index>(i)] = 1;
        }
    }
}

// =============================================================================
// Neighbor-Based Doublet Type Classification
// =============================================================================

template <typename T, bool IsCSR>
void classify_doublet_types_knn(
    const Sparse<T, IsCSR>& knn_graph,
    Array<const Index> cluster_labels,
    Array<const bool> is_doublet,
    Index n_clusters,
    Array<Index> doublet_type  // 0=singlet, 1=heterotypic, 2=homotypic
) {
    const Index n = knn_graph.primary_dim();

    SCL_CHECK_DIM(cluster_labels.len >= static_cast<Size>(n),
                  "Doublet: cluster_labels buffer too small");
    SCL_CHECK_DIM(is_doublet.len >= static_cast<Size>(n),
                  "Doublet: is_doublet buffer too small");
    SCL_CHECK_DIM(doublet_type.len >= static_cast<Size>(n),
                  "Doublet: doublet_type buffer too small");

    const bool use_parallel = n >= static_cast<Index>(config::PARALLEL_THRESHOLD);
    const size_t n_threads = scl::threading::Scheduler::get_num_threads();
    const Size workspace_size = static_cast<Size>(n_clusters) * sizeof(Index);

    scl::threading::WorkspacePool<char> workspace_pool;
    if (use_parallel) {
        workspace_pool.init(n_threads, workspace_size);
    } else {
        workspace_pool.init(1, workspace_size);
    }

    auto process_cell = [&](Index i, Index* neighbor_cluster_counts) {
        if (SCL_UNLIKELY(!is_doublet[i])) {
            doublet_type[i] = 0;  // Singlet
            return;
        }

        // Count neighbor cluster memberships
        scl::memory::zero(Array<Index>(neighbor_cluster_counts, static_cast<Size>(n_clusters)));

        auto indices = knn_graph.primary_indices_unsafe(i);
        Index len = knn_graph.primary_length_unsafe(i);

        // 4-way unrolled counting
        Index k = 0;
        for (; k + 4 <= len; k += 4) {
            Index c0 = cluster_labels[indices[k]];
            Index c1 = cluster_labels[indices[k + 1]];
            Index c2 = cluster_labels[indices[k + 2]];
            Index c3 = cluster_labels[indices[k + 3]];
            if (SCL_LIKELY(c0 >= 0 && c0 < n_clusters)) ++neighbor_cluster_counts[c0];
            if (SCL_LIKELY(c1 >= 0 && c1 < n_clusters)) ++neighbor_cluster_counts[c1];
            if (SCL_LIKELY(c2 >= 0 && c2 < n_clusters)) ++neighbor_cluster_counts[c2];
            if (SCL_LIKELY(c3 >= 0 && c3 < n_clusters)) ++neighbor_cluster_counts[c3];
        }

        for (; k < len; ++k) {
            Index neighbor = indices[k];
            Index cluster = cluster_labels[neighbor];
            if (SCL_LIKELY(cluster >= 0 && cluster < n_clusters)) {
                ++neighbor_cluster_counts[cluster];
            }
        }

        // Find top two clusters among neighbors
        Index top1_cluster = 0, top2_cluster = 0;
        Index top1_count = 0, top2_count = 0;

        for (Index c = 0; c < n_clusters; ++c) {
            if (neighbor_cluster_counts[c] > top1_count) {
                top2_cluster = top1_cluster;
                top2_count = top1_count;
                top1_cluster = c;
                top1_count = neighbor_cluster_counts[c];
            } else if (neighbor_cluster_counts[c] > top2_count) {
                top2_cluster = c;
                top2_count = neighbor_cluster_counts[c];
            }
        }

        // If significant neighbors from two different clusters, it's heterotypic
        Real total = static_cast<Real>(len);
        Real top2_frac = static_cast<Real>(top2_count) / total;

        doublet_type[i] = (top2_frac > Real(0.2) && top1_cluster != top2_cluster) ? 1 : 2;
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n),
            [&](size_t i, size_t thread_rank) {
                auto workspace = workspace_pool.get(thread_rank);
                auto counts = reinterpret_cast<Index*>(workspace);
                process_cell(static_cast<Index>(i), counts);
            });
    } else {
        auto workspace = workspace_pool.get(0);
        auto counts = reinterpret_cast<Index*>(workspace);
        for (Index i = 0; i < n; ++i) {
            process_cell(i, counts);
        }
    }
}

// =============================================================================
// Local Density-Based Doublet Score
// =============================================================================

template <typename T, bool IsCSR>
void density_doublet_score(
    const Sparse<T, IsCSR>& knn_graph,
    Array<Real> density_scores
) {
    const Index n = knn_graph.primary_dim();

    SCL_CHECK_DIM(density_scores.len >= static_cast<Size>(n),
                  "Doublet: density_scores buffer too small");

    const bool use_parallel = n >= static_cast<Index>(config::PARALLEL_THRESHOLD);

    auto compute_density = [&](Index i) {
        auto values = knn_graph.primary_values_unsafe(i);
        Index len = knn_graph.primary_length_unsafe(i);

        if (SCL_UNLIKELY(len == 0)) {
            density_scores[i] = Real(0);
            return;
        }

        // Use SIMD sum for distances
        Real sum_dist = scl::vectorize::sum(
            Array<const Real>(reinterpret_cast<const Real*>(values.ptr), static_cast<Size>(len))
        );

        Real avg_dist = sum_dist / static_cast<Real>(len);
        density_scores[i] = (SCL_LIKELY(avg_dist > config::MIN_SCORE)) ?
            Real(1) / avg_dist : Real(1) / config::MIN_SCORE;
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n),
            [&](size_t i) {
                compute_density(static_cast<Index>(i));
            });
    } else {
        for (Index i = 0; i < n; ++i) {
            compute_density(i);
        }
    }
}

// =============================================================================
// Gene Expression Variance Score (Doublets Often Have Higher Variance)
// =============================================================================

template <typename T, bool IsCSR>
void variance_doublet_score(
    const Sparse<T, IsCSR>& X,
    Index n_cells,
    Index n_genes,
    Array<const Real> gene_means,
    Array<Real> variance_scores
) {
    SCL_CHECK_DIM(variance_scores.len >= static_cast<Size>(n_cells),
                  "Doublet: variance_scores buffer too small");

    const bool use_parallel = n_cells >= static_cast<Index>(config::PARALLEL_THRESHOLD);

    auto compute_variance = [&](Index c) {
        auto indices = X.row_indices_unsafe(c);
        auto values = X.row_values_unsafe(c);
        Index len = X.row_length_unsafe(c);

        Real sum_sq_dev = Real(0);

        // 4-way unrolled for ILP
        Index k = 0;
        for (; k + 4 <= len; k += 4) {
            Index g0 = indices[k];
            Index g1 = indices[k + 1];
            Index g2 = indices[k + 2];
            Index g3 = indices[k + 3];

            Real e0 = (SCL_LIKELY(g0 < n_genes)) ? static_cast<Real>(values[k]) : Real(0);
            Real e1 = (SCL_LIKELY(g1 < n_genes)) ? static_cast<Real>(values[k + 1]) : Real(0);
            Real e2 = (SCL_LIKELY(g2 < n_genes)) ? static_cast<Real>(values[k + 2]) : Real(0);
            Real e3 = (SCL_LIKELY(g3 < n_genes)) ? static_cast<Real>(values[k + 3]) : Real(0);

            Real m0 = (SCL_LIKELY(g0 < n_genes)) ? gene_means[g0] : Real(0);
            Real m1 = (SCL_LIKELY(g1 < n_genes)) ? gene_means[g1] : Real(0);
            Real m2 = (SCL_LIKELY(g2 < n_genes)) ? gene_means[g2] : Real(0);
            Real m3 = (SCL_LIKELY(g3 < n_genes)) ? gene_means[g3] : Real(0);

            Real d0 = e0 - m0;
            Real d1 = e1 - m1;
            Real d2 = e2 - m2;
            Real d3 = e3 - m3;

            sum_sq_dev += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        }

        for (; k < len; ++k) {
            Index gene = indices[k];
            if (SCL_LIKELY(gene < n_genes)) {
                Real expr = static_cast<Real>(values[k]);
                Real dev = expr - gene_means[gene];
                sum_sq_dev += dev * dev;
            }
        }

        variance_scores[c] = (SCL_LIKELY(len > 0)) ?
            sum_sq_dev / static_cast<Real>(len) : Real(0);
    };

    if (use_parallel) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n_cells),
            [&](size_t c) {
                compute_variance(static_cast<Index>(c));
            });
    } else {
        for (Index c = 0; c < n_cells; ++c) {
            compute_variance(c);
        }
    }
}

// =============================================================================
// Combined Doublet Score
// =============================================================================

inline void combined_doublet_score(
    Array<const Real> knn_scores,
    Array<const Real> density_scores,
    Array<const Real> variance_scores,
    Array<Real> combined_scores,
    Real knn_weight = Real(0.6),
    Real density_weight = Real(0.2),
    Real variance_weight = Real(0.2)
) {
    Size n = knn_scores.len;
    SCL_CHECK_DIM(combined_scores.len >= n, "Doublet: combined_scores buffer too small");

    // Normalize weights
    Real total_weight = knn_weight + density_weight + variance_weight;
    Real inv_total = Real(1) / total_weight;
    knn_weight *= inv_total;
    density_weight *= inv_total;
    variance_weight *= inv_total;

    // Find max values with SIMD
    Real knn_max = knn_scores[0], density_max = density_scores[0], var_max = variance_scores[0];
    namespace s = scl::simd;
    const auto d = s::SimdTagFor<Real>();
    const auto lanes = s::Lanes(d);

    // SIMD min/max reduction for normalization
    Size i = 0;
    if (n >= lanes) {
        auto v_knn_max = s::Set(d, knn_max);
        auto v_den_max = s::Set(d, density_max);
        auto v_var_max = s::Set(d, var_max);

        for (; i + lanes <= n; i += lanes) {
            auto v_knn = s::Load(d, knn_scores.ptr + i);
            auto v_den = s::Load(d, density_scores.ptr + i);
            auto v_var = s::Load(d, variance_scores.ptr + i);
            v_knn_max = s::Max(v_knn_max, v_knn);
            v_den_max = s::Max(v_den_max, v_den);
            v_var_max = s::Max(v_var_max, v_var);
        }

        knn_max = s::GetLane(s::MaxOfLanes(d, v_knn_max));
        density_max = s::GetLane(s::MaxOfLanes(d, v_den_max));
        var_max = s::GetLane(s::MaxOfLanes(d, v_var_max));
    }

    for (; i < n; ++i) {
        knn_max = scl::algo::max2(knn_max, knn_scores[static_cast<Index>(i)]);
        density_max = scl::algo::max2(density_max, density_scores[static_cast<Index>(i)]);
        var_max = scl::algo::max2(var_max, variance_scores[static_cast<Index>(i)]);
    }

    // Precompute inverse max values
    Real inv_knn_max = (SCL_LIKELY(knn_max > config::MIN_SCORE)) ? Real(1) / knn_max : Real(0);
    Real inv_density_max = (SCL_LIKELY(density_max > config::MIN_SCORE)) ? Real(1) / density_max : Real(0);
    Real inv_var_max = (SCL_LIKELY(var_max > config::MIN_SCORE)) ? Real(1) / var_max : Real(0);

    // Vectorized combination
    auto v_knn_w = s::Set(d, knn_weight * inv_knn_max);
    auto v_den_w = s::Set(d, density_weight * inv_density_max);
    auto v_var_w = s::Set(d, variance_weight * inv_var_max);

    i = 0;
    for (; i + lanes <= n; i += lanes) {
        auto v_knn = s::Load(d, knn_scores.ptr + i);
        auto v_den = s::Load(d, density_scores.ptr + i);
        auto v_var = s::Load(d, variance_scores.ptr + i);

        auto v_result = s::MulAdd(v_knn, v_knn_w,
                      s::MulAdd(v_den, v_den_w,
                                s::Mul(v_var, v_var_w)));

        s::Store(v_result, d, combined_scores.ptr + i);
    }

    // Scalar cleanup
    for (; i < n; ++i) {
        Real norm_knn = knn_scores[static_cast<Index>(i)] * inv_knn_max;
        Real norm_density = density_scores[static_cast<Index>(i)] * inv_density_max;
        Real norm_var = variance_scores[static_cast<Index>(i)] * inv_var_max;

        combined_scores[static_cast<Index>(i)] = knn_weight * norm_knn +
                            density_weight * norm_density +
                            variance_weight * norm_var;
    }
}

// =============================================================================
// Full Doublet Detection Pipeline
// =============================================================================

template <typename T, bool IsCSR>
Index detect_doublets(
    const Sparse<T, IsCSR>& X,
    Index n_cells,
    Index n_genes,
    Array<Real> scores,
    Array<bool> is_doublet,
    DoubletMethod method = DoubletMethod::Scrublet,
    Real expected_rate = config::DEFAULT_DOUBLET_RATE,
    Index k_neighbors = config::DEFAULT_N_NEIGHBORS,
    uint64_t seed = 42
) {
    SCL_CHECK_DIM(scores.len >= static_cast<Size>(n_cells),
                  "Doublet: scores buffer too small");
    SCL_CHECK_DIM(is_doublet.len >= static_cast<Size>(n_cells),
                  "Doublet: is_doublet buffer too small");

    // Compute doublet scores
    switch (method) {
        case DoubletMethod::Scrublet:
        case DoubletMethod::DoubletFinder:
        case DoubletMethod::Hybrid:
        default:
            scrublet_scores(X, n_cells, n_genes, scores, 0, k_neighbors, seed);
            break;
    }

    // Estimate threshold
    Real threshold = estimate_threshold(
        Array<const Real>(scores.ptr, n_cells), expected_rate
    );

    // Call doublets
    Index n_doublets = call_doublets(
        Array<const Real>(scores.ptr, n_cells), threshold, is_doublet
    );

    return n_doublets;
}

// =============================================================================
// Remove Doublets from Matrix (Return Indices of Singlets)
// =============================================================================

inline Index get_singlet_indices(
    Array<const bool> is_doublet,
    Array<Index> singlet_indices
) {
    Index count = 0;
    for (Size i = 0; i < is_doublet.len; ++i) {
        if (!is_doublet[static_cast<Index>(i)]) {
            if (count < static_cast<Index>(singlet_indices.len)) {
                singlet_indices[count] = static_cast<Index>(i);
            }
            ++count;
        }
    }
    return count;
}

// =============================================================================
// Doublet Score Statistics
// =============================================================================

inline void doublet_score_stats(
    Array<const Real> scores,
    Real* mean,
    Real* std_dev,
    Real* median
) {
    Size n = scores.len;
    if (SCL_UNLIKELY(n == 0)) {
        *mean = Real(0);
        *std_dev = Real(0);
        *median = Real(0);
        return;
    }

    // Mean using SIMD sum
    Real sum = scl::vectorize::sum(scores);
    *mean = sum / static_cast<Real>(n);

    // Variance using SIMD
    Real var = Real(0);
    Real m = *mean;
    namespace s = scl::simd;
    const auto d = s::SimdTagFor<Real>();
    const auto lanes = s::Lanes(d);

    auto mean_vec = s::Set(d, m);
    auto acc0 = s::Zero(d);
    auto acc1 = s::Zero(d);

    Size k = 0;
    for (; k + 2 * lanes <= n; k += 2 * lanes) {
        auto v0 = s::Load(d, scores.ptr + k);
        auto v1 = s::Load(d, scores.ptr + k + lanes);
        auto d0 = s::Sub(v0, mean_vec);
        auto d1 = s::Sub(v1, mean_vec);
        acc0 = s::MulAdd(d0, d0, acc0);
        acc1 = s::MulAdd(d1, d1, acc1);
    }

    acc0 = s::Add(acc0, acc1);

    for (; k + lanes <= n; k += lanes) {
        auto v = s::Load(d, scores.ptr + k);
        auto diff = s::Sub(v, mean_vec);
        acc0 = s::MulAdd(diff, diff, acc0);
    }

    var = s::GetLane(s::SumOfLanes(d, acc0));

    for (; k < n; ++k) {
        Real diff = scores[static_cast<Index>(k)] - m;
        var += diff * diff;
    }

    *std_dev = std::sqrt(var / static_cast<Real>(n));

    // Median using efficient sort
    // PERFORMANCE: RAII memory management with unique_ptr
    auto sorted_ptr = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* sorted = sorted_ptr.get();
    scl::memory::copy_fast(
        Array<const Real>(scores.ptr, n),
        Array<Real>(sorted, n)
    );

    scl::sort::sort(Array<Real>(sorted, n));

    *median = (n % 2 == 0) ?
        (sorted[n / 2 - 1] + sorted[n / 2]) / Real(2) :
        sorted[n / 2];

    // unique_ptr automatically frees memory when going out of scope
}

// =============================================================================
// Multiplet Rate by Cell Count (10x Genomics Reference)
// =============================================================================

inline Real multiplet_rate_10x(Index n_cells_recovered) {
    // Based on 10x Genomics multiplet rate curve
    // Approximate: 0.8% per 1000 cells loaded
    Real thousands = static_cast<Real>(n_cells_recovered) / Real(1000);
    return Real(0.008) * thousands;
}

// =============================================================================
// Cluster Doublet Enrichment
// =============================================================================

inline void cluster_doublet_enrichment(
    Array<const Real> doublet_scores,
    Array<const Index> cluster_labels,
    Index n_clusters,
    Array<Real> cluster_mean_scores,
    Array<Real> cluster_doublet_fraction
) {
    Size n = doublet_scores.len;

    SCL_CHECK_DIM(cluster_mean_scores.len >= static_cast<Size>(n_clusters),
                  "Doublet: cluster_mean_scores buffer too small");
    SCL_CHECK_DIM(cluster_doublet_fraction.len >= static_cast<Size>(n_clusters),
                  "Doublet: cluster_doublet_fraction buffer too small");

    // Compute per-cluster statistics
    // PERFORMANCE: RAII memory management with unique_ptr
    auto cluster_sums_ptr = scl::memory::aligned_alloc<Real>(static_cast<Size>(n_clusters), SCL_ALIGNMENT);
    auto cluster_counts_ptr = scl::memory::aligned_alloc<Index>(static_cast<Size>(n_clusters), SCL_ALIGNMENT);
    Real* cluster_sums = cluster_sums_ptr.get();
    Index* cluster_counts = cluster_counts_ptr.get();
    scl::algo::zero(cluster_sums, static_cast<Size>(n_clusters));
    scl::algo::zero(cluster_counts, static_cast<Size>(n_clusters));

    // Global threshold for doublet calling (use median + MAD)
    Real global_mean = Real(0);
    for (Size i = 0; i < n; ++i) {
        global_mean += doublet_scores[static_cast<Index>(i)];
    }
    global_mean /= static_cast<Real>(n);

    for (Size i = 0; i < n; ++i) {
        auto c = cluster_labels[static_cast<Index>(i)];
        if (c >= 0 && c < n_clusters) {
            cluster_sums[c] += doublet_scores[static_cast<Index>(i)];
            ++cluster_counts[c];
        }
    }

    for (Index c = 0; c < n_clusters; ++c) {
        if (cluster_counts[c] > 0) {
            cluster_mean_scores[c] = cluster_sums[c] / static_cast<Real>(cluster_counts[c]);
        } else {
            cluster_mean_scores[c] = Real(0);
        }
    }

    // Compute fraction with score > global mean (enrichment proxy)
    scl::algo::zero(cluster_sums, static_cast<Size>(n_clusters));
    for (Size i = 0; i < n; ++i) {
        auto c = cluster_labels[static_cast<Index>(i)];
        if (c >= 0 && c < n_clusters && doublet_scores[static_cast<Index>(i)] > global_mean) {
            cluster_sums[c] += Real(1);
        }
    }

    for (Index c = 0; c < n_clusters; ++c) {
        if (cluster_counts[c] > 0) {
            cluster_doublet_fraction[c] = cluster_sums[c] / static_cast<Real>(cluster_counts[c]);
        } else {
            cluster_doublet_fraction[c] = Real(0);
        }
    }

    // unique_ptr automatically frees memory when going out of scope
}

} // namespace scl::kernel::doublet
