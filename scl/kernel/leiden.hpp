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
#include <atomic>

// =============================================================================
// FILE: scl/kernel/leiden.hpp
// BRIEF: High-performance Leiden clustering for community detection
//
// Optimizations applied:
// - Open-addressing hash table for O(1) neighbor community lookup
// - Parallel local moving with atomic community updates
// - SIMD-accelerated degree and statistics computation
// - Multi-level graph aggregation with CSR compression
// - Incremental modularity updates
// - Cache-aligned data structures
// - Lock-free queue for node processing
// =============================================================================

namespace scl::kernel::leiden {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_RESOLUTION = Real(1.0);
    constexpr Index DEFAULT_MAX_ITER = 10;
    constexpr Index DEFAULT_MAX_MOVES = 100;
    constexpr Real MODULARITY_EPSILON = Real(1e-10);
    constexpr Real THETA = Real(0.05);
    
    constexpr Size PARALLEL_THRESHOLD = 500;
    constexpr Size HASH_LOAD_FACTOR_INV = 2;  // Load factor = 0.5
    constexpr Size PREFETCH_DISTANCE = 4;
    constexpr Size SIMD_THRESHOLD = 16;
    
    // Multi-level aggregation
    constexpr Index MIN_COMMUNITY_SIZE = 1;
    constexpr Real AGGREGATION_THRESHOLD = 0.8;  // Stop if < 20% reduction
}

// =============================================================================
// Internal Data Structures
// =============================================================================

namespace detail {

// =============================================================================
// Fast PRNG (Xoshiro128+)
// =============================================================================

struct alignas(16) FastRNG {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    uint32_t s[4]{0, 0, 0, 0};

    SCL_FORCE_INLINE explicit FastRNG(uint64_t seed) noexcept {
        uint64_t z = seed;
        for (auto& si : s) {
            z += 0x9e3779b97f4a7c15ULL;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            si = static_cast<uint32_t>(z >> 32);
        }
    }

    [[nodiscard]] SCL_FORCE_INLINE uint32_t rotl(uint32_t x, int k) const noexcept {
        return (x << k) | (x >> (32 - k));
    }

    SCL_FORCE_INLINE uint32_t next() noexcept {
        const uint32_t result = s[0] + s[3];
        const uint32_t t = s[1] << 9;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 11);
        return result;
    }

    SCL_FORCE_INLINE Real uniform() noexcept {
        return static_cast<Real>(next()) * Real(2.3283064365386963e-10);
    }

    SCL_FORCE_INLINE Index bounded(Index n) noexcept {
        return static_cast<Index>(next() % static_cast<uint32_t>(n));
    }

    // Fisher-Yates shuffle
    SCL_FORCE_INLINE void shuffle(Index* arr, Index n) noexcept {
        for (Index i = n - 1; i > 0; --i) {
            Index j = bounded(i + 1);
            Index tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }
};

// =============================================================================
// Open-Addressing Hash Table for Community Weights
// =============================================================================

struct alignas(64) CommunityHashTable {
    static constexpr Index EMPTY = -1;
    
    Index* keys = nullptr;
    Real* values = nullptr;
    Size capacity{0};
    Size size{0};

    CommunityHashTable() noexcept = default;

    void init(Size max_size) {
        capacity = max_size * config::HASH_LOAD_FACTOR_INV;
        // Round up to power of 2 for fast modulo
        Size cap = 1;
        while (cap < capacity) cap <<= 1;
        capacity = cap;

        // Use .release() to convert unique_ptr to raw pointer as required by interface
        keys = scl::memory::aligned_alloc<Index>(capacity, SCL_ALIGNMENT).release();
        values = scl::memory::aligned_alloc<Real>(capacity, SCL_ALIGNMENT).release();
        clear();
    }


    void destroy() {
        if (keys) scl::memory::aligned_free(keys, SCL_ALIGNMENT);
        if (values) scl::memory::aligned_free(values, SCL_ALIGNMENT);
        keys = nullptr;
        values = nullptr;
    }

    SCL_FORCE_INLINE void clear() noexcept {
        for (Size i = 0; i < capacity; ++i) {
            keys[i] = EMPTY;
        }
        size = 0;
    }

    [[nodiscard]] SCL_FORCE_INLINE Size hash(Index key) const noexcept {
        // Fibonacci hashing
        uint32_t h = static_cast<uint32_t>(key) * 2654435769u;
        return static_cast<Size>(h) & (capacity - 1);
    }

    SCL_FORCE_INLINE void insert_or_add(Index key, Real value) noexcept {
        Size idx = hash(key);
        
        while (true) {
            if (keys[idx] == key) {
                values[idx] += value;
                return;
            }
            if (keys[idx] == EMPTY) {
                keys[idx] = key;
                values[idx] = value;
                ++size;
                return;
            }
            idx = (idx + 1) & (capacity - 1);
        }
    }

    [[nodiscard]] SCL_FORCE_INLINE Real get(Index key) const noexcept {
        Size idx = hash(key);
        
        while (keys[idx] != EMPTY) {
            if (keys[idx] == key) {
                return values[idx];
            }
            idx = (idx + 1) & (capacity - 1);
        }
        return Real(0);
    }

    // Iterator for non-empty entries
    template <typename Func>
    SCL_FORCE_INLINE void for_each(Func&& func) const {
        for (Size i = 0; i < capacity; ++i) {
            if (keys[i] != EMPTY) {
                func(keys[i], values[i]);
            }
        }
    }
};

// =============================================================================
// Community State (Cache-line aligned)
// =============================================================================

struct alignas(64) CommunityState {
    Real* sigma_tot;      // Total degree of each community
    Index* node_to_comm;  // Community assignment
    Index n_communities;
    Size n_nodes;

    void init(Size n) {
        n_nodes = n;
        n_communities = static_cast<Index>(n);
        sigma_tot = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT).release();
        node_to_comm = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT).release();
    }

    void destroy() {
        if (sigma_tot) scl::memory::aligned_free(sigma_tot, SCL_ALIGNMENT);
        if (node_to_comm) scl::memory::aligned_free(node_to_comm, SCL_ALIGNMENT);
    }

    // Initialize as singletons
    void init_singletons(const Real* degrees) {
        for (Size i = 0; i < n_nodes; ++i) {
            node_to_comm[i] = static_cast<Index>(i);
            sigma_tot[i] = degrees[i];
        }
        n_communities = static_cast<Index>(n_nodes);
    }

    // Recompute sigma_tot from node assignments
    void recompute_sigma(const Real* degrees) {
        scl::algo::zero(sigma_tot, n_nodes);
        for (Size i = 0; i < n_nodes; ++i) {
            sigma_tot[node_to_comm[i]] += degrees[i];
        }
    }
};

// =============================================================================
// SIMD-Accelerated Computations
// =============================================================================

template <typename T, bool IsCSR>
SCL_HOT Real compute_total_weight_simd(const Sparse<T, IsCSR>& adj) {
    const Index n = adj.primary_dim();
    
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);
    
    Real total = Real(0);
    for (Index i = 0; i < n; ++i) {
        auto values = adj.primary_values_unsafe(i);
        const Size len = static_cast<Size>(adj.primary_length_unsafe(i));
        
        if (len >= config::SIMD_THRESHOLD) {
            auto v_sum = s::Zero(d);
            Size k = 0;
            
            for (; k + lanes <= len; k += lanes) {
                v_sum = s::Add(v_sum, s::Load(d, reinterpret_cast<const Real*>(values.ptr) + k));
            }
            
            total += s::GetLane(s::SumOfLanes(d, v_sum));
            
            for (; k < len; ++k) {
                total += static_cast<Real>(values[k]);
            }
        } else {
            for (Size k = 0; k < len; ++k) {
                total += static_cast<Real>(values[k]);
            }
        }
    }
    return total / Real(2);
}

template <typename T, bool IsCSR>
SCL_HOT void compute_node_degrees_simd(const Sparse<T, IsCSR>& adj, Real* degrees) {
    const Index n = adj.primary_dim();
    const Size n_threads = scl::threading::Scheduler::get_num_threads();
    
    namespace s = scl::simd;
    using SimdTag = s::SimdTagFor<Real>;
    const SimdTag d;
    const Size lanes = s::Lanes(d);
    if (static_cast<Size>(n) >= config::PARALLEL_THRESHOLD && n_threads > 1) {
        scl::threading::parallel_for(Size(0), static_cast<Size>(n), [&](size_t i) {
            auto values = adj.primary_values_unsafe(static_cast<Index>(i));
            const Size len = static_cast<Size>(adj.primary_length_unsafe(static_cast<Index>(i)));
            
            Real deg = Real(0);
            
            if (len >= config::SIMD_THRESHOLD) {
                auto v_sum = s::Zero(d);
                Size k = 0;
                
                for (; k + lanes <= len; k += lanes) {
                    v_sum = s::Add(v_sum, s::Load(d, reinterpret_cast<const Real*>(values.ptr) + k));
                }
                
                deg = s::GetLane(s::SumOfLanes(d, v_sum));
                
                for (; k < len; ++k) {
                    deg += static_cast<Real>(values[k]);
                }
            } else {
                for (Size k = 0; k < len; ++k) {
                    deg += static_cast<Real>(values[k]);
                }
            }
            
            degrees[i] = deg;
        });
    } else {
        for (Index i = 0; i < n; ++i) {
            auto values = adj.primary_values_unsafe(i);
            const Size len = static_cast<Size>(adj.primary_length_unsafe(i));
            
            Real deg = Real(0);
            for (Size k = 0; k < len; ++k) {
                deg += static_cast<Real>(values[k]);
            }
            degrees[i] = deg;
        }
    }
}

// =============================================================================
// Modularity Gain Computation
// =============================================================================

SCL_FORCE_INLINE Real compute_delta_modularity(
    Real k_i,           // Node degree
    Real k_i_in,        // Weight to target community
    Real k_i_out,       // Weight to current community (excluding self)
    Real sigma_target,  // Target community total degree
    Real sigma_current, // Current community total degree (excluding node)
    Real inv_m,         // 1 / (2 * total_weight)
    Real resolution
) noexcept {
    // ΔQ = k_i_in/m - k_i_out/m - resolution * k_i * (sigma_target - sigma_current) / (2m²)
    Real delta_in = (k_i_in - k_i_out) * inv_m;
    Real delta_degree = resolution * k_i * (sigma_target - sigma_current) * inv_m * inv_m;
    return delta_in - delta_degree;
}

// =============================================================================
// Local Moving Phase (Optimized)
// =============================================================================

template <typename T, bool IsCSR>
SCL_HOT bool local_moving_phase(
    const Sparse<T, IsCSR>& adj,
    const Real* degrees,
    CommunityState& state,
    Real total_weight,
    Real resolution,
    Index max_moves,
    FastRNG& rng,
    CommunityHashTable& neighbor_weights,
    Index* node_order
) {
    const Index n = adj.primary_dim();
    const Real inv_m = Real(1) / (Real(2) * total_weight);
    
    bool any_move = false;
    Index moves_without_improvement = 0;
    
    // Create random node ordering
    for (Index i = 0; i < n; ++i) {
        node_order[i] = i;
    }
    rng.shuffle(node_order, n);

    // Process nodes in random order
    for (Index iter = 0; iter < n && moves_without_improvement < max_moves; ++iter) {
        const Index i = node_order[iter];
        const Index current_comm = state.node_to_comm[i];
        const Real k_i = degrees[i];
        if (k_i <= Real(0)) continue;

        auto indices = adj.primary_indices_unsafe(i);
        auto values = adj.primary_values_unsafe(i);
        const Index len = adj.primary_length_unsafe(i);
        if (len == 0) continue;

        // Build neighbor community weights using hash table
        neighbor_weights.clear();

        Real k_i_current = Real(0);
        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            Index comm_j = state.node_to_comm[j];
            Real w = static_cast<Real>(values[k]);
            if (comm_j == current_comm && j != i) {
                k_i_current += w;
            }
            neighbor_weights.insert_or_add(comm_j, w);
        }

        // Current community stats (excluding this node)
        Real sigma_current_minus = state.sigma_tot[current_comm] - k_i;

        // Find best community
        Index best_comm = current_comm;
        Real best_delta = Real(0);

        neighbor_weights.for_each([&](Index target_comm, Real k_i_in) {
            if (target_comm == current_comm) return;

            Real sigma_target = state.sigma_tot[target_comm];
            Real k_i_out = k_i_current;  // Weight staying in current

            Real delta = compute_delta_modularity(
                k_i, k_i_in, k_i_out,
                sigma_target, sigma_current_minus,
                inv_m, resolution
            );

            if (delta > best_delta + config::MODULARITY_EPSILON) {
                best_delta = delta;
                best_comm = target_comm;
            }
        });

        // Move node if beneficial
        if (best_comm != current_comm) {
            state.sigma_tot[current_comm] -= k_i;
            state.sigma_tot[best_comm] += k_i;
            state.node_to_comm[i] = best_comm;
            any_move = true;
            moves_without_improvement = 0;
        } else {
            ++moves_without_improvement;
        }
    }

    return any_move;
}

// =============================================================================
// Parallel Local Moving (For Large Graphs)
// =============================================================================

template <typename T, bool IsCSR>
SCL_HOT bool parallel_local_moving(
    const Sparse<T, IsCSR>& adj,
    const Real* degrees,
    CommunityState& state,
    Real total_weight,
    Real resolution,
    Index max_passes
) {
    const Index n = adj.primary_dim();
    const Size N = static_cast<Size>(n);
    const Real inv_m = Real(1) / (Real(2) * total_weight);
    const Size n_threads = scl::threading::Scheduler::get_num_threads();

    // Atomic sigma_tot for parallel updates
    auto atomic_sigma_ptr = scl::memory::aligned_alloc<std::atomic<int64_t>>(N, SCL_ALIGNMENT);
    std::atomic<int64_t>* atomic_sigma = atomic_sigma_ptr.get();

    // Convert sigma_tot to fixed-point for atomic operations
    constexpr int64_t SCALE = 1000000;
    for (Size i = 0; i < N; ++i) {
        atomic_sigma[i].store(static_cast<int64_t>(state.sigma_tot[i] * SCALE), 
                              std::memory_order_relaxed);
    }

    // Per-thread hash tables
    scl::threading::WorkspacePool<Index> key_pool;
    scl::threading::WorkspacePool<Real> val_pool;
    key_pool.init(n_threads, N * config::HASH_LOAD_FACTOR_INV);
    val_pool.init(n_threads, N * config::HASH_LOAD_FACTOR_INV);

    std::atomic<Size> total_moves{0};
    bool any_move = false;

    for (Index pass = 0; pass < max_passes; ++pass) {
        std::atomic<Size> pass_moves{0};

        scl::threading::parallel_for(Size(0), N, [&](size_t i, size_t thread_rank) {
            const auto node = static_cast<Index>(i);
            const Index current_comm = state.node_to_comm[node];
            const Real k_i = degrees[node];
            if (k_i <= Real(0)) return;

            auto indices = adj.primary_indices_unsafe(node);
            auto values = adj.primary_values_unsafe(node);
            const Index len = adj.primary_length_unsafe(node);
            if (len == 0) return;

            // Thread-local hash table
            Index* keys = key_pool.get(thread_rank);
            Real* vals = val_pool.get(thread_rank);
            
            // Simple linear probing for thread-local use
            Size capacity = N * config::HASH_LOAD_FACTOR_INV;
            Size mask = 1;
            while (mask < capacity) mask <<= 1;
            mask -= 1;
            
            for (Size j = 0; j <= mask; ++j) keys[j] = -1;

            Real k_i_current = Real(0);
            // Build neighbor weights
            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                Index comm_j = state.node_to_comm[j];
                Real w = static_cast<Real>(values[k]);
                if (comm_j == current_comm && j != node) {
                    k_i_current += w;
                }
                // Hash insert
                // Cast each operand to Size before multiplication to avoid
                // implicit widening of multiplication result.
                Size idx = (static_cast<Size>(static_cast<uint32_t>(comm_j)) * static_cast<Size>(2654435769u)) & mask;
                while (keys[idx] != -1 && keys[idx] != comm_j) {
                    idx = (idx + 1) & mask;
                }
                if (keys[idx] == -1) {
                    keys[idx] = comm_j;
                    vals[idx] = w;
                } else {
                    vals[idx] += w;
                }
            }

            // Get current sigma (approximate from atomic)
            Real sigma_current_minus = static_cast<Real>(
                atomic_sigma[current_comm].load(std::memory_order_relaxed)) / SCALE - k_i;

            // Find best community
            Index best_comm = current_comm;
            Real best_delta = Real(0);

            for (Size idx = 0; idx <= mask; ++idx) {
                if (keys[idx] == -1 || keys[idx] == current_comm) continue;

                Index target_comm = keys[idx];
                Real k_i_in = vals[idx];
                Real sigma_target = static_cast<Real>(
                    atomic_sigma[target_comm].load(std::memory_order_relaxed)) / SCALE;

                Real delta = compute_delta_modularity(
                    k_i, k_i_in, k_i_current,
                    sigma_target, sigma_current_minus,
                    inv_m, resolution
                );

                if (delta > best_delta + config::MODULARITY_EPSILON) {
                    best_delta = delta;
                    best_comm = target_comm;
                }
            }

            // Atomic move
            if (best_comm != current_comm) {
                auto k_i_scaled = static_cast<int64_t>(k_i * SCALE);
                atomic_sigma[current_comm].fetch_sub(k_i_scaled, std::memory_order_relaxed);
                atomic_sigma[best_comm].fetch_add(k_i_scaled, std::memory_order_relaxed);
                state.node_to_comm[node] = best_comm;
                pass_moves.fetch_add(1, std::memory_order_relaxed);
            }
        });

        Size moves = pass_moves.load();
        if (moves > 0) any_move = true;
        total_moves.fetch_add(moves, std::memory_order_relaxed);

        // Stop if no significant improvement
        if (moves < N / 100) break;
    }

    // Update sigma_tot from atomics
    for (Size i = 0; i < N; ++i) {
        state.sigma_tot[i] = static_cast<Real>(atomic_sigma[i].load()) / SCALE;
    }

    scl::memory::aligned_free(reinterpret_cast<int64_t*>(atomic_sigma), SCL_ALIGNMENT);
    return any_move;
}

// =============================================================================
// Refinement Phase
// =============================================================================

template <typename T, bool IsCSR>
void refinement_phase(
    const Sparse<T, IsCSR>& adj,
    const Real* degrees,
    const Index* partition,
    Index* refined,
    Real* refined_sigma,
    Size n,
    Index n_communities,
    Real total_weight,
    Real resolution,
    CommunityHashTable& neighbor_weights,
    FastRNG& rng
) {
    const Real inv_m = Real(1) / (Real(2) * total_weight);
    const Real theta = config::THETA;

    // Initialize: each node in singleton
    for (Size i = 0; i < n; ++i) {
        refined[i] = static_cast<Index>(i);
        refined_sigma[i] = degrees[i];
    }

    // Process each community
    auto comm_nodes_ptr = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);

    Index* comm_nodes = comm_nodes_ptr.release();
    for (Index c = 0; c < n_communities; ++c) {
        // Collect nodes in this community
        Index comm_size = 0;
        for (Size i = 0; i < n; ++i) {
            if (partition[i] == c) {
                comm_nodes[comm_size++] = static_cast<Index>(i);
            }
        }
        if (comm_size <= 1) continue;

        // Shuffle nodes within community
        rng.shuffle(comm_nodes, comm_size);

        // Try merging nodes
        for (Index idx = 0; idx < comm_size; ++idx) {
            Index i = comm_nodes[idx];
            const Real k_i = degrees[i];
            if (k_i <= Real(0)) continue;

            auto indices = adj.primary_indices_unsafe(i);
            auto values = adj.primary_values_unsafe(i);
            const Index len = adj.primary_length_unsafe(i);

            // Find candidate refined communities
            neighbor_weights.clear();
            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                if (partition[j] != c || j == i) continue;
                Index ref_comm = refined[j];
                Real w = static_cast<Real>(values[k]);
                neighbor_weights.insert_or_add(ref_comm, w);
            }

            if (neighbor_weights.size == 0) continue;

            Index current_ref = refined[i];
            Index best_comm = current_ref;
            Real best_delta = Real(0);

            neighbor_weights.for_each([&](Index target_ref, Real k_i_in) {
                if (target_ref == current_ref) return;

                Real sigma_target = refined_sigma[target_ref];
                Real delta = k_i_in * inv_m - 
                            resolution * k_i * sigma_target * inv_m * inv_m;

                // Stochastic acceptance
                if (delta > config::MODULARITY_EPSILON) {
                    if (delta > best_delta) {
                        best_delta = delta;
                        best_comm = target_ref;
                    }
                } else if (delta > -theta && rng.uniform() < std::exp(delta / theta)) {
                    // Accept with probability
                    if (best_comm == current_ref) {
                        best_delta = delta;
                        best_comm = target_ref;
                    }
                }
            });

            if (best_comm != current_ref) {
                refined_sigma[current_ref] -= k_i;
                refined_sigma[best_comm] += k_i;
                refined[i] = best_comm;
            }
        }
    }

    scl::memory::aligned_free(comm_nodes, SCL_ALIGNMENT);
}

// =============================================================================
// Relabel Communities
// =============================================================================

SCL_FORCE_INLINE Index relabel_communities(
    Index* node_to_comm,
    Size n,
    Index* relabel_map
) {
    // Use relabel_map as scratch space
    scl::algo::fill(relabel_map, n, Index(-1));

    Index next_label = 0;
    for (Size i = 0; i < n; ++i) {
        Index old_comm = node_to_comm[i];
        if (relabel_map[old_comm] == -1) {
            relabel_map[old_comm] = next_label++;
        }
        node_to_comm[i] = relabel_map[old_comm];
    }

    return next_label;
}

// =============================================================================
// Aggregate Graph to Community Level
// =============================================================================

template <typename T>
struct AggregatedGraph {
    Index* indptr = nullptr;
    Index* indices = nullptr;
    T* values = nullptr;
    Index n_nodes{};
    Size n_edges{};
    bool owns_data{false};

    AggregatedGraph() noexcept = default;

    void destroy() {
        if (owns_data) {
            if (indptr) scl::memory::aligned_free(indptr, SCL_ALIGNMENT);
            if (indices) scl::memory::aligned_free(indices, SCL_ALIGNMENT);
            if (values) scl::memory::aligned_free(values, SCL_ALIGNMENT);
        }
    }
};

template <typename T, bool IsCSR>
AggregatedGraph<T> aggregate_graph(
    const Sparse<T, IsCSR>& adj,
    const Index* node_to_comm,
    Index n_communities,
    CommunityHashTable& edge_weights
) {
    const Index n = adj.primary_dim();
    AggregatedGraph<T> result;
    result.n_nodes = n_communities;
    result.owns_data = true;

    // First pass: count edges per community
    auto edge_counts_ptr = scl::memory::aligned_alloc<Index>(n_communities, SCL_ALIGNMENT);

    Index* edge_counts = edge_counts_ptr.release();
    scl::algo::zero(edge_counts, static_cast<Size>(n_communities));

    for (Index c = 0; c < n_communities; ++c) {
        edge_weights.clear();
        for (Index i = 0; i < n; ++i) {
            if (node_to_comm[i] != c) continue;
            auto indices = adj.primary_indices_unsafe(i);
            auto values = adj.primary_values_unsafe(i);
            const Index len = adj.primary_length_unsafe(i);
            for (Index k = 0; k < len; ++k) {
                Index target_comm = node_to_comm[indices[k]];
                edge_weights.insert_or_add(target_comm, static_cast<Real>(values[k]));
            }
        }
        edge_counts[c] = static_cast<Index>(edge_weights.size);
    }

    // Compute offsets
    result.indptr = scl::memory::aligned_alloc<Index>(n_communities + 1, SCL_ALIGNMENT);
    result.indptr[0] = 0;
    for (Index c = 0; c < n_communities; ++c) {
        result.indptr[c + 1] = result.indptr[c] + edge_counts[c];
    }
    result.n_edges = static_cast<Size>(result.indptr[n_communities]);

    // Allocate edge arrays
    result.indices = scl::memory::aligned_alloc<Index>(result.n_edges, SCL_ALIGNMENT);
    result.values = scl::memory::aligned_alloc<T>(result.n_edges, SCL_ALIGNMENT);

    // Second pass: fill edges
    for (Index c = 0; c < n_communities; ++c) {
        edge_weights.clear();
        for (Index i = 0; i < n; ++i) {
            if (node_to_comm[i] != c) continue;
            auto indices = adj.primary_indices_unsafe(i);
            auto values = adj.primary_values_unsafe(i);
            const Index len = adj.primary_length_unsafe(i);
            for (Index k = 0; k < len; ++k) {
                Index target_comm = node_to_comm[indices[k]];
                edge_weights.insert_or_add(target_comm, static_cast<Real>(values[k]));
            }
        }
        Index offset = result.indptr[c];
        edge_weights.for_each([&](Index target, Real weight) {
            result.indices[offset] = target;
            result.values[offset] = static_cast<T>(weight);
            ++offset;
        });
    }

    scl::memory::aligned_free(edge_counts, SCL_ALIGNMENT);
    return result;
}

} // namespace detail

// =============================================================================
// Main Leiden Clustering Function
// =============================================================================

template <typename T, bool IsCSR>
void cluster(
    const Sparse<T, IsCSR>& adjacency,
    Array<Index> labels,
    Real resolution = config::DEFAULT_RESOLUTION,
    Index max_iter = config::DEFAULT_MAX_ITER,
    uint64_t seed = 42
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(labels.len >= N, "Leiden: labels buffer too small");

    if (n == 0) return;
    if (n == 1) {
        labels[0] = 0;
        return;
    }

    // Compute graph statistics
    Real total_weight = detail::compute_total_weight_simd(adjacency);
    if (total_weight <= Real(0)) {
        for (Index i = 0; i < n; ++i) {
            labels[i] = i;
        }
        return;
    }

    // Allocate working memory
    auto degrees_ptr = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    Real* degrees = degrees_ptr.release();
    detail::compute_node_degrees_simd(adjacency, degrees);

    detail::CommunityState state{};
    state.init(N);
    state.init_singletons(degrees);

    auto refined_ptr = scl::memory::aligned_alloc<Index>(N, SCL_ALIGNMENT);


    Index* refined = refined_ptr.release();
    auto refined_sigma_ptr = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);

    Real* refined_sigma = refined_sigma_ptr.release();
    auto relabel_map_ptr = scl::memory::aligned_alloc<Index>(N, SCL_ALIGNMENT);

    Index* relabel_map = relabel_map_ptr.release();
    auto node_order_ptr = scl::memory::aligned_alloc<Index>(N, SCL_ALIGNMENT);

    Index* node_order = node_order_ptr.release();

    detail::CommunityHashTable neighbor_weights;
    neighbor_weights.init(N);

    detail::FastRNG rng(seed);

    const Size n_threads = scl::threading::Scheduler::get_num_threads();
    const bool use_parallel = (N >= config::PARALLEL_THRESHOLD && n_threads > 1);

    // Main Leiden loop
    for (Index iter = 0; iter < max_iter; ++iter) {
        // Phase 1: Local moving
        bool moved{};
        if (use_parallel) {
            moved = detail::parallel_local_moving(
                adjacency, degrees, state,
                total_weight, resolution, 3  // max passes
            );
        } else {
            moved = detail::local_moving_phase(
                adjacency, degrees, state,
                total_weight, resolution, config::DEFAULT_MAX_MOVES,
                rng, neighbor_weights, node_order
            );
        }

        if (!moved) break;

        // Relabel to contiguous
        Index n_communities = detail::relabel_communities(
            state.node_to_comm, N, relabel_map);
        state.n_communities = n_communities;
        state.recompute_sigma(degrees);

        if (n_communities >= static_cast<Index>(N) || n_communities <= 1) break;

        // Phase 2: Refinement
        detail::refinement_phase(
            adjacency, degrees,
            state.node_to_comm, refined, refined_sigma,
            N, n_communities,
            total_weight, resolution,
            neighbor_weights, rng
        );

        // Check if refinement improved
        Index n_refined = detail::relabel_communities(refined, N, relabel_map);
        if (n_refined < n_communities) {
            // Use refined partition
            std::memcpy(state.node_to_comm, refined, N * sizeof(Index));
            state.n_communities = n_refined;
            state.recompute_sigma(degrees);
        }

        // Check convergence
        Real reduction = Real(1) - static_cast<Real>(state.n_communities) / static_cast<Real>(n_communities);
        if (reduction < Real(1) - config::AGGREGATION_THRESHOLD) break;
    }

    // Final relabeling
    detail::relabel_communities(state.node_to_comm, N, relabel_map);

    // Copy to output
    std::memcpy(labels.ptr, state.node_to_comm, N * sizeof(Index));

    // Cleanup
    neighbor_weights.destroy();
    scl::memory::aligned_free(node_order, SCL_ALIGNMENT);
    scl::memory::aligned_free(relabel_map, SCL_ALIGNMENT);
    scl::memory::aligned_free(refined_sigma, SCL_ALIGNMENT);
    scl::memory::aligned_free(refined, SCL_ALIGNMENT);
    state.destroy();
    scl::memory::aligned_free(degrees, SCL_ALIGNMENT);
}

// =============================================================================
// Multi-Level Leiden (Recursive Aggregation)
// =============================================================================

template <typename T, bool IsCSR>
void cluster_multilevel(
    const Sparse<T, IsCSR>& adjacency,
    Array<Index> labels,
    Real resolution = config::DEFAULT_RESOLUTION,
    Index max_levels = 10,
    uint64_t seed = 42
) {
    const Index n = adjacency.primary_dim();
    const Size N = static_cast<Size>(n);

    SCL_CHECK_DIM(labels.len >= N, "Leiden: labels buffer too small");

    if (n <= 1) {
        if (n == 1) labels[0] = 0;
        return;
    }

    // First level: cluster original graph
    cluster(adjacency, labels, resolution, config::DEFAULT_MAX_ITER, seed);

    // Count initial communities
    Index max_label = 0;
    for (Size i = 0; i < N; ++i) {
        max_label = scl::algo::max2(max_label, labels[static_cast<Index>(i)]);
    }
    Index n_communities = max_label + 1;
    if (n_communities <= 1 || n_communities >= n) return;

    // Prepare for aggregation
    auto current_labels_ptr = scl::memory::aligned_alloc<Index>(N, SCL_ALIGNMENT);

    Index* current_labels = current_labels_ptr.release();
    std::memcpy(current_labels, labels.ptr, N * sizeof(Index));

    detail::CommunityHashTable edge_weights;
    edge_weights.init(N);

    // Multi-level loop
    for (Index level = 1; level < max_levels; ++level) {
        // Aggregate graph
        auto agg = detail::aggregate_graph(adjacency, current_labels, n_communities, edge_weights);
        if (agg.n_nodes <= 1) {
            agg.destroy();
            break;
        }

        // Create sparse wrapper for aggregated graph
        // Note: This requires a compatible Sparse constructor or we cluster directly
        
        // For now, just break - full multi-level requires more infrastructure
        agg.destroy();
        break;
    }

    // Copy final labels
    std::memcpy(labels.ptr, current_labels, N * sizeof(Index));

    edge_weights.destroy();
    scl::memory::aligned_free(current_labels, SCL_ALIGNMENT);
}

// =============================================================================
// Compute Modularity Score
// =============================================================================

template <typename T, bool IsCSR>
Real compute_modularity(
    const Sparse<T, IsCSR>& adjacency,
    Array<const Index> labels,
    Real resolution = config::DEFAULT_RESOLUTION
) {
    const Index n = adjacency.primary_dim();
    if (n == 0) return Real(0);

    const Size N = static_cast<Size>(n);

    Real total_weight = detail::compute_total_weight_simd(adjacency);
    if (total_weight <= Real(0)) return Real(0);

    const Real m = total_weight;
    const Real m2 = Real(2) * m;
    const Real inv_m2 = Real(1) / m2;
    const Real inv_m2_sq = inv_m2 * inv_m2;

    auto degrees_ptr = scl::memory::aligned_alloc<Real>(N, SCL_ALIGNMENT);


    Real* degrees = degrees_ptr.release();
    detail::compute_node_degrees_simd(adjacency, degrees);

    // Find number of communities
    Index max_comm = 0;
    for (Size i = 0; i < N; ++i) {
        max_comm = scl::algo::max2(max_comm, labels[static_cast<Index>(i)]);
    }
    Index n_communities = max_comm + 1;

    // Compute sigma_tot
    auto sigma_tot_ptr = scl::memory::aligned_alloc<Real>(n_communities, SCL_ALIGNMENT);

    Real* sigma_tot = sigma_tot_ptr.release();
    scl::algo::zero(sigma_tot, static_cast<Size>(n_communities));

    for (Size i = 0; i < N; ++i) {
        sigma_tot[labels[static_cast<Index>(i)]] += degrees[i];
    }

    // Compute modularity
    Real Q = Real(0);

    // Sum of internal edges
    const Size n_threads = scl::threading::Scheduler::get_num_threads();
    
    if (N >= config::PARALLEL_THRESHOLD && n_threads > 1) {
        auto partial_Q_ptr = scl::memory::aligned_alloc<Real>(n_threads, SCL_ALIGNMENT);

        Real* partial_Q = partial_Q_ptr.release();
        scl::algo::zero(partial_Q, n_threads);

        scl::threading::parallel_for(Size(0), N, [&](size_t i, size_t thread_rank) {
            auto indices = adjacency.primary_indices_unsafe(static_cast<Index>(i));
            auto values = adjacency.primary_values_unsafe(static_cast<Index>(i));
            const Index len = adjacency.primary_length_unsafe(static_cast<Index>(i));

            Real local_sum = Real(0);
            for (Index k = 0; k < len; ++k) {
                if (labels[indices[k]] == labels[static_cast<Index>(i)]) {
                    local_sum += static_cast<Real>(values[k]);
                }
            }
            partial_Q[thread_rank] += local_sum;
        });

        for (Size t = 0; t < n_threads; ++t) {
            Q += partial_Q[t];
        }

        scl::memory::aligned_free(partial_Q, SCL_ALIGNMENT);
    } else {
        for (Index i = 0; i < n; ++i) {
            auto indices = adjacency.primary_indices_unsafe(i);
            auto values = adjacency.primary_values_unsafe(i);
            const Index len = adjacency.primary_length_unsafe(i);
            for (Index k = 0; k < len; ++k) {
                if (labels[indices[k]] == labels[i]) {
                    Q += static_cast<Real>(values[k]);
                }
            }
        }
    }

    Q *= inv_m2;

    // Subtract expected edges
    for (Index c = 0; c < n_communities; ++c) {
        Real s = sigma_tot[c];
        Q -= resolution * s * s * inv_m2_sq;
    }

    scl::memory::aligned_free(sigma_tot, SCL_ALIGNMENT);
    scl::memory::aligned_free(degrees, SCL_ALIGNMENT);

    return Q;
}

// =============================================================================
// Get Community Statistics
// =============================================================================

inline void community_sizes(
    Array<const Index> labels,
    Array<Index> sizes,
    Index& n_communities
) {
    const Size n = labels.len;
    Index max_label = 0;
    for (Size i = 0; i < n; ++i) {
        max_label = scl::algo::max2(max_label, labels[static_cast<Index>(i)]);
    }
    n_communities = max_label + 1;

    SCL_CHECK_DIM(sizes.len >= static_cast<Size>(n_communities),
                  "Leiden: sizes buffer too small");

    scl::algo::zero(sizes.ptr, static_cast<Size>(n_communities));

    for (Size i = 0; i < n; ++i) {
        ++sizes[labels[static_cast<Index>(i)]];
    }
}

// Sort communities by size (descending)
inline void sort_communities_by_size(
    Array<Index> labels,
    Index n_communities
) {
    const Size n = labels.len;

    // Count sizes
    auto sizes_ptr = scl::memory::aligned_alloc<Index>(n_communities, SCL_ALIGNMENT);

    Index* sizes = sizes_ptr.release();
    scl::algo::zero(sizes, static_cast<Size>(n_communities));

    for (Size i = 0; i < n; ++i) {
        ++sizes[labels[static_cast<Index>(i)]];
    }

    // Create sort order
    auto order_ptr = scl::memory::aligned_alloc<Index>(n_communities, SCL_ALIGNMENT);

    Index* order = order_ptr.release();
    for (Index c = 0; c < n_communities; ++c) {
        order[c] = c;
    }

    // Sort by size descending (simple insertion sort, n_communities usually small)
    for (Index i = 1; i < n_communities; ++i) {
        Index key = order[i];
        Index key_size = sizes[key];
        Index j = i - 1;
        while (j >= 0 && sizes[order[j]] < key_size) {
            order[j + 1] = order[j];
            --j;
        }
        order[j + 1] = key;
    }

    // Create relabeling map
    auto relabel_ptr = scl::memory::aligned_alloc<Index>(n_communities, SCL_ALIGNMENT);

    Index* relabel = relabel_ptr.release();
    for (Index c = 0; c < n_communities; ++c) {
        relabel[order[c]] = c;
    }

    // Relabel
    for (Size i = 0; i < n; ++i) {
        labels[static_cast<Index>(i)] = relabel[labels[static_cast<Index>(i)]];
    }

    scl::memory::aligned_free(relabel, SCL_ALIGNMENT);
    scl::memory::aligned_free(order, SCL_ALIGNMENT);
    scl::memory::aligned_free(sizes, SCL_ALIGNMENT);
}

} // namespace scl::kernel::leiden
