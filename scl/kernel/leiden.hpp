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
// FILE: scl/kernel/leiden.hpp
// BRIEF: Leiden clustering algorithm for community detection in sparse graphs
// Improves upon Louvain by guaranteeing connected communities
// =============================================================================

namespace scl::kernel::leiden {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_RESOLUTION = Real(1.0);
    constexpr Index DEFAULT_MAX_ITER = 10;
    constexpr Real MODULARITY_EPSILON = Real(1e-8);
    constexpr Real THETA = Real(0.05);  // Refinement randomness parameter
    constexpr Size PARALLEL_THRESHOLD = 1000;
}

// =============================================================================
// Internal Data Structures and Helpers
// =============================================================================

namespace detail {

// Simple PRNG for randomization
struct FastRNG {
    uint64_t state;

    SCL_FORCE_INLINE explicit FastRNG(uint64_t seed) noexcept : state(seed) {}

    SCL_FORCE_INLINE uint64_t next() noexcept {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        return state * 0x2545F4914F6CDD1DULL;
    }

    SCL_FORCE_INLINE Real uniform() noexcept {
        return static_cast<Real>(next() >> 11) * Real(0x1.0p-53);
    }

    SCL_FORCE_INLINE Size bounded(Size n) noexcept {
        return static_cast<Size>(next() % static_cast<uint64_t>(n));
    }
};

// =============================================================================
// Helper: Compute total graph weight
// =============================================================================

template <typename T, bool IsCSR>
SCL_FORCE_INLINE Real compute_total_weight(const Sparse<T, IsCSR>& adj) {
    const Index n = adj.primary_dim();
    Real total = Real(0);

    for (Index i = 0; i < n; ++i) {
        auto values = adj.primary_values(i);
        const Index len = adj.primary_length(i);
        for (Index k = 0; k < len; ++k) {
            total += static_cast<Real>(values[k]);
        }
    }

    return total / Real(2);
}

// =============================================================================
// Helper: Compute node degrees (weighted)
// =============================================================================

template <typename T, bool IsCSR>
void compute_node_degrees(const Sparse<T, IsCSR>& adj, Real* degrees) {
    const Index n = adj.primary_dim();

    for (Index i = 0; i < n; ++i) {
        auto values = adj.primary_values(i);
        const Index len = adj.primary_length(i);
        Real deg = Real(0);
        for (Index k = 0; k < len; ++k) {
            deg += static_cast<Real>(values[k]);
        }
        degrees[i] = deg;
    }
}

// =============================================================================
// Helper: Compute community statistics
// =============================================================================

SCL_FORCE_INLINE void compute_community_stats(
    const Index* node_to_comm,
    const Real* degrees,
    Size n,
    Real* sigma_tot,
    Index n_communities
) {
    scl::algo::zero(sigma_tot, static_cast<Size>(n_communities));

    for (Size i = 0; i < n; ++i) {
        sigma_tot[node_to_comm[i]] += degrees[i];
    }
}

// =============================================================================
// Helper: Modularity gain for moving node
// =============================================================================

SCL_FORCE_INLINE Real modularity_gain(
    Real k_i,
    Real k_i_in,
    Real sigma_tot,
    Real m,
    Real resolution
) {
    Real m2 = Real(2) * m;
    return k_i_in / m - resolution * k_i * sigma_tot / (m2 * m);
}

// =============================================================================
// Local Moving Phase
// =============================================================================

template <typename T, bool IsCSR>
bool local_moving_phase(
    const Sparse<T, IsCSR>& adj,
    const Real* degrees,
    Index* node_to_comm,
    Real* sigma_tot,
    Size n,
    Real total_weight,
    Real resolution,
    Index* neighbor_comms,
    Real* k_i_to_comm,
    Index* queue,
    bool* in_queue
) {
    const Real m = total_weight;
    bool any_move = false;

    // Initialize queue with all nodes
    Size queue_head = 0;
    Size queue_tail = n;
    for (Size i = 0; i < n; ++i) {
        queue[i] = static_cast<Index>(i);
        in_queue[i] = true;
    }

    while (queue_head < queue_tail) {
        Index i = queue[queue_head++];
        in_queue[i] = false;

        const Index current_comm = node_to_comm[i];
        const Real k_i = degrees[i];

        if (k_i <= Real(0)) continue;

        auto indices = adj.primary_indices(i);
        auto values = adj.primary_values(i);
        const Index len = adj.primary_length(i);

        // Collect neighbor communities and weights
        Index n_neighbor_comms = 0;
        Real k_i_current = Real(0);

        for (Index k = 0; k < len; ++k) {
            Index j = indices[k];
            Index comm_j = node_to_comm[j];
            Real w = static_cast<Real>(values[k]);

            if (comm_j == current_comm) {
                k_i_current += w;
            }

            bool found = false;
            for (Index c = 0; c < n_neighbor_comms; ++c) {
                if (neighbor_comms[c] == comm_j) {
                    k_i_to_comm[c] += w;
                    found = true;
                    break;
                }
            }

            if (!found) {
                neighbor_comms[n_neighbor_comms] = comm_j;
                k_i_to_comm[n_neighbor_comms] = w;
                ++n_neighbor_comms;
            }
        }

        // Find best community
        Index best_comm = current_comm;
        Real best_gain = Real(0);

        Real sigma_current_without = sigma_tot[current_comm] - k_i;
        Real loss = modularity_gain(k_i, k_i_current, sigma_current_without, m, resolution);

        for (Index c = 0; c < n_neighbor_comms; ++c) {
            Index target_comm = neighbor_comms[c];
            if (target_comm == current_comm) continue;

            Real sigma_target = sigma_tot[target_comm];
            Real k_i_target = k_i_to_comm[c];

            Real gain = modularity_gain(k_i, k_i_target, sigma_target, m, resolution);
            Real delta_q = gain - loss;

            if (delta_q > best_gain + config::MODULARITY_EPSILON) {
                best_gain = delta_q;
                best_comm = target_comm;
            }
        }

        // Move node if beneficial
        if (best_comm != current_comm) {
            sigma_tot[current_comm] -= k_i;
            sigma_tot[best_comm] += k_i;
            node_to_comm[i] = best_comm;

            any_move = true;

            // Add neighbors to queue
            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                if (!in_queue[j]) {
                    queue[queue_tail++] = j;
                    in_queue[j] = true;
                }
            }
        }

        // Reset workspace
        for (Index c = 0; c < n_neighbor_comms; ++c) {
            k_i_to_comm[c] = Real(0);
        }
    }

    return any_move;
}

// =============================================================================
// Refinement Phase: Refine partition within communities
// =============================================================================

template <typename T, bool IsCSR>
void refinement_phase(
    const Sparse<T, IsCSR>& adj,
    const Real* degrees,
    const Index* partition,   // Current partition
    Index* refined,           // Output: refined partition
    Real* sigma_tot,
    Size n,
    Index n_communities,
    Real total_weight,
    Real resolution,
    Index* neighbor_comms,
    Real* k_i_to_comm,
    FastRNG& rng
) {
    const Real m = total_weight;
    const Real theta = config::THETA;

    // Initialize refined partition: each node in singleton
    Index next_comm = 0;
    for (Size i = 0; i < n; ++i) {
        refined[i] = next_comm++;
        sigma_tot[i] = degrees[i];
    }

    // Process each community in the partition
    Index* comm_nodes = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Index* node_order = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);

    for (Index c = 0; c < n_communities; ++c) {
        // Collect nodes in this community
        Index comm_size = 0;
        for (Size i = 0; i < n; ++i) {
            if (partition[i] == c) {
                comm_nodes[comm_size++] = static_cast<Index>(i);
            }
        }

        if (comm_size <= 1) continue;

        // Create random order
        for (Index i = 0; i < comm_size; ++i) {
            node_order[i] = i;
        }
        for (Index i = comm_size - 1; i > 0; --i) {
            Index j = static_cast<Index>(rng.bounded(i + 1));
            Index tmp = node_order[i];
            node_order[i] = node_order[j];
            node_order[j] = tmp;
        }

        // Try merging nodes within this community
        for (Index idx = 0; idx < comm_size; ++idx) {
            Index i = comm_nodes[node_order[idx]];
            const Real k_i = degrees[i];

            if (k_i <= Real(0)) continue;

            auto indices = adj.primary_indices(i);
            auto values = adj.primary_values(i);
            const Index len = adj.primary_length(i);

            // Find candidate communities (refined communities of neighbors in same partition)
            Index n_candidates = 0;

            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                if (partition[j] != c) continue;  // Only consider nodes in same partition
                if (j == i) continue;

                Index ref_comm = refined[j];
                Real w = static_cast<Real>(values[k]);

                bool found = false;
                for (Index cc = 0; cc < n_candidates; ++cc) {
                    if (neighbor_comms[cc] == ref_comm) {
                        k_i_to_comm[cc] += w;
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    neighbor_comms[n_candidates] = ref_comm;
                    k_i_to_comm[n_candidates] = w;
                    ++n_candidates;
                }
            }

            if (n_candidates == 0) {
                for (Index cc = 0; cc < n_candidates; ++cc) {
                    k_i_to_comm[cc] = Real(0);
                }
                continue;
            }

            // Find best community to merge into
            Index current_ref = refined[i];
            Index best_comm = current_ref;
            Real best_gain = Real(0);

            Real sigma_current = sigma_tot[current_ref];

            for (Index cc = 0; cc < n_candidates; ++cc) {
                Index target_ref = neighbor_comms[cc];
                if (target_ref == current_ref) continue;

                Real k_i_in = k_i_to_comm[cc];
                Real sigma_target = sigma_tot[target_ref];

                // Simplified gain calculation for refinement
                Real gain = k_i_in / m - resolution * k_i * sigma_target / (Real(2) * m * m);

                // Apply randomness: accept with probability proportional to exp(gain/theta)
                if (gain > Real(0) || rng.uniform() < std::exp(gain / theta)) {
                    if (gain > best_gain) {
                        best_gain = gain;
                        best_comm = target_ref;
                    }
                }
            }

            // Merge if beneficial
            if (best_comm != current_ref && best_gain > config::MODULARITY_EPSILON) {
                sigma_tot[current_ref] -= k_i;
                sigma_tot[best_comm] += k_i;
                refined[i] = best_comm;
            }

            // Reset workspace
            for (Index cc = 0; cc < n_candidates; ++cc) {
                k_i_to_comm[cc] = Real(0);
            }
        }
    }

    scl::memory::aligned_free(node_order, SCL_ALIGNMENT);
    scl::memory::aligned_free(comm_nodes, SCL_ALIGNMENT);
}

// =============================================================================
// Relabel communities to be contiguous
// =============================================================================

SCL_FORCE_INLINE Index relabel_communities(
    Index* node_to_comm,
    Size n,
    Index* old_to_new
) {
    for (Size i = 0; i < n; ++i) {
        old_to_new[i] = -1;
    }

    Index next_label = 0;
    for (Size i = 0; i < n; ++i) {
        Index old_comm = node_to_comm[i];
        if (old_to_new[old_comm] == -1) {
            old_to_new[old_comm] = next_label++;
        }
        node_to_comm[i] = old_to_new[old_comm];
    }

    return next_label;
}

// =============================================================================
// Build aggregated graph
// =============================================================================

template <typename T, bool IsCSR>
Index aggregate_graph(
    const Sparse<T, IsCSR>& adj,
    const Index* node_to_comm,
    Index n_communities,
    Index* agg_indptr,
    Index* agg_indices,
    T* agg_values,
    Index* comm_edge_targets,
    T* comm_edge_weights
) {
    const Index n = adj.primary_dim();

    agg_indptr[0] = 0;
    Index total_edges = 0;

    for (Index c = 0; c < n_communities; ++c) {
        Index n_comm_edges = 0;

        for (Index i = 0; i < n; ++i) {
            if (node_to_comm[i] != c) continue;

            auto indices = adj.primary_indices(i);
            auto values = adj.primary_values(i);
            const Index len = adj.primary_length(i);

            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                Index target_comm = node_to_comm[j];
                T w = values[k];

                bool found = false;
                for (Index e = 0; e < n_comm_edges; ++e) {
                    if (comm_edge_targets[e] == target_comm) {
                        comm_edge_weights[e] += w;
                        found = true;
                        break;
                    }
                }

                if (!found) {
                    comm_edge_targets[n_comm_edges] = target_comm;
                    comm_edge_weights[n_comm_edges] = w;
                    ++n_comm_edges;
                }
            }
        }

        for (Index e = 0; e < n_comm_edges; ++e) {
            agg_indices[total_edges + e] = comm_edge_targets[e];
            agg_values[total_edges + e] = comm_edge_weights[e];
        }

        total_edges += n_comm_edges;
        agg_indptr[c + 1] = total_edges;

        for (Index e = 0; e < n_comm_edges; ++e) {
            comm_edge_weights[e] = T(0);
        }
    }

    return total_edges;
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

    SCL_CHECK_DIM(labels.len >= static_cast<Size>(n),
                  "Leiden: labels buffer too small");

    if (n == 0) return;

    if (n == 1) {
        labels[0] = 0;
        return;
    }

    Real total_weight = detail::compute_total_weight(adjacency);
    if (total_weight <= Real(0)) {
        for (Index i = 0; i < n; ++i) {
            labels[i] = i;
        }
        return;
    }

    // Allocate memory
    Real* degrees = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* sigma_tot = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Index* node_to_comm = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Index* refined = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Index* neighbor_comms = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Real* k_i_to_comm = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Index* old_to_new = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Index* queue = scl::memory::aligned_alloc<Index>(2 * n, SCL_ALIGNMENT);
    bool* in_queue = reinterpret_cast<bool*>(scl::memory::aligned_alloc<char>(n, SCL_ALIGNMENT));

    detail::compute_node_degrees(adjacency, degrees);

    // Initialize: each node in its own community
    for (Index i = 0; i < n; ++i) {
        node_to_comm[i] = i;
        sigma_tot[i] = degrees[i];
    }

    scl::algo::zero(k_i_to_comm, static_cast<Size>(n));
    std::memset(in_queue, 0, n);

    detail::FastRNG rng(seed);

    // Main Leiden loop
    for (Index iter = 0; iter < max_iter; ++iter) {
        // Phase 1: Local moving
        bool moved = detail::local_moving_phase(
            adjacency, degrees, node_to_comm, sigma_tot, n,
            total_weight, resolution,
            neighbor_comms, k_i_to_comm, queue, in_queue
        );

        if (!moved) break;

        // Relabel
        Index n_communities = detail::relabel_communities(node_to_comm, n, old_to_new);
        detail::compute_community_stats(node_to_comm, degrees, n, sigma_tot, n_communities);

        if (n_communities >= n || n_communities <= 1) break;

        // Phase 2: Refinement
        detail::refinement_phase(
            adjacency, degrees, node_to_comm, refined, sigma_tot, n,
            n_communities, total_weight, resolution,
            neighbor_comms, k_i_to_comm, rng
        );

        // Relabel refined partition
        Index n_refined = detail::relabel_communities(refined, n, old_to_new);

        if (n_refined >= n_communities) {
            // No refinement improvement, use partition as-is
            for (Index i = 0; i < n; ++i) {
                labels[i] = node_to_comm[i];
            }
        } else {
            // Use refined partition
            for (Index i = 0; i < n; ++i) {
                node_to_comm[i] = refined[i];
            }
            detail::compute_community_stats(node_to_comm, degrees, n, sigma_tot, n_refined);
        }

        // Check for convergence
        Index final_n = detail::relabel_communities(node_to_comm, n, old_to_new);
        if (final_n >= n_communities) break;
    }

    // Final relabeling
    detail::relabel_communities(node_to_comm, n, old_to_new);

    // Copy to output
    for (Index i = 0; i < n; ++i) {
        labels[i] = node_to_comm[i];
    }

    // Cleanup
    scl::memory::aligned_free(reinterpret_cast<char*>(in_queue), SCL_ALIGNMENT);
    scl::memory::aligned_free(queue, SCL_ALIGNMENT);
    scl::memory::aligned_free(old_to_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(k_i_to_comm, SCL_ALIGNMENT);
    scl::memory::aligned_free(neighbor_comms, SCL_ALIGNMENT);
    scl::memory::aligned_free(refined, SCL_ALIGNMENT);
    scl::memory::aligned_free(node_to_comm, SCL_ALIGNMENT);
    scl::memory::aligned_free(sigma_tot, SCL_ALIGNMENT);
    scl::memory::aligned_free(degrees, SCL_ALIGNMENT);
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

    Real total_weight = detail::compute_total_weight(adjacency);
    if (total_weight <= Real(0)) return Real(0);

    const Real m = total_weight;
    const Real m2 = Real(2) * m;

    Real* degrees = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    detail::compute_node_degrees(adjacency, degrees);

    Index max_comm = 0;
    for (Index i = 0; i < n; ++i) {
        max_comm = scl::algo::max2(max_comm, labels[i]);
    }
    Index n_communities = max_comm + 1;

    Real* sigma_tot = scl::memory::aligned_alloc<Real>(n_communities, SCL_ALIGNMENT);
    scl::algo::zero(sigma_tot, static_cast<Size>(n_communities));

    for (Index i = 0; i < n; ++i) {
        sigma_tot[labels[i]] += degrees[i];
    }

    Real Q = Real(0);

    for (Index i = 0; i < n; ++i) {
        auto indices = adjacency.primary_indices(i);
        auto values = adjacency.primary_values(i);
        const Index len = adjacency.primary_length(i);

        for (Index k = 0; k < len; ++k) {
            if (labels[indices[k]] == labels[i]) {
                Q += static_cast<Real>(values[k]);
            }
        }
    }
    Q /= m2;

    for (Index c = 0; c < n_communities; ++c) {
        Real s = sigma_tot[c];
        Q -= resolution * s * s / (m2 * m2);
    }

    scl::memory::aligned_free(sigma_tot, SCL_ALIGNMENT);
    scl::memory::aligned_free(degrees, SCL_ALIGNMENT);

    return Q;
}

// =============================================================================
// Get Community Sizes
// =============================================================================

inline void community_sizes(
    Array<const Index> labels,
    Array<Index> sizes,
    Index& n_communities
) {
    const Size n = labels.len;

    Index max_label = 0;
    for (Size i = 0; i < n; ++i) {
        max_label = scl::algo::max2(max_label, labels[i]);
    }
    n_communities = max_label + 1;

    SCL_CHECK_DIM(sizes.len >= static_cast<Size>(n_communities),
                  "Leiden: sizes buffer too small");

    scl::algo::zero(sizes.ptr, static_cast<Size>(n_communities));

    for (Size i = 0; i < n; ++i) {
        ++sizes[labels[i]];
    }
}

} // namespace scl::kernel::leiden
