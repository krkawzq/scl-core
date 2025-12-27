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
// FILE: scl/kernel/louvain.hpp
// BRIEF: Louvain clustering algorithm for community detection in sparse graphs
// =============================================================================

namespace scl::kernel::louvain {

// =============================================================================
// Configuration
// =============================================================================

namespace config {
    constexpr Real DEFAULT_RESOLUTION = Real(1.0);
    constexpr Index DEFAULT_MAX_ITER = 100;
    constexpr Real MODULARITY_EPSILON = Real(1e-8);
    constexpr Size PARALLEL_THRESHOLD = 1000;
    constexpr Index MAX_LEVELS = 100;
}

// =============================================================================
// Internal Data Structures
// =============================================================================

namespace detail {

struct CommunityInfo {
    Real* sigma_tot;     // Total weight (degree sum) of each community
    Index* comm_size;    // Number of nodes in each community
    Index* node_to_comm; // Community assignment for each node
    Index n_communities; // Current number of non-empty communities
    Size n_nodes;        // Original number of nodes
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

    return total / Real(2);  // Each edge counted twice in undirected graph
}

// =============================================================================
// Helper: Compute node degrees (weighted)
// =============================================================================

template <typename T, bool IsCSR>
void compute_node_degrees(
    const Sparse<T, IsCSR>& adj,
    Real* degrees
) {
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
// Helper: Initialize community info
// =============================================================================

SCL_FORCE_INLINE void init_communities(
    CommunityInfo& info,
    const Real* degrees,
    Size n
) {
    info.n_nodes = n;
    info.n_communities = static_cast<Index>(n);

    for (Size i = 0; i < n; ++i) {
        info.node_to_comm[i] = static_cast<Index>(i);
        info.sigma_tot[i] = degrees[i];
        info.comm_size[i] = 1;
    }
}

// =============================================================================
// Helper: Compute weight from node to community
// =============================================================================

template <typename T, bool IsCSR>
SCL_FORCE_INLINE Real compute_k_i_in(
    const Sparse<T, IsCSR>& adj,
    Index node,
    Index target_comm,
    const Index* node_to_comm
) {
    auto indices = adj.primary_indices(node);
    auto values = adj.primary_values(node);
    const Index len = adj.primary_length(node);

    Real k_in = Real(0);
    for (Index k = 0; k < len; ++k) {
        if (node_to_comm[indices[k]] == target_comm) {
            k_in += static_cast<Real>(values[k]);
        }
    }

    return k_in;
}

// =============================================================================
// Helper: Compute modularity gain
// =============================================================================

SCL_FORCE_INLINE Real modularity_gain(
    Real k_i,              // Degree of node
    Real k_i_in,           // Weight from node to target community
    Real sigma_tot,        // Total weight of target community
    Real m,                // Total graph weight
    Real resolution
) {
    // ΔQ = k_i_in / m - resolution * k_i * sigma_tot / (2 * m^2)
    Real m2 = Real(2) * m;
    return k_i_in / m - resolution * k_i * sigma_tot / (m2 * m);
}

// =============================================================================
// Local Moving Phase: Try moving each node to best neighbor community
// =============================================================================

template <typename T, bool IsCSR>
bool local_moving_phase(
    const Sparse<T, IsCSR>& adj,
    const Real* degrees,
    CommunityInfo& info,
    Real total_weight,
    Real resolution,
    Index* neighbor_comms,   // Workspace: unique neighbor communities
    Real* k_i_to_comm        // Workspace: weight to each neighbor comm
) {
    const Index n = static_cast<Index>(info.n_nodes);
    const Real m = total_weight;

    bool any_move = false;
    bool improved = true;

    while (improved) {
        improved = false;

        for (Index i = 0; i < n; ++i) {
            const Index current_comm = info.node_to_comm[i];
            const Real k_i = degrees[i];

            if (k_i <= Real(0)) continue;

            // Get neighbors and find unique communities
            auto indices = adj.primary_indices(i);
            auto values = adj.primary_values(i);
            const Index len = adj.primary_length(i);

            // Reset workspace
            Index n_neighbor_comms = 0;

            // Compute weight to current community and collect neighbor communities
            Real k_i_current = Real(0);

            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                Index comm_j = info.node_to_comm[j];
                Real w = static_cast<Real>(values[k]);

                if (comm_j == current_comm) {
                    k_i_current += w;
                }

                // Check if we've seen this community
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

            // Find best community to move to
            Index best_comm = current_comm;
            Real best_gain = Real(0);

            // Sigma_tot of current community without node i
            Real sigma_current_without = info.sigma_tot[current_comm] - k_i;

            // Modularity loss from leaving current community
            Real loss = modularity_gain(k_i, k_i_current, sigma_current_without, m, resolution);

            for (Index c = 0; c < n_neighbor_comms; ++c) {
                Index target_comm = neighbor_comms[c];
                if (target_comm == current_comm) continue;

                Real sigma_target = info.sigma_tot[target_comm];
                Real k_i_target = k_i_to_comm[c];

                // Modularity gain from joining target community
                Real gain = modularity_gain(k_i, k_i_target, sigma_target, m, resolution);
                Real delta_q = gain - loss;

                if (delta_q > best_gain + config::MODULARITY_EPSILON) {
                    best_gain = delta_q;
                    best_comm = target_comm;
                }
            }

            // Move node if beneficial
            if (best_comm != current_comm) {
                // Update community info
                info.sigma_tot[current_comm] -= k_i;
                info.sigma_tot[best_comm] += k_i;
                info.comm_size[current_comm] -= 1;
                info.comm_size[best_comm] += 1;
                info.node_to_comm[i] = best_comm;

                improved = true;
                any_move = true;
            }

            // Reset k_i_to_comm for next iteration
            for (Index c = 0; c < n_neighbor_comms; ++c) {
                k_i_to_comm[c] = Real(0);
            }
        }
    }

    return any_move;
}

// =============================================================================
// Relabel communities to be contiguous 0, 1, 2, ...
// =============================================================================

SCL_FORCE_INLINE Index relabel_communities(
    Index* node_to_comm,
    Size n,
    Index* old_to_new  // Workspace, size = n
) {
    // Initialize mapping to invalid
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
// Build aggregated graph where nodes are communities
// =============================================================================

template <typename T, bool IsCSR>
Index aggregate_graph(
    const Sparse<T, IsCSR>& adj,
    const Index* node_to_comm,
    Index n_communities,
    Index* agg_indptr,
    Index* agg_indices,
    T* agg_values,
    Index* comm_edge_targets,  // Workspace
    T* comm_edge_weights       // Workspace
) {
    const Index n = adj.primary_dim();

    // For each community, collect edges to other communities
    agg_indptr[0] = 0;
    Index total_edges = 0;

    for (Index c = 0; c < n_communities; ++c) {
        Index n_comm_edges = 0;

        // Find all nodes in this community and their neighbors
        for (Index i = 0; i < n; ++i) {
            if (node_to_comm[i] != c) continue;

            auto indices = adj.primary_indices(i);
            auto values = adj.primary_values(i);
            const Index len = adj.primary_length(i);

            for (Index k = 0; k < len; ++k) {
                Index j = indices[k];
                Index target_comm = node_to_comm[j];
                T w = values[k];

                // Find or add edge to target community
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

        // Copy to CSR arrays
        for (Index e = 0; e < n_comm_edges; ++e) {
            agg_indices[total_edges + e] = comm_edge_targets[e];
            agg_values[total_edges + e] = comm_edge_weights[e];
        }

        total_edges += n_comm_edges;
        agg_indptr[c + 1] = total_edges;

        // Reset workspace
        for (Index e = 0; e < n_comm_edges; ++e) {
            comm_edge_weights[e] = T(0);
        }
    }

    return total_edges;
}

} // namespace detail

// =============================================================================
// Main Louvain Clustering Function
// =============================================================================

template <typename T, bool IsCSR>
void cluster(
    const Sparse<T, IsCSR>& adjacency,
    Array<Index> labels,
    Real resolution = config::DEFAULT_RESOLUTION,
    Index max_iter = config::DEFAULT_MAX_ITER
) {
    const Index n = adjacency.primary_dim();

    SCL_CHECK_DIM(labels.len >= static_cast<Size>(n),
                  "Louvain: labels buffer too small");

    if (n == 0) return;

    if (n == 1) {
        labels[0] = 0;
        return;
    }

    // Compute total graph weight
    Real total_weight = detail::compute_total_weight(adjacency);
    if (total_weight <= Real(0)) {
        // No edges - each node is its own community
        for (Index i = 0; i < n; ++i) {
            labels[i] = i;
        }
        return;
    }

    // Allocate working memory
    Real* degrees = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Real* sigma_tot = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Index* comm_size = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Index* node_to_comm = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Index* neighbor_comms = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
    Real* k_i_to_comm = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    Index* old_to_new = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);

    // Compute node degrees
    detail::compute_node_degrees(adjacency, degrees);

    // Initialize community info
    detail::CommunityInfo info;
    info.sigma_tot = sigma_tot;
    info.comm_size = comm_size;
    info.node_to_comm = node_to_comm;
    detail::init_communities(info, degrees, n);

    // Initialize k_i_to_comm workspace
    scl::algo::zero(k_i_to_comm, static_cast<Size>(n));

    // First level: operate on original graph
    bool improved = detail::local_moving_phase(
        adjacency, degrees, info, total_weight, resolution,
        neighbor_comms, k_i_to_comm
    );

    // Relabel communities
    Index n_communities = detail::relabel_communities(node_to_comm, n, old_to_new);

    // Copy to output labels
    for (Index i = 0; i < n; ++i) {
        labels[i] = node_to_comm[i];
    }

    // Multi-level iteration (aggregate and repeat)
    if (improved && n_communities > 1 && n_communities < n) {
        // Allocate for aggregated graph
        Size max_edges = adjacency.nnz();
        Index* agg_indptr = scl::memory::aligned_alloc<Index>(n + 1, SCL_ALIGNMENT);
        Index* agg_indices = scl::memory::aligned_alloc<Index>(max_edges, SCL_ALIGNMENT);
        T* agg_values = scl::memory::aligned_alloc<T>(max_edges, SCL_ALIGNMENT);
        Index* comm_edge_targets = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
        T* comm_edge_weights = scl::memory::aligned_alloc<T>(n, SCL_ALIGNMENT);

        // Track original node to community mapping
        Index* original_labels = scl::memory::aligned_alloc<Index>(n, SCL_ALIGNMENT);
        for (Index i = 0; i < n; ++i) {
            original_labels[i] = labels[i];
        }

        scl::algo::zero(comm_edge_weights, static_cast<Size>(n));

        Index level = 1;
        Index current_n = n_communities;

        while (level < max_iter && improved && current_n > 1) {
            // Build aggregated graph
            Index agg_nnz = detail::aggregate_graph(
                adjacency, original_labels, current_n,
                agg_indptr, agg_indices, agg_values,
                comm_edge_targets, comm_edge_weights
            );

            if (agg_nnz == 0) break;

            // Create sparse wrapper for aggregated graph
            Sparse<T, IsCSR> agg_adj(
                agg_indptr, agg_indices, agg_values,
                current_n, current_n, agg_nnz
            );

            // Compute degrees for aggregated graph
            Real* agg_degrees = scl::memory::aligned_alloc<Real>(current_n, SCL_ALIGNMENT);
            detail::compute_node_degrees(agg_adj, agg_degrees);

            // Re-initialize community info for aggregated level
            detail::CommunityInfo agg_info;
            agg_info.sigma_tot = sigma_tot;
            agg_info.comm_size = comm_size;
            agg_info.node_to_comm = node_to_comm;
            agg_info.n_nodes = current_n;
            detail::init_communities(agg_info, agg_degrees, current_n);

            scl::algo::zero(k_i_to_comm, static_cast<Size>(current_n));

            // Local moving on aggregated graph
            improved = detail::local_moving_phase(
                agg_adj, agg_degrees, agg_info, total_weight, resolution,
                neighbor_comms, k_i_to_comm
            );

            scl::memory::aligned_free(agg_degrees, SCL_ALIGNMENT);

            if (!improved) break;

            // Relabel
            Index new_n_communities = detail::relabel_communities(
                node_to_comm, current_n, old_to_new
            );

            if (new_n_communities >= current_n) break;

            // Update original labels: map through aggregated community
            for (Index i = 0; i < n; ++i) {
                Index agg_comm = original_labels[i];
                original_labels[i] = node_to_comm[agg_comm];
            }

            current_n = new_n_communities;
            ++level;
        }

        // Copy final labels
        for (Index i = 0; i < n; ++i) {
            labels[i] = original_labels[i];
        }

        scl::memory::aligned_free(original_labels, SCL_ALIGNMENT);
        scl::memory::aligned_free(comm_edge_weights, SCL_ALIGNMENT);
        scl::memory::aligned_free(comm_edge_targets, SCL_ALIGNMENT);
        scl::memory::aligned_free(agg_values, SCL_ALIGNMENT);
        scl::memory::aligned_free(agg_indices, SCL_ALIGNMENT);
        scl::memory::aligned_free(agg_indptr, SCL_ALIGNMENT);
    }

    // Cleanup
    scl::memory::aligned_free(old_to_new, SCL_ALIGNMENT);
    scl::memory::aligned_free(k_i_to_comm, SCL_ALIGNMENT);
    scl::memory::aligned_free(neighbor_comms, SCL_ALIGNMENT);
    scl::memory::aligned_free(node_to_comm, SCL_ALIGNMENT);
    scl::memory::aligned_free(comm_size, SCL_ALIGNMENT);
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

    // Compute degree of each node
    Real* degrees = scl::memory::aligned_alloc<Real>(n, SCL_ALIGNMENT);
    detail::compute_node_degrees(adjacency, degrees);

    // Find number of communities
    Index max_comm = 0;
    for (Index i = 0; i < n; ++i) {
        max_comm = scl::algo::max2(max_comm, labels[i]);
    }
    Index n_communities = max_comm + 1;

    // Compute sigma_tot for each community
    Real* sigma_tot = scl::memory::aligned_alloc<Real>(n_communities, SCL_ALIGNMENT);
    scl::algo::zero(sigma_tot, static_cast<Size>(n_communities));

    for (Index i = 0; i < n; ++i) {
        sigma_tot[labels[i]] += degrees[i];
    }

    // Compute modularity
    Real Q = Real(0);

    // Sum of edge weights within communities
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
    Q /= m2;  // Divided by 2m, and each edge counted twice

    // Subtract expected edges under null model
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

    // Find max label
    Index max_label = 0;
    for (Size i = 0; i < n; ++i) {
        max_label = scl::algo::max2(max_label, labels[i]);
    }
    n_communities = max_label + 1;

    SCL_CHECK_DIM(sizes.len >= static_cast<Size>(n_communities),
                  "Louvain: sizes buffer too small");

    scl::algo::zero(sizes.ptr, static_cast<Size>(n_communities));

    for (Size i = 0; i < n; ++i) {
        ++sizes[labels[i]];
    }
}

// =============================================================================
// Get Nodes in Specific Community
// =============================================================================

inline void get_community_members(
    Array<const Index> labels,
    Index community,
    Array<Index> members,
    Index& n_members
) {
    n_members = 0;

    for (Size i = 0; i < labels.len; ++i) {
        if (labels[i] == community) {
            if (static_cast<Size>(n_members) < members.len) {
                members[n_members] = static_cast<Index>(i);
            }
            ++n_members;
        }
    }
}

} // namespace scl::kernel::louvain
